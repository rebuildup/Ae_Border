#include "Border.h"
#include <vector>
#include <algorithm>
#include <cfloat>
#include <limits>
#include <cstring>
#include <thread>
#include <cstdlib>
#include <cerrno>
#include <cstdint>
#include <atomic>

#ifdef _WIN32
#include <omp.h>
#undef min
#undef max
#endif

// ============================================================================
// Constants
// ============================================================================
namespace BorderConstants {
    // EDT/SDF computation
    static constexpr int EDT_INFINITY = 1 << 29;

    // Alpha/gradient thresholds (epsilon values for floating-point comparisons)
    static constexpr float ALPHA_EPSILON = 0.01f;         // For edge detection: alpha <= 0.01 or >= 0.99
    static constexpr float ALPHA_NEAR_OPAQUE = 0.99f;      // Alpha value considered "fully opaque"
    static constexpr float ALPHA_NEAR_TRANSPARENT = 0.01f; // Alpha value considered "fully transparent"
    static constexpr float ALPHA_MIDPOINT = 0.5f;          // Midpoint for solid/empty classification
    static constexpr float GRADIENT_EPSILON = 0.01f;       // Minimum gradient magnitude for edge normal calculation
    static constexpr float GRADIENT_EPSILON_FINE = 0.001f; // Finer epsilon for subpixel calculations

    // Coordinate clamping
    static constexpr float COORD_EPSILON = 0.001f;         // Small offset to prevent out-of-bounds sampling

    // Gaussian blur kernel weights (3x3 with sigma ~0.85)
    static constexpr float GAUSSIAN_KERNEL_DIVISOR = 16.0f;

    // Stroke computation
    static constexpr float STROKE_HALF = 0.5f;             // Half stroke thickness for DIRECTION_BOTH
    static constexpr float AA_RANGE_PX = 1.0f;             // Anti-aliasing range in pixels
    static constexpr float SAMPLE_GUARD_PX = 1.0f;         // Sample guard band in pixels
    static constexpr float COVERAGE_EPSILON = 0.001f;      // Minimum stroke coverage to process a pixel

    // Subpixel sampling
    static constexpr float SUBSAMPLE_OFFSET = 0.5f;        // Offset from pixel center to subpixel center
    static constexpr float SUBSAMPLE_SCALE = 0.5f;         // Scale for subpixel offset calculation
    static constexpr float SUBSAMPLE_4X_INV = 0.25f;       // Inverse of 4 for 2x2 supersampling average

    // Sobel gradient weights
    static constexpr float SOBEL_WEIGHT = 2.0f;            // Weight for center pixel in Sobel kernel
    static constexpr float SOBEL_NORMALIZATION = 8.0f;     // Sobel gradient normalization factor
    static constexpr float SOBEL_INVERSE_NORMALIZATION = (1.0f / 8.0f); // Pre-computed inverse

    // Normalized alpha space
    static constexpr float ALPHA_255 = 255.0f;             // 8-bit max alpha for normalization
    static constexpr float ALPHA_255_INV = (1.0f / 255.0f); // Pre-computed inverse

    // Smoothstep function constants
    static constexpr float SMOOTHSTEP_ZERO = 0.0f;
    static constexpr float SMOOTHSTEP_ONE = 1.0f;

    // Distance scaling for integer chamfer distance
    static constexpr float CHAMFER_SCALE = 10.0f;
    static constexpr float CHAMFER_ROUND = 0.5f;

    // Minimum gradient magnitude clamp
    static constexpr float MIN_GRADIENT_MAGNITUDE = 0.001f;

    // Dithering
    static constexpr float DITHER_CENTER = 0.5f;           // Center value for dithering (0-1 range)

    // AA range for signed distance field evaluation
    static constexpr float AA_RANGE = 0.5f;

    // Color clamping
    static constexpr float COLOR_MIN = 0.0f;
    static constexpr float COLOR_MAX = 1.0f;
}

// Clamp helper used by smoothstep
template <typename T>
constexpr T CLAMP(T value, T min, T max) noexcept {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

inline constexpr float smoothstep(float edge0, float edge1, float x) noexcept {
    const float t = CLAMP((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Fast deterministic dither for 8bpc quantization (returns [0,1]).
// AE's built-in vector renderers often dither when converting from higher precision,
// which helps avoid visible "flat" plateaus in AA ramps at 8bpc.
static inline float BorderDither8(A_long x, A_long y) noexcept {
    // 32-bit integer hash (PCG-ish mix), then take the low 8 bits.
    uint32_t n = static_cast<uint32_t>(x) * 0x1f123bb5u + static_cast<uint32_t>(y) * 0x05491333u + 0x68bc21ebu;
    n ^= n >> 15;
    n *= 0x2c1b3c6du;
    n ^= n >> 12;
    n *= 0x297a2d39u;
    n ^= n >> 15;
    return (static_cast<float>(n & 0xFFu)) / 255.0f;
}

struct EdgePoint {
    A_long x, y;
    bool isTransparent;
};

struct BorderRectI {
    A_long left, top, right, bottom; // inclusive
};

static int BorderGetInternalThreadCount(A_long work_items);

template <typename Fn>
static void BorderParallelFor(A_long count, Fn fn);

template <typename Fn>
static void BorderParallelForRange(A_long count, Fn fn);

static bool
BorderGetSolidBounds(
    const PF_EffectWorld* input,
    A_u_char threshold8,
    A_u_short threshold16,
    BorderRectI& outBounds)
{
    const A_long w = input->width;
    const A_long h = input->height;
    A_long minX = w, minY = h, maxX = -1, maxY = -1;

    const int num_threads = BorderGetInternalThreadCount(h);
    std::vector<A_long> tMinX((size_t)num_threads, w);
    std::vector<A_long> tMinY((size_t)num_threads, h);
    std::vector<A_long> tMaxX((size_t)num_threads, -1);
    std::vector<A_long> tMaxY((size_t)num_threads, -1);

    if (PF_WORLD_IS_DEEP(input)) {
        BorderParallelForRange(h, [&](A_long y0, A_long y1, int tid) {
            A_long lminX = w, lminY = h, lmaxX = -1, lmaxY = -1;
            for (A_long y = y0; y < y1; ++y) {
                PF_Pixel16* row = (PF_Pixel16*)((char*)input->data + y * input->rowbytes);
                for (A_long x = 0; x < w; ++x) {
                    if (row[x].alpha > threshold16) {
                        if (x < lminX) lminX = x;
                        if (y < lminY) lminY = y;
                        if (x > lmaxX) lmaxX = x;
                        if (y > lmaxY) lmaxY = y;
                    }
                }
            }
            tMinX[(size_t)tid] = lminX;
            tMinY[(size_t)tid] = lminY;
            tMaxX[(size_t)tid] = lmaxX;
            tMaxY[(size_t)tid] = lmaxY;
        });
    } else {
        BorderParallelForRange(h, [&](A_long y0, A_long y1, int tid) {
            A_long lminX = w, lminY = h, lmaxX = -1, lmaxY = -1;
            for (A_long y = y0; y < y1; ++y) {
                PF_Pixel8* row = (PF_Pixel8*)((char*)input->data + y * input->rowbytes);
                for (A_long x = 0; x < w; ++x) {
                    if (row[x].alpha > threshold8) {
                        if (x < lminX) lminX = x;
                        if (y < lminY) lminY = y;
                        if (x > lmaxX) lmaxX = x;
                        if (y > lmaxY) lmaxY = y;
                    }
                }
            }
            tMinX[(size_t)tid] = lminX;
            tMinY[(size_t)tid] = lminY;
            tMaxX[(size_t)tid] = lmaxX;
            tMaxY[(size_t)tid] = lmaxY;
        });
    }

    for (int t = 0; t < num_threads; ++t) {
        if (tMaxX[(size_t)t] < 0) continue;
        if (tMinX[(size_t)t] < minX) minX = tMinX[(size_t)t];
        if (tMinY[(size_t)t] < minY) minY = tMinY[(size_t)t];
        if (tMaxX[(size_t)t] > maxX) maxX = tMaxX[(size_t)t];
        if (tMaxY[(size_t)t] > maxY) maxY = tMaxY[(size_t)t];
    }

    if (maxX < minX || maxY < minY) return false;
    outBounds.left = minX;
    outBounds.top = minY;
    outBounds.right = maxX;
    outBounds.bottom = maxY;
    return true;
}

static int
BorderGetInternalThreadCount(A_long work_items)
{
    if (work_items <= 1) return 1;

    // Thread-safe cached value using atomic operations with proper memory ordering
    static std::atomic<int> cached{0};  // 0 = uninitialized
    int value = cached.load(std::memory_order_acquire);
    if (value == 0) {
        // Default to single-threaded internally.
        // After Effects already provides Multi-Frame Rendering (MFR) parallelism; additionally
        // spawning our own threads can oversubscribe CPU and even destabilize some hosts.
        // Enable internal parallelism only when explicitly requested via BORDER_THREADS.
        int threads = 1;
        char* env_buf = nullptr;
#ifdef _WIN32
        size_t len = 0;
        if (_dupenv_s(&env_buf, &len, "BORDER_THREADS") == 0 && env_buf) {
            char* endp = nullptr;
            errno = 0;
            long v = strtol(env_buf, &endp, 10);
            if (errno == 0 && endp != env_buf) threads = static_cast<int>(v);
            free(env_buf);  // Free immediately after use to avoid leak on any path
            env_buf = nullptr;
        }
#else
        if (const char* env = std::getenv("BORDER_THREADS")) {
            char* endp = nullptr;
            errno = 0;
            long v = strtol(env, &endp, 10);
            if (errno == 0 && endp != env) threads = static_cast<int>(v);
        }
#endif
        // env_buf is already freed above on Windows path; safety check removed as it's unreachable

        // If explicitly set to 0 or negative, treat as "off".
        if (threads <= 0) threads = 1;

        // Conservative cap to reduce oversubscription under AE Multi-Frame Rendering.
        if (threads > 8) threads = 8;
        if (threads < 1) threads = 1;

        // Use release semantics to ensure the computed value is visible to other threads
        cached.store(threads, std::memory_order_release);
        value = threads;
    }

    int t = cached.load(std::memory_order_relaxed);
    if (t > static_cast<int>(work_items)) t = static_cast<int>(work_items);
    if (t < 1) t = 1;
    return t;
}

template <typename Fn>
static void
BorderParallelFor(A_long count, Fn fn)
{
    const int num_threads = BorderGetInternalThreadCount(count);
    if (num_threads <= 1) {
        for (A_long i = 0; i < count; ++i) fn(i);
        return;
    }

    const A_long chunk = (count + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    threads.reserve((size_t)num_threads);
    for (int t = 0; t < num_threads; ++t) {
        const A_long start = (A_long)t * chunk;
        const A_long end = MIN(start + chunk, count);
        if (start >= end) break;
        threads.emplace_back([=, &fn]() {
            for (A_long i = start; i < end; ++i) fn(i);
        });
    }
    for (auto& th : threads) th.join();
}

template <typename Fn>
static void
BorderParallelForRange(A_long count, Fn fn)
{
    const int num_threads = BorderGetInternalThreadCount(count);
    if (num_threads <= 1) {
        fn(0, count, 0);
        return;
    }

    const A_long chunk = (count + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    threads.reserve((size_t)num_threads);
    for (int t = 0; t < num_threads; ++t) {
        const A_long start = (A_long)t * chunk;
        const A_long end = MIN(start + chunk, count);
        if (start >= end) break;
        threads.emplace_back([=, &fn]() { fn(start, end, t); });
    }
    for (auto& th : threads) th.join();
}

static PF_Err
About(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    PF_SPRINTF(out_data->return_msg,
        "%s v%d.%d\r%s",
        STR(StrID_Name),
        MAJOR_VERSION,
        MINOR_VERSION,
        STR(StrID_Description));
    return PF_Err_NONE;
}

static PF_Err
GlobalSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    // Use the same packed constant as the PiPL to avoid version mismatch warnings.
    out_data->my_version = BORDER_VERSION_VALUE;

    // Keep these numeric flags in sync with BorderPiPL.r (PiPLTool requires literals).
    // out_flags  = PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_PIX_INDEPENDENT |
    //              PF_OutFlag_USE_OUTPUT_EXTENT | PF_OutFlag_I_EXPAND_BUFFER
    // out_flags2 = PF_OutFlag2_SUPPORTS_THREADED_RENDERING | PF_OutFlag2_REVEALS_ZERO_ALPHA
    static constexpr A_long kOutFlags  = 0x02000640;
    static constexpr A_long kOutFlags2 = 0x08000080;
    out_data->out_flags  = kOutFlags;
    out_data->out_flags2 = kOutFlags2;

    return PF_Err_NONE;
}

static PF_Err
ParamsSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err        err = PF_Err_NONE;
    PF_ParamDef    def;

    AEFX_CLR_STRUCT(def);

    PF_ADD_FLOAT_SLIDERX(STR(StrID_Thickness_Param_Name),
        BORDER_THICKNESS_MIN,
        BORDER_THICKNESS_MAX,
        0,
        100,
        2,
        PF_Precision_TENTHS,
        0,
        0,
        THICKNESS_DISK_ID);

    AEFX_CLR_STRUCT(def);

    def.flags = PF_ParamFlag_NONE;
    def.u.cd.value.red = 255;
    def.u.cd.value.green = 0;
    def.u.cd.value.blue = 0;
    def.u.cd.value.alpha = 255;

    PF_ADD_COLOR(STR(StrID_Color_Param_Name),
        def.u.cd.value.red,
        def.u.cd.value.green,
        def.u.cd.value.blue,
        COLOR_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_SLIDER(STR(StrID_Threshold_Param_Name),
        BORDER_THRESHOLD_MIN,
        BORDER_THRESHOLD_MAX,
        0,
        255,
        0,
        THRESHOLD_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_POPUP(STR(StrID_Direction_Param_Name),
        3,
        DIRECTION_BOTH,
        "Both Directions|Inside|Outside",
        DIRECTION_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_CHECKBOX(STR(StrID_ShowLineOnly_Param_Name),
        "Show line only",
        FALSE,
        0,
        SHOW_LINE_ONLY_DISK_ID);

    out_data->num_params = BORDER_NUM_PARAMS;

    return err;
}

static inline float BorderDownscaleFactor(const PF_RationalScale& scale)
{
    if (scale.den == 0) return 1.0f;
    float v = static_cast<float>(scale.num) / static_cast<float>(scale.den);
    return (v > 0.0f) ? v : 1.0f;
}

static inline void BorderComputePixelThickness(
    PF_InData* in_data,
    PF_FpLong thicknessParam,
    A_long direction,
    float& outStrokeThicknessPx,  // stroke thickness on the chosen side (pixels)
    A_long& outExpansionPx)       // expansion required (pixels)
{
    float numX = static_cast<float>(in_data->downsample_x.num);
    float numY = static_cast<float>(in_data->downsample_y.num);
    if (numX == 0.0f || numY == 0.0f) {
        outStrokeThicknessPx = 0.0f;
        outExpansionPx = 0;
        return;
    }
    float downsize_x = static_cast<float>(in_data->downsample_x.den) / numX;
    float downsize_y = static_cast<float>(in_data->downsample_y.den) / numY;
    float resolution_factor = MIN(downsize_x, downsize_y);

    float pixelThickness;
    if (thicknessParam <= 0.0f) {
        pixelThickness = 0.0f;
    } else if (thicknessParam <= 10.0f) {
        pixelThickness = static_cast<float>(thicknessParam);
    } else {
        float normalizedThickness = static_cast<float>((thicknessParam - 10.0f) / 90.0f);
        pixelThickness = 10.0f + 40.0f * sqrtf(normalizedThickness);
    }

    float outsideThicknessPx = 0.0f;
    if (direction == DIRECTION_OUTSIDE) {
        outsideThicknessPx = pixelThickness;
    } else if (direction == DIRECTION_BOTH) {
        outsideThicknessPx = pixelThickness * BorderConstants::STROKE_HALF;
    }

    // This is used by the renderer for stroke evaluation.
    outStrokeThicknessPx = (direction == DIRECTION_BOTH) ? (pixelThickness * BorderConstants::STROKE_HALF) : pixelThickness;

    // Expand by outside thickness only (inside doesn't need extra pixels).
    if (outsideThicknessPx > 0.0f) {
        outExpansionPx = static_cast<A_long>(ceil((outsideThicknessPx / resolution_factor) + BorderConstants::AA_RANGE_PX + BorderConstants::SAMPLE_GUARD_PX));
    } else {
        outExpansionPx = 0;
    }
}

static PF_Err
FrameSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;

    PF_FpLong thickness = params[BORDER_THICKNESS]->u.fs_d.value;
    A_long direction = params[BORDER_DIRECTION]->u.pd.value;

    float strokeThicknessPx = 0.0f;
    A_long expansionPx = 0;
    BorderComputePixelThickness(in_data, thickness, direction, strokeThicknessPx, expansionPx);
    [[maybe_unused]] float strokeThicknessPxUnused = strokeThicknessPx;

    // IMPORTANT: In PF_Cmd_FRAME_SETUP, the input layer param is not valid per the SDK docs.
    // Use in_data->width/height as the base buffer size for expansion.
    const A_long srcW = in_data->width;
    const A_long srcH = in_data->height;

    if (expansionPx > 0) {
        // Check for integer overflow before computing expanded dimensions
        // expansionPx * 2 could overflow, and adding to srcW could overflow
        if (expansionPx > INT32_MAX / 2) {
            return PF_Err_OUT_OF_MEMORY;
        }
        A_long expansion = expansionPx * 2;
        if (srcW > INT32_MAX - expansion || srcH > INT32_MAX - expansion) {
            return PF_Err_OUT_OF_MEMORY;
        }
        out_data->width = srcW + expansion;
        out_data->height = srcH + expansion;
        out_data->origin.x = expansionPx;
        out_data->origin.y = expansionPx;
    } else {
        out_data->width = srcW;
        out_data->height = srcH;
        out_data->origin.x = 0;
        out_data->origin.y = 0;
    }

    return err;
}
// Computes signed distance to the *alpha edge* (alpha == threshold) with subpixel precision.
//
// Why this exists:
// - The previous method computed distance to the nearest opposite-class pixel *center*, which creates
//   "plateaus" (repeated identical coverage/alpha values) along near-horizontal/vertical strokes.
// - This version:
//   1) Extracts a 4-neighbor boundary set from the thresholded alpha mask.
//   2) Runs an exact Euclidean distance transform (EDT) to the nearest boundary pixel.
//   3) Refines each boundary pixel center into a subpixel edge point by solving alpha(x)=threshold
//      using a Sobel alpha gradient, then measures distance to that edge point.
static PF_Err
ComputeSignedDistanceField_EdgeEDT(
    const PF_EffectWorld* input,
    A_u_char threshold8,
    [[maybe_unused]] A_u_short threshold16,
    std::vector<float>& signedDist,
    A_long width,
    A_long height)
{
    const int INF = BorderConstants::EDT_INFINITY;
    const A_long w = width;
    const A_long h = height;

    // Check for integer overflow in w * h calculation
    if (w > 0 && h > INT32_MAX / w) {
        return PF_Err_OUT_OF_MEMORY;
    }

    signedDist.clear();

    if (w <= 0 || h <= 0) return PF_Err_NONE;

    // Alpha threshold in normalized 0..1 space. Threshold==0 means "auto" (0.5).
    const float thresholdNorm = (threshold8 == 0) ? BorderConstants::ALPHA_MIDPOINT : (threshold8 * BorderConstants::ALPHA_255_INV);

    // Get alpha at a pixel, bilinear interpolated for subpixel positions (normalized 0..1).
    auto getAlpha = [&](float fx, float fy) -> float {
        fx = CLAMP(fx, 0.0f, static_cast<float>(w - 1) - BorderConstants::COORD_EPSILON);
        fy = CLAMP(fy, 0.0f, static_cast<float>(h - 1) - BorderConstants::COORD_EPSILON);

        int x0 = static_cast<int>(fx);
        int y0 = static_cast<int>(fy);
        int x1 = MIN(x0 + 1, w - 1);
        int y1 = MIN(y0 + 1, h - 1);
        float tx = fx - x0;
        float ty = fy - y0;

        if (PF_WORLD_IS_DEEP(input)) {
            PF_Pixel16* row0 = (PF_Pixel16*)((char*)input->data + y0 * input->rowbytes);
            PF_Pixel16* row1 = (PF_Pixel16*)((char*)input->data + y1 * input->rowbytes);
            float a00 = row0[x0].alpha / (float)PF_MAX_CHAN16;
            float a10 = row0[x1].alpha / (float)PF_MAX_CHAN16;
            float a01 = row1[x0].alpha / (float)PF_MAX_CHAN16;
            float a11 = row1[x1].alpha / (float)PF_MAX_CHAN16;
            float a0 = a00 + (a10 - a00) * tx;
            float a1 = a01 + (a11 - a01) * tx;
            return a0 + (a1 - a0) * ty;
        } else {
            PF_Pixel8* row0 = (PF_Pixel8*)((char*)input->data + y0 * input->rowbytes);
            PF_Pixel8* row1 = (PF_Pixel8*)((char*)input->data + y1 * input->rowbytes);
            float a00 = row0[x0].alpha * BorderConstants::ALPHA_255_INV;
            float a10 = row0[x1].alpha * BorderConstants::ALPHA_255_INV;
            float a01 = row1[x0].alpha * BorderConstants::ALPHA_255_INV;
            float a11 = row1[x1].alpha * BorderConstants::ALPHA_255_INV;
            float a0 = a00 + (a10 - a00) * tx;
            float a1 = a01 + (a11 - a01) * tx;
            return a0 + (a1 - a0) * ty;
        }
    };

    auto isSolid = [&](A_long x, A_long y) -> bool {
        float alpha = getAlpha(static_cast<float>(x), static_cast<float>(y));
        return alpha > thresholdNorm;
    };

    // 1D squared EDT (Felzenszwalb & Huttenlocher), with optional nearest-site index output.
    auto edt1d = [&](const int* f, int n, int* d, int* arg, std::vector<int>& v, std::vector<float>& z) {
        bool anyFinite = false;
        for (int i = 0; i < n; ++i) {
            if (f[i] < INF) { anyFinite = true; break; }
        }
        if (!anyFinite) {
            for (int i = 0; i < n; ++i) {
                d[i] = INF;
                if (arg) arg[i] = 0;
            }
            return;
        }

        if (static_cast<int>(v.size()) < n) v.resize(n);
        if (static_cast<int>(z.size()) < n + 1) z.resize(n + 1);

        const float INF_F = std::numeric_limits<float>::infinity();

        int k = 0;
        v[0] = 0;
        z[0] = -INF_F;
        z[1] = INF_F;

        auto sep = [&](int i, int u)->float {
            return ((float)(f[u] + u * u) - (float)(f[i] + i * i)) / (2.0f * (u - i));
        };

        for (int q = 1; q < n; ++q) {
            float s = sep(v[k], q);
            while (k > 0 && s <= z[k]) {
                --k;
                s = sep(v[k], q);
            }
            ++k;
            v[k] = q;
            z[k] = s;
            z[k + 1] = INF_F;
        }

        k = 0;
        for (int q = 0; q < n; ++q) {
            while (z[k + 1] < static_cast<float>(q)) ++k;
            const int site = v[k];
            const int dx = q - site;
            d[q] = dx * dx + f[site];
            if (arg) arg[q] = site;
        }
    };

    // Solid + boundary masks.
    std::vector<A_u_char> solidMask(static_cast<size_t>(w) * static_cast<size_t>(h), 0);
    BorderParallelFor(h, [&](A_long y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (A_long x = 0; x < w; ++x) {
            solidMask[row + static_cast<size_t>(x)] = isSolid(x, y) ? 1 : 0;
        }
    });

    std::vector<A_u_char> edgeMask(static_cast<size_t>(w) * static_cast<size_t>(h), 0);
    BorderParallelFor(h, [&](A_long y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (A_long x = 0; x < w; ++x) {
            const size_t i = row + static_cast<size_t>(x);
            const A_u_char s = solidMask[i];
            bool edge = false;
            if (x > 0) edge |= (solidMask[i - 1] != s);
            if (x + 1 < w) edge |= (solidMask[i + 1] != s);
            if (y > 0) edge |= (solidMask[i - (size_t)w] != s);
            if (y + 1 < h) edge |= (solidMask[i + (size_t)w] != s);
            edgeMask[i] = edge ? 1 : 0;
        }
    });

    bool anyEdge = false;
    for (size_t i = 0; i < edgeMask.size(); ++i) {
        if (edgeMask[i]) { anyEdge = true; break; }
    }
    if (!anyEdge) {
        signedDist.resize(static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f);
        return PF_Err_NONE;
    }

    // Subpixel edge points for boundary pixels by solving alpha(x)=threshold along the alpha gradient.
    std::vector<float> edgeX(static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f);
    std::vector<float> edgeY(static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f);
    BorderParallelFor(h, [&](A_long y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (A_long x = 0; x < w; ++x) {
            const size_t i = row + (size_t)x;
            if (!edgeMask[i]) continue;

            const float ax = (float)x;
            const float ay = (float)y;

            auto a = [&](A_long sx, A_long sy) -> float {
                sx = CLAMP(sx, (A_long)0, w - 1);
                sy = CLAMP(sy, (A_long)0, h - 1);
                return getAlpha((float)sx, (float)sy);
            };

            // Sobel on alpha (normalized).
            const float a00 = a(x - 1, y - 1);
            const float a10 = a(x,     y - 1);
            const float a20 = a(x + 1, y - 1);
            const float a01 = a(x - 1, y);
            const float a21 = a(x + 1, y);
            const float a02 = a(x - 1, y + 1);
            const float a12 = a(x,     y + 1);
            const float a22 = a(x + 1, y + 1);

            const float gradX = (a20 + BorderConstants::SOBEL_WEIGHT * a21 + a22) - (a00 + BorderConstants::SOBEL_WEIGHT * a01 + a02);
            const float gradY = (a02 + BorderConstants::SOBEL_WEIGHT * a12 + a22) - (a00 + BorderConstants::SOBEL_WEIGHT * a10 + a20);

            float gx = gradX * BorderConstants::SOBEL_INVERSE_NORMALIZATION;
            float gy = gradY * BorderConstants::SOBEL_INVERSE_NORMALIZATION;
            float gmag = sqrtf(gx * gx + gy * gy);
            if (!std::isfinite(gmag) || gmag < BorderConstants::MIN_GRADIENT_MAGNITUDE) { gmag = 1.0f; }

            // Fallback normal from mask if alpha gradient is too small (flat alpha regions).
            if (gmag < 1e-5f) {
                float mx = 0.0f;
                float my = 0.0f;
                const A_u_char s = solidMask[i];
                if (x > 0) mx += (solidMask[i - 1] ? 1.0f : 0.0f) - (s ? 1.0f : 0.0f);
                if (x + 1 < w) mx += (solidMask[i + 1] ? 1.0f : 0.0f) - (s ? 1.0f : 0.0f);
                if (y > 0) my += (solidMask[i - (size_t)w] ? 1.0f : 0.0f) - (s ? 1.0f : 0.0f);
                if (y + 1 < h) my += (solidMask[i + (size_t)w] ? 1.0f : 0.0f) - (s ? 1.0f : 0.0f);
                gmag = sqrtf(mx * mx + my * my);
                if (gmag > 1e-5f) {
                    gx = mx / gmag;
                    gy = my / gmag;
                } else {
                    gx = 1.0f;
                    gy = 0.0f;
                    gmag = 1.0f;
                }
            }

            const float nx = gx / gmag;
            const float ny = gy / gmag;
            const float alpha0 = getAlpha(ax, ay);

            float t = (thresholdNorm - alpha0) / gmag;
            t = CLAMP(t, -BorderConstants::STROKE_HALF, BorderConstants::STROKE_HALF);

            edgeX[i] = ax + nx * t;
            edgeY[i] = ay + ny * t;
        }
    });

    // Build 0/INF grid for boundary sites.
    std::vector<int> fEdge(static_cast<size_t>(w) * static_cast<size_t>(h), INF);
    BorderParallelFor(h, [&](A_long y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (A_long x = 0; x < w; ++x) {
            const size_t i = row + (size_t)x;
            if (edgeMask[i]) fEdge[i] = 0;
        }
    });

    // 2D EDT: first pass over rows.
    std::vector<int> tmpD(static_cast<size_t>(w) * static_cast<size_t>(h));
    std::vector<int> tmpArgX(static_cast<size_t>(w) * static_cast<size_t>(h));
    BorderParallelFor(h, [&](A_long y) {
        thread_local std::vector<int> lineIn, lineOut, lineArg, vBuf;
        thread_local std::vector<float> zBuf;
        lineIn.resize(static_cast<size_t>(w));
        lineOut.resize(static_cast<size_t>(w));
        lineArg.resize(static_cast<size_t>(w));

        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (A_long x = 0; x < w; ++x) lineIn[static_cast<size_t>(x)] = fEdge[row + static_cast<size_t>(x)];

        edt1d(lineIn.data(), static_cast<int>(w), lineOut.data(), lineArg.data(), vBuf, zBuf);
        for (A_long x = 0; x < w; ++x) {
            const size_t i = row + static_cast<size_t>(x);
            tmpD[i] = lineOut[static_cast<size_t>(x)];
            tmpArgX[i] = lineArg[static_cast<size_t>(x)];
        }
    });

    // 2D EDT: second pass over columns, propagating nearest site coordinates.
    std::vector<int> nearX(static_cast<size_t>(w) * static_cast<size_t>(h));
    std::vector<int> nearY(static_cast<size_t>(w) * static_cast<size_t>(h));
    BorderParallelFor(w, [&](A_long x) {
        thread_local std::vector<int> lineIn, lineOut, lineArg, vBuf;
        thread_local std::vector<float> zBuf;
        lineIn.resize(static_cast<size_t>(h));
        lineOut.resize(static_cast<size_t>(h));
        lineArg.resize(static_cast<size_t>(h));

        for (A_long y = 0; y < h; ++y) lineIn[static_cast<size_t>(y)] = tmpD[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)];

        edt1d(lineIn.data(), static_cast<int>(h), lineOut.data(), lineArg.data(), vBuf, zBuf);
        for (A_long y = 0; y < h; ++y) {
            const int sy = lineArg[static_cast<size_t>(y)];
            const size_t site = static_cast<size_t>(sy) * static_cast<size_t>(w) + static_cast<size_t>(x);
            const size_t i = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
            nearX[i] = tmpArgX[site];
            nearY[i] = sy;
        }
    });

    signedDist.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    BorderParallelFor(h, [&](A_long y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (A_long x = 0; x < w; ++x) {
            const size_t i = row + (size_t)x;
            const bool solid = (solidMask[i] != 0);

            const int sx = nearX[i];
            const int sy = nearY[i];
            const size_t si = (size_t)sy * (size_t)w + (size_t)sx;

            float ex = (float)sx;
            float ey = (float)sy;
            if (edgeMask[si]) {
                ex = edgeX[si];
                ey = edgeY[si];
            }

            const float dx = static_cast<float>(x) - ex;
            const float dy = static_cast<float>(y) - ey;
            float dist = sqrtf(dx * dx + dy * dy);
            if (!std::isfinite(dist)) dist = 0.0f;
            signedDist[i] = solid ? dist : -dist;
        }
    });

    return PF_Err_NONE;
}

template <typename GetAlphaAtFn>
static void
ComputeSignedDistanceField_EdgeEDT_FromAlpha(
    A_long width,
    A_long height,
    float thresholdNorm,
    GetAlphaAtFn getAlphaAt,
    std::vector<float>& signedDist)
{
    const int INF = BorderConstants::EDT_INFINITY;
    const A_long w = width;
    const A_long h = height;

    // Check for integer overflow in w * h calculation
    if (w > 0 && h > INT32_MAX / w) {
        signedDist.clear();
        return;
    }

    signedDist.clear();
    if (w <= 0 || h <= 0) return;

    // Additional check for size_t overflow
    if (static_cast<size_t>(w) > SIZE_MAX / static_cast<size_t>(h)) {
        signedDist.clear();
        return;
    }

    // Clamp + read.
    auto alphaAt = [&](A_long x, A_long y) -> float {
        x = CLAMP(x, static_cast<A_long>(0), w - 1);
        y = CLAMP(y, static_cast<A_long>(0), h - 1);
        return getAlphaAt(x, y);
    };

    // 1D squared EDT (Felzenszwalb & Huttenlocher), with optional nearest-site index output.
    auto edt1d = [&](const int* f, int n, int* d, int* arg, std::vector<int>& v, std::vector<float>& z) {
        bool anyFinite = false;
        for (int i = 0; i < n; ++i) {
            if (f[i] < INF) { anyFinite = true; break; }
        }
        if (!anyFinite) {
            for (int i = 0; i < n; ++i) {
                d[i] = INF;
                if (arg) arg[i] = 0;
            }
            return;
        }

        if (static_cast<int>(v.size()) < n) v.resize(n);
        if (static_cast<int>(z.size()) < n + 1) z.resize(n + 1);

        const float INF_F = std::numeric_limits<float>::infinity();

        int k = 0;
        v[0] = 0;
        z[0] = -INF_F;
        z[1] = INF_F;

        auto sep = [&](int i, int u)->float {
            return ((static_cast<float>(f[u] + u * u)) - (static_cast<float>(f[i] + i * i))) / (2.0f * (u - i));
        };

        for (int q = 1; q < n; ++q) {
            float s = sep(v[k], q);
            while (k > 0 && s <= z[k]) {
                --k;
                s = sep(v[k], q);
            }
            ++k;
            v[k] = q;
            z[k] = s;
            z[k + 1] = INF_F;
        }

        k = 0;
        for (int q = 0; q < n; ++q) {
            while (z[k + 1] < static_cast<float>(q)) ++k;
            const int site = v[k];
            const int dx = q - site;
            d[q] = dx * dx + f[site];
            if (arg) arg[q] = site;
        }
    };

    // Solid + boundary masks.
    std::vector<A_u_char> solidMask(static_cast<size_t>(w) * static_cast<size_t>(h), 0);
    BorderParallelFor(h, [&](A_long y) {
        const size_t row = (size_t)y * (size_t)w;
        for (A_long x = 0; x < w; ++x) {
            const float a = alphaAt(x, y);
            solidMask[row + (size_t)x] = (a > thresholdNorm) ? 1 : 0;
        }
    });

    std::vector<A_u_char> edgeMask(static_cast<size_t>(w) * static_cast<size_t>(h), 0);
    BorderParallelFor(h, [&](A_long y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (A_long x = 0; x < w; ++x) {
            const size_t i = row + static_cast<size_t>(x);
            const A_u_char s = solidMask[i];
            bool edge = false;
            if (x > 0) edge |= (solidMask[i - 1] != s);
            if (x + 1 < w) edge |= (solidMask[i + 1] != s);
            if (y > 0) edge |= (solidMask[i - (size_t)w] != s);
            if (y + 1 < h) edge |= (solidMask[i + (size_t)w] != s);
            edgeMask[i] = edge ? 1 : 0;
        }
    });

    bool anyEdge = false;
    for (size_t i = 0; i < edgeMask.size(); ++i) {
        if (edgeMask[i]) { anyEdge = true; break; }
    }
    if (!anyEdge) {
        signedDist.resize((size_t)w * (size_t)h, 0.0f);
        return;
    }

    // Subpixel edge points for boundary pixels by solving alpha(x)=threshold along the alpha gradient.
    std::vector<float> edgeX((size_t)w * (size_t)h, 0.0f);
    std::vector<float> edgeY((size_t)w * (size_t)h, 0.0f);
    BorderParallelFor(h, [&](A_long y) {
        const size_t row = (size_t)y * (size_t)w;
        for (A_long x = 0; x < w; ++x) {
            const size_t i = row + (size_t)x;
            if (!edgeMask[i]) continue;

            // Sobel on alpha (normalized).
            const float a00 = alphaAt(x - 1, y - 1);
            const float a10 = alphaAt(x,     y - 1);
            const float a20 = alphaAt(x + 1, y - 1);
            const float a01 = alphaAt(x - 1, y);
            const float a21 = alphaAt(x + 1, y);
            const float a02 = alphaAt(x - 1, y + 1);
            const float a12 = alphaAt(x,     y + 1);
            const float a22 = alphaAt(x + 1, y + 1);

            const float gradX = (a20 + BorderConstants::SOBEL_WEIGHT * a21 + a22) - (a00 + BorderConstants::SOBEL_WEIGHT * a01 + a02);
            const float gradY = (a02 + BorderConstants::SOBEL_WEIGHT * a12 + a22) - (a00 + BorderConstants::SOBEL_WEIGHT * a10 + a20);

            float gx = gradX * BorderConstants::SOBEL_INVERSE_NORMALIZATION;
            float gy = gradY * BorderConstants::SOBEL_INVERSE_NORMALIZATION;
            float gmag = sqrtf(gx * gx + gy * gy);
            if (!std::isfinite(gmag) || gmag < BorderConstants::MIN_GRADIENT_MAGNITUDE) { gmag = 1.0f; }

            // Fallback normal from mask if alpha gradient is too small (flat alpha regions).
            if (gmag < 1e-5f) {
                float mx = 0.0f;
                float my = 0.0f;
                const A_u_char s = solidMask[i];
                if (x > 0) mx += (solidMask[i - 1] ? 1.0f : 0.0f) - (s ? 1.0f : 0.0f);
                if (x + 1 < w) mx += (solidMask[i + 1] ? 1.0f : 0.0f) - (s ? 1.0f : 0.0f);
                if (y > 0) my += (solidMask[i - (size_t)w] ? 1.0f : 0.0f) - (s ? 1.0f : 0.0f);
                if (y + 1 < h) my += (solidMask[i + (size_t)w] ? 1.0f : 0.0f) - (s ? 1.0f : 0.0f);
                gmag = sqrtf(mx * mx + my * my);
                if (gmag > 1e-5f) {
                    gx = mx / gmag;
                    gy = my / gmag;
                } else {
                    gx = 1.0f;
                    gy = 0.0f;
                    gmag = 1.0f;
                }
            }

            const float nx = gx / gmag;
            const float ny = gy / gmag;
            const float alpha0 = alphaAt(x, y);

            float t = (thresholdNorm - alpha0) / gmag;
            t = CLAMP(t, -BorderConstants::STROKE_HALF, BorderConstants::STROKE_HALF);

            edgeX[i] = (float)x + nx * t;
            edgeY[i] = (float)y + ny * t;
        }
    });

    // Build 0/INF grid for boundary sites.
    std::vector<int> fEdge(static_cast<size_t>(w) * static_cast<size_t>(h), INF);
    BorderParallelFor(h, [&](A_long y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (A_long x = 0; x < w; ++x) {
            const size_t i = row + (size_t)x;
            if (edgeMask[i]) fEdge[i] = 0;
        }
    });

    // 2D EDT: first pass over rows.
    std::vector<int> tmpD(static_cast<size_t>(w) * static_cast<size_t>(h));
    std::vector<int> tmpArgX(static_cast<size_t>(w) * static_cast<size_t>(h));
    BorderParallelFor(h, [&](A_long y) {
        thread_local std::vector<int> lineIn, lineOut, lineArg, vBuf;
        thread_local std::vector<float> zBuf;
        lineIn.resize(static_cast<size_t>(w));
        lineOut.resize(static_cast<size_t>(w));
        lineArg.resize(static_cast<size_t>(w));

        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (A_long x = 0; x < w; ++x) lineIn[static_cast<size_t>(x)] = fEdge[row + static_cast<size_t>(x)];

        edt1d(lineIn.data(), static_cast<int>(w), lineOut.data(), lineArg.data(), vBuf, zBuf);
        for (A_long x = 0; x < w; ++x) {
            const size_t i = row + static_cast<size_t>(x);
            tmpD[i] = lineOut[static_cast<size_t>(x)];
            tmpArgX[i] = lineArg[static_cast<size_t>(x)];
        }
    });

    // 2D EDT: second pass over columns, propagating nearest site coordinates.
    std::vector<int> nearX(static_cast<size_t>(w) * static_cast<size_t>(h));
    std::vector<int> nearY(static_cast<size_t>(w) * static_cast<size_t>(h));
    BorderParallelFor(w, [&](A_long x) {
        thread_local std::vector<int> lineIn, lineOut, lineArg, vBuf;
        thread_local std::vector<float> zBuf;
        lineIn.resize(static_cast<size_t>(h));
        lineOut.resize(static_cast<size_t>(h));
        lineArg.resize(static_cast<size_t>(h));

        for (A_long y = 0; y < h; ++y) lineIn[static_cast<size_t>(y)] = tmpD[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)];

        edt1d(lineIn.data(), static_cast<int>(h), lineOut.data(), lineArg.data(), vBuf, zBuf);
        for (A_long y = 0; y < h; ++y) {
            const int sy = lineArg[static_cast<size_t>(y)];
            const size_t site = static_cast<size_t>(sy) * static_cast<size_t>(w) + static_cast<size_t>(x);
            const size_t i = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
            nearX[i] = tmpArgX[site];
            nearY[i] = sy;
        }
    });

    signedDist.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    BorderParallelFor(h, [&](A_long y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (A_long x = 0; x < w; ++x) {
            const size_t i = row + (size_t)x;
            const bool solid = (solidMask[i] != 0);

            const int sx = nearX[i];
            const int sy = nearY[i];
            const size_t si = (size_t)sy * (size_t)w + (size_t)sx;

            float ex = (float)sx;
            float ey = (float)sy;
            if (edgeMask[si]) {
                ex = edgeX[si];
                ey = edgeY[si];
            }

            const float dx = static_cast<float>(x) - ex;
            const float dy = static_cast<float>(y) - ey;
            float dist = sqrtf(dx * dx + dy * dy);
            if (!std::isfinite(dist)) dist = 0.0f;
            signedDist[i] = solid ? dist : -dist;
        }
    });
}

template <typename IsSolidFn>
static void
ComputeSignedDistanceFieldFromSolid(
    A_long width,
    A_long height,
    IsSolidFn isSolid,
    std::vector<int>& signedDist) // scaled by 10
{
    const int INF = BorderConstants::EDT_INFINITY;
    const A_long w = width;
    const A_long h = height;

    // Check for integer overflow in w * h calculation
    if (w > 0 && h > INT32_MAX / w) {
        signedDist.clear();
        return;
    }

    // Additional check for size_t overflow
    if (static_cast<size_t>(w) > SIZE_MAX / static_cast<size_t>(h)) {
        signedDist.clear();
        return;
    }

    signedDist.assign(w * h, 0);

    // 1D squared EDT (Felzenszwalb & Huttenlocher).
    // Uses scratch buffers to avoid per-row/col allocations (important for tiled renders).
    auto edt1d = [&](const int* f, int n, int* d, std::vector<int>& v, std::vector<float>& z) {
        if ((int)v.size() < n) v.resize(n);
        if ((int)z.size() < n + 1) z.resize(n + 1);

        const float INF_F = std::numeric_limits<float>::infinity();

        int k = 0;
        v[0] = 0;
        z[0] = -INF_F;
        z[1] = INF_F;

        auto sep = [&](int i, int u)->float {
            return ((float)(f[u] + u * u) - (float)(f[i] + i * i)) / (2.0f * (u - i));
        };

        for (int q = 1; q < n; ++q) {
            float s = sep(v[k], q);
            while (k > 0 && s <= z[k]) {
                --k;
                s = sep(v[k], q);
            }
            ++k;
            v[k] = q;
            z[k] = s;
            z[k + 1] = INF_F;
        }

        k = 0;
        for (int q = 0; q < n; ++q) {
            while (z[k + 1] < (float)q) ++k;
            int dx = q - v[k];
            d[q] = dx * dx + f[v[k]];
        }
    };

    // Scratch buffer reused across calls (thread-safe across AE's threaded rendering).
    thread_local std::vector<int> tmp;

    auto edt2d = [&](const std::vector<int>& f2d, std::vector<int>& d2d) {
        // Column pass writes every element; no need to prefill with INF.
        d2d.resize(static_cast<size_t>(w) * static_cast<size_t>(h));

        // Row pass writes every element; no need to prefill.
        if (tmp.size() != static_cast<size_t>(w) * static_cast<size_t>(h)) tmp.resize(static_cast<size_t>(w) * static_cast<size_t>(h));

        BorderParallelFor(h, [&](A_long y) {
            thread_local std::vector<int> vScratch;
            thread_local std::vector<float> zScratch;
            thread_local std::vector<int> lineIn;
            thread_local std::vector<int> lineOut;
            lineIn.resize(static_cast<size_t>(w));
            lineOut.resize(static_cast<size_t>(w));
            const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
            for (A_long x = 0; x < w; ++x) lineIn[static_cast<size_t>(x)] = f2d[row + static_cast<size_t>(x)];
            edt1d(lineIn.data(), static_cast<int>(w), lineOut.data(), vScratch, zScratch);
            for (A_long x = 0; x < w; ++x) tmp[row + static_cast<size_t>(x)] = lineOut[static_cast<size_t>(x)];
        });

        BorderParallelFor(w, [&](A_long x) {
            thread_local std::vector<int> vScratch;
            thread_local std::vector<float> zScratch;
            thread_local std::vector<int> lineIn;
            thread_local std::vector<int> lineOut;
            lineIn.resize(static_cast<size_t>(h));
            lineOut.resize(static_cast<size_t>(h));
            for (A_long y = 0; y < h; ++y) lineIn[static_cast<size_t>(y)] = tmp[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)];
            edt1d(lineIn.data(), static_cast<int>(h), lineOut.data(), vScratch, zScratch);
            for (A_long y = 0; y < h; ++y) d2d[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = lineOut[static_cast<size_t>(y)];
        });
    };

    // Reuse large buffers across frames to reduce heap churn (thread-safe under MFR).
    thread_local std::vector<A_u_char> solidMask;
    thread_local std::vector<int> fFg;
    thread_local std::vector<int> fBg;
    thread_local std::vector<int> dtFg2;
    thread_local std::vector<int> dtBg2;

    const size_t n = (size_t)w * (size_t)h;
    if (solidMask.size() != n) solidMask.assign(n, 0);
    if (fFg.size() != n) fFg.resize(n);
    if (fBg.size() != n) fBg.resize(n);

    // Build 0/INF grids for foreground/background feature points and cache solidity.
    // Write both arrays per pixel to avoid an O(N) fill pass.
    BorderParallelFor(h, [&](A_long y) {
        const size_t row = (size_t)y * (size_t)w;
        for (A_long x = 0; x < w; ++x) {
            const bool solid = isSolid(x, y);
            const size_t i = row + (size_t)x;
            solidMask[i] = solid ? 1 : 0;
            fFg[i] = solid ? 0 : INF;
            fBg[i] = solid ? INF : 0;
        }
    });

    edt2d(fFg, dtFg2);
    edt2d(fBg, dtBg2);

    // Signed distance to the nearest opposite-class pixel center (pixels), scaled by 10.
    // NOTE: The 0.5px center correction is applied later in the sampler.
    signedDist.resize(n);
    BorderParallelFor(static_cast<A_long>(n), [&](A_long ii) {
        const size_t i = static_cast<size_t>(ii);
        const bool solid = (solidMask[i] != 0);
        const int d2 = solid ? dtBg2[i] : dtFg2[i];
        float dist = sqrtf(static_cast<float>(d2));
        if (!std::isfinite(dist)) dist = 0.0f;
        const int sd = static_cast<int>(floorf(dist * 10.0f + 0.5f));
        signedDist[i] = solid ? sd : -sd;
    });
}

static PF_Err
PreRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    PF_RenderRequest req = extra->input->output_request;

    // SmartFX checkout ids must be unique within a pre-render.
    // Keep them positive to satisfy stricter hosts.
    const A_long kCheckoutIdInput  = 1;
    const A_long kCheckoutIdBounds = 2;

    PF_CheckoutResult in_result;
    AEFX_CLR_STRUCT(in_result);

    // 1) Bounds checkout with empty request_rect: lets AE compute input max_result_rect
    // without requiring us to render pixels. This is the recommended way to get stable
    // layer bounds in SmartFX.
    PF_CheckoutResult bounds_result;
    AEFX_CLR_STRUCT(bounds_result);
    {
        PF_RenderRequest boundsReq = req;
        boundsReq.rect.left = boundsReq.rect.top = boundsReq.rect.right = boundsReq.rect.bottom = 0; // empty
        ERR(extra->cb->checkout_layer(in_data->effect_ref, BORDER_INPUT, kCheckoutIdBounds, &boundsReq,
            in_data->current_time, in_data->time_step, in_data->time_scale, &bounds_result));
    }

    // 2) Pixel checkout for the requested rect.
    ERR(extra->cb->checkout_layer(in_data->effect_ref, BORDER_INPUT, kCheckoutIdInput, &req,
        in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));

    // Store the checkout_id we provided to checkout_layer() for pixels.
    extra->output->pre_render_data = (void*)(intptr_t)kCheckoutIdInput;

    // Do not claim extra pixels via SmartFX (avoids (25::237) result/max rect errors).
    extra->output->result_rect     = extra->input->output_request.rect;
    extra->output->max_result_rect = bounds_result.max_result_rect;

    return err;
}
static PF_Err
SmartRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    A_long checkout_id = static_cast<A_long>(reinterpret_cast<intptr_t>(extra->input->pre_render_data));

    PF_EffectWorld* input = NULL;
    PF_EffectWorld* output = NULL;

    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, checkout_id, &input));
    if (err) return err;
    if (!input) return PF_Err_BAD_CALLBACK_PARAM;

    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
    if (err) return err;

    PF_FpLong thickness = BORDER_THICKNESS_DFLT;
    PF_Pixel8 color = { 255, 0, 0, 255 };
    A_u_char threshold = BORDER_THRESHOLD_DFLT;
    A_long direction = BORDER_DIRECTION_DFLT;
    PF_Boolean showLineOnly = FALSE;

    PF_ParamDef param;

    AEFX_CLR_STRUCT(param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_THICKNESS, in_data->current_time, in_data->time_step, in_data->time_scale, &param));
    if (!err) thickness = param.u.fs_d.value;
    PF_CHECKIN_PARAM(in_data, &param);

    AEFX_CLR_STRUCT(param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_COLOR, in_data->current_time, in_data->time_step, in_data->time_scale, &param));
    if (!err) color = param.u.cd.value;
    PF_CHECKIN_PARAM(in_data, &param);

    AEFX_CLR_STRUCT(param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_THRESHOLD, in_data->current_time, in_data->time_step, in_data->time_scale, &param));
    if (!err) threshold = (A_u_char)param.u.sd.value;
    PF_CHECKIN_PARAM(in_data, &param);

    AEFX_CLR_STRUCT(param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_DIRECTION, in_data->current_time, in_data->time_step, in_data->time_scale, &param));
    if (!err) direction = param.u.pd.value;
    PF_CHECKIN_PARAM(in_data, &param);

    AEFX_CLR_STRUCT(param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_SHOW_LINE_ONLY, in_data->current_time, in_data->time_step, in_data->time_scale, &param));
    if (!err) showLineOnly = param.u.bd.value;
    PF_CHECKIN_PARAM(in_data, &param);

    float numX = static_cast<float>(in_data->downsample_x.num);
    float numY = static_cast<float>(in_data->downsample_y.num);
    if (numX == 0.0f || numY == 0.0f) {
        return PF_Err_BAD_CALLBACK_PARAM;
    }
    float downsize_x = static_cast<float>(in_data->downsample_x.den) / numX;
    float downsize_y = static_cast<float>(in_data->downsample_y.den) / numY;
    float resolution_factor = MIN(downsize_x, downsize_y);

    float pixelThickness;
    if (thickness <= 0.0f) {
        pixelThickness = 0.0f;
    }
    else if (thickness <= 10.0f) {
        pixelThickness = static_cast<float>(thickness);
    }
    else {
        float normalizedThickness = static_cast<float>((thickness - 10.0f) / 90.0f);
        pixelThickness = 10.0f + 40.0f * sqrtf(normalizedThickness);
    }

    A_long thicknessInt = (A_long)(pixelThickness / resolution_factor + 0.5f);
    float thicknessF = static_cast<float>(thicknessInt);
    // BOTH draws half inside + half outside so visible幅=thicknessF
    float strokeThicknessF = (direction == DIRECTION_BOTH) ? thicknessF * BorderConstants::STROKE_HALF : thicknessF;

    if (input && output) {
        // Calculate offset between input and output (output may be expanded)
        // We want output pixel (outX,outY) to map to input (x,y) at the same comp space.
        // If output is expanded (origin more negative), we need a positive offset to index into it.
        // Cast PF_LayerDef to PF_EffectWorld to access origin_h/origin_v
        PF_EffectWorld* inputWorld = reinterpret_cast<PF_EffectWorld*>(input);
        PF_EffectWorld* outputWorld = reinterpret_cast<PF_EffectWorld*>(output);
        A_long offsetX = inputWorld->origin_h - outputWorld->origin_h;
        A_long offsetY = inputWorld->origin_v - outputWorld->origin_v;

        // Generate signed distance field (fast chamfer, scaled by 10)
        std::vector<float> signedDist;
        // Treat Threshold==0 as "auto" and use 50% alpha to match the perceived AA edge.
        A_u_char thresholdSdf8 = (threshold == 0) ? 128 : threshold;
        A_u_short thresholdSdf16 = (A_u_short)(thresholdSdf8 * (PF_MAX_CHAN16 / 255.0f) + 0.5f);
        ERR(ComputeSignedDistanceField_EdgeEDT(input, thresholdSdf8, thresholdSdf16, signedDist,
            input->width, input->height));

        // STEP 1 & 2: Clear output and optionally copy source (optimized with memset/memcpy)
        const A_long outW = output->width;
        const A_long outH = output->height;
        const A_long inW = input->width;
        const A_long inH = input->height;

        if (PF_WORLD_IS_DEEP(output)) {
            const size_t pxSize = sizeof(PF_Pixel16);
            BorderParallelFor(outH, [&](A_long oy) {
                PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + oy * output->rowbytes);
                const A_long iy = oy - offsetY;
                const bool inYRange = (iy >= 0 && iy < inH);
                
                if (!showLineOnly && inYRange) {
                    // Clear left margin
                    if (offsetX > 0) {
                        memset(outData, 0, (size_t)offsetX * pxSize);
                    }
                    // Copy source pixels
                    const A_long startX = (offsetX > 0) ? offsetX : (A_long)0;
                    const A_long endX = (offsetX + inW < outW) ? (offsetX + inW) : outW;
                    if (endX > startX) {
                        PF_Pixel16* inData = (PF_Pixel16*)((char*)input->data + iy * input->rowbytes);
                        const A_long srcStartX = startX - offsetX;
                        memcpy(outData + startX, inData + srcStartX, (size_t)(endX - startX) * pxSize);
                    }
                    // Clear right margin
                    if (endX < outW) {
                        memset(outData + endX, 0, (size_t)(outW - endX) * pxSize);
                    }
                } else {
                    // Clear entire row
                    memset(outData, 0, (size_t)outW * pxSize);
                }
            });
        } else {
            const size_t pxSize = sizeof(PF_Pixel8);
            BorderParallelFor(outH, [&](A_long oy) {
                PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + oy * output->rowbytes);
                const A_long iy = oy - offsetY;
                const bool inYRange = (iy >= 0 && iy < inH);
                
                if (!showLineOnly && inYRange) {
                    // Clear left margin
                    if (offsetX > 0) {
                        memset(outData, 0, (size_t)offsetX * pxSize);
                    }
                    // Copy source pixels
                    const A_long startX = (offsetX > 0) ? offsetX : (A_long)0;
                    const A_long endX = (offsetX + inW < outW) ? (offsetX + inW) : outW;
                    if (endX > startX) {
                        PF_Pixel8* inData = (PF_Pixel8*)((char*)input->data + iy * input->rowbytes);
                        const A_long srcStartX = startX - offsetX;
                        memcpy(outData + startX, inData + srcStartX, (size_t)(endX - startX) * pxSize);
                    }
                    // Clear right margin
                    if (endX < outW) {
                        memset(outData + endX, 0, (size_t)(outW - endX) * pxSize);
                    }
                } else {
                    // Clear entire row
                    memset(outData, 0, (size_t)outW * pxSize);
                }
            });
        }

        // STEP 3: Draw the border using signed distance.
        if (strokeThicknessF <= 0.0f) {
            ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, checkout_id));
            return err; // stroke width 0 → nothing to draw
        }

        // SDF data pointer
        const float* sdfData = signedDist.data();

        // Get source alpha at subpixel position (bilinear interpolated)
        auto getSourceAlpha = [&](float fx, float fy) -> float {
            fx = CLAMP(fx, 0.0f, static_cast<float>(inW - 1) - BorderConstants::COORD_EPSILON);
            fy = CLAMP(fy, 0.0f, static_cast<float>(inH - 1) - BorderConstants::COORD_EPSILON);

            int x0 = static_cast<int>(fx);
            int y0 = static_cast<int>(fy);
            int x1 = MIN(x0 + 1, inW - 1);
            int y1 = MIN(y0 + 1, inH - 1);
            float tx = fx - x0;
            float ty = fy - y0;
            
            if (PF_WORLD_IS_DEEP(input)) {
                PF_Pixel16* row0 = (PF_Pixel16*)((char*)input->data + y0 * input->rowbytes);
                PF_Pixel16* row1 = (PF_Pixel16*)((char*)input->data + y1 * input->rowbytes);
                float a00 = row0[x0].alpha / (float)PF_MAX_CHAN16;
                float a10 = row0[x1].alpha / (float)PF_MAX_CHAN16;
                float a01 = row1[x0].alpha / (float)PF_MAX_CHAN16;
                float a11 = row1[x1].alpha / (float)PF_MAX_CHAN16;
                float a0 = a00 + (a10 - a00) * tx;
                float a1 = a01 + (a11 - a01) * tx;
                return a0 + (a1 - a0) * ty;
            } else {
                PF_Pixel8* row0 = (PF_Pixel8*)((char*)input->data + y0 * input->rowbytes);
                PF_Pixel8* row1 = (PF_Pixel8*)((char*)input->data + y1 * input->rowbytes);
                float a00 = row0[x0].alpha / 255.0f;
                float a10 = row0[x1].alpha / 255.0f;
                float a01 = row1[x0].alpha / 255.0f;
                float a11 = row1[x1].alpha / 255.0f;
                float a0 = a00 + (a10 - a00) * tx;
                float a1 = a01 + (a11 - a01) * tx;
                return a0 + (a1 - a0) * ty;
            }
        };

        // Hermite interpolated SDF lookup for smooth anti-aliasing
        auto getSDF_Hermite = [=](float fx, float fy) -> float {
            // Clamp to valid range
            fx = CLAMP(fx, 0.0f, static_cast<float>(inW - 1) - BorderConstants::COORD_EPSILON);
            fy = CLAMP(fy, 0.0f, static_cast<float>(inH - 1) - BorderConstants::COORD_EPSILON);

            int x0 = static_cast<int>(fx);
            int y0 = static_cast<int>(fy);
            int x1 = MIN(x0 + 1, inW - 1);
            int y1 = MIN(y0 + 1, inH - 1);
            float tx = fx - x0;
            float ty = fy - y0;
            
            // Apply smoothstep for Hermite interpolation (smoother than linear)
            float hx = smoothstep(0.0f, 1.0f, tx);
            float hy = smoothstep(0.0f, 1.0f, ty);

            // Sample 4 corners
            // SDF is now stored as float, no need to scale
            float d00 = sdfData[static_cast<size_t>(y0) * static_cast<size_t>(inW) + static_cast<size_t>(x0)];
            float d10 = sdfData[static_cast<size_t>(y0) * static_cast<size_t>(inW) + static_cast<size_t>(x1)];
            float d01 = sdfData[static_cast<size_t>(y1) * static_cast<size_t>(inW) + static_cast<size_t>(x0)];
            float d11 = sdfData[static_cast<size_t>(y1) * static_cast<size_t>(inW) + static_cast<size_t>(x1)];
            
            // Hermite interpolation using smoothstep weights
            float d0 = d00 + (d10 - d00) * hx;
            float d1 = d01 + (d11 - d01) * hx;
            float sdf = d0 + (d1 - d0) * hy;
            
            return sdf;
        };

        // Sharp AA range (0.5px) with more samples for smooth edges
        const float AA_RANGE_SM = BorderConstants::AA_RANGE;

        if (PF_WORLD_IS_DEEP(output)) {
            PF_Pixel16 edge_color;
            edge_color.alpha = PF_MAX_CHAN16;
            edge_color.red = PF_BYTE_TO_CHAR(color.red);
            edge_color.green = PF_BYTE_TO_CHAR(color.green);
            edge_color.blue = PF_BYTE_TO_CHAR(color.blue);

            // Parallel processing of rows - 8-sample MSAA for smooth sharp edges
            // Capture output by reference (needed for data access), all other values by value
            BorderParallelFor(outH, [&output, getSDF_Hermite, edge_color, offsetX, offsetY, inW, inH, outW,
                                      strokeThicknessF, direction, showLineOnly, AA_RANGE_SM](A_long oy) {
                PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + oy * output->rowbytes);
                const float fy = (float)(oy - offsetY);
                if (fy < 0.0f || fy >= (float)inH) return;

                for (A_long ox = 0; ox < outW; ++ox) {
                    const float fx = (float)(ox - offsetX);
                    if (fx < 0.0f || fx >= (float)inW) continue;

                    // 4x Supersampling (2x2 grid) for shape layer-like AA
                    // This ensures neighboring pixels have different coverage values
                    float totalCoverage = 0.0f;

                    // 2x2 subpixel grid with offsets: -0.25, +0.25
                    for (int sy = 0; sy < 2; ++sy) {
                        for (int sx = 0; sx < 2; ++sx) {
                            float subX = fx + (static_cast<float>(sx) - BorderConstants::STROKE_HALF) * BorderConstants::SUBSAMPLE_SCALE;
                            float subY = fy + (static_cast<float>(sy) - BorderConstants::STROKE_HALF) * BorderConstants::SUBSAMPLE_SCALE;

                            // Get SDF at subpixel position
                            float sdfC = getSDF_Hermite(subX, subY);

                            // Evaluate distance based on direction
                            float evalDist;
                            if (direction == DIRECTION_INSIDE) {
                                if (sdfC < 0.0f) continue;
                                evalDist = sdfC;
                            } else if (direction == DIRECTION_OUTSIDE) {
                                if (sdfC > 0.0f) continue;
                                evalDist = -sdfC;
                            } else {
                                evalDist = fabsf(sdfC);
                            }

                            // Calculate gradient at subpixel position
                            float gradX = getSDF_Hermite(subX + 0.25f, subY) - getSDF_Hermite(subX - 0.25f, subY);
                            float gradY = getSDF_Hermite(subX, subY + 0.25f) - getSDF_Hermite(subX, subY - 0.25f);
                            float gradMag = sqrtf(gradX * gradX + gradY * gradY) * 2.0f;  // Scale back to per-pixel
                            if (!std::isfinite(gradMag) || gradMag < BorderConstants::MIN_GRADIENT_MAGNITUDE) gradMag = BorderConstants::MIN_GRADIENT_MAGNITUDE;

                            // Distance to stroke edge in pixels
                            float distToEdge = (evalDist - strokeThicknessF) / gradMag;

                            // Coverage for this subpixel
                            float subCov = BorderConstants::STROKE_HALF - distToEdge;
                            subCov = CLAMP(subCov, 0.0f, 1.0f);
                            totalCoverage += subCov;
                        }
                    }

                    // Average coverage from 4 subpixels
                    float strokeCov = totalCoverage * BorderConstants::SUBSAMPLE_4X_INV;
                    if (strokeCov < BorderConstants::COVERAGE_EPSILON) continue;

                    PF_Pixel16 dst = outData[ox];
                    float dstAlphaNorm = dst.alpha / (float)PF_MAX_CHAN16;

                    if (showLineOnly) {
                        // Straight alpha: color at full intensity, alpha only for transparency
                        dst.red   = edge_color.red;
                        dst.green = edge_color.green;
                        dst.blue  = edge_color.blue;
                        dst.alpha = static_cast<A_u_short>(PF_MAX_CHAN16 * strokeCov + 0.5f);
                    } else {
                        // Straight alpha compositing (Porter-Duff over)
                        float strokeA = strokeCov;
                        float invStrokeA = 1.0f - strokeA;
                        float strokeR = edge_color.red / (float)PF_MAX_CHAN16;
                        float strokeG = edge_color.green / (float)PF_MAX_CHAN16;
                        float strokeB = edge_color.blue / (float)PF_MAX_CHAN16;
                        float outA = strokeA + dstAlphaNorm * invStrokeA;
                        float outR, outG, outB;
                        if (outA > BorderConstants::COVERAGE_EPSILON) {
                            // Straight alpha: divide by output alpha to get unmultiplied color
                            float dstR = dst.red / (float)PF_MAX_CHAN16;
                            float dstG = dst.green / (float)PF_MAX_CHAN16;
                            float dstB = dst.blue / (float)PF_MAX_CHAN16;
                            outR = (strokeR * strokeA + dstR * dstAlphaNorm * invStrokeA) / outA;
                            outG = (strokeG * strokeA + dstG * dstAlphaNorm * invStrokeA) / outA;
                            outB = (strokeB * strokeA + dstB * dstAlphaNorm * invStrokeA) / outA;
                        } else {
                            // Fully transparent: use stroke color (won't be visible anyway)
                            outR = strokeR;
                            outG = strokeG;
                            outB = strokeB;
                        }
                        dst.alpha = static_cast<A_u_short>(CLAMP(outA, BorderConstants::COLOR_MIN, BorderConstants::COLOR_MAX) * PF_MAX_CHAN16 + 0.5f);
                        dst.red   = static_cast<A_u_short>(CLAMP(outR, BorderConstants::COLOR_MIN, BorderConstants::COLOR_MAX) * PF_MAX_CHAN16 + 0.5f);
                        dst.green = static_cast<A_u_short>(CLAMP(outG, BorderConstants::COLOR_MIN, BorderConstants::COLOR_MAX) * PF_MAX_CHAN16 + 0.5f);
                        dst.blue  = static_cast<A_u_short>(CLAMP(outB, BorderConstants::COLOR_MIN, BorderConstants::COLOR_MAX) * PF_MAX_CHAN16 + 0.5f);
                    }

                    outData[ox] = dst;
                }
            });
        }
        else {
            // Parallel processing of rows (8-bit) - RGSS for smoother edges
            // Capture output by reference (needed for data access), all other values by value
            BorderParallelFor(outH, [&output, getSDF_Hermite, color, offsetX, offsetY, inW, inH, outW,
                                      strokeThicknessF, direction, showLineOnly, AA_RANGE_SM](A_long oy) {
                PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + oy * output->rowbytes);
                const float fy = (float)(oy - offsetY);
                if (fy < 0.0f || fy >= (float)inH) return;

                for (A_long ox = 0; ox < outW; ++ox) {
                    const float fx = (float)(ox - offsetX);
                    if (fx < 0.0f || fx >= (float)inW) continue;

                    // 4x Supersampling (2x2 grid) for shape layer-like AA
                    float totalCoverage = 0.0f;

                    for (int sy = 0; sy < 2; ++sy) {
                        for (int sx = 0; sx < 2; ++sx) {
                            float subX = fx + (static_cast<float>(sx) - BorderConstants::STROKE_HALF) * BorderConstants::SUBSAMPLE_SCALE;
                            float subY = fy + (static_cast<float>(sy) - BorderConstants::STROKE_HALF) * BorderConstants::SUBSAMPLE_SCALE;

                            float sdfC = getSDF_Hermite(subX, subY);

                            float evalDist;
                            if (direction == DIRECTION_INSIDE) {
                                if (sdfC < 0.0f) continue;
                                evalDist = sdfC;
                            } else if (direction == DIRECTION_OUTSIDE) {
                                if (sdfC > 0.0f) continue;
                                evalDist = -sdfC;
                            } else {
                                evalDist = fabsf(sdfC);
                            }

                            float gradX = getSDF_Hermite(subX + 0.25f, subY) - getSDF_Hermite(subX - 0.25f, subY);
                            float gradY = getSDF_Hermite(subX, subY + 0.25f) - getSDF_Hermite(subX, subY - 0.25f);
                            float gradMag = sqrtf(gradX * gradX + gradY * gradY) * 2.0f;
                            if (!std::isfinite(gradMag) || gradMag < BorderConstants::MIN_GRADIENT_MAGNITUDE) gradMag = BorderConstants::MIN_GRADIENT_MAGNITUDE;

                            float distToEdge = (evalDist - strokeThicknessF) / gradMag;
                            float subCov = CLAMP(BorderConstants::STROKE_HALF - distToEdge, 0.0f, 1.0f);
                            totalCoverage += subCov;
                        }
                    }

                    float strokeCov = totalCoverage * 0.25f;
                    if (strokeCov < BorderConstants::COVERAGE_EPSILON) continue;

                    // Dither coverage before 8bpc quantization to avoid visible plateaus.
                    float strokeA = strokeCov;
                    strokeA = CLAMP(strokeA + (BorderDither8(ox, oy) - BorderConstants::DITHER_CENTER) / 255.0f, 0.0f, 1.0f);

                    PF_Pixel8 dst = outData[ox];
                    float dstAlphaNorm = dst.alpha / (float)PF_MAX_CHAN8;

                    if (showLineOnly) {
                        // Straight alpha: color at full intensity, alpha only for transparency
                        dst.red   = color.red;
                        dst.green = color.green;
                        dst.blue  = color.blue;
                        dst.alpha = static_cast<A_u_char>(PF_MAX_CHAN8 * strokeA + 0.5f);
                    } else {
                        // Straight alpha compositing (Porter-Duff over)
                        float invStrokeA = 1.0f - strokeA;
                        float strokeR = color.red / (float)PF_MAX_CHAN8;
                        float strokeG = color.green / (float)PF_MAX_CHAN8;
                        float strokeB = color.blue / (float)PF_MAX_CHAN8;
                        float outA = strokeA + dstAlphaNorm * invStrokeA;
                        float outR, outG, outB;
                        if (outA > BorderConstants::COVERAGE_EPSILON) {
                            // Straight alpha: divide by output alpha to get unmultiplied color
                            float dstR = dst.red / (float)PF_MAX_CHAN8;
                            float dstG = dst.green / (float)PF_MAX_CHAN8;
                            float dstB = dst.blue / (float)PF_MAX_CHAN8;
                            outR = (strokeR * strokeA + dstR * dstAlphaNorm * invStrokeA) / outA;
                            outG = (strokeG * strokeA + dstG * dstAlphaNorm * invStrokeA) / outA;
                            outB = (strokeB * strokeA + dstB * dstAlphaNorm * invStrokeA) / outA;
                        } else {
                            // Fully transparent: use stroke color (won't be visible anyway)
                            outR = strokeR;
                            outG = strokeG;
                            outB = strokeB;
                        }
                        dst.alpha = static_cast<A_u_char>(CLAMP(outA, BorderConstants::COLOR_MIN, BorderConstants::COLOR_MAX) * PF_MAX_CHAN8 + 0.5f);
                        dst.red   = static_cast<A_u_char>(CLAMP(outR, BorderConstants::COLOR_MIN, BorderConstants::COLOR_MAX) * PF_MAX_CHAN8 + 0.5f);
                        dst.green = static_cast<A_u_char>(CLAMP(outG, BorderConstants::COLOR_MIN, BorderConstants::COLOR_MAX) * PF_MAX_CHAN8 + 0.5f);
                        dst.blue  = static_cast<A_u_char>(CLAMP(outB, BorderConstants::COLOR_MIN, BorderConstants::COLOR_MAX) * PF_MAX_CHAN8 + 0.5f);
                    }

                    outData[ox] = dst;
                }
            });
        }
    }

    ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, checkout_id));

    return err;
}

static PF_Err
Render(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;

    PF_EffectWorld* input = &params[BORDER_INPUT]->u.ld;

    PF_FpLong thickness = params[BORDER_THICKNESS]->u.fs_d.value;
    PF_Pixel8 color = params[BORDER_COLOR]->u.cd.value;
    A_u_char threshold = static_cast<A_u_char>(params[BORDER_THRESHOLD]->u.sd.value);
    A_long direction = params[BORDER_DIRECTION]->u.pd.value;
    PF_Boolean showLineOnly = params[BORDER_SHOW_LINE_ONLY]->u.bd.value;

    float downsize_x = static_cast<float>(in_data->downsample_x.den) / static_cast<float>(in_data->downsample_x.num);
    float downsize_y = static_cast<float>(in_data->downsample_y.den) / static_cast<float>(in_data->downsample_y.num);
    float resolution_factor = MIN(downsize_x, downsize_y);

    float pixelThickness;
    if (thickness <= 0.0f) {
        pixelThickness = 0.0f;
    } else if (thickness <= 10.0f) {
        pixelThickness = static_cast<float>(thickness);
    } else {
        float normalizedThickness = static_cast<float>((thickness - 10.0f) / 90.0f);
        pixelThickness = 10.0f + 40.0f * sqrtf(normalizedThickness);
    }

    A_long thicknessInt = static_cast<A_long>(pixelThickness / resolution_factor + 0.5f);
    float thicknessF = static_cast<float>(thicknessInt);
    float strokeThicknessF = (direction == DIRECTION_BOTH) ? thicknessF * BorderConstants::STROKE_HALF : thicknessF;

    // Calculate relative origin of input from output (output may be expanded)
    // Cast PF_LayerDef to PF_EffectWorld to access origin_h/origin_v
    PF_EffectWorld* inputWorld = reinterpret_cast<PF_EffectWorld*>(input);
    PF_EffectWorld* outputWorld = reinterpret_cast<PF_EffectWorld*>(output);
    const A_long originX = inputWorld->origin_h - outputWorld->origin_h;
    const A_long originY = inputWorld->origin_v - outputWorld->origin_v;

    const A_long outW = output->width;
    const A_long outH = output->height;
    const A_long inW = input->width;
    const A_long inH = input->height;

    const A_long outStartX = MAX((A_long)0, originX);
    const A_long outStartY = MAX((A_long)0, originY);
    const A_long inStartX = MAX((A_long)0, -originX);
    const A_long inStartY = MAX((A_long)0, -originY);
    const A_long copyW = MIN(inW - inStartX, outW - outStartX);
    const A_long copyH = MIN(inH - inStartY, outH - outStartY);

    // Treat Threshold==0 as "auto" and use 50% alpha to match the perceived AA edge.
    const A_u_char thresholdSdf8 = (threshold == 0) ? 128 : threshold;
    const A_u_short thresholdSdf16 = (A_u_short)(thresholdSdf8 * (PF_MAX_CHAN16 / 255.0f) + 0.5f);

    // Output base: either fully transparent (line-only) or input copied into expanded output
    // with only the non-overlapping bands cleared.
    if (PF_WORLD_IS_DEEP(output)) {
        const size_t pxSize = sizeof(PF_Pixel16);
        BorderParallelFor(outH, [&](A_long oy) {
            PF_Pixel16* outRow = (PF_Pixel16*)((char*)output->data + oy * output->rowbytes);
            const bool overlapsY = (!showLineOnly && copyH > 0 && oy >= outStartY && oy < (outStartY + copyH));
            if (!overlapsY) {
                memset(outRow, 0, (size_t)outW * pxSize);
                return;
            }

            // Clear only the left/right bands (outside the copied source span).
            if (outStartX > 0) {
                memset(outRow, 0, (size_t)outStartX * pxSize);
            }
            const A_long rightStart = outStartX + copyW;
            if (rightStart < outW) {
                memset(outRow + rightStart, 0, (size_t)(outW - rightStart) * pxSize);
            }

            // Copy the contiguous source span.
            const A_long iy = (oy - outStartY) + inStartY;
            PF_Pixel16* inRow = (PF_Pixel16*)((char*)input->data + iy * input->rowbytes);
            if (copyW > 0) {
                memcpy(outRow + outStartX, inRow + inStartX, (size_t)copyW * pxSize);
            }
        });
    } else {
        const size_t pxSize = sizeof(PF_Pixel8);
        BorderParallelFor(outH, [&](A_long oy) {
            PF_Pixel8* outRow = (PF_Pixel8*)((char*)output->data + oy * output->rowbytes);
            const bool overlapsY = (!showLineOnly && copyH > 0 && oy >= outStartY && oy < (outStartY + copyH));
            if (!overlapsY) {
                memset(outRow, 0, (size_t)outW * pxSize);
                return;
            }

            if (outStartX > 0) {
                memset(outRow, 0, (size_t)outStartX * pxSize);
            }
            const A_long rightStart = outStartX + copyW;
            if (rightStart < outW) {
                memset(outRow + rightStart, 0, (size_t)(outW - rightStart) * pxSize);
            }

            const A_long iy = (oy - outStartY) + inStartY;
            PF_Pixel8* inRow = (PF_Pixel8*)((char*)input->data + iy * input->rowbytes);
            if (copyW > 0) {
                memcpy(outRow + outStartX, inRow + inStartX, (size_t)copyW * pxSize);
            }
        });
    }

    if (strokeThicknessF <= 0.0f) {
        return err;
    }

    // Sharp AA range (0.5px) for crisp edges
    const float AA_RANGE = BorderConstants::AA_RANGE;
    const float MAX_EVAL_DIST = strokeThicknessF + AA_RANGE + 1.0f;

    // Compute a tight ROI for the SDF / stroke evaluation.
    BorderRectI solidIn;
    if (!BorderGetSolidBounds(input, thresholdSdf8, thresholdSdf16, solidIn)) {
        // Fully transparent input => nothing to stroke.
        return err;
    }

    const A_long pad = (A_long)ceilf(MAX_EVAL_DIST + 3.0f);
    const A_long solidOutL = solidIn.left + originX;
    const A_long solidOutT = solidIn.top + originY;
    const A_long solidOutR = solidIn.right + originX;
    const A_long solidOutB = solidIn.bottom + originY;

    const A_long roiLeft = CLAMP(solidOutL - pad, (A_long)0, outW - 1);
    const A_long roiTop = CLAMP(solidOutT - pad, (A_long)0, outH - 1);
    const A_long roiRight = CLAMP(solidOutR + pad, (A_long)0, outW - 1);
    const A_long roiBottom = CLAMP(solidOutB + pad, (A_long)0, outH - 1);

    const A_long roiW = roiRight - roiLeft + 1;
    [[maybe_unused]] const A_long roiH = roiBottom - roiTop + 1;

    const float thresholdNorm = thresholdSdf8 / 255.0f;
    auto getAlphaRoi = [&](A_long lx, A_long ly) -> float {
        const A_long gx = roiLeft + lx;
        const A_long gy = roiTop + ly;
        const A_long ix = gx - originX;
        const A_long iy = gy - originY;
        if (ix < 0 || ix >= inW || iy < 0 || iy >= inH) return 0.0f;

        if (PF_WORLD_IS_DEEP(input)) {
            PF_Pixel16* p = (PF_Pixel16*)((char*)input->data + iy * input->rowbytes) + ix;
            return p->alpha / (float)PF_MAX_CHAN16;
        } else {
            PF_Pixel8* p = (PF_Pixel8*)((char*)input->data + iy * input->rowbytes) + ix;
            return p->alpha / 255.0f;
        }
    };

    std::vector<float> signedDist;
    ComputeSignedDistanceField_EdgeEDT_FromAlpha(roiW, roiH, thresholdNorm, getAlphaRoi, signedDist);

    // Hermite SDF lookup for smooth anti-aliasing
    auto sampleSDF = [&](float fx, float fy) -> float {
        fx = CLAMP(fx, static_cast<float>(roiLeft), static_cast<float>(roiRight) - BorderConstants::COORD_EPSILON);
        fy = CLAMP(fy, static_cast<float>(roiTop), static_cast<float>(roiBottom) - BorderConstants::COORD_EPSILON);
        float lx = fx - static_cast<float>(roiLeft);
        float ly = fy - static_cast<float>(roiTop);
        int x0 = static_cast<int>(lx);
        int y0 = static_cast<int>(ly);
        int x1 = MIN(x0 + 1, (int)roiW - 1);
        int y1 = MIN(y0 + 1, (int)roiH - 1);
        float tx = lx - x0;
        float ty = ly - y0;
        
        // Apply smoothstep for Hermite interpolation (smoother than linear)
        float sx = smoothstep(BorderConstants::SMOOTHSTEP_ZERO, BorderConstants::SMOOTHSTEP_ONE, tx);
        float sy = smoothstep(BorderConstants::SMOOTHSTEP_ZERO, BorderConstants::SMOOTHSTEP_ONE, ty);

        float d00 = signedDist[(size_t)y0 * (size_t)roiW + (size_t)x0];
        float d10 = signedDist[(size_t)y0 * (size_t)roiW + (size_t)x1];
        float d01 = signedDist[(size_t)y1 * (size_t)roiW + (size_t)x0];
        float d11 = signedDist[(size_t)y1 * (size_t)roiW + (size_t)x1];

        // Hermite interpolation using smoothstep weights
        float d0 = d00 + (d10 - d00) * sx;
        float d1 = d01 + (d11 - d01) * sx;
        return d0 + (d1 - d0) * sy;
    };

    auto strokeSampleCoverage = [&](float sdf) -> float {
        // sdf: + inside, - outside (pixels, boundary at 0)
        // Use smoothstep for anti-aliasing at stroke edges
        float evalDist;
        if (direction == DIRECTION_INSIDE) {
            if (sdf < 0.0f) return 0.0f; // outside the shape
            evalDist = sdf;
        } else if (direction == DIRECTION_OUTSIDE) {
            if (sdf > 0.0f) return 0.0f; // inside the shape
            evalDist = -sdf;
        } else {
            evalDist = fabsf(sdf);
        }

        // Smooth anti-aliasing: full coverage inside, smooth falloff at edge
        // AA_RANGE defines the width of the anti-aliasing transition (1 pixel)
        if (evalDist <= strokeThicknessF - AA_RANGE) {
            return 1.0f; // fully inside stroke
        } else if (evalDist >= strokeThicknessF + AA_RANGE) {
            return 0.0f; // fully outside stroke
        } else {
            // Smooth transition using smoothstep for high-quality AA
            return 1.0f - smoothstep(strokeThicknessF - AA_RANGE, strokeThicknessF + AA_RANGE, evalDist);
        }
    };

    if (PF_WORLD_IS_DEEP(output)) {
        PF_Pixel16 edge_color;
        edge_color.alpha = PF_MAX_CHAN16;
        edge_color.red = PF_BYTE_TO_CHAR(color.red);
        edge_color.green = PF_BYTE_TO_CHAR(color.green);
        edge_color.blue = PF_BYTE_TO_CHAR(color.blue);

        const A_long roiH = roiBottom - roiTop + 1;
        // Capture output and lambdas by reference, all other values by value
        BorderParallelFor(roiH, [&output, &sampleSDF, &strokeSampleCoverage, edge_color, roiLeft, roiRight, roiTop, showLineOnly](A_long ly) {
            const A_long oy = roiTop + ly;
            PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + oy * output->rowbytes);
            for (A_long ox = roiLeft; ox <= roiRight; ++ox) {
                // Single sample with subpixel alpha correction
                float sdfC = sampleSDF((float)ox, (float)oy);
                float strokeCoverage = strokeSampleCoverage(sdfC);
                if (strokeCoverage < BorderConstants::COVERAGE_EPSILON) continue;

                PF_Pixel16 dst = outData[ox];
                float dstAlphaNorm = dst.alpha / (float)PF_MAX_CHAN16;

                if (showLineOnly) {
                    // Straight alpha: color at full intensity, alpha only for transparency
                    dst.red = edge_color.red;
                    dst.green = edge_color.green;
                    dst.blue = edge_color.blue;
                    dst.alpha = static_cast<A_u_short>(PF_MAX_CHAN16 * strokeCoverage + 0.5f);
                } else {
                    // Straight alpha compositing (Porter-Duff over)
                    float strokeA = strokeCoverage;
                    float invStrokeA = 1.0f - strokeA;

                    float strokeR = edge_color.red / (float)PF_MAX_CHAN16;
                    float strokeG = edge_color.green / (float)PF_MAX_CHAN16;
                    float strokeB = edge_color.blue / (float)PF_MAX_CHAN16;

                    float outA = strokeA + dstAlphaNorm * invStrokeA;

                    float outR, outG, outB;
                    if (outA > 0.001f) {
                        // Straight alpha: divide by output alpha to get unmultiplied color
                        float dstR = dst.red / (float)PF_MAX_CHAN16;
                        float dstG = dst.green / (float)PF_MAX_CHAN16;
                        float dstB = dst.blue / (float)PF_MAX_CHAN16;
                        outR = (strokeR * strokeA + dstR * dstAlphaNorm * invStrokeA) / outA;
                        outG = (strokeG * strokeA + dstG * dstAlphaNorm * invStrokeA) / outA;
                        outB = (strokeB * strokeA + dstB * dstAlphaNorm * invStrokeA) / outA;
                    } else {
                        // Fully transparent: use stroke color (won't be visible anyway)
                        outR = strokeR;
                        outG = strokeG;
                        outB = strokeB;
                    }

                    dst.alpha = (A_u_short)(CLAMP(outA, 0.0f, 1.0f) * PF_MAX_CHAN16 + 0.5f);
                    dst.red = (A_u_short)(CLAMP(outR, 0.0f, 1.0f) * PF_MAX_CHAN16 + 0.5f);
                    dst.green = (A_u_short)(CLAMP(outG, 0.0f, 1.0f) * PF_MAX_CHAN16 + 0.5f);
                    dst.blue = (A_u_short)(CLAMP(outB, 0.0f, 1.0f) * PF_MAX_CHAN16 + 0.5f);
                }

                outData[ox] = dst;
            }
        });
    } else {
        const A_long roiH = roiBottom - roiTop + 1;
        // Capture output and lambdas by reference, all other values by value
        BorderParallelFor(roiH, [&output, &sampleSDF, &strokeSampleCoverage, color, roiLeft, roiRight, roiTop, showLineOnly](A_long ly) {
            const A_long oy = roiTop + ly;
            PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + oy * output->rowbytes);
            for (A_long ox = roiLeft; ox <= roiRight; ++ox) {
                // Single sample with subpixel alpha correction
                float sdfC = sampleSDF((float)ox, (float)oy);
                float strokeCoverage = strokeSampleCoverage(sdfC);
                if (strokeCoverage < BorderConstants::COVERAGE_EPSILON) continue;

                // Dither coverage before 8bpc quantization to avoid visible plateaus.
                float strokeA = strokeCoverage;
                strokeA = CLAMP(strokeA + (BorderDither8(ox, oy) - 0.5f) / 255.0f, 0.0f, 1.0f);

                PF_Pixel8 dst = outData[ox];
                float dstAlphaNorm = dst.alpha / (float)PF_MAX_CHAN8;

                if (showLineOnly) {
                    // Straight alpha: color at full intensity, alpha only for transparency
                    dst.red = color.red;
                    dst.green = color.green;
                    dst.blue = color.blue;
                    dst.alpha = static_cast<A_u_char>(PF_MAX_CHAN8 * strokeA + 0.5f);
                } else {
                    // Straight alpha compositing (Porter-Duff over)
                    float invStrokeA = 1.0f - strokeA;

                    float strokeR = color.red / (float)PF_MAX_CHAN8;
                    float strokeG = color.green / (float)PF_MAX_CHAN8;
                    float strokeB = color.blue / (float)PF_MAX_CHAN8;

                    float outA = strokeA + dstAlphaNorm * invStrokeA;

                    float outR, outG, outB;
                    if (outA > 0.001f) {
                        // Straight alpha: divide by output alpha to get unmultiplied color
                        float dstR = dst.red / (float)PF_MAX_CHAN8;
                        float dstG = dst.green / (float)PF_MAX_CHAN8;
                        float dstB = dst.blue / (float)PF_MAX_CHAN8;
                        outR = (strokeR * strokeA + dstR * dstAlphaNorm * invStrokeA) / outA;
                        outG = (strokeG * strokeA + dstG * dstAlphaNorm * invStrokeA) / outA;
                        outB = (strokeB * strokeA + dstB * dstAlphaNorm * invStrokeA) / outA;
                    } else {
                        // Fully transparent: use stroke color (won't be visible anyway)
                        outR = strokeR;
                        outG = strokeG;
                        outB = strokeB;
                    }

                    dst.alpha = (A_u_char)(CLAMP(outA, 0.0f, 1.0f) * PF_MAX_CHAN8 + 0.5f);
                    dst.red = (A_u_char)(CLAMP(outR, 0.0f, 1.0f) * PF_MAX_CHAN8 + 0.5f);
                    dst.green = (A_u_char)(CLAMP(outG, 0.0f, 1.0f) * PF_MAX_CHAN8 + 0.5f);
                    dst.blue = (A_u_char)(CLAMP(outB, 0.0f, 1.0f) * PF_MAX_CHAN8 + 0.5f);
                }

                outData[ox] = dst;
            }
        });
    }

    return err;
}

extern "C" DllExport
PF_Err PluginDataEntryFunction2(
    PF_PluginDataPtr inPtr,
    PF_PluginDataCB2 inPluginDataCallBackPtr,
    SPBasicSuite* inSPBasicSuitePtr,
    const char* inHostName,
    const char* inHostVersion)
{
    PF_Err result = PF_Err_INVALID_CALLBACK;

    result = PF_REGISTER_EFFECT_EXT2(
        inPtr,
        inPluginDataCallBackPtr,
        "Border",
        "361do Border",
        "361do_plugins",
        AE_RESERVED_INFO,
        "EffectMain",
        "https://github.com/rebuildup/Ae_Border");

    return result;
}

extern "C" DllExport PF_Err
EffectMain(
    PF_Cmd            cmd,
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output,
    void* extra)
{
    PF_Err        err = PF_Err_NONE;

    try {
        switch (cmd) {
        case PF_Cmd_ABOUT:
            err = About(in_data, out_data, params, output);
            break;

        case PF_Cmd_GLOBAL_SETUP:
            err = GlobalSetup(in_data, out_data, params, output);
            break;

        case PF_Cmd_PARAMS_SETUP:
            err = ParamsSetup(in_data, out_data, params, output);
            break;

        case PF_Cmd_FRAME_SETUP:
            err = FrameSetup(in_data, out_data, params, output);
            break;

        case PF_Cmd_RENDER:
            err = Render(in_data, out_data, params, output);
            break;

        case PF_Cmd_SMART_PRE_RENDER:
        case PF_Cmd_SMART_RENDER:
            // We do not advertise Smart Render in out_flags2
            err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
            break;
        }
    }
    catch (PF_Err& thrown_err) {
        err = thrown_err;
    }
    return err;
}
