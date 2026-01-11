#include "Border.h"
#include <vector>
#include <algorithm>
#include <cfloat>
#include <limits>
#include <cstring>
#include <thread>
#include <cstdlib>
#include <cerrno>

#ifdef _WIN32
#include <omp.h>
#undef min
#undef max
#endif

// Clamp helper used by smoothstep
template <typename T>
T CLAMP(T value, T min, T max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

inline float smoothstep(float edge0, float edge1, float x) {
    float t = CLAMP((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
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
    PF_EffectWorld* input,
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

    static int cached = -1;
    if (cached < 0) {
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
            if (errno == 0 && endp != env_buf) threads = (int)v;
        }
#else
        if (const char* env = std::getenv("BORDER_THREADS")) {
            char* endp = nullptr;
            errno = 0;
            long v = strtol(env, &endp, 10);
            if (errno == 0 && endp != env) threads = (int)v;
        }
#endif
        if (env_buf) free(env_buf);

        // If explicitly set to 0 or negative, treat as "off".
        if (threads <= 0) threads = 1;

        // Conservative cap to reduce oversubscription under AE Multi-Frame Rendering.
        if (threads > 8) threads = 8;
        if (threads < 1) threads = 1;
        cached = threads;
    }

    int t = cached;
    if (t > (int)work_items) t = (int)work_items;
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

    suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
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
    float v = (float)scale.num / (float)scale.den;
    return (v > 0.0f) ? v : 1.0f;
}

static inline void BorderComputePixelThickness(
    PF_InData* in_data,
    PF_FpLong thicknessParam,
    A_long direction,
    float& outStrokeThicknessPx,  // stroke thickness on the chosen side (pixels)
    A_long& outExpansionPx)       // expansion required (pixels)
{
    float downsize_x = (float)in_data->downsample_x.den / (float)in_data->downsample_x.num;
    float downsize_y = (float)in_data->downsample_y.den / (float)in_data->downsample_y.num;
    float resolution_factor = MIN(downsize_x, downsize_y);

    float pixelThickness;
    if (thicknessParam <= 0.0f) {
        pixelThickness = 0.0f;
    } else if (thicknessParam <= 10.0f) {
        pixelThickness = (float)thicknessParam;
    } else {
        float normalizedThickness = (float)((thicknessParam - 10.0f) / 90.0f);
        pixelThickness = 10.0f + 40.0f * sqrtf(normalizedThickness);
    }

    float outsideThicknessPx = 0.0f;
    if (direction == DIRECTION_OUTSIDE) {
        outsideThicknessPx = pixelThickness;
    } else if (direction == DIRECTION_BOTH) {
        outsideThicknessPx = pixelThickness * 0.5f;
    }

    // This is used by the renderer for stroke evaluation.
    outStrokeThicknessPx = (direction == DIRECTION_BOTH) ? (pixelThickness * 0.5f) : pixelThickness;

    // Expand by outside thickness only (inside doesn't need extra pixels).
    const float AA_RANGE_PX = 1.0f;
    const float SAMPLE_GUARD_PX = 1.0f;
    if (outsideThicknessPx > 0.0f) {
        outExpansionPx = (A_long)ceil((outsideThicknessPx / resolution_factor) + AA_RANGE_PX + SAMPLE_GUARD_PX);
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
    (void)strokeThicknessPx;

    // IMPORTANT: In PF_Cmd_FRAME_SETUP, the input layer param is not valid per the SDK docs.
    // Use in_data->width/height as the base buffer size for expansion.
    const A_long srcW = in_data->width;
    const A_long srcH = in_data->height;

    if (expansionPx > 0) {
        out_data->width = srcW + expansionPx * 2;
        out_data->height = srcH + expansionPx * 2;
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
// Fast 3-4 chamfer distance transform (integer, scaled by 10) to compute
// signed distance to the alpha edge. Positive inside opaque region, negative outside.
//
// NOTE:
// The previous 3-4 chamfer distance had directional bias that could appear as a lateral
// "shift" of the stroke (especially noticeable on diagonals / sharp corners).
// We use an exact (grid) Euclidean Distance Transform (EDT) to avoid anisotropy.
static PF_Err
ComputeSignedDistanceField(
    PF_EffectWorld* input,
    A_u_char threshold8,
    A_u_short threshold16,
    std::vector<int>& signedDist, // scaled by 10
    A_long width,
    A_long height)
{
    const int INF = 1 << 29;
    const A_long w = width;
    const A_long h = height;

    signedDist.clear();

    auto isSolid = [&](A_long x, A_long y)->bool {
        if (PF_WORLD_IS_DEEP(input)) {
            PF_Pixel16* p = (PF_Pixel16*)((char*)input->data + y * input->rowbytes) + x;
            return p->alpha > threshold16;
        } else {
            PF_Pixel8* p = (PF_Pixel8*)((char*)input->data + y * input->rowbytes) + x;
            return p->alpha > threshold8;
        }
    };

    // 1D squared EDT (Felzenszwalb & Huttenlocher).
    // Uses scratch buffers to avoid per-row/col allocations (important for SMART_RENDER tiling).
    auto edt1d = [&](const int* f, int n, int* d, std::vector<int>& v, std::vector<float>& z) {
        if ((int)v.size() < n) v.resize(n);
        if ((int)z.size() < n + 1) z.resize(n + 1);

        const float INF_F = std::numeric_limits<float>::infinity();

        int k = 0;
        v[0] = 0;
        z[0] = -INF_F;
        z[1] =  INF_F;

        auto sep = [&](int i, int u)->float {
            // Intersection of parabolas from i and u:
            // s = ((f[u] + u^2) - (f[i] + i^2)) / (2u - 2i)
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

    // 2D EDT: first along X (rows), then along Y (columns).
    // Scratch buffers reused across calls (thread-safe across AE's threaded rendering).
    thread_local std::vector<int> tmp;

    auto edt2d = [&](const std::vector<int>& f2d, std::vector<int>& d2d) {
        // Column pass writes every element; no need to prefill with INF.
        d2d.resize((size_t)w * (size_t)h);

        // Row pass writes every element; no need to prefill.
        if (tmp.size() != (size_t)w * (size_t)h) tmp.resize((size_t)w * (size_t)h);

        // Pass 1: rows
        BorderParallelFor(h, [&](A_long y) {
            thread_local std::vector<int> vScratch;
            thread_local std::vector<float> zScratch;
            thread_local std::vector<int> lineIn;
            thread_local std::vector<int> lineOut;
            lineIn.resize((size_t)w);
            lineOut.resize((size_t)w);
            const size_t row = (size_t)y * (size_t)w;
            for (A_long x = 0; x < w; ++x) lineIn[(size_t)x] = f2d[row + (size_t)x];
            edt1d(lineIn.data(), (int)w, lineOut.data(), vScratch, zScratch);
            for (A_long x = 0; x < w; ++x) tmp[row + (size_t)x] = lineOut[(size_t)x];
        });

        // Pass 2: columns
        BorderParallelFor(w, [&](A_long x) {
            thread_local std::vector<int> vScratch;
            thread_local std::vector<float> zScratch;
            thread_local std::vector<int> lineIn;
            thread_local std::vector<int> lineOut;
            lineIn.resize((size_t)h);
            lineOut.resize((size_t)h);
            for (A_long y = 0; y < h; ++y) lineIn[(size_t)y] = tmp[(size_t)y * (size_t)w + (size_t)x];
            edt1d(lineIn.data(), (int)h, lineOut.data(), vScratch, zScratch);
            for (A_long y = 0; y < h; ++y) d2d[(size_t)y * (size_t)w + (size_t)x] = lineOut[(size_t)y];
        });
    };

    // Build 0/INF grids for foreground/background feature points and cache solidity.
    // dtFg2: squared distance to nearest solid pixel center
    // dtBg2: squared distance to nearest transparent pixel center
    std::vector<A_u_char> solidMask((size_t)w * (size_t)h, 0);
    std::vector<int> fFg((size_t)w * (size_t)h);
    std::vector<int> fBg((size_t)w * (size_t)h);
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

    std::vector<int> dtFg2, dtBg2;
    edt2d(fFg, dtFg2);
    edt2d(fBg, dtBg2);

    // Signed distance to the nearest opposite-class pixel center (pixels), scaled by 10.
    // NOTE: The 0.5px center correction is applied later in the sampler.
    BorderParallelFor(h, [&](A_long y) {
        const size_t row = (size_t)y * (size_t)w;
        for (A_long x = 0; x < w; ++x) {
            const size_t i = row + (size_t)x;
            const bool solid = (solidMask[i] != 0);
            const int d2 = solid ? dtBg2[i] : dtFg2[i];
            const float dist = sqrtf((float)d2);
            const int sd = (int)floorf(dist * 10.0f + 0.5f);
            signedDist[i] = solid ? sd : -sd;
        }
    });

    return PF_Err_NONE;
}

template <typename IsSolidFn>
static void
ComputeSignedDistanceFieldFromSolid(
    A_long width,
    A_long height,
    IsSolidFn isSolid,
    std::vector<int>& signedDist) // scaled by 10
{
    const int INF = 1 << 29;
    const A_long w = width;
    const A_long h = height;

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
        d2d.resize((size_t)w * (size_t)h);

        // Row pass writes every element; no need to prefill.
        if (tmp.size() != (size_t)w * (size_t)h) tmp.resize((size_t)w * (size_t)h);

        BorderParallelFor(h, [&](A_long y) {
            thread_local std::vector<int> vScratch;
            thread_local std::vector<float> zScratch;
            thread_local std::vector<int> lineIn;
            thread_local std::vector<int> lineOut;
            lineIn.resize((size_t)w);
            lineOut.resize((size_t)w);
            const size_t row = (size_t)y * (size_t)w;
            for (A_long x = 0; x < w; ++x) lineIn[(size_t)x] = f2d[row + (size_t)x];
            edt1d(lineIn.data(), (int)w, lineOut.data(), vScratch, zScratch);
            for (A_long x = 0; x < w; ++x) tmp[row + (size_t)x] = lineOut[(size_t)x];
        });

        BorderParallelFor(w, [&](A_long x) {
            thread_local std::vector<int> vScratch;
            thread_local std::vector<float> zScratch;
            thread_local std::vector<int> lineIn;
            thread_local std::vector<int> lineOut;
            lineIn.resize((size_t)h);
            lineOut.resize((size_t)h);
            for (A_long y = 0; y < h; ++y) lineIn[(size_t)y] = tmp[(size_t)y * (size_t)w + (size_t)x];
            edt1d(lineIn.data(), (int)h, lineOut.data(), vScratch, zScratch);
            for (A_long y = 0; y < h; ++y) d2d[(size_t)y * (size_t)w + (size_t)x] = lineOut[(size_t)y];
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
    BorderParallelFor((A_long)n, [&](A_long ii) {
        const size_t i = (size_t)ii;
        const bool solid = (solidMask[i] != 0);
        const int d2 = solid ? dtBg2[i] : dtFg2[i];
        const float dist = sqrtf((float)d2);
        const int sd = (int)floorf(dist * 10.0f + 0.5f);
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

    A_long checkout_id = (A_long)(intptr_t)extra->input->pre_render_data;

    PF_EffectWorld* input = NULL;
    PF_EffectWorld* output = NULL;

    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, checkout_id, &input));
    if (err) return err;

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

    float downsize_x = static_cast<float>(in_data->downsample_x.den) / static_cast<float>(in_data->downsample_x.num);
    float downsize_y = static_cast<float>(in_data->downsample_y.den) / static_cast<float>(in_data->downsample_y.num);
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
    float strokeThicknessF = (direction == DIRECTION_BOTH) ? thicknessF * 0.5f : thicknessF;

    if (input && output) {
        // Calculate offset between input and output (output may be expanded)
        // We want output pixel (outX,outY) to map to input (x,y) at the same comp space.
        // If output is expanded (origin more negative), we need a positive offset to index into it.
        A_long offsetX = input->origin_x - output->origin_x;
        A_long offsetY = input->origin_y - output->origin_y;

        // Generate signed distance field (fast chamfer, scaled by 10)
        std::vector<int> signedDist;
        // Treat Threshold==0 as "auto" and use 50% alpha to match the perceived AA edge.
        A_u_char thresholdSdf8 = (threshold == 0) ? 128 : threshold;
        A_u_short thresholdSdf16 = thresholdSdf8 * 257;
        ERR(ComputeSignedDistanceField(input, thresholdSdf8, thresholdSdf16, signedDist,
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
        const int* sdfData = signedDist.data();

        // Get source alpha at subpixel position (bilinear interpolated)
        auto getSourceAlpha = [&](float fx, float fy) -> float {
            fx = CLAMP(fx, 0.0f, (float)(inW - 1) - 0.001f);
            fy = CLAMP(fy, 0.0f, (float)(inH - 1) - 0.001f);
            
            int x0 = (int)fx;
            int y0 = (int)fy;
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

        // Bilinear interpolated SDF lookup for smooth anti-aliasing
        auto getSDF_Bilinear = [=](float fx, float fy) -> float {
            // Clamp to valid range
            fx = CLAMP(fx, 0.0f, (float)(inW - 1) - 0.001f);
            fy = CLAMP(fy, 0.0f, (float)(inH - 1) - 0.001f);
            
            int x0 = (int)fx;
            int y0 = (int)fy;
            int x1 = MIN(x0 + 1, inW - 1);
            int y1 = MIN(y0 + 1, inH - 1);
            float tx = fx - x0;
            float ty = fy - y0;
            
            // Sample 4 corners
            float d00 = sdfData[(size_t)y0 * (size_t)inW + (size_t)x0] * 0.1f;
            float d10 = sdfData[(size_t)y0 * (size_t)inW + (size_t)x1] * 0.1f;
            float d01 = sdfData[(size_t)y1 * (size_t)inW + (size_t)x0] * 0.1f;
            float d11 = sdfData[(size_t)y1 * (size_t)inW + (size_t)x1] * 0.1f;
            
            // Bilinear interpolation
            float d0 = d00 + (d10 - d00) * tx;
            float d1 = d01 + (d11 - d01) * tx;
            float sdf = d0 + (d1 - d0) * ty;
            
            // Apply 0.5px boundary correction
            if (sdf > 0.0f) sdf -= 0.5f;
            else if (sdf < 0.0f) sdf += 0.5f;
            
            return sdf;
        };

        // AA range of 1.0px for smoother edges
        const float AA_RANGE_SM = 1.0f;

        if (PF_WORLD_IS_DEEP(output)) {
            PF_Pixel16 edge_color;
            edge_color.alpha = PF_MAX_CHAN16;
            edge_color.red = PF_BYTE_TO_CHAR(color.red);
            edge_color.green = PF_BYTE_TO_CHAR(color.green);
            edge_color.blue = PF_BYTE_TO_CHAR(color.blue);

            // Parallel processing of rows - 2-sample diagonal AA for smoother edges
            BorderParallelFor(outH, [&, getSDF_Bilinear, edge_color, offsetX, offsetY, inW, inH, outW, 
                                      strokeThicknessF, direction, showLineOnly, AA_RANGE_SM](A_long oy) {
                PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + oy * output->rowbytes);
                const float fy = (float)(oy - offsetY);
                if (fy < 0.0f || fy >= (float)inH) return;

                for (A_long ox = 0; ox < outW; ++ox) {
                    const float fx = (float)(ox - offsetX);
                    if (fx < 0.0f || fx >= (float)inW) continue;

                    // Rotated Grid Super Sampling (RGSS) - 4 samples rotated 45 degrees
                    // More effective for diagonal edges than standard 2x2 grid
                    float totalCoverage = 0.0f;
                    int validSamples = 0;
                    
                    // RGSS pattern - optimized for diagonal edges
                    const float offsets[4][2] = {
                        {-0.125f, -0.375f}, { 0.375f, -0.125f},
                        {-0.375f,  0.125f}, { 0.125f,  0.375f}
                    };
                    
                    for (int s = 0; s < 4; ++s) {
                        float sx = fx + offsets[s][0];
                        float sy = fy + offsets[s][1];
                        
                        if (sx < 0.0f || sx >= (float)(inW - 1) || sy < 0.0f || sy >= (float)(inH - 1)) continue;
                        
                        float sdfC = getSDF_Bilinear(sx, sy);
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
                        
                        if (evalDist > strokeThicknessF + AA_RANGE_SM) continue;

                        // Calculate coverage using smoothstep
                        float sampleCov;
                        if (evalDist <= strokeThicknessF - AA_RANGE_SM) {
                            sampleCov = 1.0f;
                        } else {
                            sampleCov = 1.0f - smoothstep(strokeThicknessF - AA_RANGE_SM, strokeThicknessF + AA_RANGE_SM, evalDist);
                        }
                        totalCoverage += sampleCov;
                        validSamples++;
                    }
                    
                    if (validSamples == 0) continue;
                    float strokeCov = totalCoverage / (float)validSamples;
                    if (strokeCov < 0.001f) continue;

                    PF_Pixel16 dst = outData[ox];
                    float dstAlphaNorm = dst.alpha / (float)PF_MAX_CHAN16;

                    if (showLineOnly) {
                        // Straight alpha: color at full intensity, alpha only for transparency
                        dst.red   = edge_color.red;
                        dst.green = edge_color.green;
                        dst.blue  = edge_color.blue;
                        dst.alpha = (A_u_short)(PF_MAX_CHAN16 * strokeCov + 0.5f);
                    } else {
                        // Straight alpha compositing (Porter-Duff over)
                        float strokeA = strokeCov;
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
                        dst.red   = (A_u_short)(CLAMP(outR, 0.0f, 1.0f) * PF_MAX_CHAN16 + 0.5f);
                        dst.green = (A_u_short)(CLAMP(outG, 0.0f, 1.0f) * PF_MAX_CHAN16 + 0.5f);
                        dst.blue  = (A_u_short)(CLAMP(outB, 0.0f, 1.0f) * PF_MAX_CHAN16 + 0.5f);
                    }

                    outData[ox] = dst;
                }
            });
        }
        else {
            // Parallel processing of rows (8-bit) - 2-sample diagonal AA for smoother edges
            BorderParallelFor(outH, [&, getSDF_Bilinear, color, offsetX, offsetY, inW, inH, outW, 
                                      strokeThicknessF, direction, showLineOnly, AA_RANGE_SM](A_long oy) {
                PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + oy * output->rowbytes);
                const float fy = (float)(oy - offsetY);
                if (fy < 0.0f || fy >= (float)inH) return;

                for (A_long ox = 0; ox < outW; ++ox) {
                    const float fx = (float)(ox - offsetX);
                    if (fx < 0.0f || fx >= (float)inW) continue;

                    // Rotated Grid Super Sampling (RGSS) - 4 samples rotated 45 degrees
                    // More effective for diagonal edges than standard 2x2 grid
                    float totalCoverage = 0.0f;
                    int validSamples = 0;
                    
                    // RGSS pattern - optimized for diagonal edges
                    const float offsets[4][2] = {
                        {-0.125f, -0.375f}, { 0.375f, -0.125f},
                        {-0.375f,  0.125f}, { 0.125f,  0.375f}
                    };
                    
                    for (int s = 0; s < 4; ++s) {
                        float sx = fx + offsets[s][0];
                        float sy = fy + offsets[s][1];
                        
                        if (sx < 0.0f || sx >= (float)(inW - 1) || sy < 0.0f || sy >= (float)(inH - 1)) continue;
                        
                        float sdfC = getSDF_Bilinear(sx, sy);
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
                        
                        if (evalDist > strokeThicknessF + AA_RANGE_SM) continue;

                        // Calculate coverage using smoothstep
                        float sampleCov;
                        if (evalDist <= strokeThicknessF - AA_RANGE_SM) {
                            sampleCov = 1.0f;
                        } else {
                            sampleCov = 1.0f - smoothstep(strokeThicknessF - AA_RANGE_SM, strokeThicknessF + AA_RANGE_SM, evalDist);
                        }
                        totalCoverage += sampleCov;
                        validSamples++;
                    }
                    
                    if (validSamples == 0) continue;
                    float strokeCov = totalCoverage / (float)validSamples;
                    if (strokeCov < 0.001f) continue;

                    PF_Pixel8 dst = outData[ox];
                    float dstAlphaNorm = dst.alpha / (float)PF_MAX_CHAN8;

                    if (showLineOnly) {
                        // Straight alpha: color at full intensity, alpha only for transparency
                        dst.red   = color.red;
                        dst.green = color.green;
                        dst.blue  = color.blue;
                        dst.alpha = (A_u_char)(PF_MAX_CHAN8 * strokeCov + 0.5f);
                    } else {
                        // Straight alpha compositing (Porter-Duff over)
                        float strokeA = strokeCov;
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
                        dst.red   = (A_u_char)(CLAMP(outR, 0.0f, 1.0f) * PF_MAX_CHAN8 + 0.5f);
                        dst.green = (A_u_char)(CLAMP(outG, 0.0f, 1.0f) * PF_MAX_CHAN8 + 0.5f);
                        dst.blue  = (A_u_char)(CLAMP(outB, 0.0f, 1.0f) * PF_MAX_CHAN8 + 0.5f);
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
    A_u_char threshold = (A_u_char)params[BORDER_THRESHOLD]->u.sd.value;
    A_long direction = params[BORDER_DIRECTION]->u.pd.value;
    PF_Boolean showLineOnly = params[BORDER_SHOW_LINE_ONLY]->u.bd.value;

    float downsize_x = (float)in_data->downsample_x.den / (float)in_data->downsample_x.num;
    float downsize_y = (float)in_data->downsample_y.den / (float)in_data->downsample_y.num;
    float resolution_factor = MIN(downsize_x, downsize_y);

    float pixelThickness;
    if (thickness <= 0.0f) {
        pixelThickness = 0.0f;
    } else if (thickness <= 10.0f) {
        pixelThickness = (float)thickness;
    } else {
        float normalizedThickness = (float)((thickness - 10.0f) / 90.0f);
        pixelThickness = 10.0f + 40.0f * sqrtf(normalizedThickness);
    }

    A_long thicknessInt = (A_long)(pixelThickness / resolution_factor + 0.5f);
    float thicknessF = (float)thicknessInt;
    float strokeThicknessF = (direction == DIRECTION_BOTH) ? thicknessF * 0.5f : thicknessF;

    const A_long originX = in_data->output_origin_x;
    const A_long originY = in_data->output_origin_y;

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
    const A_u_short thresholdSdf16 = thresholdSdf8 * 257;

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

    const float AA_RANGE = 1.0f;
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
    const A_long roiH = roiBottom - roiTop + 1;
    (void)roiH;

    auto isSolidRoi = [&](A_long lx, A_long ly) -> bool {
        const A_long gx = roiLeft + lx;
        const A_long gy = roiTop + ly;
        const A_long ix = gx - originX;
        const A_long iy = gy - originY;
        if (ix < 0 || ix >= inW || iy < 0 || iy >= inH) return false;
        if (PF_WORLD_IS_DEEP(input)) {
            PF_Pixel16* p = (PF_Pixel16*)((char*)input->data + iy * input->rowbytes) + ix;
            return p->alpha > thresholdSdf16;
        } else {
            PF_Pixel8* p = (PF_Pixel8*)((char*)input->data + iy * input->rowbytes) + ix;
            return p->alpha > thresholdSdf8;
        }
    };

    std::vector<int> signedDist;
    ComputeSignedDistanceFieldFromSolid(roiW, roiH, isSolidRoi, signedDist);

    // Get source alpha at position for subpixel edge correction
    auto getSourceAlphaRender = [&](float fx, float fy) -> float {
        int ix = (int)fx - originX;
        int iy = (int)fy - originY;
        if (ix < 0 || ix >= inW || iy < 0 || iy >= inH) return 0.0f;
        
        if (PF_WORLD_IS_DEEP(input)) {
            PF_Pixel16* p = (PF_Pixel16*)((char*)input->data + iy * input->rowbytes) + ix;
            return p->alpha / (float)PF_MAX_CHAN16;
        } else {
            PF_Pixel8* p = (PF_Pixel8*)((char*)input->data + iy * input->rowbytes) + ix;
            return p->alpha / 255.0f;
        }
    };

    // Bilinear SDF lookup for smooth anti-aliasing
    auto sampleSDF = [&](float fx, float fy) -> float {
        fx = CLAMP(fx, (float)roiLeft, (float)roiRight - 0.001f);
        fy = CLAMP(fy, (float)roiTop, (float)roiBottom - 0.001f);
        float lx = fx - (float)roiLeft;
        float ly = fy - (float)roiTop;
        int x0 = (int)lx;
        int y0 = (int)ly;
        int x1 = MIN(x0 + 1, (int)roiW - 1);
        int y1 = MIN(y0 + 1, (int)roiH - 1);
        float tx = lx - x0;
        float ty = ly - y0;

        int d00 = signedDist[(size_t)y0 * (size_t)roiW + (size_t)x0];
        int d10 = signedDist[(size_t)y0 * (size_t)roiW + (size_t)x1];
        int d01 = signedDist[(size_t)y1 * (size_t)roiW + (size_t)x0];
        int d11 = signedDist[(size_t)y1 * (size_t)roiW + (size_t)x1];

        float d0 = d00 + (d10 - d00) * tx;
        float d1 = d01 + (d11 - d01) * tx;
        float d = d0 + (d1 - d0) * ty;

        float sdf = d * 0.1f;
        if (sdf > 0.0f) sdf -= 0.5f;
        else if (sdf < 0.0f) sdf += 0.5f;
        
        return sdf;
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
        BorderParallelFor(roiH, [&](A_long ly) {
            const A_long oy = roiTop + ly;
            PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + oy * output->rowbytes);
            for (A_long ox = roiLeft; ox <= roiRight; ++ox) {
                // Single sample with subpixel alpha correction
                float sdfC = sampleSDF((float)ox, (float)oy);
                float strokeCoverage = strokeSampleCoverage(sdfC);
                if (strokeCoverage < 0.001f) continue;

                PF_Pixel16 dst = outData[ox];
                float dstAlphaNorm = dst.alpha / (float)PF_MAX_CHAN16;

                if (showLineOnly) {
                    // Straight alpha: color at full intensity, alpha only for transparency
                    dst.red = edge_color.red;
                    dst.green = edge_color.green;
                    dst.blue = edge_color.blue;
                    dst.alpha = (A_u_short)(PF_MAX_CHAN16 * strokeCoverage + 0.5f);
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
        BorderParallelFor(roiH, [&](A_long ly) {
            const A_long oy = roiTop + ly;
            PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + oy * output->rowbytes);
            for (A_long ox = roiLeft; ox <= roiRight; ++ox) {
                // Single sample with subpixel alpha correction
                float sdfC = sampleSDF((float)ox, (float)oy);
                float strokeCoverage = strokeSampleCoverage(sdfC);
                if (strokeCoverage < 0.001f) continue;

                PF_Pixel8 dst = outData[ox];
                float dstAlphaNorm = dst.alpha / (float)PF_MAX_CHAN8;

                if (showLineOnly) {
                    // Straight alpha: color at full intensity, alpha only for transparency
                    dst.red = color.red;
                    dst.green = color.green;
                    dst.blue = color.blue;
                    dst.alpha = (A_u_char)(PF_MAX_CHAN8 * strokeCoverage + 0.5f);
                } else {
                    // Straight alpha compositing (Porter-Duff over)
                    float strokeA = strokeCoverage;
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

        // Fallback SmartFX handlers:
        // We do not advertise Smart Render in out_flags2, but hosts may still call these
        // if they have cached older PiPL flags. Implementing them avoids hard errors.
        case PF_Cmd_SMART_PRE_RENDER:
            err = PreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra));
            break;

        case PF_Cmd_SMART_RENDER:
            err = SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra));
            break;
        }
    }
    catch (PF_Err& thrown_err) {
        err = thrown_err;
    }
    return err;
}
