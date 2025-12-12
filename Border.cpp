#include "Border.h"
#include <vector>
#include <algorithm>
#include <cfloat>
#include <limits>

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

    out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_PIX_INDEPENDENT | PF_OutFlag_USE_OUTPUT_EXTENT;
    // REVEALS_ZERO_ALPHA is important for Shape/Text layers: AE may otherwise crop
    // the renderable rect to non-zero alpha, which prevents drawing Outside bounds.
    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER |
                           PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
                           PF_OutFlag2_REVEALS_ZERO_ALPHA;

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

    signedDist.assign(w * h, 0);

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
    thread_local std::vector<int> vScratch;
    thread_local std::vector<float> zScratch;
    thread_local std::vector<int> lineIn;
    thread_local std::vector<int> lineOut;
    thread_local std::vector<int> tmp;

    auto edt2d = [&](const std::vector<int>& f2d, std::vector<int>& d2d) {
        d2d.assign(w * h, INF);

        // Ensure temp buffer matches current size.
        if (tmp.size() != (size_t)w * (size_t)h) tmp.assign((size_t)w * (size_t)h, INF);

        // Pass 1: rows
        for (A_long y = 0; y < h; ++y) {
            lineIn.resize((size_t)w);
            lineOut.resize((size_t)w);
            for (A_long x = 0; x < w; ++x) lineIn[(size_t)x] = f2d[(size_t)y * (size_t)w + (size_t)x];

            edt1d(lineIn.data(), (int)w, lineOut.data(), vScratch, zScratch);
            for (A_long x = 0; x < w; ++x) tmp[(size_t)y * (size_t)w + (size_t)x] = lineOut[(size_t)x];
        }

        // Pass 2: columns
        for (A_long x = 0; x < w; ++x) {
            lineIn.resize((size_t)h);
            lineOut.resize((size_t)h);
            for (A_long y = 0; y < h; ++y) lineIn[(size_t)y] = tmp[(size_t)y * (size_t)w + (size_t)x];

            edt1d(lineIn.data(), (int)h, lineOut.data(), vScratch, zScratch);

            for (A_long y = 0; y < h; ++y) d2d[(size_t)y * (size_t)w + (size_t)x] = lineOut[(size_t)y];
        }
    };

    // Build 0/INF grids for foreground/background feature points.
    // dtFg: distance to nearest solid pixel center
    // dtBg: distance to nearest transparent pixel center
    std::vector<int> fFg(w * h, INF);
    std::vector<int> fBg(w * h, INF);
    for (A_long y = 0; y < h; ++y) {
        for (A_long x = 0; x < w; ++x) {
            bool solid = isSolid(x, y);
            size_t i = (size_t)y * (size_t)w + (size_t)x;
            if (solid) {
                fFg[i] = 0;
            } else {
                fBg[i] = 0;
            }
        }
    }

    std::vector<int> dtFg2, dtBg2;
    edt2d(fFg, dtFg2);
    edt2d(fBg, dtBg2);

    // Signed distance (pixels), scaled by 10.
    // sdf = distToBg - distToFg  => positive inside, negative outside.
    for (A_long y = 0; y < h; ++y) {
        for (A_long x = 0; x < w; ++x) {
            size_t i = (size_t)y * (size_t)w + (size_t)x;
            float distToFg = sqrtf((float)dtFg2[i]);
            float distToBg = sqrtf((float)dtBg2[i]);
            float sdf = distToBg - distToFg;
            signedDist[i] = (int)floorf(sdf * 10.0f + (sdf >= 0.0f ? 0.5f : -0.5f));
        }
    }

    return PF_Err_NONE;
}
static PF_Err
PreRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    PF_RenderRequest req = extra->input->output_request;
    PF_CheckoutResult in_result;
    AEFX_CLR_STRUCT(in_result);

    PF_ParamDef thickness_param;
    AEFX_CLR_STRUCT(thickness_param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_THICKNESS, in_data->current_time, in_data->time_step, in_data->time_scale, &thickness_param));

    PF_ParamDef direction_param;
    AEFX_CLR_STRUCT(direction_param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_DIRECTION, in_data->current_time, in_data->time_step, in_data->time_scale, &direction_param));

    PF_FpLong thickness = thickness_param.u.fs_d.value;
    A_long direction = direction_param.u.pd.value;

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

    float downsize_x = static_cast<float>(in_data->downsample_x.den) / static_cast<float>(in_data->downsample_x.num);
    float downsize_y = static_cast<float>(in_data->downsample_y.den) / static_cast<float>(in_data->downsample_y.num);
    float resolution_factor = MIN(downsize_x, downsize_y);

    float effectiveThickness = pixelThickness;
    if (direction == DIRECTION_BOTH) {
        effectiveThickness = pixelThickness / 2.0f;
    }

    A_long borderExpansion = (A_long)ceil(effectiveThickness / resolution_factor) + 5;

    // Ask AE for a slightly larger input region so we can sample beyond the request
    // when drawing the border. Output rect itself must stay within the host's request.
    req.rect.left   -= borderExpansion;
    req.rect.top    -= borderExpansion;
    req.rect.right  += borderExpansion;
    req.rect.bottom += borderExpansion;
    ERR(extra->cb->checkout_layer(in_data->effect_ref, BORDER_INPUT, BORDER_INPUT, &req, in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));

    // Store the checkout_id we provided to checkout_layer() (the second BORDER_INPUT argument).
    // Not all SDKs expose an id field in PF_CheckoutResult.
    extra->output->pre_render_data = (void*)(intptr_t)BORDER_INPUT;

    PF_Rect request_rect = extra->input->output_request.rect;

    // For Shape/Text layers the layer bounds can be tight; when drawing Outside/Both we need
    // pixels beyond the requested rect. SmartFX supports this via RETURNS_EXTRA_PIXELS.
    // If we don't set it, AE may clip the stroke to the layer bounds regardless of "Grow Bounds".
    PF_Rect result_rect = request_rect;
    PF_Rect max_rect = in_result.max_result_rect;

    auto expand_rect = [](PF_Rect& r, A_long e) {
        r.left   -= e;
        r.top    -= e;
        r.right  += e;
        r.bottom += e;
    };

    auto intersect_rect = [](const PF_Rect& a, const PF_Rect& b) -> PF_Rect {
        PF_Rect o;
        o.left   = MAX(a.left,   b.left);
        o.top    = MAX(a.top,    b.top);
        o.right  = MIN(a.right,  b.right);
        o.bottom = MIN(a.bottom, b.bottom);
        if (o.right < o.left)   o.right = o.left;
        if (o.bottom < o.top)   o.bottom = o.top;
        return o;
    };

    if (direction != DIRECTION_INSIDE && borderExpansion > 0) {
        extra->output->flags |= PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS;

        expand_rect(result_rect, borderExpansion);

        // Keep result_rect within max_rect. If upstream limits max_result_rect to the layer bounds,
        // we cannot output beyond that. With PF_OutFlag2_REVEALS_ZERO_ALPHA, AE is less likely to
        // crop max_result_rect for Shape/Text layers.
        result_rect = intersect_rect(result_rect, max_rect);
    }

    extra->output->result_rect     = result_rect;
    extra->output->max_result_rect = max_rect;

    PF_CHECKIN_PARAM(in_data, &thickness_param);
    PF_CHECKIN_PARAM(in_data, &direction_param);

    return err;
}
static PF_Err
SmartRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

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

        // STEP 1: Clear the output buffer to transparency
        if (PF_WORLD_IS_DEEP(output)) {
            for (A_long y = 0; y < output->height; y++) {
                PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + y * output->rowbytes);
                for (A_long x = 0; x < output->width; x++) {
                    outData[x].alpha = 0;
                    outData[x].red = 0;
                    outData[x].green = 0;
                    outData[x].blue = 0;
                }
            }
        }
        else {
            for (A_long y = 0; y < output->height; y++) {
                PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + y * output->rowbytes);
                for (A_long x = 0; x < output->width; x++) {
                    outData[x].alpha = 0;
                    outData[x].red = 0;
                    outData[x].green = 0;
                    outData[x].blue = 0;
                }
            }
        }

        // STEP 2: Copy the source layer if not "show line only"
        if (!showLineOnly) {
            if (PF_WORLD_IS_DEEP(output)) {
                for (A_long oy = 0; oy < output->height; ++oy) {
                    A_long iy = oy - offsetY;
                    if (iy < 0 || iy >= input->height) continue;

                    PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + oy * output->rowbytes);
                    PF_Pixel16* inData  = (PF_Pixel16*)((char*)input->data  + iy * input->rowbytes);

                    for (A_long ox = 0; ox < output->width; ++ox) {
                        A_long ix = ox - offsetX;
                        if (ix < 0 || ix >= input->width) continue;
                        outData[ox] = inData[ix];
                    }
                }
            }
            else {
                for (A_long oy = 0; oy < output->height; ++oy) {
                    A_long iy = oy - offsetY;
                    if (iy < 0 || iy >= input->height) continue;

                    PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + oy * output->rowbytes);
                    PF_Pixel8* inData  = (PF_Pixel8*)((char*)input->data  + iy * input->rowbytes);

                    for (A_long ox = 0; ox < output->width; ++ox) {
                        A_long ix = ox - offsetX;
                        if (ix < 0 || ix >= input->width) continue;
                        outData[ox] = inData[ix];
                    }
                }
            }
        }

        // STEP 3: Draw the border using signed distance. We supersample 2x2 per pixel
        // with bilinear SDF lookup to get stable AA on diagonals and at Inside/Outside edges.
        if (strokeThicknessF <= 0.0f) {
            ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, checkout_id));
            return err; // stroke width 0 → nothing to draw
        }
        const float AA_RANGE = 1.0f; // feather width (pixels)

        auto sampleSDF = [&](float fx, float fy) -> float {
            // Bilinear sample of signedDist (stored scaled by 10)
            fx = CLAMP(fx, 0.0f, input->width  - 1.001f);
            fy = CLAMP(fy, 0.0f, input->height - 1.001f);
            int x0 = (int)fx;
            int y0 = (int)fy;
            int x1 = MIN(x0 + 1, input->width  - 1);
            int y1 = MIN(y0 + 1, input->height - 1);
            float tx = fx - x0;
            float ty = fy - y0;

            int d00 = signedDist[y0 * input->width + x0];
            int d10 = signedDist[y0 * input->width + x1];
            int d01 = signedDist[y1 * input->width + x0];
            int d11 = signedDist[y1 * input->width + x1];

            float d0 = d00 + (d10 - d00) * tx;
            float d1 = d01 + (d11 - d01) * tx;
            float d = d0 + (d1 - d0) * ty;
            // Convert back to pixels. Our SDF is based on EDT between pixel centers,
            // so the boundary is halfway between opposite-class pixels.
            // Adjust by ±0.5px so distance is measured to the boundary (pixel edge).
            float sdf = d * 0.1f;
            if (sdf > 0.0f) sdf -= 0.5f;
            else if (sdf < 0.0f) sdf += 0.5f;
            return sdf;
        };

        const float sampleOffsets[4][2] = {
            {-0.25f, -0.25f}, { 0.25f, -0.25f},
            {-0.25f,  0.25f}, { 0.25f,  0.25f}
        };

        auto sdfCenterAdjusted = [&](A_long ix, A_long iy) -> float {
            // Center sample without bilinear (fast path), adjusted to boundary (pixel edge).
            ix = CLAMP(ix, (A_long)0, input->width - 1);
            iy = CLAMP(iy, (A_long)0, input->height - 1);
            float sdf = signedDist[(size_t)iy * (size_t)input->width + (size_t)ix] * 0.1f;
            if (sdf > 0.0f) sdf -= 0.5f;
            else if (sdf < 0.0f) sdf += 0.5f;
            return sdf;
        };

        const float MAX_EVAL_DIST = strokeThicknessF + AA_RANGE + 1.0f; // small guard band

        auto strokeSampleCoverage = [&](float sdf) -> float {
            // sdf: + inside, - outside (pixels, boundary at 0)
            // Build a soft half-space mask around the boundary to avoid hard sign cuts
            // which can look like a 1px lateral bias on AA edges due to interpolation.
            float sideMask = 1.0f;
            if (direction == DIRECTION_INSIDE) {
                // 0 when clearly outside, 1 when inside/on boundary
                sideMask = smoothstep(-AA_RANGE, 0.0f, sdf);
                sdf = CLAMP(sdf, 0.0f, 1e9f);
            } else if (direction == DIRECTION_OUTSIDE) {
                // 0 when clearly inside, 1 when outside/on boundary
                sideMask = 1.0f - smoothstep(0.0f, AA_RANGE, sdf);
                sdf = CLAMP(-sdf, 0.0f, 1e9f);
            } else {
                sdf = fabsf(sdf);
            }

            if (sdf > strokeThicknessF + AA_RANGE) return 0.0f;
            float stroke = 1.0f - smoothstep(strokeThicknessF, strokeThicknessF + AA_RANGE, sdf);
            return stroke * sideMask;
        };

        if (PF_WORLD_IS_DEEP(output)) {
            PF_Pixel16 edge_color;
            edge_color.alpha = PF_MAX_CHAN16;
            edge_color.red = PF_BYTE_TO_CHAR(color.red);
            edge_color.green = PF_BYTE_TO_CHAR(color.green);
            edge_color.blue = PF_BYTE_TO_CHAR(color.blue);

            for (A_long oy = 0; oy < output->height; ++oy) {
                PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + oy * output->rowbytes);
                for (A_long ox = 0; ox < output->width; ++ox) {
                    A_long ix = ox - offsetX;
                    A_long iy = oy - offsetY;
                    if (ix < 0 || ix >= input->width || iy < 0 || iy >= input->height) continue;

                    PF_Pixel16 dst = outData[ox];
                    float dstAlphaNorm = dst.alpha / (float)PF_MAX_CHAN16;

                    // Fast reject: most pixels are far from the boundary, avoid 2x2 bilinear sampling.
                    {
                        float sdfC = sdfCenterAdjusted(ix, iy);
                        float distC;
                        if (direction == DIRECTION_INSIDE) {
                            if (sdfC < -MAX_EVAL_DIST) continue;
                            distC = (sdfC > 0.0f) ? sdfC : 0.0f;
                        } else if (direction == DIRECTION_OUTSIDE) {
                            if (sdfC > MAX_EVAL_DIST) continue;
                            distC = (sdfC < 0.0f) ? -sdfC : 0.0f;
                        } else {
                            distC = fabsf(sdfC);
                        }
                        if (distC > MAX_EVAL_DIST) continue;
                    }

                    // 2x2 supersample
                    float strokeCoverage = 0.0f;
                    for (int s = 0; s < 4; ++s) {
                        // Sample in input pixel space (x/y are input indices).
                        float fx = (float)ix + 0.5f + sampleOffsets[s][0];
                        float fy = (float)iy + 0.5f + sampleOffsets[s][1];
                        float sdf = sampleSDF(fx, fy); // + inside, - outside
                        strokeCoverage += strokeSampleCoverage(sdf);
                    }

                    strokeCoverage *= 0.25f; // average 4 samples

                    if (strokeCoverage < 0.001f) continue;

                    if (showLineOnly) {
                        // Line only: just draw the stroke (premultiplied)
                        dst.red   = (A_u_short)(edge_color.red   * strokeCoverage);
                        dst.green = (A_u_short)(edge_color.green * strokeCoverage);
                        dst.blue  = (A_u_short)(edge_color.blue  * strokeCoverage);
                        dst.alpha = (A_u_short)(PF_MAX_CHAN16    * strokeCoverage);
                    } else {
                        // Proper premultiplied alpha compositing: stroke OVER dst
                        float strokeA    = strokeCoverage;
                        float invStrokeA = 1.0f - strokeA;

                        float dstR = dst.red   / (float)PF_MAX_CHAN16; // already premultiplied
                        float dstG = dst.green / (float)PF_MAX_CHAN16;
                        float dstB = dst.blue  / (float)PF_MAX_CHAN16;

                        float strokeR = edge_color.red   / (float)PF_MAX_CHAN16;
                        float strokeG = edge_color.green / (float)PF_MAX_CHAN16;
                        float strokeB = edge_color.blue  / (float)PF_MAX_CHAN16;

                        float outA = strokeA + dstAlphaNorm * invStrokeA;

                        // Composite premultiplied: dst is already premultiplied, so weight by (1-strokeA) only.
                        float outR = strokeR * strokeA + dstR * invStrokeA;
                        float outG = strokeG * strokeA + dstG * invStrokeA;
                        float outB = strokeB * strokeA + dstB * invStrokeA;

                        dst.alpha = (A_u_short)(CLAMP(outA, 0.0f, 1.0f) * PF_MAX_CHAN16 + 0.5f);
                        dst.red   = (A_u_short)(CLAMP(outR, 0.0f, 1.0f) * PF_MAX_CHAN16 + 0.5f);
                        dst.green = (A_u_short)(CLAMP(outG, 0.0f, 1.0f) * PF_MAX_CHAN16 + 0.5f);
                        dst.blue  = (A_u_short)(CLAMP(outB, 0.0f, 1.0f) * PF_MAX_CHAN16 + 0.5f);
                    }

                    outData[ox] = dst;
                }
            }
        }
        else {
            for (A_long oy = 0; oy < output->height; ++oy) {
                PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + oy * output->rowbytes);
                for (A_long ox = 0; ox < output->width; ++ox) {
                    A_long ix = ox - offsetX;
                    A_long iy = oy - offsetY;
                    if (ix < 0 || ix >= input->width || iy < 0 || iy >= input->height) continue;

                    PF_Pixel8 dst = outData[ox];
                    float dstAlphaNorm = dst.alpha / (float)PF_MAX_CHAN8;

                    // Fast reject: most pixels are far from the boundary, avoid 2x2 bilinear sampling.
                    {
                        float sdfC = sdfCenterAdjusted(ix, iy);
                        float distC;
                        if (direction == DIRECTION_INSIDE) {
                            if (sdfC < -MAX_EVAL_DIST) continue;
                            distC = (sdfC > 0.0f) ? sdfC : 0.0f;
                        } else if (direction == DIRECTION_OUTSIDE) {
                            if (sdfC > MAX_EVAL_DIST) continue;
                            distC = (sdfC < 0.0f) ? -sdfC : 0.0f;
                        } else {
                            distC = fabsf(sdfC);
                        }
                        if (distC > MAX_EVAL_DIST) continue;
                    }

                    float strokeCoverage = 0.0f;
                    for (int s = 0; s < 4; ++s) {
                        // Sample in input pixel space (x/y are input indices).
                        float fx = (float)ix + 0.5f + sampleOffsets[s][0];
                        float fy = (float)iy + 0.5f + sampleOffsets[s][1];
                        float sdf = sampleSDF(fx, fy);
                        strokeCoverage += strokeSampleCoverage(sdf);
                    }

                    strokeCoverage *= 0.25f;

                    if (strokeCoverage < 0.001f) continue;
                    
                    if (showLineOnly) {
                        // Line only: just draw the stroke (premultiplied)
                        dst.red   = (A_u_char)(color.red   * strokeCoverage);
                        dst.green = (A_u_char)(color.green * strokeCoverage);
                        dst.blue  = (A_u_char)(color.blue  * strokeCoverage);
                        dst.alpha = (A_u_char)(PF_MAX_CHAN8 * strokeCoverage);
                    } else {
                        // Proper premultiplied alpha compositing: stroke OVER dst
                        float strokeA    = strokeCoverage;
                        float invStrokeA = 1.0f - strokeA;

                        float dstR = dst.red   / (float)PF_MAX_CHAN8; // premultiplied
                        float dstG = dst.green / (float)PF_MAX_CHAN8;
                        float dstB = dst.blue  / (float)PF_MAX_CHAN8;

                        float strokeR = color.red   / (float)PF_MAX_CHAN8;
                        float strokeG = color.green / (float)PF_MAX_CHAN8;
                        float strokeB = color.blue  / (float)PF_MAX_CHAN8;

                        float outA = strokeA + dstAlphaNorm * invStrokeA;

                        float outR = strokeR * strokeA + dstR * invStrokeA;
                        float outG = strokeG * strokeA + dstG * invStrokeA;
                        float outB = strokeB * strokeA + dstB * invStrokeA;

                        dst.alpha = (A_u_char)(CLAMP(outA, 0.0f, 1.0f) * PF_MAX_CHAN8 + 0.5f);
                        dst.red   = (A_u_char)(CLAMP(outR, 0.0f, 1.0f) * PF_MAX_CHAN8 + 0.5f);
                        dst.green = (A_u_char)(CLAMP(outG, 0.0f, 1.0f) * PF_MAX_CHAN8 + 0.5f);
                        dst.blue  = (A_u_char)(CLAMP(outB, 0.0f, 1.0f) * PF_MAX_CHAN8 + 0.5f);
                    }

                    outData[ox] = dst;
                }
            }
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

        case PF_Cmd_RENDER:
            err = Render(in_data, out_data, params, output);
            break;

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
