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
    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

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
        1,
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
static PF_Err
ComputeSignedDistanceField(
    PF_EffectWorld* input,
    A_u_char threshold8,
    A_u_short threshold16,
    std::vector<int>& signedDist, // scaled by 10
    A_long width,
    A_long height)
{
    const int INF = std::numeric_limits<int>::max() / 4;
    signedDist.assign(width * height, 0);

    auto idx = [width](A_long x, A_long y) { return y * width + x; };

    std::vector<int> distInside(width * height, INF);
    std::vector<int> distOutside(width * height, INF);

    auto isSolid = [&](A_long x, A_long y)->bool {
        if (PF_WORLD_IS_DEEP(input)) {
            PF_Pixel16* p = (PF_Pixel16*)((char*)input->data + y * input->rowbytes) + x;
            return p->alpha > threshold16;
        }
        else {
            PF_Pixel8* p = (PF_Pixel8*)((char*)input->data + y * input->rowbytes) + x;
            return p->alpha > threshold8;
        }
    };

    for (A_long y = 0; y < height; ++y) {
        for (A_long x = 0; x < width; ++x) {
            bool solid = isSolid(x, y);
            if (solid) {
                distOutside[idx(x, y)] = 0; // seeds for outside distances
            }
            else {
                distInside[idx(x, y)] = 0;  // seeds for inside distances
            }
        }
    }

    const int wStraight = 10; // scaled by 10
    const int wDiag = 14;     // approx sqrt(2)*10

    auto forwardPass = [&](std::vector<int>& dist) {
        for (A_long y = 0; y < height; ++y) {
            for (A_long x = 0; x < width; ++x) {
                int v = dist[idx(x, y)];
                if (v == 0) continue;
                int best = v;
                if (x > 0)                 best = std::min(best, dist[idx(x - 1, y)] + wStraight);
                if (y > 0)                 best = std::min(best, dist[idx(x, y - 1)] + wStraight);
                if (x > 0 && y > 0)        best = std::min(best, dist[idx(x - 1, y - 1)] + wDiag);
                if (x + 1 < width && y > 0)best = std::min(best, dist[idx(x + 1, y - 1)] + wDiag);
                dist[idx(x, y)] = best;
            }
        }
    };

    auto backwardPass = [&](std::vector<int>& dist) {
        for (A_long y = height - 1; y >= 0; --y) {
            for (A_long x = width - 1; x >= 0; --x) {
                int v = dist[idx(x, y)];
                int best = v;
                if (x + 1 < width)          best = std::min(best, dist[idx(x + 1, y)] + wStraight);
                if (y + 1 < height)         best = std::min(best, dist[idx(x, y + 1)] + wStraight);
                if (x + 1 < width && y + 1 < height) best = std::min(best, dist[idx(x + 1, y + 1)] + wDiag);
                if (x > 0 && y + 1 < height) best = std::min(best, dist[idx(x - 1, y + 1)] + wDiag);
                dist[idx(x, y)] = best;
            }
        }
    };

    forwardPass(distInside);
    forwardPass(distOutside);
    backwardPass(distInside);
    backwardPass(distOutside);

    signedDist.resize(width * height);
    for (A_long y = 0; y < height; ++y) {
        for (A_long x = 0; x < width; ++x) {
            bool solid = isSolid(x, y);
            int d = solid ? distInside[idx(x, y)] : distOutside[idx(x, y)];
            signedDist[idx(x, y)] = solid ? d : -d;
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

    ERR(extra->cb->checkout_layer(in_data->effect_ref, BORDER_INPUT, BORDER_INPUT, &req, in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));

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

    extra->output->pre_render_data = (void*)(intptr_t)BORDER_INPUT;

    PF_Rect request_rect = req.rect;

    extra->output->result_rect = in_result.result_rect;

    extra->output->result_rect.left = MAX(request_rect.left, in_result.result_rect.left - borderExpansion);
    extra->output->result_rect.top = MAX(request_rect.top, in_result.result_rect.top - borderExpansion);
    extra->output->result_rect.right = MIN(request_rect.right, in_result.result_rect.right + borderExpansion);
    extra->output->result_rect.bottom = MIN(request_rect.bottom, in_result.result_rect.bottom + borderExpansion);

    extra->output->max_result_rect = in_result.max_result_rect;
    extra->output->max_result_rect.left = MAX(request_rect.left, in_result.max_result_rect.left - borderExpansion);
    extra->output->max_result_rect.top = MAX(request_rect.top, in_result.max_result_rect.top - borderExpansion);
    extra->output->max_result_rect.right = MIN(request_rect.right, in_result.max_result_rect.right + borderExpansion);
    extra->output->max_result_rect.bottom = MIN(request_rect.bottom, in_result.max_result_rect.bottom + borderExpansion);

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
    // For BOTH we split thickness half inside/outside to match visual width
    float strokeThicknessF = (direction == DIRECTION_BOTH) ? thicknessF * 0.5f : thicknessF;

    if (input && output) {
        // Calculate offset between input and output
        A_long offsetX = input->origin_x - output->origin_x;
        A_long offsetY = input->origin_y - output->origin_y;

        // Generate signed distance field (fast chamfer, scaled by 10)
        std::vector<int> signedDist;
        A_u_short threshold16 = threshold * 257;
        ERR(ComputeSignedDistanceField(input, threshold, threshold16, signedDist,
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
                for (A_long y = 0; y < input->height; y++) {
                    A_long outY = y + offsetY;

                    if (outY < 0 || outY >= output->height) continue;

                    PF_Pixel16* inData = (PF_Pixel16*)((char*)input->data + y * input->rowbytes);
                    PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + outY * output->rowbytes);

                    for (A_long x = 0; x < input->width; x++) {
                        A_long outX = x + offsetX;

                        if (outX < 0 || outX >= output->width) continue;

                        outData[outX] = inData[x];
                    }
                }
            }
            else {
                for (A_long y = 0; y < input->height; y++) {
                    A_long outY = y + offsetY;

                    if (outY < 0 || outY >= output->height) continue;

                    PF_Pixel8* inData = (PF_Pixel8*)((char*)input->data + y * input->rowbytes);
                    PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + outY * output->rowbytes);

                    for (A_long x = 0; x < input->width; x++) {
                        A_long outX = x + offsetX;

                        if (outX < 0 || outX >= output->width) continue;

                        outData[outX] = inData[x];
                    }
                }
            }
        }

        // STEP 3: Draw the border using signed distance (fast, O(N))
        const float AA_RANGE = 1.0f; // 1 pixel smoothing

        if (PF_WORLD_IS_DEEP(output)) {
            PF_Pixel16 edge_color;
            edge_color.alpha = PF_MAX_CHAN16;
            edge_color.red = PF_BYTE_TO_CHAR(color.red);
            edge_color.green = PF_BYTE_TO_CHAR(color.green);
            edge_color.blue = PF_BYTE_TO_CHAR(color.blue);

            for (A_long y = 0; y < input->height; y++) {
                A_long outY = y + offsetY;
                if (outY < 0 || outY >= output->height) continue;
                PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + outY * output->rowbytes);
                PF_Pixel16* srcData = (PF_Pixel16*)((char*)input->data + y * input->rowbytes);

                for (A_long x = 0; x < input->width; x++) {
                    A_long outX = x + offsetX;
                    if (outX < 0 || outX >= output->width) continue;

                    int signed10 = signedDist[y * input->width + x];
                    float sdf = signed10 * 0.1f;          // + inside, - outside

                    // Select which side to draw
                    float dist;
                    switch (direction) {
                    case DIRECTION_INSIDE:
                        if (sdf <= 0.0f) continue; // only inside
                        dist = sdf;
                        break;
                    case DIRECTION_OUTSIDE:
                        if (sdf >= 0.0f) continue; // only outside
                        dist = -sdf;
                        break;
                    default: // both
                        dist = fabsf(sdf);
                        break;
                    }

                    if (dist > strokeThicknessF + AA_RANGE) continue;

                    // Edge fade-in (different slope for outside to avoid gaps)
                    float aEdge = (direction == DIRECTION_OUTSIDE)
                        ? 1.0f                                // keep full strength at the edge for outside
                        : smoothstep(0.0f, AA_RANGE, dist);   // fade in for inside/both
                    // Fade-out after stroke thickness
                    float aOut = 1.0f - smoothstep(strokeThicknessF, strokeThicknessF + AA_RANGE, dist);
                    float coverage = aEdge * aOut;

                    PF_Pixel16 src = srcData[x];
                    PF_Pixel16 dst = outData[outX];

                    // Blend stroke over existing output (which may already contain source)
                    // For OUTSIDE, don't paint over opaque source pixels
                    dst.red   = (A_u_short)(edge_color.red   * coverage + dst.red   * (1.0f - coverage));
                    dst.green = (A_u_short)(edge_color.green * coverage + dst.green * (1.0f - coverage));
                    dst.blue  = (A_u_short)(edge_color.blue  * coverage + dst.blue  * (1.0f - coverage));

                    if (direction == DIRECTION_INSIDE) {
                        dst.alpha = MAX(src.alpha, (A_u_short)(PF_MAX_CHAN16 * coverage));
                    } else if (direction == DIRECTION_OUTSIDE) {
                        // treat stroke as behind: keep existing alpha, add stroke alpha where transparent
                        dst.alpha = MAX(dst.alpha, (A_u_short)(PF_MAX_CHAN16 * coverage));
                    } else { // BOTH
                        dst.alpha = MAX(dst.alpha, (A_u_short)(PF_MAX_CHAN16 * coverage));
                    }

                    outData[outX] = dst;
                }
            }
        }
        else {
            for (A_long y = 0; y < input->height; y++) {
                A_long outY = y + offsetY;
                if (outY < 0 || outY >= output->height) continue;
                PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + outY * output->rowbytes);
                PF_Pixel8* srcData = (PF_Pixel8*)((char*)input->data + y * input->rowbytes);

                for (A_long x = 0; x < input->width; x++) {
                    A_long outX = x + offsetX;
                    if (outX < 0 || outX >= output->width) continue;

                    int signed10 = signedDist[y * input->width + x];
                    float sdf = signed10 * 0.1f;

                    float dist;
                    switch (direction) {
                    case DIRECTION_INSIDE:
                        if (sdf <= 0.0f) continue;
                        dist = sdf;
                        break;
                    case DIRECTION_OUTSIDE:
                        if (sdf >= 0.0f) continue;
                        dist = -sdf;
                        break;
                    default:
                        dist = fabsf(sdf);
                        break;
                    }

                    if (dist > strokeThicknessF + AA_RANGE) continue;

                    float aEdge = (direction == DIRECTION_OUTSIDE)
                        ? 1.0f
                        : smoothstep(0.0f, AA_RANGE, dist);
                    float aOut = 1.0f - smoothstep(strokeThicknessF, strokeThicknessF + AA_RANGE, dist);
                    float coverage = aEdge * aOut;

                    PF_Pixel8 src = srcData[x];
                    PF_Pixel8 dst = outData[outX];

                    dst.red   = (A_u_char)(color.red   * coverage + dst.red   * (1.0f - coverage));
                    dst.green = (A_u_char)(color.green * coverage + dst.green * (1.0f - coverage));
                    dst.blue  = (A_u_char)(color.blue  * coverage + dst.blue  * (1.0f - coverage));

                    if (direction == DIRECTION_INSIDE) {
                        dst.alpha = MAX(src.alpha, (A_u_char)(PF_MAX_CHAN8 * coverage));
                    } else if (direction == DIRECTION_OUTSIDE) {
                        dst.alpha = MAX(dst.alpha, (A_u_char)(PF_MAX_CHAN8 * coverage));
                    } else {
                        dst.alpha = MAX(dst.alpha, (A_u_char)(PF_MAX_CHAN8 * coverage));
                    }

                    outData[outX] = dst;
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
