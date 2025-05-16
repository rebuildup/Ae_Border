#include "Border.h"
#include <vector>
#include <algorithm>
#include <cfloat>

#ifdef _WIN32
#include <omp.h>
#endif

struct EdgePoint {
    A_long x, y;
    bool isTransparent;
};

template <typename T>
T CLAMP(T value, T min, T max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
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
    out_data->my_version = PF_VERSION(MAJOR_VERSION,
        MINOR_VERSION,
        BUG_VERSION,
        STAGE_VERSION,
        BUILD_VERSION);

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
static PF_Boolean
IsEdgePixel8(
    PF_EffectWorld* input,
    A_long         x,
    A_long         y,
    A_u_char       threshold,
    A_long         thickness,
    A_long         direction)
{
    if (thickness <= 0) {
        return FALSE;
    }

    A_long effectiveThickness = thickness;
    if (direction == DIRECTION_BOTH) {
        effectiveThickness = (thickness + 1) / 2;
    }

    PF_Boolean isTransparent = TRUE;
    PF_Boolean isValidPosition = FALSE;

    if (x >= 0 && y >= 0 && x < input->width && y < input->height) {
        isValidPosition = TRUE;
        PF_Pixel8* currentPixel = (PF_Pixel8*)((char*)input->data + y * input->rowbytes + x * sizeof(PF_Pixel8));
        isTransparent = (currentPixel->alpha <= threshold);
    }

    if (!isValidPosition && direction == DIRECTION_INSIDE) {
        return FALSE;
    }

    if (isValidPosition) {
        if ((direction == DIRECTION_INSIDE && isTransparent) ||
            (direction == DIRECTION_OUTSIDE && !isTransparent)) {
            return FALSE;
        }
    }

    float minDistSquared = (float)(effectiveThickness * effectiveThickness + 1);
    PF_Boolean foundEdge = FALSE;

    A_long searchRadius = effectiveThickness + 1;

    for (A_long dy = -searchRadius; dy <= searchRadius; dy++) {
        for (A_long dx = -searchRadius; dx <= searchRadius; dx++) {
            if (dx == 0 && dy == 0) continue;

            A_long distSquared = dx * dx + dy * dy;
            if (distSquared > effectiveThickness * effectiveThickness) {
                continue;
            }

            A_long nx = x + dx;
            A_long ny = y + dy;

            PF_Boolean neighborIsTransparent = TRUE;
            PF_Boolean neighborIsValid = FALSE;

            if (nx >= 0 && ny >= 0 && nx < input->width && ny < input->height) {
                neighborIsValid = TRUE;
                PF_Pixel8* neighborPixel = (PF_Pixel8*)((char*)input->data + ny * input->rowbytes + nx * sizeof(PF_Pixel8));
                neighborIsTransparent = (neighborPixel->alpha <= threshold);
            }

            if (!isValidPosition && !neighborIsValid) {
                continue;
            }

            PF_Boolean isEdge = FALSE;

            switch (direction) {
            case DIRECTION_BOTH:
                isEdge = (isTransparent != neighborIsTransparent);
                break;
            case DIRECTION_INSIDE:
                isEdge = (!isTransparent && neighborIsTransparent);
                break;
            case DIRECTION_OUTSIDE:
                isEdge = (isTransparent && !neighborIsTransparent);
                break;
            }

            if (isEdge) {
                if (distSquared < minDistSquared) {
                    minDistSquared = (float)distSquared;
                    foundEdge = TRUE;
                }
            }
        }
    }

    return (foundEdge && minDistSquared <= (effectiveThickness * effectiveThickness));
}
static PF_Boolean
IsEdgePixel16(
    PF_EffectWorld* input,
    A_long         x,
    A_long         y,
    A_u_short      threshold,
    A_long         thickness,
    A_long         direction)
{
    if (thickness <= 0) {
        return FALSE;
    }

    A_long effectiveThickness = thickness;
    if (direction == DIRECTION_BOTH) {
        effectiveThickness = (thickness + 1) / 2;
    }

    PF_Boolean isTransparent = TRUE;
    PF_Boolean isValidPosition = FALSE;

    if (x >= 0 && y >= 0 && x < input->width && y < input->height) {
        isValidPosition = TRUE;
        PF_Pixel16* currentPixel = (PF_Pixel16*)((char*)input->data + y * input->rowbytes + x * sizeof(PF_Pixel16));
        isTransparent = (currentPixel->alpha <= threshold);
    }

    if (!isValidPosition && direction == DIRECTION_INSIDE) {
        return FALSE;
    }

    if (isValidPosition) {
        if ((direction == DIRECTION_INSIDE && isTransparent) ||
            (direction == DIRECTION_OUTSIDE && !isTransparent)) {
            return FALSE;
        }
    }

    float minDistSquared = (float)(effectiveThickness * effectiveThickness + 1);
    PF_Boolean foundEdge = FALSE;

    A_long searchRadius = effectiveThickness + 1;

    for (A_long dy = -searchRadius; dy <= searchRadius; dy++) {
        for (A_long dx = -searchRadius; dx <= searchRadius; dx++) {
            if (dx == 0 && dy == 0) continue;

            A_long distSquared = dx * dx + dy * dy;
            if (distSquared > effectiveThickness * effectiveThickness) {
                continue;
            }

            A_long nx = x + dx;
            A_long ny = y + dy;

            PF_Boolean neighborIsTransparent = TRUE;
            PF_Boolean neighborIsValid = FALSE;

            if (nx >= 0 && ny >= 0 && nx < input->width && ny < input->height) {
                neighborIsValid = TRUE;
                PF_Pixel16* neighborPixel = (PF_Pixel16*)((char*)input->data + ny * input->rowbytes + nx * sizeof(PF_Pixel16));
                neighborIsTransparent = (neighborPixel->alpha <= threshold);
            }

            if (!isValidPosition && !neighborIsValid) {
                continue;
            }

            PF_Boolean isEdge = FALSE;

            switch (direction) {
            case DIRECTION_BOTH:
                isEdge = (isTransparent != neighborIsTransparent);
                break;
            case DIRECTION_INSIDE:
                isEdge = (!isTransparent && neighborIsTransparent);
                break;
            case DIRECTION_OUTSIDE:
                isEdge = (isTransparent && !neighborIsTransparent);
                break;
            }

            if (isEdge) {
                if (distSquared < minDistSquared) {
                    minDistSquared = (float)distSquared;
                    foundEdge = TRUE;
                }
            }
        }
    }

    return (foundEdge && minDistSquared <= (effectiveThickness * effectiveThickness));
}

static PF_Err
GenerateDistanceField(
    PF_EffectWorld* input,
    A_u_char threshold8,
    A_u_short threshold16,
    A_long direction,
    std::vector<float>& distanceField,
    A_long width,
    A_long height,
    A_long offsetX,
    A_long offsetY)
{
    PF_Err err = PF_Err_NONE;

    const float MAX_DISTANCE = FLT_MAX;
    distanceField.resize(width * height, MAX_DISTANCE);

    std::vector<EdgePoint> edgePoints;
    edgePoints.reserve(width + height);

    for (A_long y = 0; y < height; y++) {
        for (A_long x = 0; x < width; x++) {
            A_long inX = x - offsetX;
            A_long inY = y - offsetY;

            if (inX < -1 || inY < -1 || inX > input->width || inY > input->height) {
                continue;
            }

            bool isTransparent = true;
            bool isValidPosition = false;

            if (inX >= 0 && inY >= 0 && inX < input->width && inY < input->height) {
                isValidPosition = true;
                if (PF_WORLD_IS_DEEP(input)) {
                    PF_Pixel16* pixel = (PF_Pixel16*)((char*)input->data + inY * input->rowbytes + inX * sizeof(PF_Pixel16));
                    isTransparent = (pixel->alpha <= threshold16);
                }
                else {
                    PF_Pixel8* pixel = (PF_Pixel8*)((char*)input->data + inY * input->rowbytes + inX * sizeof(PF_Pixel8));
                    isTransparent = (pixel->alpha <= threshold8);
                }
            }

            if (!isValidPosition && direction == DIRECTION_INSIDE) {
                continue;
            }

            if (isValidPosition) {
                if ((direction == DIRECTION_INSIDE && isTransparent) ||
                    (direction == DIRECTION_OUTSIDE && !isTransparent)) {
                    continue;
                }
            }

            bool foundEdge = false;

            const int dx[4] = { 1, 0, -1, 0 };
            const int dy[4] = { 0, 1, 0, -1 };

            for (int i = 0; i < 4; i++) {
                A_long nx = inX + dx[i];
                A_long ny = inY + dy[i];

                bool neighborTransparent = true;
                bool neighborIsValid = false;

                if (nx >= 0 && ny >= 0 && nx < input->width && ny < input->height) {
                    neighborIsValid = true;
                    if (PF_WORLD_IS_DEEP(input)) {
                        PF_Pixel16* neighbor = (PF_Pixel16*)((char*)input->data + ny * input->rowbytes + nx * sizeof(PF_Pixel16));
                        neighborTransparent = (neighbor->alpha <= threshold16);
                    }
                    else {
                        PF_Pixel8* neighbor = (PF_Pixel8*)((char*)input->data + ny * input->rowbytes + nx * sizeof(PF_Pixel8));
                        neighborTransparent = (neighbor->alpha <= threshold8);
                    }
                }

                if (!isValidPosition && !neighborIsValid) {
                    continue;
                }

                bool isEdge = false;

                switch (direction) {
                case DIRECTION_BOTH:
                    isEdge = (isTransparent != neighborTransparent);
                    break;
                case DIRECTION_INSIDE:
                    isEdge = (!isTransparent && neighborTransparent);
                    break;
                case DIRECTION_OUTSIDE:
                    isEdge = (isTransparent && !neighborTransparent);
                    break;
                }

                if (isEdge) {
                    foundEdge = true;
                    break;
                }
            }

            if (foundEdge) {
                EdgePoint ep = { x, y, isTransparent };
                edgePoints.push_back(ep);
                distanceField[y * width + x] = 0.0f;
            }
        }
    }

    if (edgePoints.empty()) {
        return err;
    }

#ifdef _WIN32
#pragma omp parallel for
#endif
    for (int i = 0; i < edgePoints.size(); i++) {
        EdgePoint& ep = edgePoints[i];

        const int regionSize = 50;

        for (A_long dy = -regionSize; dy <= regionSize; dy++) {
            for (A_long dx = -regionSize; dx <= regionSize; dx++) {
                A_long nx = ep.x + dx;
                A_long ny = ep.y + dy;

                if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                    continue;
                }

                float distSq = dx * dx + dy * dy;

                if (distSq > regionSize * regionSize) {
                    continue;
                }

                unsigned int index = ny * width + nx;
#ifdef _WIN32
#pragma omp critical
#endif
                {
                    if (distSq < distanceField[index]) {
                        distanceField[index] = distSq;
                    }
                }
            }
        }
    }

#ifdef _WIN32
#pragma omp parallel for
#endif
    for (A_long i = 0; i < width * height; i++) {
        if (distanceField[i] < MAX_DISTANCE) {
            distanceField[i] = sqrtf(distanceField[i]);
        }
        else {
            distanceField[i] = -1.0f;
        }
    }

    return err;
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
        pixelThickness = thickness;
    }
    else {
        float normalizedThickness = (thickness - 10.0f) / 90.0f;
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
        pixelThickness = thickness;
    }
    else {
        float normalizedThickness = (thickness - 10.0f) / 90.0f;
        pixelThickness = 10.0f + 40.0f * sqrtf(normalizedThickness);
    }

    A_long thicknessInt = (A_long)(pixelThickness / resolution_factor + 0.5f);

    if (input && output) {
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
            A_long offsetX = input->origin_x - output->origin_x;
            A_long offsetY = input->origin_y - output->origin_y;

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

        // STEP 3: Draw the border
        A_long search_margin = thicknessInt + 5;
        A_long search_left = -search_margin;
        A_long search_top = -search_margin;
        A_long search_right = input->width + search_margin;
        A_long search_bottom = input->height + search_margin;

        A_long offsetX = input->origin_x - output->origin_x;
        A_long offsetY = input->origin_y - output->origin_y;

        if (PF_WORLD_IS_DEEP(output)) {
            A_u_short threshold16 = threshold * 257;
            PF_Pixel16 edge_color;

            edge_color.alpha = PF_MAX_CHAN16;
            edge_color.red = PF_BYTE_TO_CHAR(color.red);
            edge_color.green = PF_BYTE_TO_CHAR(color.green);
            edge_color.blue = PF_BYTE_TO_CHAR(color.blue);

            for (A_long y = search_top; y < search_bottom; y++) {
                for (A_long x = search_left; x < search_right; x++) {
                    if (IsEdgePixel16(input, x, y, threshold16, thicknessInt, direction)) {
                        A_long outX = x + offsetX;
                        A_long outY = y + offsetY;

                        if (outX >= 0 && outX < output->width && outY >= 0 && outY < output->height) {
                            PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + outY * output->rowbytes);
                            outData[outX] = edge_color;
                        }
                    }
                }
            }
        }
        else {
            for (A_long y = search_top; y < search_bottom; y++) {
                for (A_long x = search_left; x < search_right; x++) {
                    if (IsEdgePixel8(input, x, y, threshold, thicknessInt, direction)) {
                        A_long outX = x + offsetX;
                        A_long outY = y + offsetY;

                        if (outX >= 0 && outX < output->width && outY >= 0 && outY < output->height) {
                            PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + outY * output->rowbytes);
                            outData[outX].alpha = PF_MAX_CHAN8;
                            outData[outX].red = color.red;
                            outData[outX].green = color.green;
                            outData[outX].blue = color.blue;
                        }
                    }
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
        "ADBE Border",
        "Border",
        AE_RESERVED_INFO,
        "EffectMain",
        "https://github.com/rebuildup/Ae_Border");

    return result;
}

PF_Err
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