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

// Define our own clamp function since AEFX_CLAMP isn't available
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

    // Enable critical flags for proper rendering
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

    // Stroke Width parameter - using float slider
    PF_ADD_FLOAT_SLIDERX(STR(StrID_Thickness_Param_Name),
        BORDER_THICKNESS_MIN,
        BORDER_THICKNESS_MAX,
        0,
        100,
        1,
        PF_Precision_TENTHS, // Control decimal places
        0,
        0,
        THICKNESS_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // Color parameter - Default to red
    def.flags = PF_ParamFlag_NONE;
    def.u.cd.value.red = 255;    // Red
    def.u.cd.value.green = 0;    // No Green
    def.u.cd.value.blue = 0;     // No Blue
    def.u.cd.value.alpha = 255;  // Full Alpha

    PF_ADD_COLOR(STR(StrID_Color_Param_Name),
        def.u.cd.value.red,
        def.u.cd.value.green,
        def.u.cd.value.blue,
        COLOR_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // Threshold parameter
    PF_ADD_SLIDER(STR(StrID_Threshold_Param_Name),
        BORDER_THRESHOLD_MIN,
        BORDER_THRESHOLD_MAX,
        0,
        255,
        0,
        THRESHOLD_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // Direction parameter - fixed format with pipe delimiter
    PF_ADD_POPUP(STR(StrID_Direction_Param_Name),
        3,                             // Number of choices
        DIRECTION_BOTH,                // Default choice (Both Directions)
        "Both Directions|Inside|Outside", // Options as pipe-delimited string
        DIRECTION_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // Show Line Only checkbox
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
    // Skip processing if thickness is 0
    if (thickness <= 0) {
        return FALSE;
    }

    // Get effective thickness - halve it for "Both Directions" mode
    A_long effectiveThickness = thickness;
    if (direction == DIRECTION_BOTH) {
        effectiveThickness = (thickness + 1) / 2; // Add 1 before division to round up for odd values
    }

    // For pixels outside the input buffer, we need custom handling
    PF_Boolean isTransparent = TRUE; // Default for out-of-bounds pixels
    PF_Boolean isValidPosition = FALSE;

    // Check if we're inside the input buffer
    if (x >= 0 && y >= 0 && x < input->width && y < input->height) {
        isValidPosition = TRUE;
        // Get current pixel's alpha value
        PF_Pixel8* currentPixel = (PF_Pixel8*)((char*)input->data + y * input->rowbytes + x * sizeof(PF_Pixel8));
        isTransparent = (currentPixel->alpha <= threshold);
    }

    // If we're outside the layer and want "inside" direction only, return early
    if (!isValidPosition && direction == DIRECTION_INSIDE) {
        return FALSE;
    }

    // If we're inside the layer, check direction
    if (isValidPosition) {
        // Early return based on direction and current pixel transparency
        if ((direction == DIRECTION_INSIDE && isTransparent) ||
            (direction == DIRECTION_OUTSIDE && !isTransparent)) {
            return FALSE;
        }
    }

    // Track the shortest distance to a transparency transition
    float minDistSquared = (float)(effectiveThickness * effectiveThickness + 1); // Start beyond the threshold
    PF_Boolean foundEdge = FALSE;

    // Use a larger search radius to ensure we don't miss any edges
    A_long searchRadius = effectiveThickness + 1;

    // First, let's find the closest edge by checking surrounding pixels
    for (A_long dy = -searchRadius; dy <= searchRadius; dy++) {
        for (A_long dx = -searchRadius; dx <= searchRadius; dx++) {
            // Skip the current pixel
            if (dx == 0 && dy == 0) continue;

            // Early distance check to speed up processing
            A_long distSquared = dx * dx + dy * dy;
            if (distSquared > effectiveThickness * effectiveThickness) {
                continue; // Skip if too far away
            }

            A_long nx = x + dx;
            A_long ny = y + dy;

            // Determine neighbor transparency
            PF_Boolean neighborIsTransparent = TRUE; // Default for out-of-bounds
            PF_Boolean neighborIsValid = FALSE;

            if (nx >= 0 && ny >= 0 && nx < input->width && ny < input->height) {
                neighborIsValid = TRUE;
                // Pixel is in bounds, get its actual value
                PF_Pixel8* neighborPixel = (PF_Pixel8*)((char*)input->data + ny * input->rowbytes + nx * sizeof(PF_Pixel8));
                neighborIsTransparent = (neighborPixel->alpha <= threshold);
            }

            // If both current position and neighbor are outside layer, skip
            if (!isValidPosition && !neighborIsValid) {
                continue;
            }

            // Check if this is an edge based on direction
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

            // If it's an edge, update the minimum distance
            if (isEdge) {
                if (distSquared < minDistSquared) {
                    minDistSquared = (float)distSquared;
                    foundEdge = TRUE;
                }
            }
        }
    }

    // If we found an edge and it's within the thickness, this is an edge pixel
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
    // Skip processing if thickness is 0
    if (thickness <= 0) {
        return FALSE;
    }

    // Get effective thickness - halve it for "Both Directions" mode
    A_long effectiveThickness = thickness;
    if (direction == DIRECTION_BOTH) {
        effectiveThickness = (thickness + 1) / 2; // Add 1 before division to round up for odd values
    }

    // For pixels outside the input buffer, we need custom handling
    PF_Boolean isTransparent = TRUE; // Default for out-of-bounds pixels
    PF_Boolean isValidPosition = FALSE;

    // Check if we're inside the input buffer
    if (x >= 0 && y >= 0 && x < input->width && y < input->height) {
        isValidPosition = TRUE;
        // Get current pixel's alpha value
        PF_Pixel16* currentPixel = (PF_Pixel16*)((char*)input->data + y * input->rowbytes + x * sizeof(PF_Pixel16));
        isTransparent = (currentPixel->alpha <= threshold);
    }

    // If we're outside the layer and want "inside" direction only, return early
    if (!isValidPosition && direction == DIRECTION_INSIDE) {
        return FALSE;
    }

    // If we're inside the layer, check direction
    if (isValidPosition) {
        // Early return based on direction and current pixel transparency
        if ((direction == DIRECTION_INSIDE && isTransparent) ||
            (direction == DIRECTION_OUTSIDE && !isTransparent)) {
            return FALSE;
        }
    }

    // Track the shortest distance to a transparency transition
    float minDistSquared = (float)(effectiveThickness * effectiveThickness + 1); // Start beyond the threshold
    PF_Boolean foundEdge = FALSE;

    // Use a larger search radius to ensure we don't miss any edges
    A_long searchRadius = effectiveThickness + 1;

    // First, let's find the closest edge by checking surrounding pixels
    for (A_long dy = -searchRadius; dy <= searchRadius; dy++) {
        for (A_long dx = -searchRadius; dx <= searchRadius; dx++) {
            // Skip the current pixel
            if (dx == 0 && dy == 0) continue;

            // Early distance check to speed up processing
            A_long distSquared = dx * dx + dy * dy;
            if (distSquared > effectiveThickness * effectiveThickness) {
                continue; // Skip if too far away
            }

            A_long nx = x + dx;
            A_long ny = y + dy;

            // Determine neighbor transparency
            PF_Boolean neighborIsTransparent = TRUE; // Default for out-of-bounds
            PF_Boolean neighborIsValid = FALSE;

            if (nx >= 0 && ny >= 0 && nx < input->width && ny < input->height) {
                neighborIsValid = TRUE;
                // Pixel is in bounds, get its actual value
                PF_Pixel16* neighborPixel = (PF_Pixel16*)((char*)input->data + ny * input->rowbytes + nx * sizeof(PF_Pixel16));
                neighborIsTransparent = (neighborPixel->alpha <= threshold);
            }

            // If both current position and neighbor are outside layer, skip
            if (!isValidPosition && !neighborIsValid) {
                continue;
            }

            // Check if this is an edge based on direction
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

            // If it's an edge, update the minimum distance
            if (isEdge) {
                if (distSquared < minDistSquared) {
                    minDistSquared = (float)distSquared;
                    foundEdge = TRUE;
                }
            }
        }
    }

    // If we found an edge and it's within the thickness, this is an edge pixel
    return (foundEdge && minDistSquared <= (effectiveThickness * effectiveThickness));
}

// Optimized distance field generation
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

    // Initialize distance field to "far" values
    const float MAX_DISTANCE = FLT_MAX;
    distanceField.resize(width * height, MAX_DISTANCE);

    // Find edge pixels first - this dramatically reduces processing time
    std::vector<EdgePoint> edgePoints;
    edgePoints.reserve(width + height); // Reserve reasonable space

    // First pass: identify edge pixels
    for (A_long y = 0; y < height; y++) {
        for (A_long x = 0; x < width; x++) {
            // Calculate input coordinates
            A_long inX = x - offsetX;
            A_long inY = y - offsetY;

            // Skip if not near input layer
            if (inX < -1 || inY < -1 || inX > input->width || inY > input->height) {
                continue;
            }

            // Get current pixel transparency
            bool isTransparent = true; // Default for out-of-bounds pixels
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

            // Early return for inside/outside directions
            if (!isValidPosition && direction == DIRECTION_INSIDE) {
                continue;
            }

            if (isValidPosition) {
                if ((direction == DIRECTION_INSIDE && isTransparent) ||
                    (direction == DIRECTION_OUTSIDE && !isTransparent)) {
                    continue;
                }
            }

            // Check neighboring pixels for transparency difference
            bool foundEdge = false;

            // Only check the minimum needed neighbors (4-connected is sufficient for edge detection)
            const int dx[4] = { 1, 0, -1, 0 };
            const int dy[4] = { 0, 1, 0, -1 };

            for (int i = 0; i < 4; i++) {
                A_long nx = inX + dx[i];
                A_long ny = inY + dy[i];

                // Determine neighbor transparency
                bool neighborTransparent = true; // Default for out-of-bounds
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

                // If both positions are outside layer, skip
                if (!isValidPosition && !neighborIsValid) {
                    continue;
                }

                // Check if this is an edge based on direction
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

            // If this is an edge pixel, add to list and set distance to 0
            if (foundEdge) {
                EdgePoint ep = { x, y, isTransparent };
                edgePoints.push_back(ep);
                distanceField[y * width + x] = 0.0f; // Edge pixels have distance 0
            }
        }
    }

    // If no edge pixels found, return early
    if (edgePoints.empty()) {
        return err;
    }

    // Compute the distance field using a fast approximation
    // This is much faster than checking every pixel against every edge
#ifdef _WIN32
#pragma omp parallel for
#endif
    for (int i = 0; i < edgePoints.size(); i++) {
        EdgePoint& ep = edgePoints[i];

        // Process a region around each edge point
        const int regionSize = 50; // Adjust based on max thickness

        for (A_long dy = -regionSize; dy <= regionSize; dy++) {
            for (A_long dx = -regionSize; dx <= regionSize; dx++) {
                A_long nx = ep.x + dx;
                A_long ny = ep.y + dy;

                // Skip if out of bounds
                if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                    continue;
                }

                // Calculate squared distance
                float distSq = dx * dx + dy * dy;

                // Early optimization: skip if we're too far away
                if (distSq > regionSize * regionSize) {
                    continue;
                }

                // Atomic update of minimum distance
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

    // Convert squared distances to actual distances
#ifdef _WIN32
#pragma omp parallel for
#endif
    for (A_long i = 0; i < width * height; i++) {
        if (distanceField[i] < MAX_DISTANCE) {
            distanceField[i] = sqrtf(distanceField[i]);
        }
        else {
            distanceField[i] = -1.0f; // Mark as "no edge influence"
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

    // Checkout the input buffer
    ERR(extra->cb->checkout_layer(in_data->effect_ref, BORDER_INPUT, BORDER_INPUT, &req, in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));

    // Get thickness parameter
    PF_ParamDef thickness_param;
    AEFX_CLR_STRUCT(thickness_param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_THICKNESS, in_data->current_time, in_data->time_step, in_data->time_scale, &thickness_param));

    // Get direction parameter
    PF_ParamDef direction_param;
    AEFX_CLR_STRUCT(direction_param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_DIRECTION, in_data->current_time, in_data->time_step, in_data->time_scale, &direction_param));

    // Process thickness
    PF_FpLong thickness = thickness_param.u.fs_d.value;
    A_long direction = direction_param.u.pd.value;

    // Apply non-linear scaling
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

    // Calculate downsampling factors
    float downsize_x = static_cast<float>(in_data->downsample_x.den) / static_cast<float>(in_data->downsample_x.num);
    float downsize_y = static_cast<float>(in_data->downsample_y.den) / static_cast<float>(in_data->downsample_y.num);
    float resolution_factor = MIN(downsize_x, downsize_y);

    // Calculate effective thickness
    float effectiveThickness = pixelThickness;
    if (direction == DIRECTION_BOTH) {
        effectiveThickness = pixelThickness / 2.0f;
    }

    // Apply resolution factor - divide by resolution factor
    // Add a very generous margin to ensure border is fully displayed
    A_long borderExpansion = (A_long)ceil(effectiveThickness / resolution_factor) + 50;

    // Store input checkout ID
    extra->output->pre_render_data = (void*)BORDER_INPUT;

    // Set expanded output rects
    extra->output->result_rect = in_result.result_rect;
    extra->output->result_rect.left -= borderExpansion;
    extra->output->result_rect.top -= borderExpansion;
    extra->output->result_rect.right += borderExpansion;
    extra->output->result_rect.bottom += borderExpansion;

    extra->output->max_result_rect = in_result.max_result_rect;
    extra->output->max_result_rect.left -= borderExpansion;
    extra->output->max_result_rect.top -= borderExpansion;
    extra->output->max_result_rect.right += borderExpansion;
    extra->output->max_result_rect.bottom += borderExpansion;

    // Clean up
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

    // Get input checkout ID
    A_long checkout_id = (A_long)extra->input->pre_render_data;

    // Get input and output
    PF_EffectWorld* input = NULL;
    PF_EffectWorld* output = NULL;

    // Checkout pixels
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, checkout_id, &input));
    if (err) return err;

    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
    if (err) return err;

    // Parameter variables
    PF_FpLong thickness = BORDER_THICKNESS_DFLT;
    PF_Pixel8 color = { 255, 0, 0, 255 }; // Default to red
    A_u_char threshold = BORDER_THRESHOLD_DFLT;
    A_long direction = BORDER_DIRECTION_DFLT;
    PF_Boolean showLineOnly = FALSE;

    // Get parameters
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

    // Calculate downsampling factors
    float downsize_x = static_cast<float>(in_data->downsample_x.den) / static_cast<float>(in_data->downsample_x.num);
    float downsize_y = static_cast<float>(in_data->downsample_y.den) / static_cast<float>(in_data->downsample_y.num);
    float resolution_factor = MIN(downsize_x, downsize_y);

    // Process thickness
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

    // Apply resolution factor
    A_long thicknessInt = (A_long)(pixelThickness / resolution_factor + 0.5f);

    if (input && output) {
        // ===============================================
        // STEP 1: Clear the output buffer to transparency
        // ===============================================
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

        // ==========================================
        // STEP 2: Copy the source layer if not "show line only"
        // ==========================================
        if (!showLineOnly) {
            // First, we need to calculate the proper alignment for the input layer
            // Since the output buffer has been expanded, we need to determine where the
            // input layer should be positioned within it

            // Get the offset between origins - this is the critical calculation
            A_long offsetX = input->origin_x - output->origin_x;
            A_long offsetY = input->origin_y - output->origin_y;

            if (PF_WORLD_IS_DEEP(output)) {
                // 16-bit processing
                for (A_long y = 0; y < input->height; y++) {
                    // Calculate output y-coordinate
                    A_long outY = y + offsetY;

                    // Skip if out of bounds
                    if (outY < 0 || outY >= output->height) continue;

                    PF_Pixel16* inData = (PF_Pixel16*)((char*)input->data + y * input->rowbytes);
                    PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + outY * output->rowbytes);

                    for (A_long x = 0; x < input->width; x++) {
                        // Calculate output x-coordinate
                        A_long outX = x + offsetX;

                        // Skip if out of bounds
                        if (outX < 0 || outX >= output->width) continue;

                        // Copy exact pixel
                        outData[outX] = inData[x];
                    }
                }
            }
            else {
                // 8-bit processing
                for (A_long y = 0; y < input->height; y++) {
                    // Calculate output y-coordinate
                    A_long outY = y + offsetY;

                    // Skip if out of bounds
                    if (outY < 0 || outY >= output->height) continue;

                    PF_Pixel8* inData = (PF_Pixel8*)((char*)input->data + y * input->rowbytes);
                    PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + outY * output->rowbytes);

                    for (A_long x = 0; x < input->width; x++) {
                        // Calculate output x-coordinate
                        A_long outX = x + offsetX;

                        // Skip if out of bounds
                        if (outX < 0 || outX >= output->width) continue;

                        // Copy exact pixel
                        outData[outX] = inData[x];
                    }
                }
            }
        }

        // ==========================================
        // STEP 3: Draw the border
        // ==========================================
        // Create a border using the same coordinate system as the input layer
        // Then map these coordinates to the output buffer

        // First define our search area - extend beyond input layer bounds
        A_long search_margin = thicknessInt + 5; // Add safety margin
        A_long search_left = -search_margin;
        A_long search_top = -search_margin;
        A_long search_right = input->width + search_margin;
        A_long search_bottom = input->height + search_margin;

        // Get the offset between origins
        A_long offsetX = input->origin_x - output->origin_x;
        A_long offsetY = input->origin_y - output->origin_y;

        if (PF_WORLD_IS_DEEP(output)) {
            // 16-bit processing
            A_u_short threshold16 = threshold * 257;
            PF_Pixel16 edge_color;

            edge_color.alpha = PF_MAX_CHAN16;
            edge_color.red = PF_BYTE_TO_CHAR(color.red);
            edge_color.green = PF_BYTE_TO_CHAR(color.green);
            edge_color.blue = PF_BYTE_TO_CHAR(color.blue);

            // Loop through the extended search area in input coordinates
            for (A_long y = search_top; y < search_bottom; y++) {
                for (A_long x = search_left; x < search_right; x++) {
                    // Check if this is a border pixel in input coordinates
                    if (IsEdgePixel16(input, x, y, threshold16, thicknessInt, direction)) {
                        // Convert to output coordinates
                        A_long outX = x + offsetX;
                        A_long outY = y + offsetY;

                        // Check if within output bounds
                        if (outX >= 0 && outX < output->width && outY >= 0 && outY < output->height) {
                            // Draw the border pixel
                            PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + outY * output->rowbytes);
                            outData[outX] = edge_color;
                        }
                    }
                }
            }
        }
        else {
            // 8-bit processing
            for (A_long y = search_top; y < search_bottom; y++) {
                for (A_long x = search_left; x < search_right; x++) {
                    // Check if this is a border pixel in input coordinates
                    if (IsEdgePixel8(input, x, y, threshold, thicknessInt, direction)) {
                        // Convert to output coordinates
                        A_long outX = x + offsetX;
                        A_long outY = y + offsetY;

                        // Check if within output bounds
                        if (outX >= 0 && outX < output->width && outY >= 0 && outY < output->height) {
                            // Draw the border pixel
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

    // Cleanup
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

    // This is just a fallback for non-SmartFX rendering (should not be called)
    // The real rendering is handled by SmartRender

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
        "Border", // Name
        "ADBE Border", // Match Name
        "361do_plug", // Category
        AE_RESERVED_INFO, // Reserved Info
        "EffectMain",     // Entry point
        "https://www.adobe.com");    // support URL

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