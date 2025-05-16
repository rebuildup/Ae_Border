/*******************************************************************/
/*                                                                 */
/*                      ADOBE CONFIDENTIAL                         */
/*                   _ _ _ _ _ _ _ _ _ _ _ _                     */
/*                                                                 */
/* Copyright 2007-2023 Adobe Inc.                                  */
/* All Rights Reserved.                                            */
/*                                                                 */
/* NOTICE:  All information contained herein is, and remains the   */
/* property of Adobe Inc. and its suppliers, if                    */
/* any.  The intellectual and technical concepts contained         */
/* herein are proprietary to Adobe Inc. and its                    */
/* suppliers and may be covered by U.S. and Foreign Patents,       */
/* patents in process, and are protected by trade secret or        */
/* copyright law.  Dissemination of this information or            */
/* reproduction of this material is strictly forbidden unless      */
/* prior written permission is obtained from Adobe Inc.            */
/* Incorporated.                                                   */
/*                                                                 */
/*******************************************************************/

/*    Border.cpp

    This effect generates a line between transparent and non-transparent pixels.

    Revision History

    Version        Change                                                Engineer    Date
    =======        ======                                                ========    ======
    1.0            Original plugin                                        bbb        6/1/2002
    1.1            Added transparency edge detection                    anon        5/11/2025
    1.2            Ultra-fast multi-resolution approach                 anon        5/15/2025
*/

#include "Border.h"
#include <vector>
#include <algorithm>
#include <cfloat> // For FLT_MAX
#include <cstring> // For memset

#ifdef _WIN32
#include <omp.h>  // For OpenMP parallel processing
#endif

// Define our own clamp function
template <typename T>
T CLAMP(T value, T min, T max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// Compact edge point struct
struct EdgePoint {
    A_short x, y;
};

// Structure to hold a downsampled version of the input buffer
struct DownsampledBuffer {
    std::vector<unsigned char> alpha;
    int width;
    int height;
    int scale;
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

// Create a downsampled version of the input for faster processing
static void CreateDownsampledBuffer(
    PF_EffectWorld* input,
    A_u_char threshold,
    DownsampledBuffer& buffer,
    int scale)
{
    // Calculate downsampled dimensions
    buffer.width = (input->width + scale - 1) / scale;
    buffer.height = (input->height + scale - 1) / scale;
    buffer.scale = scale;

    // Allocate space for the downsampled alpha channel only
    buffer.alpha.resize(buffer.width * buffer.height, 0);

    // Fill in the downsampled buffer
    for (int y = 0; y < input->height; y += scale) {
        for (int x = 0; x < input->width; x += scale) {
            // Get current pixel alpha value
            A_u_char alphaValue = 0;

            if (PF_WORLD_IS_DEEP(input)) {
                PF_Pixel16* pixel = (PF_Pixel16*)((char*)input->data + y * input->rowbytes + x * sizeof(PF_Pixel16));
                alphaValue = PF_CHAR_TO_BYTE(pixel->alpha);
            }
            else {
                PF_Pixel8* pixel = (PF_Pixel8*)((char*)input->data + y * input->rowbytes + x * sizeof(PF_Pixel8));
                alphaValue = pixel->alpha;
            }

            // Store as binary transparency (0 or 1)
            unsigned char isTransparent = (alphaValue <= threshold) ? 0 : 1;

            // Store in the downsampled buffer
            int dx = x / scale;
            int dy = y / scale;
            buffer.alpha[dy * buffer.width + dx] = isTransparent;
        }
    }
}

// Finds edges in the downsampled buffer
static void FindEdges(
    const DownsampledBuffer& buffer,
    std::vector<EdgePoint>& edges,
    A_long direction)
{
    // Clear the output
    edges.clear();
    edges.reserve(buffer.width * 2 + buffer.height * 2); // Reasonable initial capacity

    // Find edges in the downsampled buffer
    for (int y = 0; y < buffer.height; y++) {
        for (int x = 0; x < buffer.width; x++) {
            // Get current transparency
            unsigned char isTransparent = buffer.alpha[y * buffer.width + x];

            // Check neighboring pixels (4-connected is sufficient)
            const int dx[4] = { 1, 0, -1, 0 };
            const int dy[4] = { 0, 1, 0, -1 };

            bool isEdge = false;

            for (int i = 0; i < 4; i++) {
                int nx = x + dx[i];
                int ny = y + dy[i];

                // Skip if neighbor is out of bounds
                if (nx < 0 || nx >= buffer.width || ny < 0 || ny >= buffer.height) {
                    // Treat out-of-bounds as transparent
                    if (direction != DIRECTION_INSIDE && isTransparent) {
                        isEdge = true;
                        break;
                    }
                    continue;
                }

                // Get neighbor transparency
                unsigned char neighborTransparent = buffer.alpha[ny * buffer.width + nx];

                // Check if this is an edge based on direction
                if (direction == DIRECTION_BOTH) {
                    if (isTransparent != neighborTransparent) {
                        isEdge = true;
                        break;
                    }
                }
                else if (direction == DIRECTION_INSIDE) {
                    if (!isTransparent && neighborTransparent == 0) {
                        isEdge = true;
                        break;
                    }
                }
                else if (direction == DIRECTION_OUTSIDE) {
                    if (isTransparent == 0 && neighborTransparent) {
                        isEdge = true;
                        break;
                    }
                }
            }

            // Store edge points in the original coordinate system
            if (isEdge) {
                EdgePoint ep;
                ep.x = x;
                ep.y = y;
                edges.push_back(ep);
            }
        }
    }
}

// Ultra-fast border drawing using circle primitives
static PF_Err
DrawBorders(
    PF_EffectWorld* output,
    const std::vector<EdgePoint>& edges,
    const DownsampledBuffer& buffer,
    PF_Pixel8 color,
    float thickness,
    bool showLineOnly,
    A_long offsetX,
    A_long offsetY)
{
    PF_Err err = PF_Err_NONE;

    // Only process if we have edges and valid thickness
    if (edges.empty() || thickness <= 0) {
        return err;
    }

    // Create a mask buffer to track which output pixels we've already processed
    std::vector<bool> processedPixels(output->width * output->height, false);

    // Convert border thickness from downsampled to original scale
    float actualThickness = thickness * buffer.scale;

    // For AA, expand slightly
    float outerRadius = actualThickness;
    float innerRadius = MAX(0.0f, actualThickness - 1.5f);
    float radiusSq = outerRadius * outerRadius;

    // Now draw the border around each edge pixel
    int borderSize = (int)ceil(actualThickness);

    if (PF_WORLD_IS_DEEP(output)) {
        // 16-bit version
        PF_Pixel16 colorDeep;
        colorDeep.red = PF_BYTE_TO_CHAR(color.red);
        colorDeep.green = PF_BYTE_TO_CHAR(color.green);
        colorDeep.blue = PF_BYTE_TO_CHAR(color.blue);
        colorDeep.alpha = PF_MAX_CHAN16;

        // Process each edge point
#ifdef _WIN32
#pragma omp parallel for
#endif
        for (int i = 0; i < edges.size(); i++) {
            // Convert downsampled edge to full-scale center
            int centerX = edges[i].x * buffer.scale + offsetX;
            int centerY = edges[i].y * buffer.scale + offsetY;

            // Draw a circle around this edge point
            for (int y = MAX(0, centerY - borderSize); y <= MIN(output->height - 1, centerY + borderSize); y++) {
                for (int x = MAX(0, centerX - borderSize); x <= MIN(output->width - 1, centerX + borderSize); x++) {
                    // Skip if this pixel has already been processed
                    unsigned int pixelIndex = y * output->width + x;
                    if (processedPixels[pixelIndex]) continue;

                    // Calculate squared distance to edge center
                    float dx = (float)(x - centerX);
                    float dy = (float)(y - centerY);
                    float distSq = dx * dx + dy * dy;

                    // Skip if outside border radius
                    if (distSq > radiusSq) continue;

                    // Mark as processed regardless of result
                    processedPixels[pixelIndex] = true;

                    // Calculate opacity based on distance (anti-aliasing)
                    float distance = sqrtf(distSq);
                    float opacity = 1.0f;

                    if (distance > innerRadius) {
                        opacity = (outerRadius - distance) / (outerRadius - innerRadius);
                        opacity = CLAMP(opacity, 0.0f, 1.0f);
                    }

                    // Apply pixel
                    PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + y * output->rowbytes + x * sizeof(PF_Pixel16));

                    if (showLineOnly) {
                        // Just set the color with opacity
                        outData->red = colorDeep.red;
                        outData->green = colorDeep.green;
                        outData->blue = colorDeep.blue;
                        outData->alpha = (A_u_short)(opacity * PF_MAX_CHAN16);
                    }
                    else {
                        // Blend with existing pixel
                        float existingAlpha = (float)outData->alpha / PF_MAX_CHAN16;
                        float edgeAlpha = opacity;
                        float resultAlpha = edgeAlpha + existingAlpha * (1.0f - edgeAlpha);

                        if (resultAlpha > 0.0f) {
                            // Blend colors
                            float blend = edgeAlpha / resultAlpha;
                            outData->red = (A_u_short)(colorDeep.red * blend + outData->red * (1.0f - blend));
                            outData->green = (A_u_short)(colorDeep.green * blend + outData->green * (1.0f - blend));
                            outData->blue = (A_u_short)(colorDeep.blue * blend + outData->blue * (1.0f - blend));
                            outData->alpha = (A_u_short)(resultAlpha * PF_MAX_CHAN16);
                        }
                    }
                }
            }
        }
    }
    else {
        // 8-bit version
        // Process each edge point
#ifdef _WIN32
#pragma omp parallel for
#endif
        for (int i = 0; i < edges.size(); i++) {
            // Convert downsampled edge to full-scale center
            int centerX = edges[i].x * buffer.scale + offsetX;
            int centerY = edges[i].y * buffer.scale + offsetY;

            // Draw a circle around this edge point
            for (int y = MAX(0, centerY - borderSize); y <= MIN(output->height - 1, centerY + borderSize); y++) {
                for (int x = MAX(0, centerX - borderSize); x <= MIN(output->width - 1, centerX + borderSize); x++) {
                    // Skip if this pixel has already been processed
                    unsigned int pixelIndex = y * output->width + x;
#ifdef _WIN32
#pragma omp critical
#endif
                    {
                        if (processedPixels[pixelIndex]) continue;
                        processedPixels[pixelIndex] = true; // Mark as processed
                    }

                    // Calculate squared distance to edge center
                    float dx = (float)(x - centerX);
                    float dy = (float)(y - centerY);
                    float distSq = dx * dx + dy * dy;

                    // Skip if outside border radius
                    if (distSq > radiusSq) continue;

                    // Calculate opacity based on distance (anti-aliasing)
                    float distance = sqrtf(distSq);
                    float opacity = 1.0f;

                    if (distance > innerRadius) {
                        opacity = (outerRadius - distance) / (outerRadius - innerRadius);
                        opacity = CLAMP(opacity, 0.0f, 1.0f);
                    }

                    // Apply pixel
                    PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + y * output->rowbytes + x * sizeof(PF_Pixel8));

                    if (showLineOnly) {
                        // Just set the color with opacity
                        outData->red = color.red;
                        outData->green = color.green;
                        outData->blue = color.blue;
                        outData->alpha = (A_u_char)(opacity * PF_MAX_CHAN8);
                    }
                    else {
                        // Blend with existing pixel
                        float existingAlpha = (float)outData->alpha / PF_MAX_CHAN8;
                        float edgeAlpha = opacity;
                        float resultAlpha = edgeAlpha + existingAlpha * (1.0f - edgeAlpha);

                        if (resultAlpha > 0.0f) {
                            // Blend colors
                            float blend = edgeAlpha / resultAlpha;
                            outData->red = (A_u_char)(color.red * blend + outData->red * (1.0f - blend));
                            outData->green = (A_u_char)(color.green * blend + outData->green * (1.0f - blend));
                            outData->blue = (A_u_char)(color.blue * blend + outData->blue * (1.0f - blend));
                            outData->alpha = (A_u_char)(resultAlpha * PF_MAX_CHAN8);
                        }
                    }
                }
            }
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
    // Use a reasonable margin
    A_long borderExpansion = (A_long)ceil(effectiveThickness / resolution_factor) + 5;

    // Store input checkout ID
    extra->output->pre_render_data = (void*)BORDER_INPUT;

    // Store the request rect for reference - this is important!
    PF_Rect request_rect = extra->input->output_request.rect;

    // First, set up our desired result rect based on input
    PF_Rect desired_result = in_result.result_rect;
    desired_result.left -= borderExpansion;
    desired_result.top -= borderExpansion;
    desired_result.right += borderExpansion;
    desired_result.bottom += borderExpansion;

    // Now intersect with the request rect to ensure we don't exceed it
    extra->output->result_rect.left = MAX(desired_result.left, request_rect.left);
    extra->output->result_rect.top = MAX(desired_result.top, request_rect.top);
    extra->output->result_rect.right = MIN(desired_result.right, request_rect.right);
    extra->output->result_rect.bottom = MIN(desired_result.bottom, request_rect.bottom);

    // Set max_result_rect - should be contained within request rect as well
    extra->output->max_result_rect = extra->output->result_rect;

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
    float thicknessFloat = pixelThickness / resolution_factor;

    if (input && output && thicknessFloat > 0) {
        // ===============================================
        // STEP 1: Clear the output buffer to transparency
        // ===============================================
        if (PF_WORLD_IS_DEEP(output)) {
#ifdef _WIN32
#pragma omp parallel for
#endif
            for (A_long y = 0; y < output->height; y++) {
                PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + y * output->rowbytes);
                std::memset(outData, 0, output->width * sizeof(PF_Pixel16));
            }
        }
        else {
#ifdef _WIN32
#pragma omp parallel for
#endif
            for (A_long y = 0; y < output->height; y++) {
                PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + y * output->rowbytes);
                std::memset(outData, 0, output->width * sizeof(PF_Pixel8));
            }
        }

        // ==========================================
        // STEP 2: Copy the source layer if not "show line only"
        // ==========================================
        if (!showLineOnly) {
            // Calculate the proper alignment for the input layer
            A_long offsetX = input->origin_x - output->origin_x;
            A_long offsetY = input->origin_y - output->origin_y;

            if (PF_WORLD_IS_DEEP(output)) {
                // 16-bit processing
#ifdef _WIN32
#pragma omp parallel for
#endif
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
#ifdef _WIN32
#pragma omp parallel for
#endif
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
        // STEP 3: Multi-scale border detection and drawing
        // ==========================================

        // Calculate optimal downsampling scale based on border thickness
        // Larger borders can use more aggressive downsampling
        int downsampleScale = 1;
        if (thicknessFloat <= 3.0f) {
            downsampleScale = 1; // No downsampling for very thin borders
        }
        else if (thicknessFloat <= 10.0f) {
            downsampleScale = 2; // 2x downsampling for medium borders
        }
        else if (thicknessFloat <= 30.0f) {
            downsampleScale = 4; // 4x downsampling for thick borders
        }
        else {
            downsampleScale = 8; // 8x downsampling for very thick borders
        }

        // Get the offset between origins
        A_long offsetX = input->origin_x - output->origin_x;
        A_long offsetY = input->origin_y - output->origin_y;

        // Create downsampled buffer
        DownsampledBuffer buffer;
        CreateDownsampledBuffer(input, threshold, buffer, downsampleScale);

        // Find edges in downsampled space
        std::vector<EdgePoint> edges;
        FindEdges(buffer, edges, direction);

        // Draw borders directly from edge points
        // This massively reduces computation by directly drawing border primitives
        // without computing a full distance field
        ERR(DrawBorders(output, edges, buffer, color, thicknessFloat / downsampleScale, showLineOnly, offsetX / downsampleScale, offsetY / downsampleScale));
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
        "Border", // Category
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