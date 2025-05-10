/*******************************************************************/
/*                                                                 */
/*                      ADOBE CONFIDENTIAL                         */
/*                   _ _ _ _ _ _ _ _ _ _ _ _ _                     */
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
*/

#include "Border.h"

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

    // Use SLIDER instead of FLOAT_SLIDERX
    // This makes the slider display range from 0-100 while still allowing the full range internally
    PF_ADD_SLIDER(STR(StrID_Thickness_Param_Name),
        BORDER_THICKNESS_MIN,
        BORDER_THICKNESS_MAX,
        BORDER_THICKNESS_MIN,
        BORDER_THICKNESS_MAX,
        BORDER_THICKNESS_DFLT,
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
        BORDER_THRESHOLD_MIN,
        BORDER_THRESHOLD_MAX,
        BORDER_THRESHOLD_DFLT,
        THRESHOLD_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // Direction parameter using the pipe-delimited format from Skeleton example
    char* directionStrings[3] = {
        STR(StrID_Direction_Both),
        STR(StrID_Direction_Inside),
        STR(StrID_Direction_Outside)
    };

    PF_ADD_POPUP(STR(StrID_Direction_Param_Name),
        3,                             // Number of choices
        DIRECTION_BOTH,                // Default choice (Both Directions)
        directionStrings[0],           // Name for the popup
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
    // Boundary check - important to avoid crashes
    if (x < 0 || y < 0 || x >= input->width || y >= input->height) {
        return FALSE; // Skip pixels outside the bounds
    }

    // Skip processing if thickness is 0
    if (thickness <= 0) {
        return FALSE;
    }

    // Get current pixel's alpha value
    PF_Pixel8* currentPixel = (PF_Pixel8*)((char*)input->data + y * input->rowbytes + x * sizeof(PF_Pixel8));
    PF_Boolean isTransparent = (currentPixel->alpha <= threshold);

    // Early return based on direction and current pixel transparency
    if ((direction == DIRECTION_INSIDE && isTransparent) ||
        (direction == DIRECTION_OUTSIDE && !isTransparent)) {
        return FALSE;
    }

    // Check surrounding pixels within the thickness radius
    for (A_long dy = -thickness; dy <= thickness; dy++) {
        for (A_long dx = -thickness; dx <= thickness; dx++) {
            // Skip the current pixel
            if (dx == 0 && dy == 0) continue;

            // Calculate distance from current pixel (using squared distance for efficiency)
            A_long distSquared = (dx * dx + dy * dy);
            A_long thicknessSquared = thickness * thickness;

            // Only check pixels within the thickness radius
            if (distSquared <= thicknessSquared) {
                A_long nx = x + dx;
                A_long ny = y + dy;

                // Boundary check - critical to prevent crashes
                if (nx >= 0 && ny >= 0 && nx < input->width && ny < input->height) {
                    PF_Pixel8* neighborPixel = (PF_Pixel8*)((char*)input->data + ny * input->rowbytes + nx * sizeof(PF_Pixel8));
                    PF_Boolean neighborIsTransparent = (neighborPixel->alpha <= threshold);

                    // Check for edge based on direction
                    switch (direction) {
                    case DIRECTION_BOTH:
                        // If one is transparent and the other is not, this is an edge
                        if (isTransparent != neighborIsTransparent) {
                            return TRUE;
                        }
                        break;

                    case DIRECTION_INSIDE:
                        // Inside: current pixel is non-transparent and neighbor is transparent
                        if (!isTransparent && neighborIsTransparent) {
                            return TRUE;
                        }
                        break;

                    case DIRECTION_OUTSIDE:
                        // Outside: current pixel is transparent and neighbor is non-transparent
                        if (isTransparent && !neighborIsTransparent) {
                            return TRUE;
                        }
                        break;
                    }
                }
                // For pixels outside the frame boundary, treat them as transparent
                else if (!isTransparent && (direction == DIRECTION_INSIDE || direction == DIRECTION_BOTH)) {
                    // If we're looking for Inside edges and the current pixel is non-transparent,
                    // treat the out-of-bounds pixel as transparent -> this is an edge
                    return TRUE;
                }
                else if (isTransparent && (direction == DIRECTION_OUTSIDE || direction == DIRECTION_BOTH)) {
                    // If we're looking for Outside edges and the current pixel is transparent,
                    // treat the out-of-bounds pixel as non-transparent -> this is an edge
                    return TRUE;
                }
            }
        }
    }

    return FALSE;
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
    // Boundary check
    if (x < 0 || y < 0 || x >= input->width || y >= input->height) {
        return FALSE;
    }

    // Skip processing if thickness is 0
    if (thickness <= 0) {
        return FALSE;
    }

    // Get current pixel's alpha value
    PF_Pixel16* currentPixel = (PF_Pixel16*)((char*)input->data + y * input->rowbytes + x * sizeof(PF_Pixel16));
    PF_Boolean isTransparent = (currentPixel->alpha <= threshold);

    // Early return based on direction and current pixel transparency
    if ((direction == DIRECTION_INSIDE && isTransparent) ||
        (direction == DIRECTION_OUTSIDE && !isTransparent)) {
        return FALSE;
    }

    // Check surrounding pixels within the thickness radius
    for (A_long dy = -thickness; dy <= thickness; dy++) {
        for (A_long dx = -thickness; dx <= thickness; dx++) {
            // Skip the current pixel
            if (dx == 0 && dy == 0) continue;

            // Calculate distance from current pixel (using squared distance for efficiency)
            A_long distSquared = (dx * dx + dy * dy);
            A_long thicknessSquared = thickness * thickness;

            // Only check pixels within the thickness radius
            if (distSquared <= thicknessSquared) {
                A_long nx = x + dx;
                A_long ny = y + dy;

                // Boundary check
                if (nx >= 0 && ny >= 0 && nx < input->width && ny < input->height) {
                    PF_Pixel16* neighborPixel = (PF_Pixel16*)((char*)input->data + ny * input->rowbytes + nx * sizeof(PF_Pixel16));
                    PF_Boolean neighborIsTransparent = (neighborPixel->alpha <= threshold);

                    // Check for edge based on direction
                    switch (direction) {
                    case DIRECTION_BOTH:
                        // If one is transparent and the other is not, this is an edge
                        if (isTransparent != neighborIsTransparent) {
                            return TRUE;
                        }
                        break;

                    case DIRECTION_INSIDE:
                        // Inside: current pixel is non-transparent and neighbor is transparent
                        if (!isTransparent && neighborIsTransparent) {
                            return TRUE;
                        }
                        break;

                    case DIRECTION_OUTSIDE:
                        // Outside: current pixel is transparent and neighbor is non-transparent
                        if (isTransparent && !neighborIsTransparent) {
                            return TRUE;
                        }
                        break;
                    }
                }
                // For pixels outside the frame boundary, treat them as transparent
                else if (!isTransparent && (direction == DIRECTION_INSIDE || direction == DIRECTION_BOTH)) {
                    // If we're looking for Inside edges and the current pixel is non-transparent,
                    // treat the out-of-bounds pixel as transparent -> this is an edge
                    return TRUE;
                }
                else if (isTransparent && (direction == DIRECTION_OUTSIDE || direction == DIRECTION_BOTH)) {
                    // If we're looking for Outside edges and the current pixel is transparent,
                    // treat the out-of-bounds pixel as non-transparent -> this is an edge
                    return TRUE;
                }
            }
        }
    }

    return FALSE;
}

// SmartFX Pre-Render function
static PF_Err
PreRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    PF_RenderRequest req = extra->input->output_request;
    PF_CheckoutResult in_result;

    // Checkout the input buffer - we need the entire frame to detect edges
    ERR(extra->cb->checkout_layer(in_data->effect_ref, BORDER_INPUT, BORDER_INPUT, &req, in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));

    // Use the checkout ID directly
    extra->output->pre_render_data = (void*)BORDER_INPUT;

    // Pass through the input's rects to the output
    extra->output->result_rect = in_result.result_rect;
    extra->output->max_result_rect = in_result.max_result_rect;

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

    // Get the pre-render data (checkout ID)
    A_long checkout_id = (A_long)extra->input->pre_render_data;

    // Get the input and output
    PF_EffectWorld* input = NULL;
    PF_EffectWorld* output = NULL;

    // Checkout the input pixels
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, checkout_id, &input));
    if (err) return err;

    // Checkout the output buffer
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
    if (err) return err;

    // Parameter variables
    A_long thickness = BORDER_THICKNESS_DFLT;
    PF_Pixel8 color = { 255, 0, 0, 255 }; // Default to red
    A_u_char threshold = BORDER_THRESHOLD_DFLT;
    A_long direction = BORDER_DIRECTION_DFLT;
    PF_Boolean showLineOnly = FALSE;

    // Get parameter values
    PF_ParamDef param;

    // Get thickness parameter
    AEFX_CLR_STRUCT(param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_THICKNESS, in_data->current_time, in_data->time_step, in_data->time_scale, &param));
    if (!err) thickness = param.u.sd.value; // Getting integer value
    PF_CHECKIN_PARAM(in_data, &param);

    // Get color parameter
    AEFX_CLR_STRUCT(param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_COLOR, in_data->current_time, in_data->time_step, in_data->time_scale, &param));
    if (!err) color = param.u.cd.value;
    PF_CHECKIN_PARAM(in_data, &param);

    // Get threshold parameter
    AEFX_CLR_STRUCT(param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_THRESHOLD, in_data->current_time, in_data->time_step, in_data->time_scale, &param));
    if (!err) threshold = (A_u_char)param.u.sd.value;
    PF_CHECKIN_PARAM(in_data, &param);

    // Get direction parameter
    AEFX_CLR_STRUCT(param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_DIRECTION, in_data->current_time, in_data->time_step, in_data->time_scale, &param));
    if (!err) direction = param.u.pd.value;
    PF_CHECKIN_PARAM(in_data, &param);

    // Get show line only parameter
    AEFX_CLR_STRUCT(param);
    ERR(PF_CHECKOUT_PARAM(in_data, BORDER_SHOW_LINE_ONLY, in_data->current_time, in_data->time_step, in_data->time_scale, &param));
    if (!err) showLineOnly = param.u.bd.value;
    PF_CHECKIN_PARAM(in_data, &param);

    // Calculate downsampling factors exactly as in MultiSlicer
    float downsize_x = static_cast<float>(in_data->downsample_x.den) / static_cast<float>(in_data->downsample_x.num);
    float downsize_y = static_cast<float>(in_data->downsample_y.den) / static_cast<float>(in_data->downsample_y.num);
    float resolution_factor = MIN(downsize_x, downsize_y);

    // Scale thickness from 0-100 display range to 0-2000 internal range
    A_long scaledThickness = thickness * 20; // Convert display range to internal range

    // Apply resolution factor to get exact pixel calculation
    A_long thicknessInt = (A_long)(scaledThickness * resolution_factor);

    if (input && output) {
        // First, copy the input to output if we're not in "show line only" mode
        if (!showLineOnly) {
            ERR(suites.WorldTransformSuite1()->copy(
                in_data->effect_ref,
                input,
                output,
                NULL,
                NULL
            ));
        }
        else {
            // If "show line only", start with a blank (transparent) output
            // Manual fill with transparency (all zeros)
            if (PF_WORLD_IS_DEEP(output)) {
                // 16-bit fill
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
                // 8-bit fill
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
        }

        // Process based on bit depth
        if (PF_WORLD_IS_DEEP(output)) {
            // 16-bit processing
            A_u_short threshold16 = threshold * 257; // Convert 8-bit to 16-bit threshold
            PF_Pixel16 edge_color;

            // Convert 8-bit color to 16-bit
            edge_color.alpha = PF_MAX_CHAN16;
            edge_color.red = PF_BYTE_TO_CHAR(color.red);
            edge_color.green = PF_BYTE_TO_CHAR(color.green);
            edge_color.blue = PF_BYTE_TO_CHAR(color.blue);

            for (A_long y = 0; y < output->height; y++) {
                PF_Pixel16* outData = (PF_Pixel16*)((char*)output->data + y * output->rowbytes);

                for (A_long x = 0; x < output->width; x++) {
                    if (IsEdgePixel16(input, x, y, threshold16, thicknessInt, direction)) {
                        // This is an edge pixel, draw with border color
                        outData[x] = edge_color;
                    }
                }
            }
        }
        else {
            // 8-bit processing
            for (A_long y = 0; y < output->height; y++) {
                PF_Pixel8* outData = (PF_Pixel8*)((char*)output->data + y * output->rowbytes);

                for (A_long x = 0; x < output->width; x++) {
                    if (IsEdgePixel8(input, x, y, threshold, thicknessInt, direction)) {
                        // This is an edge pixel, draw with border color
                        outData[x].alpha = PF_MAX_CHAN8;
                        outData[x].red = color.red;
                        outData[x].green = color.green;
                        outData[x].blue = color.blue;
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