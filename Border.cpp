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

/*	Border.cpp

	This effect generates a line between transparent and non-transparent pixels.

	Revision History

	Version		Change													Engineer	Date
	=======		======													========	======
	1.0			Original plugin											bbb			6/1/2002
	1.1			Added transparency edge detection						anon		5/11/2025
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
	PF_Err		err = PF_Err_NONE;
	PF_ParamDef	def;

	AEFX_CLR_STRUCT(def);

	// Thickness parameter
	PF_ADD_FLOAT_SLIDERX(STR(StrID_Thickness_Param_Name),
		BORDER_THICKNESS_MIN,
		BORDER_THICKNESS_MAX,
		BORDER_THICKNESS_MIN,
		BORDER_THICKNESS_MAX,
		BORDER_THICKNESS_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		THICKNESS_DISK_ID);

	AEFX_CLR_STRUCT(def);

	// Color parameter
	PF_ADD_COLOR(STR(StrID_Color_Param_Name),
		PF_HALF_CHAN8,
		PF_MAX_CHAN8,
		PF_MAX_CHAN8,
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

	// Show Line Only checkbox
	PF_ADD_CHECKBOX(STR(StrID_ShowLineOnly_Param_Name),
		"Show line only",
		FALSE,
		0,
		SHOW_LINE_ONLY_DISK_ID);

	out_data->num_params = BORDER_NUM_PARAMS;

	return err;
}

// Function to determine if a pixel is part of an edge between transparent and non-transparent areas
static PF_Boolean
IsEdgePixel8(
	PF_EffectWorld* input,
	A_long         x,
	A_long         y,
	A_u_char       threshold,
	A_long         thickness)
{
	// Boundary check
	if (x < 0 || y < 0 || x >= input->width || y >= input->height) {
		return FALSE;
	}

	// Get current pixel's alpha value
	PF_Pixel8* currentPixel = (PF_Pixel8*)((char*)input->data + y * input->rowbytes + x * sizeof(PF_Pixel8));
	PF_Boolean isTransparent = (currentPixel->alpha <= threshold);

	// Check surrounding pixels within the thickness radius
	for (A_long dy = -thickness; dy <= thickness; dy++) {
		for (A_long dx = -thickness; dx <= thickness; dx++) {
			// Skip the current pixel
			if (dx == 0 && dy == 0) continue;

			// Calculate distance from current pixel
			A_long dist = (A_long)sqrt((double)(dx * dx + dy * dy));

			// Only check pixels within the thickness radius
			if (dist <= thickness) {
				A_long nx = x + dx;
				A_long ny = y + dy;

				// Boundary check
				if (nx >= 0 && ny >= 0 && nx < input->width && ny < input->height) {
					PF_Pixel8* neighborPixel = (PF_Pixel8*)((char*)input->data + ny * input->rowbytes + nx * sizeof(PF_Pixel8));
					PF_Boolean neighborIsTransparent = (neighborPixel->alpha <= threshold);

					// If one is transparent and the other is not, this is an edge
					if (isTransparent != neighborIsTransparent) {
						return TRUE;
					}
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
	A_long         thickness)
{
	// Boundary check
	if (x < 0 || y < 0 || x >= input->width || y >= input->height) {
		return FALSE;
	}

	// Get current pixel's alpha value
	PF_Pixel16* currentPixel = (PF_Pixel16*)((char*)input->data + y * input->rowbytes + x * sizeof(PF_Pixel16));
	PF_Boolean isTransparent = (currentPixel->alpha <= threshold);

	// Check surrounding pixels within the thickness radius
	for (A_long dy = -thickness; dy <= thickness; dy++) {
		for (A_long dx = -thickness; dx <= thickness; dx++) {
			// Skip the current pixel
			if (dx == 0 && dy == 0) continue;

			// Calculate distance from current pixel
			A_long dist = (A_long)sqrt((double)(dx * dx + dy * dy));

			// Only check pixels within the thickness radius
			if (dist <= thickness) {
				A_long nx = x + dx;
				A_long ny = y + dy;

				// Boundary check
				if (nx >= 0 && ny >= 0 && nx < input->width && ny < input->height) {
					PF_Pixel16* neighborPixel = (PF_Pixel16*)((char*)input->data + ny * input->rowbytes + nx * sizeof(PF_Pixel16));
					PF_Boolean neighborIsTransparent = (neighborPixel->alpha <= threshold);

					// If one is transparent and the other is not, this is an edge
					if (isTransparent != neighborIsTransparent) {
						return TRUE;
					}
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

	// Use the checkout ID directly - this will be more reliable
	extra->output->pre_render_data = (void*)BORDER_INPUT;

	// Simply pass through the input's rects to the output
	extra->output->result_rect = in_result.result_rect;
	extra->output->max_result_rect = in_result.max_result_rect;

	return err;
}

// SmartFX Render function
static PF_Err
SmartRender(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_SmartRenderExtra* extra)
{
	PF_Err err = PF_Err_NONE;

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

	if (input && output) {
		// Simple copy from input to output for now to make sure we see something
		for (A_long y = 0; y < output->height; y++) {
			char* srcRow = (char*)input->data + y * input->rowbytes;
			char* dstRow = (char*)output->data + y * output->rowbytes;

			// Copy the row bytes
			memcpy(dstRow, srcRow, MIN(input->rowbytes, output->rowbytes));
		}

		// Add your border effect processing here once the basic copy works
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
		"EffectMain",	// Entry point
		"https://www.adobe.com");	// support URL

	return result;
}

PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output,
	void* extra)
{
	PF_Err		err = PF_Err_NONE;

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