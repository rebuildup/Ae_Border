#define NOMINMAX
#include "Border.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <atomic>
#include <limits>

// -----------------------------------------------------------------------------
// Constants & Helpers
// -----------------------------------------------------------------------------

static const float INF_DIST = 1e9f;

template <typename T>
static inline T Clamp(T val, T minVal, T maxVal) {
    return std::max(minVal, std::min(val, maxVal));
}

// -----------------------------------------------------------------------------
// Pixel Traits
// -----------------------------------------------------------------------------

template <typename PixelT>
struct BorderPixelTraits;

template <>
struct BorderPixelTraits<PF_Pixel> {
    using ChannelType = A_u_char;
    static constexpr float MAX_VAL = 255.0f;
    static inline float ToFloat(ChannelType v) { return static_cast<float>(v); }
    static inline ChannelType FromFloat(float v) { return static_cast<ChannelType>(Clamp(v, 0.0f, MAX_VAL) + 0.5f); }
};

template <>
struct BorderPixelTraits<PF_Pixel16> {
    using ChannelType = A_u_short;
    static constexpr float MAX_VAL = 32768.0f;
    static inline float ToFloat(ChannelType v) { return static_cast<float>(v); }
    static inline ChannelType FromFloat(float v) { return static_cast<ChannelType>(Clamp(v, 0.0f, MAX_VAL) + 0.5f); }
};

template <>
struct BorderPixelTraits<PF_PixelFloat> {
    using ChannelType = PF_FpShort;
    static constexpr float MAX_VAL = 1.0f;
    static inline float ToFloat(ChannelType v) { return static_cast<float>(v); }
    static inline ChannelType FromFloat(float v) { return static_cast<ChannelType>(v); }
};

// -----------------------------------------------------------------------------
// Euclidean Distance Transform (EDT) Helpers
// -----------------------------------------------------------------------------

// 1D Squared Distance Transform using Parabolic Lower Envelope
// grid: input/output array of squared distances
// width: number of elements
static void EDT_1D(std::vector<float>& grid, int width) {
    std::vector<float> z(width + 1);
    std::vector<int> v(width);
    
    int k = 0;
    v[0] = 0;
    z[0] = -INF_DIST;
    z[1] = INF_DIST;

    for (int q = 1; q < width; q++) {
        float s = ((grid[q] + q * q) - (grid[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
        while (s <= z[k]) {
            k--;
            s = ((grid[q] + q * q) - (grid[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k + 1] = INF_DIST;
    }

    k = 0;
    for (int q = 0; q < width; q++) {
        while (z[k + 1] < q) {
            k++;
        }
        float dx = (float)(q - v[k]);
        grid[q] = dx * dx + grid[v[k]];
    }
}

// -----------------------------------------------------------------------------
// Rendering
// -----------------------------------------------------------------------------

template <typename Pixel>
static PF_Err RenderGeneric(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output) {
    PF_EffectWorld* input = &params[BORDER_INPUT]->u.ld;
    
    const int width = output->width;
    const int height = output->height;
    
    if (width <= 0 || height <= 0) return PF_Err_NONE;

    const A_u_char* input_base = reinterpret_cast<const A_u_char*>(input->data);
    A_u_char* output_base = reinterpret_cast<A_u_char*>(output->data);
    const A_long input_rowbytes = input->rowbytes;
    const A_long output_rowbytes = output->rowbytes;

    // Parameters
    float thickness = static_cast<float>(params[BORDER_THICKNESS]->u.fs_d.value);
    PF_Pixel color_param = params[BORDER_COLOR]->u.cd.value;
    float threshold = static_cast<float>(params[BORDER_THRESHOLD]->u.sd.value) * (BorderPixelTraits<Pixel>::MAX_VAL / 255.0f);
    int direction = params[BORDER_DIRECTION]->u.pd.value;
    bool show_line_only = params[BORDER_SHOW_LINE_ONLY]->u.bd.value;

    // Convert color param to target depth
    Pixel border_color;
    border_color.alpha = BorderPixelTraits<Pixel>::FromFloat(BorderPixelTraits<Pixel>::MAX_VAL);
    border_color.red = BorderPixelTraits<Pixel>::FromFloat(color_param.red * (BorderPixelTraits<Pixel>::MAX_VAL / 255.0f));
    border_color.green = BorderPixelTraits<Pixel>::FromFloat(color_param.green * (BorderPixelTraits<Pixel>::MAX_VAL / 255.0f));
    border_color.blue = BorderPixelTraits<Pixel>::FromFloat(color_param.blue * (BorderPixelTraits<Pixel>::MAX_VAL / 255.0f));

    // Initialize SDF buffers
    // We need two buffers: one for distance to "inside" (foreground), one for distance to "outside" (background).
    // Initialize with INF for opposite type, 0 for same type.
    std::vector<float> dist_inside(width * height);
    std::vector<float> dist_outside(width * height);

    // 1. Initialization Pass
    // Parallelize initialization
    int num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;

    auto init_buffers = [&](int start_y, int end_y) {
        for (int y = start_y; y < end_y; ++y) {
            const Pixel* row = reinterpret_cast<const Pixel*>(input_base + y * input_rowbytes);
            for (int x = 0; x < width; ++x) {
                float alpha = BorderPixelTraits<Pixel>::ToFloat(row[x].alpha);
                bool is_inside = alpha > threshold;
                
                int idx = y * width + x;
                if (is_inside) {
                    dist_inside[idx] = 0.0f;
                    dist_outside[idx] = INF_DIST;
                } else {
                    dist_inside[idx] = INF_DIST;
                    dist_outside[idx] = 0.0f;
                }
            }
        }
    };

    int rows_per_thread = (height + num_threads - 1) / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        int start = i * rows_per_thread;
        int end = std::min(start + rows_per_thread, height);
        if (start < end) threads.emplace_back(init_buffers, start, end);
    }
    for (auto& t : threads) t.join();
    threads.clear();

    // 2. EDT Pass 1: Horizontal
    // Process each row independently
    auto edt_horizontal = [&](int start_y, int end_y) {
        std::vector<float> row_buf(width);
        for (int y = start_y; y < end_y; ++y) {
            // Inside
            for(int x=0; x<width; ++x) row_buf[x] = dist_inside[y * width + x];
            EDT_1D(row_buf, width);
            for(int x=0; x<width; ++x) dist_inside[y * width + x] = row_buf[x];

            // Outside
            for(int x=0; x<width; ++x) row_buf[x] = dist_outside[y * width + x];
            EDT_1D(row_buf, width);
            for(int x=0; x<width; ++x) dist_outside[y * width + x] = row_buf[x];
        }
    };

    for (int i = 0; i < num_threads; ++i) {
        int start = i * rows_per_thread;
        int end = std::min(start + rows_per_thread, height);
        if (start < end) threads.emplace_back(edt_horizontal, start, end);
    }
    for (auto& t : threads) t.join();
    threads.clear();

    // 3. EDT Pass 2: Vertical
    // Process each column independently
    // Need to transpose or just stride? Striding is bad for cache, but for EDT_1D we need a vector.
    // Let's copy column to vector, process, copy back.
    
    int cols_per_thread = (width + num_threads - 1) / num_threads;
    auto edt_vertical = [&](int start_x, int end_x) {
        std::vector<float> col_buf(height);
        for (int x = start_x; x < end_x; ++x) {
            // Inside
            for(int y=0; y<height; ++y) col_buf[y] = dist_inside[y * width + x];
            EDT_1D(col_buf, height);
            for(int y=0; y<height; ++y) dist_inside[y * width + x] = col_buf[y];

            // Outside
            for(int y=0; y<height; ++y) col_buf[y] = dist_outside[y * width + x];
            EDT_1D(col_buf, height);
            for(int y=0; y<height; ++y) dist_outside[y * width + x] = col_buf[y];
        }
    };

    for (int i = 0; i < num_threads; ++i) {
        int start = i * cols_per_thread;
        int end = std::min(start + cols_per_thread, width);
        if (start < end) threads.emplace_back(edt_vertical, start, end);
    }
    for (auto& t : threads) t.join();
    threads.clear();

    // 4. Render Pass
    auto render_rows = [&](int start_y, int end_y) {
        for (int y = start_y; y < end_y; ++y) {
            Pixel* out_row = reinterpret_cast<Pixel*>(output_base + y * output_rowbytes);
            const Pixel* in_row = reinterpret_cast<const Pixel*>(input_base + y * input_rowbytes);
            
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                float d_in = std::sqrt(dist_inside[idx]);
                float d_out = std::sqrt(dist_outside[idx]);
                
                // Signed distance: negative inside, positive outside
                // But here: d_in is 0 inside, >0 outside (distance to nearest inside pixel)
                // Wait, dist_inside initialized to 0 inside. So d_in is distance FROM inside set?
                // No, EDT computes distance to nearest 0.
                // If inside pixels are 0, then d_in is distance to nearest inside pixel.
                // So inside the object, d_in is 0. Outside, it increases.
                
                // We want distance to the EDGE.
                // Edge is where transition happens.
                // d_out: distance to nearest outside pixel. Inside object, it increases. Outside, it is 0.
                
                // SDF = d_in - d_out?
                // Inside: d_in = 0, d_out > 0. SDF = -d_out (negative inside).
                // Outside: d_in > 0, d_out = 0. SDF = d_in (positive outside).
                // Correct.
                
                float sdf = d_in - d_out;
                
                // Adjust SDF based on direction
                // Thickness T.
                // Both: -T/2 to T/2.
                // Inside: -T to 0.
                // Outside: 0 to T.
                
                float alpha_factor = 0.0f;
                float aa_width = 1.0f; // 1 pixel AA
                
                if (direction == DIRECTION_BOTH) {
                    // Border is centered at 0. Width T.
                    // Range [-T/2, T/2].
                    // Distance from center of border: abs(sdf).
                    // Coverage = 1 - smoothstep(T/2 - 0.5, T/2 + 0.5, abs(sdf))
                    float half_t = thickness * 0.5f;
                    float d = std::abs(sdf);
                    alpha_factor = 1.0f - Clamp((d - half_t + 0.5f), 0.0f, 1.0f);
                } else if (direction == DIRECTION_INSIDE) {
                    // Border from -T to 0.
                    // We want 1 when sdf in [-T, 0].
                    // Distance from center (-T/2): abs(sdf - (-T/2)) = abs(sdf + T/2)
                    // Same logic as Both but shifted.
                    float half_t = thickness * 0.5f;
                    float d = std::abs(sdf + half_t);
                    alpha_factor = 1.0f - Clamp((d - half_t + 0.5f), 0.0f, 1.0f);
                } else { // Outside
                    // Border from 0 to T.
                    // Center T/2.
                    float half_t = thickness * 0.5f;
                    float d = std::abs(sdf - half_t);
                    alpha_factor = 1.0f - Clamp((d - half_t + 0.5f), 0.0f, 1.0f);
                }

                // Composite
                // If show_line_only, just show border color * alpha_factor.
                // Else, composite border over input.
                
                Pixel p_in = in_row[x];
                Pixel p_out;
                
                float border_a = BorderPixelTraits<Pixel>::ToFloat(border_color.alpha) * alpha_factor;
                
                if (show_line_only) {
                    p_out.red = border_color.red;
                    p_out.green = border_color.green;
                    p_out.blue = border_color.blue;
                    p_out.alpha = BorderPixelTraits<Pixel>::FromFloat(border_a);
                } else {
                    // Simple alpha blending: Border over Input
                    // out = border * border_a + input * (1 - border_a)
                    // Note: Pre-multiplied alpha handling might be needed if AE expects it.
                    // Assuming straight alpha for simplicity or standard AE compositing.
                    // AE usually uses pre-multiplied alpha.
                    // If pre-multiplied:
                    // out.a = border.a + input.a * (1 - border.a)
                    // out.rgb = border.rgb + input.rgb * (1 - border.a)
                    // But border_color from param is usually straight.
                    
                    float in_a = BorderPixelTraits<Pixel>::ToFloat(p_in.alpha);
                    float ba_norm = alpha_factor; // 0..1
                    
                    // Let's assume straight alpha blending for RGB, then premultiply?
                    // Or just standard lerp.
                    
                    float out_a = in_a + (BorderPixelTraits<Pixel>::MAX_VAL - in_a) * ba_norm; // Approximation
                    
                    // Correct over operator:
                    // out = src + dst * (1 - src_a)
                    // But here we are painting a stroke.
                    // Let's just mix based on alpha_factor.
                    
                    float r = BorderPixelTraits<Pixel>::ToFloat(border_color.red);
                    float g = BorderPixelTraits<Pixel>::ToFloat(border_color.green);
                    float b = BorderPixelTraits<Pixel>::ToFloat(border_color.blue);
                    
                    float ir = BorderPixelTraits<Pixel>::ToFloat(p_in.red);
                    float ig = BorderPixelTraits<Pixel>::ToFloat(p_in.green);
                    float ib = BorderPixelTraits<Pixel>::ToFloat(p_in.blue);
                    
                    p_out.red = BorderPixelTraits<Pixel>::FromFloat(r * ba_norm + ir * (1.0f - ba_norm));
                    p_out.green = BorderPixelTraits<Pixel>::FromFloat(g * ba_norm + ig * (1.0f - ba_norm));
                    p_out.blue = BorderPixelTraits<Pixel>::FromFloat(b * ba_norm + ib * (1.0f - ba_norm));
                    p_out.alpha = BorderPixelTraits<Pixel>::FromFloat(std::max(in_a, border_a * BorderPixelTraits<Pixel>::MAX_VAL)); // Max alpha
                }
                out_row[x] = p_out;
            }
        }
    };

    for (int i = 0; i < num_threads; ++i) {
        int start = i * rows_per_thread;
        int end = std::min(start + rows_per_thread, height);
        if (start < end) threads.emplace_back(render_rows, start, end);
    }
    for (auto& t : threads) t.join();

    return PF_Err_NONE;
}

static PF_Err Render(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output) {
    int bpp = (output->width > 0) ? (output->rowbytes / output->width) : 0;
    if (bpp == sizeof(PF_PixelFloat)) {
        return RenderGeneric<PF_PixelFloat>(in_data, out_data, params, output);
    } else if (bpp == sizeof(PF_Pixel16)) {
        return RenderGeneric<PF_Pixel16>(in_data, out_data, params, output);
    } else {
        return RenderGeneric<PF_Pixel>(in_data, out_data, params, output);
    }
}

static PF_Err
About(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
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
GlobalSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    out_data->my_version = PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION, STAGE_VERSION, BUILD_VERSION);
    out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_PIX_INDEPENDENT;
    out_data->out_flags2 = PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_SUPPORTS_THREADED_RENDERING;
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

    // Checkout input
    ERR(extra->cb->checkout_layer(in_data->effect_ref,
        BORDER_INPUT,
        BORDER_INPUT,
        &req,
        in_data->current_time,
        in_data->time_step,
        in_data->time_scale,
        &in_result));

    // Set result rects
    if (!err) {
        UnionLRect(&in_result.result_rect, &extra->output->result_rect);
        UnionLRect(&in_result.max_result_rect, &extra->output->max_result_rect);
    }

    return err;
}

static PF_Err
SmartRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    PF_EffectWorld* input_world = NULL;
    PF_EffectWorld* output_world = NULL;
    PF_WorldSuite2* wsP = NULL;

    // Get World Suite
    ERR(AEFX_AcquireSuite(in_data, out_data, kPFWorldSuite, kPFWorldSuiteVersion2, "PFWorldSuite", (void**)&wsP));

    if (!err) {
        // Checkout input/output
        ERR((extra->cb->checkout_layer_pixels(in_data->effect_ref, BORDER_INPUT, &input_world)));
        ERR(extra->cb->checkout_output(in_data->effect_ref, &output_world));
    }

    if (!err && input_world && output_world) {
        // Checkout parameters
        PF_ParamDef p[BORDER_NUM_PARAMS];
        PF_ParamDef* pp[BORDER_NUM_PARAMS];
        
        // Input
        AEFX_CLR_STRUCT(p[BORDER_INPUT]);
        p[BORDER_INPUT].u.ld = *input_world;
        pp[BORDER_INPUT] = &p[BORDER_INPUT];

        // Params
        for (int i = 1; i < BORDER_NUM_PARAMS; ++i) {
            PF_Checkout_Value(in_data, out_data, i, in_data->current_time, in_data->time_step, in_data->time_scale, &p[i]);
            pp[i] = &p[i];
        }

        // Call Render
        int bpp = (output_world->width > 0) ? (output_world->rowbytes / output_world->width) : 0;
        if (bpp == sizeof(PF_PixelFloat)) {
            err = RenderGeneric<PF_PixelFloat>(in_data, out_data, pp, output_world);
        } else if (bpp == sizeof(PF_Pixel16)) {
            err = RenderGeneric<PF_Pixel16>(in_data, out_data, pp, output_world);
        } else {
            err = RenderGeneric<PF_Pixel>(in_data, out_data, pp, output_world);
        }
        
        // Checkin params
        for (int i = 1; i < BORDER_NUM_PARAMS; ++i) {
            PF_Checkin_Param(in_data, out_data, i, &p[i]);
        }
    }
    
    if (wsP) AEFX_ReleaseSuite(in_data, out_data, kPFWorldSuite, kPFWorldSuiteVersion2, "PFWorldSuite");

    return err;
}

extern "C" DllExport
PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr,
    PF_PluginDataCB2 inPluginDataCallBackPtr,
    SPBasicSuite * inSPBasicSuitePtr,
    const char* inHostName,
    const char* inHostVersion)
{
    PF_Err result = PF_Err_INVALID_CALLBACK;
    result = PF_REGISTER_EFFECT_EXT2(
        inPtr,
        inPluginDataCallBackPtr,
        "Border", // Name
        "361do Border", // Match Name
        "361do_plugins", // Category
        AE_RESERVED_INFO,
        "EffectMain",
        "https://github.com/rebuildup/Ae_Border");
    return result;
}

extern "C" DllExport
PF_Err EffectMain(PF_Cmd cmd,
    PF_InData * in_data,
    PF_OutData * out_data,
    PF_ParamDef * params[],
    PF_LayerDef * output,
    void* extra)
{
    PF_Err err = PF_Err_NONE;
    try {
        switch (cmd) {
        case PF_Cmd_ABOUT: err = About(in_data, out_data, params, output); break;
        case PF_Cmd_GLOBAL_SETUP: err = GlobalSetup(in_data, out_data, params, output); break;
        case PF_Cmd_PARAMS_SETUP: err = ParamsSetup(in_data, out_data, params, output); break;
        case PF_Cmd_RENDER: err = Render(in_data, out_data, params, output); break;
        case PF_Cmd_SMART_PRE_RENDER: err = PreRender(in_data, out_data, (PF_PreRenderExtra*)extra); break;
        case PF_Cmd_SMART_RENDER: err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra); break;
        default: break;
        }
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}
