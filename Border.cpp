#define NOMINMAX
#include "Border.h"

#include <vector>
#include <algorithm>
#include <cmath>
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
// v, z: pre-allocated buffers
static void EDT_1D(std::vector<float>& grid, int width, std::vector<int>& v, std::vector<float>& z) {
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
    for (int y = 0; y < height; ++y) {
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

    // 2. EDT Pass 1: Horizontal
    std::vector<float> row_buf(width);
    std::vector<int> v(width);
    std::vector<float> z(width + 1);
    for (int y = 0; y < height; ++y) {
        // Inside
        for(int x=0; x<width; ++x) row_buf[x] = dist_inside[y * width + x];
        EDT_1D(row_buf, width, v, z);
        for(int x=0; x<width; ++x) dist_inside[y * width + x] = row_buf[x];

        // Outside
        for(int x=0; x<width; ++x) row_buf[x] = dist_outside[y * width + x];
        EDT_1D(row_buf, width, v, z);
        for(int x=0; x<width; ++x) dist_outside[y * width + x] = row_buf[x];
    }

    // 3. EDT Pass 2: Vertical
    std::vector<float> col_buf(height);
    std::vector<int> v_col(height);
    std::vector<float> z_col(height + 1);
    for (int x = 0; x < width; ++x) {
        // Inside
        for(int y=0; y<height; ++y) col_buf[y] = dist_inside[y * width + x];
        EDT_1D(col_buf, height, v_col, z_col);
        for(int y=0; y<height; ++y) dist_inside[y * width + x] = col_buf[y];

        // Outside
        for(int y=0; y<height; ++y) col_buf[y] = dist_outside[y * width + x];
        EDT_1D(col_buf, height, v_col, z_col);
        for(int y=0; y<height; ++y) dist_outside[y * width + x] = col_buf[y];
    }

    // 4. Render Pass
    for (int y = 0; y < height; ++y) {
        Pixel* out_row = reinterpret_cast<Pixel*>(output_base + y * output_rowbytes);
        const Pixel* in_row = reinterpret_cast<const Pixel*>(input_base + y * input_rowbytes);
        
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float d_in = std::sqrt(dist_inside[idx]);
            float d_out = std::sqrt(dist_outside[idx]);
            
            float sdf = d_in - d_out;
            
            float alpha_factor = 0.0f;
            float aa_width = 1.0f; // 1 pixel AA
            
            if (direction == DIRECTION_BOTH) {
                float half_t = thickness * 0.5f;
                float d = std::abs(sdf);
                alpha_factor = 1.0f - Clamp((d - half_t + 0.5f), 0.0f, 1.0f);
            } else if (direction == DIRECTION_INSIDE) {
                float half_t = thickness * 0.5f;
                float d = std::abs(sdf + half_t);
                alpha_factor = 1.0f - Clamp((d - half_t + 0.5f), 0.0f, 1.0f);
            } else { // Outside
                float half_t = thickness * 0.5f;
                float d = std::abs(sdf - half_t);
                alpha_factor = 1.0f - Clamp((d - half_t + 0.5f), 0.0f, 1.0f);
            }

            Pixel p_in = in_row[x];
            Pixel p_out;
            
            float border_a = BorderPixelTraits<Pixel>::ToFloat(border_color.alpha) * alpha_factor;
            
            if (show_line_only) {
                p_out.red = border_color.red;
                p_out.green = border_color.green;
                p_out.blue = border_color.blue;
                p_out.alpha = BorderPixelTraits<Pixel>::FromFloat(border_a);
            } else {
                float in_a = BorderPixelTraits<Pixel>::ToFloat(p_in.alpha);
                float ba_norm = alpha_factor; // 0..1
                
                float out_a = in_a + (BorderPixelTraits<Pixel>::MAX_VAL - in_a) * ba_norm;
                
                float r = BorderPixelTraits<Pixel>::ToFloat(border_color.red);
                float g = BorderPixelTraits<Pixel>::ToFloat(border_color.green);
                float b = BorderPixelTraits<Pixel>::ToFloat(border_color.blue);
                
                float ir = BorderPixelTraits<Pixel>::ToFloat(p_in.red);
                float ig = BorderPixelTraits<Pixel>::ToFloat(p_in.green);
                float ib = BorderPixelTraits<Pixel>::ToFloat(p_in.blue);
                
                p_out.red = BorderPixelTraits<Pixel>::FromFloat(r * ba_norm + ir * (1.0f - ba_norm));
                p_out.green = BorderPixelTraits<Pixel>::FromFloat(g * ba_norm + ig * (1.0f - ba_norm));
                p_out.blue = BorderPixelTraits<Pixel>::FromFloat(b * ba_norm + ib * (1.0f - ba_norm));
                p_out.alpha = BorderPixelTraits<Pixel>::FromFloat(std::max(in_a, border_a * BorderPixelTraits<Pixel>::MAX_VAL));
            }
            out_row[x] = p_out;
        }
    }

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
    out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE;
    return PF_Err_NONE;
}

static PF_Err
ParamsSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    AEFX_CLR_STRUCT(def);

    PF_ADD_FLOAT_SLIDERX(
        "Thickness",
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

    PF_ADD_COLOR(
        "Color",
        PF_HALF_CHAN8,
        PF_MAX_CHAN8,
        PF_MAX_CHAN8,
        COLOR_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_SLIDER(
        "Threshold",
        BORDER_THRESHOLD_MIN,
        BORDER_THRESHOLD_MAX,
        BORDER_THRESHOLD_MIN,
        BORDER_THRESHOLD_MAX,
        BORDER_THRESHOLD_DFLT,
        THRESHOLD_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_POPUP(
        "Direction",
        3,
        BORDER_DIRECTION_DFLT,
        "Both|Inside|Outside",
        DIRECTION_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_CHECKBOXX(
        "Show Line Only",
        FALSE,
        0,
        SHOW_LINE_ONLY_DISK_ID);

    out_data->num_params = BORDER_NUM_PARAMS;
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
        default: break;
        }
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}
