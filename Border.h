/*
    Border.h
*/

#pragma once

#ifndef BORDER_H
#define BORDER_H

typedef unsigned char		u_char;
typedef unsigned short		u_short;
typedef unsigned short		u_int16;
typedef unsigned long		u_long;
typedef short int			int16;
#define PF_TABLE_BITS	12
#define PF_TABLE_SZ_16	4096

#define BORDER_THICKNESS_MIN     0.0
#define BORDER_THICKNESS_MAX     2000.0
#define BORDER_THICKNESS_DFLT    1.0

#define BORDER_THRESHOLD_MIN     0
#define BORDER_THRESHOLD_MAX     255
#define BORDER_THRESHOLD_DFLT    1

// Direction options
enum {
    DIRECTION_BOTH = 1,
    DIRECTION_INSIDE,
    DIRECTION_OUTSIDE
};

#define BORDER_DIRECTION_DFLT    DIRECTION_BOTH

// Conversion macros for 8/16 bit color
#define PF_BYTE_TO_CHAR(b)      ((b) * PF_MAX_CHAN16 / PF_MAX_CHAN8)
#define PF_CHAR_TO_BYTE(c)      ((c) * PF_MAX_CHAN8 / PF_MAX_CHAN16)

enum {
    BORDER_INPUT = 0,
    BORDER_THICKNESS,
    BORDER_COLOR,
    BORDER_THRESHOLD,
    BORDER_DIRECTION,
    BORDER_SHOW_LINE_ONLY,
    BORDER_NUM_PARAMS
};
```c
/*
    Border.h
*/

#pragma once

#ifndef BORDER_H
#define BORDER_H

typedef unsigned char		u_char;
typedef unsigned short		u_short;
typedef unsigned short		u_int16;
typedef unsigned long		u_long;
typedef short int			int16;
#define PF_TABLE_BITS	12
#define PF_TABLE_SZ_16	4096

#define BORDER_THICKNESS_MIN     0.0
#define BORDER_THICKNESS_MAX     2000.0
#define BORDER_THICKNESS_DFLT    1.0

#define BORDER_THRESHOLD_MIN     0
#define BORDER_THRESHOLD_MAX     255
#define BORDER_THRESHOLD_DFLT    1

// Direction options
enum {
    DIRECTION_BOTH = 1,
    DIRECTION_INSIDE,
    DIRECTION_OUTSIDE
};

#define BORDER_DIRECTION_DFLT    DIRECTION_BOTH

// Conversion macros for 8/16 bit color
#define PF_BYTE_TO_CHAR(b)      ((b) * PF_MAX_CHAN16 / PF_MAX_CHAN8)
#define PF_CHAR_TO_BYTE(c)      ((c) * PF_MAX_CHAN8 / PF_MAX_CHAN16)

enum {
    BORDER_INPUT = 0,
    BORDER_THICKNESS,
    BORDER_COLOR,
    BORDER_THRESHOLD,
    BORDER_DIRECTION,
    BORDER_SHOW_LINE_ONLY,
    BORDER_NUM_PARAMS
};

enum {
    THICKNESS_DISK_ID = 1,
    COLOR_DISK_ID,
    THRESHOLD_DISK_ID,
    DIRECTION_DISK_ID,
    SHOW_LINE_ONLY_DISK_ID
};

// Define Smart Render structs locally to avoid missing header issues
typedef struct PF_SmartRenderCallbacks_Local {
    PF_Err (*checkout_layer_pixels)(PF_ProgPtr effect_ref, PF_ParamIndex index, PF_EffectWorld **pixels);
    PF_Err (*checkout_output)(PF_ProgPtr effect_ref, PF_EffectWorld **output);
    PF_Err (*checkin_layer_pixels)(PF_ProgPtr effect_ref, PF_ParamIndex index);
    PF_Err (*is_layer_pixel_data_valid)(PF_ProgPtr effect_ref, PF_ParamIndex index, PF_Boolean *valid);
} PF_SmartRenderCallbacks_Local;

typedef struct PF_SmartRenderExtra_Local {
    PF_SmartRenderCallbacks_Local *cb;
    void *unused;
} PF_SmartRenderExtra_Local;

typedef struct BorderInfo {
    A_long      thickness;
    PF_Pixel    color;
    A_u_char    threshold;
    A_long      direction;
    PF_Boolean  showLineOnly;
} BorderInfo, * BorderInfoP, ** BorderInfoH;

extern "C" {

    DllExport
        PF_Err
        EffectMain(
            PF_Cmd			cmd,
            PF_InData* in_data,
            PF_OutData* out_data,
            PF_ParamDef* params[],
            PF_LayerDef* output,
            PF_SmartRenderExtra_Local* extra);

}

#endif // BORDER_H
```