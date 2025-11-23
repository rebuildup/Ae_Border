/*
    Border.h
*/

#pragma once

#ifndef BORDER_H
#define BORDER_H

#define PF_DEEP_COLOR_AWARE 1

#include "AEConfig.h"

#ifdef AE_OS_WIN
    typedef unsigned short PixelType;
    #include <Windows.h>
#endif

#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEFX_ChannelDepthTpl.h"
#include "AEGP_SuiteHandler.h"

#include "Border_Strings.h"

/* Versioning information */
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

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
    BORDER_NUM_PARAMS
};

enum {
    THICKNESS_DISK_ID = 1,
    COLOR_DISK_ID,
    THRESHOLD_DISK_ID,
    DIRECTION_DISK_ID,
    SHOW_LINE_ONLY_DISK_ID
};

typedef struct BorderInfo {
    PF_FpLong thickness;
    PF_Boolean showLineOnly;
} BorderInfo, *BorderInfoP, **BorderInfoH;

extern "C" {

    DllExport
    PF_Err
    EffectMain(
        PF_Cmd			cmd,
        PF_InData		*in_data,
        PF_OutData		*out_data,
        PF_ParamDef		*params[],
        PF_LayerDef		*output,
        void			*extra);

}

#endif // BORDER_H