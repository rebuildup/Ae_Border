#include "AEConfig.h"
#include "AE_Effect.h"       // for PF_OutFlag macros & PF_VERSION
#include "AE_EffectVers.h"
#include "Border_Version.h"

// Precompute flag values for the PiPL expression parser.
enum {
    BORDER_OUTFLAGS  = (PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_PIX_INDEPENDENT | PF_OutFlag_USE_OUTPUT_EXTENT),
    BORDER_OUTFLAGS2 = (PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_SUPPORTS_THREADED_RENDERING)
};

#ifndef AE_OS_WIN
    #include <AE_General.r>
#endif
    
resource 'PiPL' (16000) {
    {    /* array properties: 12 elements */
        /* [1] */
        Kind {
            AEEffect
        },
        /* [2] */
        Name {
            "Border"
        },
        /* [3] */
        Category {
            "361do_plugins"
        },
#ifdef AE_OS_WIN
    #ifdef AE_PROC_INTELx64
        CodeWin64X86 {"EffectMain"},
    #endif
#else
    #ifdef AE_OS_MAC
        CodeMacIntel64 {"EffectMain"},
        CodeMacARM64 {"EffectMain"},
    #endif
#endif
        /* [6] */
        AE_PiPL_Version {
            2,
            0
        },
        /* [7] */
        AE_Effect_Spec_Version {
            PF_PLUG_IN_VERSION,
            PF_PLUG_IN_SUBVERS
        },
        /* [8] */
        AE_Effect_Version {
            BORDER_VERSION_VALUE
        },
        /* [9] */
        AE_Effect_Info_Flags {
            0
        },
        /* [10] */
        AE_Effect_Global_OutFlags { BORDER_OUTFLAGS },
        AE_Effect_Global_OutFlags_2 { BORDER_OUTFLAGS2 },
        /* [11] */
        AE_Effect_Match_Name {
            "361do Border"
        },
        /* [12] */
        AE_Reserved_Info {
            0
        },
        /* [13] */
        AE_Effect_Support_URL {
            "https://github.com/rebuildup/Ae_Border"
        }
    }
};
