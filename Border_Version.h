#pragma once

// Border effect version definitions shared between runtime and PiPL.
// Keep the PiPL and GlobalSetup in sync by deriving the packed value
// directly from the individual fields via PF_VERSION.
#include "AE_EffectVers.h"

#define MAJOR_VERSION 1
#define MINOR_VERSION 0
#define BUG_VERSION 0
#define STAGE_VERSION PF_Stage_DEVELOP
#define BUILD_VERSION 1

#define BORDER_VERSION_VALUE PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION, STAGE_VERSION, BUILD_VERSION)

