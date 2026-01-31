#pragma once

// Border effect version definitions shared between runtime and PiPL.
// Use PF_VERSION macro to ensure consistency and avoid calculation errors.
#define MAJOR_VERSION 1
#define MINOR_VERSION 0
#define BUG_VERSION 0
// 0 = develop, 1 = alpha, 2 = beta, 3 = release (per AE SDK)
#define STAGE_VERSION PF_Stage_DEVELOP
#define BUILD_VERSION 2

// Use PF_VERSION macro instead of manual calculation to prevent errors
#define BORDER_VERSION_VALUE PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION, STAGE_VERSION, BUILD_VERSION)
