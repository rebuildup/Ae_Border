#pragma once

// Border effect version definitions shared between runtime and PiPL.
// Use an explicit packed value to keep PiPL and code in sync and avoid
// PiPL parser issues with expressions on Windows.
#define MAJOR_VERSION 1
#define MINOR_VERSION 0
#define BUG_VERSION 0
// 0 = develop, 1 = alpha, 2 = beta, 3 = release (per AE SDK)
#define STAGE_VERSION 0
#define BUILD_VERSION 1

// PF_VERSION(1,0,0,PF_Stage_DEVELOP,1) evaluated offline: 0x00080001 (524289)
#define BORDER_VERSION_VALUE 524289

