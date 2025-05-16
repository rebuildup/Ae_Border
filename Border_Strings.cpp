#include "Border.h"

typedef struct {
    A_u_long    index;
    A_char      str[256];
} TableString;



TableString g_strs[StrID_NUMTYPES] = {
    StrID_NONE,                      "",
    StrID_Name,                      "Border",
    StrID_Description,               "Generates a line between transparent and non-transparent pixels.\rCopyright 2023-2025.",
    StrID_Thickness_Param_Name,      "Stroke Width",
    StrID_Color_Param_Name,          "Color",
    StrID_Threshold_Param_Name,      "Threshold",
    StrID_Direction_Param_Name,      "Direction",
    StrID_Direction_Both,            "Both Directions",
    StrID_Direction_Inside,          "Inside",
    StrID_Direction_Outside,         "Outside",
    StrID_ShowLineOnly_Param_Name,   "Show Line Only",
};


char* GetStringPtr(int strNum)
{
    return g_strs[strNum].str;
}