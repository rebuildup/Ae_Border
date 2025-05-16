#pragma once

typedef enum {
    StrID_NONE,
    StrID_Name,
    StrID_Description,
    StrID_Thickness_Param_Name,
    StrID_Color_Param_Name,
    StrID_Threshold_Param_Name,
    StrID_Direction_Param_Name,
    StrID_Direction_Both,     
    StrID_Direction_Inside,   
    StrID_Direction_Outside,  
    StrID_ShowLineOnly_Param_Name,
    StrID_NUMTYPES
} StrIDType;

char* GetStringPtr(int strNum);