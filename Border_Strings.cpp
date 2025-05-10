/*******************************************************************/
/*                                                                 */
/*                      ADOBE CONFIDENTIAL                         */
/*                   _ _ _ _ _ _ _ _ _ _ _ _ _                     */
/*                                                                 */
/* Copyright 2007-2023 Adobe Inc.                                  */
/* All Rights Reserved.                                            */
/*                                                                 */
/* NOTICE:  All information contained herein is, and remains the   */
/* property of Adobe Inc. and its suppliers, if                    */
/* any.  The intellectual and technical concepts contained         */
/* herein are proprietary to Adobe Inc. and its                    */
/* suppliers and may be covered by U.S. and Foreign Patents,       */
/* patents in process, and are protected by trade secret or        */
/* copyright law.  Dissemination of this information or            */
/* reproduction of this material is strictly forbidden unless      */
/* prior written permission is obtained from Adobe Inc.            */
/* Incorporated.                                                   */
/*                                                                 */
/*******************************************************************/

#include "Border.h"

typedef struct {
	A_u_long	index;
	A_char		str[256];
} TableString;



TableString g_strs[StrID_NUMTYPES] = {
    StrID_NONE,                      "",
    StrID_Name,                      "Border",
    StrID_Description,               "Generates a line between transparent and non-transparent pixels.\rCopyright 2023-2025.",
    StrID_Thickness_Param_Name,      "Thickness",
    StrID_Color_Param_Name,          "Color",
    StrID_Threshold_Param_Name,      "Threshold",
    StrID_ShowLineOnly_Param_Name,   "Show Line Only",
};


char	*GetStringPtr(int strNum)
{
	return g_strs[strNum].str;
}
	