# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \globals.py                                                       #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Thursday, March 11th 2021, 4:13:00 pm                       #
# Last Modified : Thursday, March 11th 2021, 4:13:00 pm                       #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
random_state = 9876
discrete =  ["Year_Built","Year_Remod_Add","Bsmt_Full_Bath","Bsmt_Half_Bath",
    "Full_Bath","Half_Bath","Bedroom_AbvGr","Kitchen_AbvGr","TotRms_AbvGrd",
    "Fireplaces","Garage_Cars","Mo_Sold","Year_Sold","Age"]
continuous = ["Lot_Frontage","Lot_Area","Mas_Vnr_Area","BsmtFin_SF_1","BsmtFin_SF_2",
    "Bsmt_Unf_SF","Total_Bsmt_SF","First_Flr_SF","Second_Flr_SF","Low_Qual_Fin_SF",
    "Gr_Liv_Area","Garage_Area","Wood_Deck_SF","Open_Porch_SF","Enclosed_Porch",
    "Three_season_porch","Screen_Porch","Pool_Area","Misc_Val"]
numeric = ["Lot_Frontage","Lot_Area","Mas_Vnr_Area","BsmtFin_SF_1","BsmtFin_SF_2",
    "Bsmt_Unf_SF","Total_Bsmt_SF","First_Flr_SF","Second_Flr_SF","Low_Qual_Fin_SF",
    "Gr_Liv_Area","Garage_Area","Wood_Deck_SF","Open_Porch_SF","Enclosed_Porch",
    "Three_season_porch","Screen_Porch","Pool_Area","Misc_Val",
    "Year_Built","Year_Remod_Add","Bsmt_Full_Bath","Bsmt_Half_Bath",
    "Full_Bath","Half_Bath","Bedroom_AbvGr","Kitchen_AbvGr","TotRms_AbvGrd",
    "Fireplaces","Garage_Cars","Mo_Sold","Year_Sold"]
n_nominal_levels = 189
nominal = ['MS_SubClass', 'MS_Zoning', 'Street', 'Alley', 'Land_Contour', 'Lot_Config', 'Neighborhood',
 'Condition_1', 'Condition_2', 'Bldg_Type', 'House_Style', 'Roof_Style', 'Roof_Matl',
 'Exterior_1st', 'Exterior_2nd', 'Mas_Vnr_Type', 'Foundation', 'Heating', 'Central_Air',
 'Garage_Type', 'Misc_Feature', 'Sale_Type', 'Sale_Condition']

ordinal = ['BsmtFin_Type_1', 'BsmtFin_Type_2', 'Bsmt_Cond', 'Bsmt_Exposure', 
'Bsmt_Qual', 'Electrical', 'Exter_Cond', 'Exter_Qual', 'Fence', 'Fireplace_Qu', 
'Functional', 'Garage_Cond', 'Garage_Finish', 'Garage_Qual', 'Heating_QC', 'Kitchen_Qual', 
'Land_Slope', 'Lot_Shape', 'Overall_Cond', 'Overall_Qual', 'Paved_Drive', 'Pool_QC', 'Utilities']

ordinal_map = {'BsmtFin_Type_1': {'ALQ': 5, 'BLQ': 4, 'GLQ': 6, 'LwQ': 2, 'No_Basement': 0, 'Rec': 3, 'Unf': 1},
 'BsmtFin_Type_2': {'ALQ': 5, 'BLQ': 4, 'GLQ': 6, 'LwQ': 2, 'No_Basement': 0, 'Rec': 3, 'Unf': 1},
 'Bsmt_Cond': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Basement': 0, 'Poor': 1, 'Typical': 3},
 'Bsmt_Exposure': {'Av': 3, 'Gd': 4, 'Mn': 2, 'No': 1, 'No_Basement': 0},
 'Bsmt_Qual': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Basement': 0, 'Poor': 1, 'Typical': 3},
 'Electrical': {'FuseA': 4, 'FuseF': 2, 'FuseP': 1, 'Mix': 0, 'SBrkr': 5, 'Unknown': 3},
 'Exter_Cond': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'Poor': 0, 'Typical': 2},
 'Exter_Qual': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'Poor': 0, 'Typical': 2},
 'Fence': {'Good_Privacy': 4, 'Good_Wood': 2, 'Minimum_Privacy': 3, 'Minimum_Wood_Wire': 1,'No_Fence': 0},
 'Fireplace_Qu': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Fireplace': 0, 'Poor': 1, 'Typical': 3},
 'Functional': {'Maj1': 3, 'Maj2': 2, 'Min1': 5, 'Min2': 6, 'Mod': 4, 'Sal': 0, 'Sev': 1, 'Typ': 7},
 'Garage_Cond': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Garage': 0, 'Poor': 1, 'Typical': 3},
 'Garage_Finish': {'Fin': 3, 'No_Garage': 0, 'RFn': 2, 'Unf': 1},
 'Garage_Qual': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Garage': 0, 'Poor': 1, 'Typical': 3},
 'Heating_QC': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'Poor': 0, 'Typical': 2},
 'Kitchen_Qual': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'Poor': 0, 'Typical': 2},
 'Land_Slope': {'Gtl': 0, 'Mod': 1, 'Sev': 2},
 'Lot_Shape': {'Irregular': 0, 'Moderately_Irregular': 1, 'Regular': 3, 'Slightly_Irregular': 2},
 'Overall_Cond': {'Above_Average': 5, 'Average': 4,'Below_Average': 3,'Excellent': 8,'Fair': 2,
                  'Good': 6,'Poor': 1,'Very_Excellent': 9,'Very_Good': 7,'Very_Poor': 0},
 'Overall_Qual': {'Above_Average': 5,'Average': 4,'Below_Average': 3,'Excellent': 8,'Fair': 2,
                  'Good': 6,'Poor': 1,'Very_Excellent': 9,'Very_Good': 7,'Very_Poor': 0},
 'Paved_Drive': {'Dirt_Gravel': 0, 'Partial_Pavement': 1, 'Paved': 2},
 'Pool_QC': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'No_Pool': 0, 'Typical': 2},
 'Utilities': {'AllPub': 2, 'NoSeWa': 0, 'NoSewr': 1}}