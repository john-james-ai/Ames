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
data_paths = {
    "raw": "../data/raw/",
    "interim": "../data/interim/",
    "processed": "../data/processed/",
    "metadata": "../data/metadata/"
}

categoricals = {
    "nominal":  [    
        "MS_SubClass",
        "MS_Zoning",
        "Street",
        "Alley",
        "Land_Contour",
        "Lot_Config",
        "Neighborhood",
        "Condition_1",
        "Condition_2",
        "Bldg_Type",
        "House_Style",
        "Roof_Style",
        "Roof_Matl",
        "Exterior_1st",
        "Exterior_2nd",
        "Mas_Vnr_Type",
        "Foundation",
        "Heating",
        "Central_Air",
        "Garage_Type",
        "Misc_Feature",
        "Sale_Type",
        "Sale_Condition"],
    "ordinal": [
        "Lot_Shape",
        "Utilities",
        "Land_Slope",
        "Overall_Qual",
        "Overall_Cond",
        "Exter_Qual",
        "Exter_Cond",
        "Bsmt_Qual",
        "Bsmt_Cond",
        "Bsmt_Exposure",
        "BsmtFin_Type_1",
        "BsmtFin_Type_2",
        "Heating_QC",
        "Electrical",
        "Kitchen_Qual",
        "Functional",
        "Fireplace_Qu",
        "Garage_Finish",
        "Garage_Qual",
        "Garage_Cond",
        "Paved_Drive",
        "Pool_QC",
        "Fence"]
}

numericals = {
    "discrete": [
        "Year_Built",
        "Year_Remod_Add",
        "Bsmt_Full_Bath",
        "Bsmt_Half_Bath",
        "Full_Bath",
        "Half_Bath",
        "Bedroom_AbvGr",
        "Kitchen_AbvGr",
        "TotRms_AbvGrd",
        "Fireplaces",
        "Garage_Cars",
        "Mo_Sold",
        "Year_Sold"],

    "continuous": [
        "Lot_Frontage",
        "Lot_Area",
        "Mas_Vnr_Area",
        "BsmtFin_SF_1",
        "BsmtFin_SF_2",
        "Bsmt_Unf_SF",
        "Total_Bsmt_SF",
        "First_Flr_SF",
        "Second_Flr_SF",
        "Low_Qual_Fin_SF",
        "Gr_Liv_Area",
        "Garage_Area",
        "Wood_Deck_SF",
        "Open_Porch_SF",
        "Enclosed_Porch",
        "Three_season_porch",
        "Screen_Porch",
        "Pool_Area",
        "Misc_Val"]
}