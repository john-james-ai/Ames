# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \eda.py                                                           #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Tuesday, March 9th 2021, 11:06:05 pm                        #
# Last Modified : Tuesday, March 9th 2021, 11:06:05 pm                        #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
import os
import pandas as pd
import numpy as np
from tabulate import tabulate
import pprint

from globals import data_paths, categoricals
from data import Ames
# --------------------------------------------------------------------------- #
def categorical_metadata(X):
    summary = pd.DataFrame()
    detail = pd.DataFrame()

    for var_type, variables in categoricals.items():        
        detail = pd.DataFrame()
        filename = var_type + ".csv"
        for variable in variables:
            # Summary information
            d = {"Type": var_type, "Variable": variable, "nLevels": X[variable].nunique()} 
            df = pd.DataFrame(data=d,index=[0])
            summary = pd.concat([summary,df], axis=0)
            # Detail
            df = pd.DataFrame(data=X.value_counts(subset=[variable])).reset_index()
            df.columns = ["Levels", "Counts"]
            df["Type"] = var_type
            df["Variable"] = variable
            df = df[["Type", "Variable", "Levels", "Counts"]]            
            detail = pd.concat([detail,df], axis=0)
        detail.to_csv(os.path.join(data_paths["metadata"],filename), index=False)

    summary.to_csv(os.path.join(data_paths["metadata"],"categorical_summary.csv"), index=False)    
    print(f"{detail.shape[0]} categorical variables and levels.")        
    print(summary)

def create_ordinal_map():
    filename = "ordered.csv"
    codes = pd.read_csv(os.path.join(data_paths["metadata"], filename))
    ordinal_map = {}
    levels = {}

    variables = codes.groupby("Variable")
    for variable, group in variables:
        levels = dict(zip(group["Levels"], group["Order"]))
        ordinal_map[variable] = levels
    pp = pprint.PrettyPrinter(compact=True, width=100)
    pp.pprint(ordinal_map)
    
def create_nominal_map():
    filename = "categorical_summary.csv"
    categoricals = pd.read_csv(os.path.join(data_paths["metadata"], filename))
    nominals = categoricals[categoricals["Type"]=="nominal"]["Variable"]
    pp = pprint.PrettyPrinter(compact=True, width=100)
    pp.pprint(list(nominals))

def main():
    # ames = Ames()
    # X, y = ames.read()   
    # categorical_metadata(X)
    create_nominal_map()

if __name__ == "__main__":
    main()
#%%
        




