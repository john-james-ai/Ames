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

from globals import data_paths, categoricals
from data import Ames
# --------------------------------------------------------------------------- #
def categorical_metadata(X):
    summary = pd.DataFrame()
    detail = pd.DataFrame()

    for var_type, variables in categoricals.items():        
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

    summary.to_csv(os.path.join(data_paths["metadata"],"categorical_summary.csv"), index=False)
    detail.to_csv(os.path.join(data_paths["metadata"],"categorical_detail.csv"), index=False)
    print(f"{detail.shape[0]} categorical variables and levels.")        
    print(summary)

def main():
    ames = Ames()
    X, y = ames.read()   
    categorical_metadata(X)

if __name__ == "__main__":
    main()
#%%
        




