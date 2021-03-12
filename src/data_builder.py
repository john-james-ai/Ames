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
import pandas as pd
import numpy as np
import fnmatch
import os
from tabulate import tabulate
# --------------------------------------------------------------------------- #
class DataBuilder:
    """Combines all training set splits into a single file."""
    def __init__(self, inpath="../data/raw/", outpath="../data/interim/"):
        self._inpath = inpath
        self._outpath = outpath
        self.X_train = None
        self.y_train = None

    def build_data(self):

        train = pd.DataFrame()
        
        for filename in os.listdir(self._inpath):
            if fnmatch.fnmatch(filename, "*train.csv"):                
                df = pd.read_csv(os.path.join(self._inpath, filename))
                train = pd.concat((train, df), axis=0)
        
        self.X_train = train.loc[:,train.columns != "Sale_Price"]      
        self.y_train = train.loc[:,train.columns == "Sale_Price"]      
        self.X_train.to_csv(os.path.join(self._outpath, "X_train.csv"), index=False)          
        self.y_train.to_csv(os.path.join(self._outpath, "y_train.csv"), index=False)          
    
    def summary(self):
        print("_"*40)
        print("X_Train")
        print("_"*40)
        print(self.X_train.info())
        print(self.X_train.shape)
        percent_missing = self.X_train.isnull().sum() * 100 / len(self.X_train)
        missing = pd.DataFrame({"Column": self.X_train.columns,
                                "Percent Missing": percent_missing})
        missing.sort_values("Percent Missing", inplace=True, ascending=False)
        print("\n\nMissing Values")
        print(tabulate(missing[missing["Percent Missing"]>0], headers="keys", showindex=False))

def main():
    builder = DataBuilder()
    builder.build_data()   
    builder.summary()

if __name__ == "__main__":
    main()
#%%
        




