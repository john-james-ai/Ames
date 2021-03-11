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
        self._train = pd.DataFrame()
        self._train_filenames = []
        self._numeric_stats = pd.DataFrame()
        self._categorical_stats = pd.DataFrame()

    def build_data(self):
        
        for filename in os.listdir(self._inpath):
            if fnmatch.fnmatch(filename, "*train.csv"):                
                df = pd.read_csv(os.path.join(self._inpath, filename))
                self._train = pd.concat((self._train, df), axis=0)
        print(f"Training data shape is {self._train.shape}")
        X_train = self._train.loc[:,self._train.columns != "Sale_Price"]      
        y_train = self._train.loc[:,self._train.columns == "Sale_Price"]      
        X_train.to_csv(os.path.join(self._outpath, "X_train.csv"), index=False)          
        y_train.to_csv(os.path.join(self._outpath, "y_train.csv"), index=False)          
    
    def summary(self):
        self._numeric_stats = self._train.describe(include=[np.number]).T
        self._categorical_stats = self._train.describe(include=[object]).T
        percent_missing = self._train.isnull().sum() * 100 / len(self._train)
        missing = pd.DataFrame({"Column": self._train.columns,
                                "Percent Missing": percent_missing})
        missing.sort_values("Percent Missing", inplace=True, ascending=False)
        print("Numeric Variable Descriptive Statistics")
        print(tabulate(self._numeric_stats, headers="keys"))
        print("\n\nCategorical Variable Descriptive Statistics")
        print(tabulate(self._categorical_stats, headers="keys"))
        print("\n\nMissing Values")
        print(tabulate(missing[missing["Percent Missing"]>0], headers="keys", showindex=False))

def main():
    builder = DataBuilder()
    builder.build_data()   
    builder.summary()

if __name__ == "__main__":
    main()
#%%
        




