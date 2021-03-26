# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \analysis.py                                                      #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Tuesday, March 23rd 2021, 10:30:24 pm                       #
# Last Modified : Tuesday, March 23rd 2021, 10:30:24 pm                       #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# --------------------------------------------------------------------------- #
class Analysis:
    def __init__(self, filepath="../reports/scores.csv"):
        self._filepath = filepath        
        self.read()

    def read(self):
        self.scores_ = pd.read_csv(self._filepath)        

    def performance_by_cv(self, return_best=False):
        #If multiple trials, group by group and estimator and take means before 
        df = self.scores_.pivot(index=["Group","Estimator"], columns="FileSplit", values="Test RMSE")        
        df["Average"] = df.mean(axis=1)
        df.sort_values(by="Average", inplace=True)        
        if return_best:
            best = df.groupby("Group").head(1)
            return best
        return df

    def grouped_summary(self):
        df = self.scores_.groupby(by=["Group","Estimator"]).mean()
        return df[["Score","Fit Time", "Test RMSE", "Test Time"]].sort_values(by="Test RMSE")

def main():
    analysis = Analysis()
    print(analysis.performance_by_cv(return_best=True))
    print(analysis.performance_by_cv())
    print(analysis.grouped_summary())

if __name__ == "__main__":
    main()        

#%%