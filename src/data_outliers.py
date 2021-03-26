# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \data_outliers.py                                                 #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Wednesday, March 24th 2021, 10:19:45 am                     #
# Last Modified : Wednesday, March 24th 2021, 10:46:39 am                     #
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
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from utils import get_formulas
from pipeline_v2 import continuous, discrete
from sklearn.neighbors import LocalOutlierFactor

from data import AmesData
# --------------------------------------------------------------------------- #
class Outliers:
    def __init__(self, features):
        self._features = features        

    def set_features(features):
        self._features = features

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        clf = LocalOutlierFactor(n_neighbors=2)
        self.lof_ = clf.fit_predict(X)
        self.n_outliers_ = len(self.lof_[self.lof_ == -1])
        mask = np.argwhere(self.lof_==-1)
        self.outliers_ = X[mask,]
        print(outliers.shape)

    def save(self, filename):
        directory = "../reports/"
        filepath = directory + filename
        self.outliers_.to_csv(filepath, index=False)

        
def main():
    # Get Data
    filename = "../data/external/Ames_data.csv"
    ames = pd.read_csv(filename)
    X = ames.drop(columns=["PID"])
    y = ames["Sale_Price"].to_numpy()
    outliers = Outliers(continuous)
    outliers.fit(X).predict(X)    
    print(outliers.outliers_)
    outliers.save(filename="outliers_continuous.csv")
    
    outliers.set_features(discrete)
    outliers.fit(X).predict(X)
    print(outliers.outliers_)
    outliers.save(filename="outliers_discrete.csv")    


if __name__ == "__main__":
    main()        

#%%