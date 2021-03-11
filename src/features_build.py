# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \build_features.py                                                #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Monday, March 8th 2021, 11:46:37 pm                         #
# Last Modified : Tuesday, March 9th 2021, 10:50:21 pm                        #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
# Load libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV
from tabulate import tabulate
# --------------------------------------------------------------------------- #
class FeatureSelector:
    def __init__(self, inpath="../../data/interim/"):
        self._inpath = inpath
        self._features = {}
        self._X = None
        self._y = None

    def select(self, estimator):
        if not self._X:
            self._X = pd.read_csv(os.path.join(self._inpath,"X_train.csv"))
            self._y = pd.read_csv(os.path.join(self._inpath,"y_train.csv"))

        features = list(self._X.columns)

        selector = RFECV(estimator=estimator, scoring="mean_squared_error", n_jobs=4)
        selector.fit(self._X, self._y)
        self._features[estimator.__class__.__name__] = features[selector.support_]

    def summary(self):
        df = pd.DataFrame(data=self._features)
        print(tabulate(df, headers="keys"))

def main():
    estimators = [
        LinearRegression(),
        LassoCV(),
        RidgeCV(),
        ElasticNetCV(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        AdaBoostRegressor(),
        GradientBoostingRegressor()
    ]

    selector = FeatureSelector()
    for estimator in estimators:
        selector.select(estimator)
    selector.summary()

if __name__ == "__main__":
    main()
#%%