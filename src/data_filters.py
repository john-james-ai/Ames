# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \process_data.py                                                  #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Wednesday, March 10th 2021, 12:03:50 am                     #
# Last Modified : Wednesday, March 10th 2021, 12:03:51 am                     #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
# =========================================================================== #
#                               1. LIBRARIES                                  #
# =========================================================================== #
#%%
# System and python libraries
from abc import ABC, abstractmethod
import datetime
import glob
from joblib import dump, load
import os
import pickle
# Manipulating, analyzing and processing data
from collections import OrderedDict
import itertools
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

# Feature and model selection and evaluation
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_regression
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

# Regression based estimators
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

# Tree-based estimators
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# Visualizing data
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

# Utilities
from utils import notify, Persist, validate, convert, comment

# Global Variables
from globals import discrete, continuous, numeric, n_nominal_levels
from globals import nominal, ordinal, ordinal_map, all_features

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# =========================================================================== #
#                            1. BASE FILTER                                   #
# =========================================================================== #  
class BaseFilter(BaseEstimator, TransformerMixin, ABC):
    def __init__(self, all_features=all_features):
        self._all_features = all_features        
    
    def report(self, X):
        message = f"The following {len(self.features_removed_)} were removed from the data.\n{self.features_removed_}."

# =========================================================================== #
#                         2. COLLINEARITY FILTER                              #
# =========================================================================== #  
class CollinearityFilter(BaseFilter):
    def __init__(self, threshold=0.80, alpha=0.05, all_features=all_features):
        self._threshold = threshold
        self._alpha = alpha
        self._all_features = all_features

    def _report(self, X):
        classname = self.__class__.__name__
        message = f"The following {len(self.features_removed_)} were removed from the data.\n{self.features_removed_}."
        comment.regarding(classname, message)
        
    def _fit(self, X, y=None):
        correlations = pd.DataFrame()
        columns = X.columns.tolist()
        
        # Perform pairwise correlation coefficient calculations
        for col_a, col_b in itertools.combinations(columns,2):
            r, p = pearsonr(X.loc[:,col_a], X.loc[:,col_b])
            d = {"Columns": col_a + "__" + col_b, "A": col_a, "B": col_b,
                 "Correlation": r, "p-value": p}
            df = pd.DataFrame(data=d, index=[0])
            correlations = pd.concat((correlations, df), axis=0)

        # Now compute correlation between features and target.
        relevance = pd.DataFrame()
        for column in columns:
            r, p = pearsonr(X.loc[:,column], y)
            d = {"Feature": column, "Correlation": r, "p-value": p}
            df = pd.DataFrame(data=d, index=[0])            

        # Obtain observations above correlation threshold and below alpha
        self.suspects_ = correlations[(correlations["Correlation"] >= self._threshold) & (correlations["p-value"] <= self._alpha)]
        if self.suspects_.shape[0] == 0:
            self.X_ = X
            return 

        # Iterate over suspects and determine column to remove based upon
        # correlation with target
        to_remove = []
        for index, row in suspects.iterrows():
            if np.abs(relevance[relevance["Feature"] == row["A"]]) > \
                np.abs(relevance[relevance["Feature"] == row["B"]]):
                to_remove.append(row["B"])
            else:
                to_remove.append(row["A"])
        
        self.X_ = X.drop(columns=to_remove)
        self.features_removed_ += to_remove
        self._fit(self.X_)

    def fit(self, X, y=None):
        self.features_removed_ = []
        self._fit(x, y)        
        return self

    def transform(self, X):
        self._report(X)
        return self.X_

# =========================================================================== #
#                            2. BORUTA FILTER                                 #
# =========================================================================== #  

                