# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Data Mining                                                       #
# File    : \mymain.py                                                        #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Data-Mining/                     #
# --------------------------------------------------------------------------- #
# Created       : Tuesday, March 9th 2021, 12:24:24 am                        #
# Last Modified : Tuesday, March 9th 2021, 12:24:24 am                        #
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
import datetime
import glob
from joblib import dump, load
import os
import pickle
# Manipulating, analyzing and processing data
from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy as sp
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
from utils import Notify, Persist


if __name__ == "__main__":
    notify = Notify(verbose=True)    
    main()
#%%
