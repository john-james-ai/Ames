# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \lab.py                                                           #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Friday, March 12th 2021, 1:28:12 pm                         #
# Last Modified : Friday, March 12th 2021, 1:28:12 pm                         #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from data import AmesData
from globals import nominal
data = AmesData()
X, y = data.get()

X = X[nominal]
fn = X.columns
print(X.shape)
ohe = OneHotEncoder()
X = ohe.fit_transform(X)
print(X.shape)
print(ohe.categories_)
print(ohe.get_feature_names(fn))
X = ohe.inverse_transform(X)
print(X.shape)
#%%