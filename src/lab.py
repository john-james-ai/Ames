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

filepath = "../data/external/Ames_data.csv"
df = pd.read_csv(filepath)
df1 = df.select_dtypes(include=[object])
df2 = df.select_dtypes(exclude=[object])
print(df1.info())
print(df2.info())
print(df.shape[1])
total = df1.shape[1] + df2.shape[1]
print(total)

#%%