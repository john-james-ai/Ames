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
import os

from globals import data_paths

enc = OneHotEncoder()

summary = "categorical_summary.csv"
X_filename = "X_train.csv"
codes = pd.read_csv(os.path.join(data_paths["metadata"], summary))
X = pd.read_csv(os.path.join(data_paths["interim"], X_filename))

nominals = codes[codes["Type"] == "nominal"]["Variable"]

X_nom = X[nominals]   
print(f"X_nom.shape is {X_nom.shape}") 
print(f"X.shape before is {X.shape}")
X.drop(nominals, axis=1, inplace=True)
print(f"X.shape after is {X.shape}")
X_enc = enc.fit_transform(X_nom).toarray()
X_enc_df = pd.DataFrame(data=X_enc)
X_enc_df.columns = enc.get_feature_names()

print(f"X_nom.shape is {X_enc_df.shape}") 
print(X_enc_df.head())
X = pd.concat([X, X_enc_df], axis=1)
print(f"Final X.shape is {X.shape}")
