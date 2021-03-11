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
#%%
import numpy as np
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tabulate import tabulate
# --------------------------------------------------------------------------- #
class DataframeFunctionTransformer():
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self

def standard_onehot(df):
    # Extract numeric and categorical data
    numeric_features = df.select_dtypes(exclude=["object"])
    numeric_feature_names = list(numeric_features.columns)
    categorical_features = df.select_dtypes(exclude=[np.number])
    categorical_feature_names = list(categorical_features.columns)

    # Standardize
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(numeric_features)
    df_numeric = pd.DataFrame(data=numeric_features, columns=numeric_feature_names)

    # One-Hot
    encoder = OneHotEncoder()
    categorical_features = encoder.fit_transform(categorical_features)
    df_categorical = pd.DataFrame(data=categorical_features, columns=categorical_feature_names)

    df_transformed = pd.concat([df_numeric, df_categorical], axis=1)
    return df_transformed

class DataProcessor:
    def __init__(self, inpath="../../data/interim/", outpath="../../data/processed/"):
        self._inpath = inpath
        self._outpath = outpath

    def process(self):
        df = pd.read_csv(os.path.join(self._inpath, "train.csv"))
        print(df.info())
        # Remove Garage_Yr_Blt, 5% missing values
        df.drop(columns=["Garage_Yr_Blt"], inplace=True)
        
        # Transform target variable
        df["Sale_Price_Log"] = np.log(df["Sale_Price"])
        
        # Remove known outliers 
        df = df.loc[df["Gr_Liv_Area"]<=4000]

        # Separate features from target
        X_train = df.drop(["Sale_Price", "Sale_Price_Log"], axis=1)
        y_train = df["Sale_Price_Log"]
        
        #Standardize Numeric Data
        numeric_features = list(X_train.select_dtypes(exclude=['object']).columns)
        scaler = StandardScaler()
        X_train[numeric_features] =  scaler.fit_transform(X_train[numeric_features])        

        # Save as processed data
        X_train.to_csv(os.path.join(self._outpath,"X_train.csv"), index=False)
        y_train.to_csv(os.path.join(self._outpath,"y_train.csv"), index=False)
    
    def summary(self):
        X_train = pd.read_csv(os.path.join(self._outpath,"X_train.csv"))
        y_train = pd.read_csv(os.path.join(self._outpath,"y_train.csv"))
        X_quant_stats = X_train.describe(include=[np.number]).T
        X_qual_stats =  X_train.describe(include=[object]).T
        y_quant_stats = y_train.describe()
        print("Training set summary")
        print(f"Observations: {X_train.shape[0]} Features: {X_train.shape[1]}")
        print("\n\nQuantitative Statistics for Numeric Variables")
        print(tabulate(X_quant_stats, headers="keys"))
        print("\n\nQualitative Statistics for Categorical Variables")
        print(tabulate(X_qual_stats, headers="keys"))
        print("\n\nQuantitative Statistics for Target Variable")
        print(tabulate(y_quant_stats, headers="keys"))


def main():
    dp = DataProcessor()
    dp.process()
    dp.summary()

if __name__ == "__main__":
    main()
#%%
