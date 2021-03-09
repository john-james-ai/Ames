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
# Load libraries
import pandas as pd
import scipy as sp
import numpy as np
import xgboost
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, ColumnTransformer
from sklearn.pipeline import make_pipeline
import seaborn as sns
import lightGBM
# --------------------------------------------------------------------------- #
#                               PIPELINE BUILDER                              #
# --------------------------------------------------------------------------- #
class PipelineBuilder:
    """Builds the end-to-end machine learning pipeline."""
    def __init__(self, X, y=None):
        self._X = X
        self._y = y

    def build_preprocessor(self):
        continuous = list(self._X.select_dtypes(include=["float64"]).columns)
        categorical = list(self._X.select_dtypes(include=["object"]).columns)
        discrete = list(self._X.select_dtypes(include=["int"]).columns)

        continuous_pipeline = make_pipeline(
            IterativeImputer(),
            PowerTransformer(method="yeo-johnson", standardize=False),
            StandardScaler()
        )

        categorical_pipeline = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="unknown"),
            OneHotEncoder()
        )

        discrete_pipeline = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=-1)
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('continuous', continuous_pipeline, continuous),
                ('categorical', categorical_pipeline, categorical),
                ('discrete', discrete_pipeline, discrete)
            ]
        )
        return preprocessor

# --------------------------------------------------------------------------- #
#                                  EVALUATE                                   #
# --------------------------------------------------------------------------- #
def evaluate(pipeline, X_train, y_train, X_test, y_test, verbose = True):
    """ 
    Trains model pipeline and evaluates model on test data. Returns the original
    model, RMSE on log Sales Price, and testing RMSE.
    """

    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_score = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_score = np.sqrt(mean_squared_error(y_test, y_test_pred))

    if verbose:
        print(f"Algorithm: {pipeline.named_steps['algorithm'].__class__.__name__}")
        print(f"Train RMSE: {train_score}")
        print(f"Test RMSE: {test_score}")

    return pipeline.named_steps['algorithm'], train_score, test_score
# --------------------------------------------------------------------------- #
def evaluate_regressors(X_train, y_train, X_test, y_test):
    regressors = [
        LinearRegression(),
        LassoCV(),
        RidgeCV()
    ]

    preprocessor = build_preprocessor(X_train)    

    for r in regressors:
        pipe = Pipeline(steps = [
            ('preprocessor', preprocessor),
            ('algorithm',r)
        ])

        evaluate(pipe, X_train, y_train, X_test, y_test)    
# --------------------------------------------------------------------------- #
#                                  GET DATA                                   #
# --------------------------------------------------------------------------- #
def get_data(set_id):
    directory = "../../data/raw/"
    train_file = os.path.join(directory, str(set_id) + "_train.csv")
    test_file =  os.path.join(directory, str(set_id) + "_test.csv")
    test_file_y = os.path.join(directory, str(set_id) + "_test_y.csv")

    train = pd.read_csv(train_file)
    X_train = train.drop("SalePrice", axis=1)
    y_train = np.log(train.loc[:,df.columns == "SalePrice"])
    X_test = pd.read_csv(test_file)
    y_test = pd.read_csv(test_file_y)
    return X_train, y_train, X_test, y_test

# --------------------------------------------------------------------------- #
#                                    MAIN                                     #
# --------------------------------------------------------------------------- #    
def main():
