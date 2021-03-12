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
from collections import OrderedDict
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.externals import joblib
import seaborn as sns
import lightGBM
# --------------------------------------------------------------------------- #
#                               DATA PREPROCESSING                            #
# --------------------------------------------------------------------------- #
def preprocess(X,y):
    """Preprocesses the data before feature engineering."""
    # Drop unnecessary columns
    X = X.drop(columns=["Latitude", "Longitude"])
    # Drop known outliers based upon DeCock's recommendation
    X = X[X["Gr_Live_Area"] <= 4000] 

    return X
# --------------------------------------------------------------------------- #
def add_features(X):
    X["Age"] = X["Year_Sold"] - X["Year_Built"]
    return X
# --------------------------------------------------------------------------- #
def transform_target(y):
    y["Sale_Price_Log"] = np.log(y["Sale_Price"])    
    y = y.drop(columns=["Sale_Price"])
    return y
# --------------------------------------------------------------------------- #    
def encode_ordinal(X):
    """Encodes ordinal variables as integers in ascending order of positive price affect."""
    filename = "ordered.csv"
    codes = pd.read_csv(os.path.join(data_paths["metadata"], filename))

    #TODO: Fix this.
    variables = codes.groupby("Variable")
    for variable, group in variables:
        for seq, level in zip(group["Order"], group["Levels"]):
            X.replace({level:seq}, inplace=True)

# --------------------------------------------------------------------------- #
#                               PIPELINE BUILDER                              #
# --------------------------------------------------------------------------- #
class PipelineBuilder:
    """Builds the end-to-end machine learning pipeline."""
    def __init__(self, X, y=None):
        self._X = X
        self._y = y
        self._pipelines = OrderedDict()
        self._preprocessor = None
        self._selector = None


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
            IterativeImputer(),
            OneHotEncoder()
        )

        discrete_pipeline = make_pipeline(
            IterativeImputer(),
            StandardScaler()
        )

        self._preprocessor = ColumnTransformer(
            transformers=[
                ('continuous', continuous_pipeline, continuous),
                ('categorical', categorical_pipeline, categorical),
                ('discrete', discrete_pipeline, discrete)
            ]
        )

    def build_feature_selector(self, estimator, min_features=3):
        self._selector = RFECV(estimator, min_features_to_select=min_features)

    def build_pipelines(self, estimators=[]):

        self._pipelines = OrderedDict()
        for e in estimators:
            self._pipelines[e.__class__.__name__] = Pipeline(steps=[
                ('preprocessor', self._preprocessor),
                ('selector', self._selector),
                ('algorithm': e)
            ])
    
    def get_pipelines(self):
        return self._pipelines


        

# --------------------------------------------------------------------------- #
#                                  EVALUATOR                                  #
# --------------------------------------------------------------------------- #
class Evaluator:
    """Trains and evaluates a pipeline"""

    def __init__(self, pipeline, X_train, y_train, X_test, y_test, verbose = True):
        self._pipeline = pipeline
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._verbose = verbose

    def evaluate(self):

        self._pipeline.fit(self._X_train, self._y_train)
        y_train_pred = pipeline.predict(self._X_train)
        y_test_pred = pipeline.predict(self._X_test)

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
