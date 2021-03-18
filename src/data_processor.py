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
from utils import notify, Persistence

# Global Variables
from globals import random_state, discrete, continuous, numeric, n_nominal_levels
from globals import nominal, ordinal, ordinal_map

# =========================================================================== #
#                          2. DATA SCREENER                                   #
# =========================================================================== #    
class DataScreener(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_map=ordinal_map):
        self._ordinal_map = ordinal_map

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        """Creation, removal, and encoding of features."""
        notify.entering(__class__.__name__, "transform")
        # Add an age feature and remove year built
        X["Age"] = X["Year_Sold"] - X["Year_Built"]
        X["Age"].fillna(X["Age"].median())

        # Remove longitude and latitude
        X = X.drop(columns=["Latitude", "Longitude"])

        # Remove outliers 
        idx = X[X["Gr_Liv_Area"] < 4000].index.tolist()
        X = X.iloc[idx]
        y = y.iloc[idx]

        notify.leaving(__class__.__name__, "transform")

        return X, y        
# =========================================================================== #
#                        3. DATA PREPROCESSING                                #
# =========================================================================== #
class ContinuousPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X,  **transform_params):       
        notify.entering(__class__.__name__, "transform")
        # Create imputer and power transformer objects
        imputer = IterativeImputer()
        power = PowerTransformer(method="yeo-johnson", standardize=True)
        
        # Perform imputation of continuous variables
        X[continuous] = imputer.fit_transform(X[continuous])

        # Perform power transformations to make data closer to Guassian distribution
        # Data is standardized as well
        X[continuous] = power.fit_transform(X[continuous])

        notify.leaving(__class__.__name__, "transform")
        
        return X
# --------------------------------------------------------------------------- #
class CategoricalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X,  **transform_params):       
        notify.entering(__class__.__name__, "transform")
        categorical = list(X.select_dtypes(include=["object"]).columns)
        # Create imputer object
        imputer = SimpleImputer(strategy="most_frequent")
        
        # Perform imputation of categorical variables to most frequent
        X[categorical] = imputer.fit_transform(X[categorical])

        notify.leaving(__class__.__name__, "transform")
        
        return X        
# --------------------------------------------------------------------------- #
class DiscretePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X,  **transform_params):       
        notify.entering(__class__.__name__, "transform")
        # Create imputer and scaler objects
        imputer = SimpleImputer(strategy="most_frequent")
        scaler = StandardScaler()        
        
        # Perform imputation of discrete variables to most frequent
        X[discrete] = imputer.fit_transform(X[discrete])
        X[discrete] = scaler.fit_transform(X[discrete])
        
        notify.leaving(__class__.__name__, "transform")

        return X        


# =========================================================================== #
#                            4. ENCODERS                                      #
# =========================================================================== #
class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_map=ordinal_map):
        self._ordinal_map = ordinal_map

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X,  **transform_params):       
        notify.entering(__class__.__name__, "transform")
        for variable, mappings in self._ordinal_map.items():
            for k,v in mappings.items():
                X[variable].replace({k:v}, inplace=True)       

        # Scale data as continuous 
        scaler = StandardScaler()        
        X[ordinal] = scaler.fit_transform(X[ordinal])   

        notify.leaving(__class__.__name__, "transform")                  

        return X
# --------------------------------------------------------------------------- #
class NominalEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, nominal=nominal):
        self._nominal = nominal

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X,  **transform_params):       
        """Converting nominal variables to one-hot representation."""
        notify.entering(__class__.__name__, "transform")
        n = X.shape[0]
        # Extract nominal from X
        nominal = pd.Series(self._nominal)
        features = X.columns
        nominal_features = features[features.isin(nominal)]
        X_nominal = X[nominal_features]   
        X.drop(nominal_features, axis=1, inplace=True)    
        n_other_features = X.shape[1]

        # Encode nominal and store in dataframe with feature names
        enc = OneHotEncoder()
        X_nominal = enc.fit_transform(X_nominal).toarray()
        X_nominal = pd.DataFrame(data=X_nominal)
        X_nominal.columns = enc.get_feature_names()

        # Concatenate X with X_nominal and validate    
        X = pd.concat([X, X_nominal], axis=1)
        expected_shape = (n,n_other_features+n_total_feature_levels)
        assert(X.shape == expected_shape), "Error in Encode Nominal. X shape doesn't match expected."

        notify.leaving(__class__.__name__, "transform")

        return X
# =========================================================================== #
#                          5. TARGET TRANSFORMER                              #
# =========================================================================== #
class TargetTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self,y, **fit_params):
        return self
    
    def transform(self, y, **transform_params):
        notify.entering(__class__.__name__, "transform")        
        d = {"Sale_Price": np.log(y["Sale_Price"].values)}
        df = pd.DataFrame(data=d)
        notify.leaving(__class__.__name__, "transform")
        return df

    def inverse_transform(self, y, **transform_params):
        notify.entering(__class__.__name__, "inverse_transform")
        
        d = {"Sale_Price": np.exp(y["Sale_Price"].values)}
        df = pd.DataFrame(data=d)
        notify.leaving(__class__.__name__, "inverse_transform")        
        return df
# =========================================================================== #
#                             6. AMES DATA                                    #
# =========================================================================== #   
class AmesData:
    """ Obtains processed data if exists, processes raw data otherwise."""
    def __init__(self):
        self._train_directory = "../data/train/"
        self._processed_directory = "../data/processed/"
        self._X_filename = "X_train.csv"
        self._y_filename = "y_train.csv"
        
    
    def process(self, X, y, **transform_params):
        """Screens, preprocesses and transforms the data."""
        notify.entering(__class__.__name__, "transform")
        
        # Screen data of outliers and non-informative features
        screener = DataScreener()
        screener.fit(X, y)
        X, y = screener.transform(X, y)

        # Execute feature preprocessors
        preprocessors = [ContinuousPreprocessor(), 
                         CategoricalPreprocessor(), DiscretePreprocessor(),
                         OrdinalEncoder()]        
        for preprocessor in preprocessors:
            x4mr = preprocessor
            x4mr.fit(X, y)
            X = x4mr.transform(X)

        # Transform Target
        x4mr = TargetTransformer()
        x4mr.fit(y)                    
        y = x4mr.transform(y)

        # Save data
        X_filepath = self._processed_directory + self._X_filename
        y_filepath = self._processed_directory + self._y_filename
        X.to_csv(X_filepath)
        y.to_csv(y_filepath)

        notify.leaving(__class__.__name__, "transform")        
        return X, y

    def get(self):
        """Obtains processed data if extant, otherwise, processes raw data"""
        X_filepath = self._processed_directory + self._X_filename
        if os.path.exists(X_filepath):
            y_filepath = self._processed_directory + self._y_filename
            X = pd.read_csv(X_filepath)
            y = pd.read_csv(y_filepath)
        else:
            X_filepath = self._train_directory + self._X_filename
            y_filepath = self._train_directory + self._y_filename
            X = pd.read_csv(X_filepath)
            y = pd.read_csv(y_filepath)
            X, y = self.process(X,y)
        
        return X, y
