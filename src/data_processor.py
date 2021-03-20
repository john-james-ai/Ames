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
from utils import notify, Persist, validate, convert

# Global Variables
from globals import discrete, continuous, numeric, n_nominal_levels
from globals import nominal, ordinal, ordinal_map

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# =========================================================================== #
#                          1. DATA CLEARNER                                   #
# =========================================================================== #  
class DataCleaner:
    def __init__(self):
        pass
    def run(self, X, y=None):
        X.replace(to_replace="NA", value=np.NaN, inplace=True)
        X["Garage_Yr_Blt"].replace(to_replace=2207, value=2007, inplace=True)
        return X, y


# =========================================================================== #
#                          2. DATA SCREENER                                   #
# =========================================================================== #    
class DataScreener:
    def __init__(self, ordinal_map=ordinal_map):
        self._ordinal_map = ordinal_map

    def run(self, X, y=None, **fit_params):
        notify.entering(__class__.__name__, "run")
        # Add an age feature and remove year built
        X["Age"] = X["Year_Sold"] - X["Year_Built"]
        X["Age"].fillna(X["Age"].median())

        # Remove longitude and latitude
        X = X.drop(columns=["Latitude", "Longitude"])

        # Remove outliers 
        idx = X[(X["Gr_Liv_Area"] <= 4000) & (X["Garage_Yr_Blt"]<=2010)].index.tolist()
        X = X.iloc[idx]
        y = y.iloc[idx]

        notify.leaving(__class__.__name__, "run")
        return X, y

# =========================================================================== #
#                         3. DATA AUGMENTATION                                #
# =========================================================================== #    
class DataAugmentor:
    def __init__(self):
        pass
    def run(self, X, y=None):
        notify.entering(__class__.__name__, "run")
        # Add an age feature and remove year built
        X["Age"] = X["Year_Sold"] - X["Year_Built"]
        X["Age"].fillna(X["Age"].median())        

        # Add age feature for garage.
        X["Garage_Age"] = X["Year_Sold"] - X["Garage_Yr_Blt"]
        X["Garage_Age"].fillna(value=0,inplace=True)        




        notify.leaving(__class__.__name__, "run")
        return X, y



# =========================================================================== #
#                        4. DATA PREPROCESSING                                #
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
        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()        
        
        # Perform imputation of discrete variables to most frequent
        X[discrete] = imputer.fit_transform(X[discrete])
        X[discrete] = scaler.fit_transform(X[discrete])
        
        notify.leaving(__class__.__name__, "transform")

        return X        


# =========================================================================== #
#                            5. ENCODERS                                      #
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
    """Accepts nominal features (only) and converts to One-Hot representation."""
    
    def __init__(self):        
        pass

    def fit(self, X, y=None, **fit_params):
        notify.entering(__class__.__name__, "fit")
        self.original_features_ = X.columns.tolist()
        self.ohe_ = OneHotEncoder()
        self.ohe_.fit(X)
        notify.leaving(__class__.__name__, "fit")
        return self
    
    def transform(self, X,  **transform_params):       
        """Converting nominal variables to one-hot representation."""        
        notify.entering(__class__.__name__, "transform")

        self.X_ = self.ohe_.transform(X).toarray()
        self.transformed_features_ = self.ohe_.get_feature_names(self.original_features_).tolist()        
        self.X_df_ = pd.DataFrame(self.X_, columns=self.transformed_features_)
        self.to_original_ = {}
        self.to_transformed_ = {}
        
        for i in self.transformed_features_:
            for j in self.original_features_:
                self.to_transformed_[j] = self.to_transformed_[j] or []  
                if j in i:
                    self.to_original_[i] = j
                    self.to_transformed_[j].append(i)
                    break        

        notify.leaving(__class__.__name__, "transform")
        return self.X_df_ 
        
    def inverse_transform(self, X,  **transform_params):               
        return self.ohe_.inverse_transform(X)

    def get_original(self, transformed):
        original = []
        for i in transformed:            
            for k,v in self.to_original_.items():
                if i == k:
                    original.append(v)
                    break
        return original

    def get_transformed(self, original):
        transformed = []
        for i in original:
            transformed += self.to_transformed_[i]            
        return transformed

# =========================================================================== #
#                          6. TARGET TRANSFORMER                              #
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

