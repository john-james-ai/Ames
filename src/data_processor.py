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
#                          1. DATA CLEANER                                    #
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
    def __init__(self):
        pass

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
#                         3. FEATURE ENGINEERING                              #
# =========================================================================== #    
class FeatureEngineer:
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
#                      4. CONTINUOUS  PREPROCESSING                           #
# =========================================================================== #
class ContinuousPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, continuous=continuous):
        self._continuous = continuous

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X,  **transform_params):       
        notify.entering(__class__.__name__, "transform")
        # Impute missing values as linear function of other features
        imputer = IterativeImputer()
        X[self._continuous] = imputer.fit_transform(X[self._continuous])

        # Power transformation to make feature distributions closer to Guassian
        power = PowerTransformer(method="yeo-johnson", standardize=False)
        X[self._continuous] = power.fit_transform(X[self._continuous])

        notify.leaving(__class__.__name__, "transform")
        
        return X
# =========================================================================== #
#                        5. DISCRETE PREPROCESSING                            #
# =========================================================================== #
class DiscretePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="mean", discrete=discrete):
        self._strategy = strategy
        self._discrete = discrete

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X,  **transform_params):       
        notify.entering(__class__.__name__, "transform")
        # Missing discrete variables will be imputed according to the strategy provided
        # Default strategy is the mean.
        imputer = SimpleImputer(strategy=self._strategy)
        X[self._discrete] = imputer.fit_transform(X[self._discrete])

        # Standardize discrete variables to zero mean and unit variance
        scaler = StandardScaler()        
        X[self._discrete] = scaler.fit_transform(X[self._discrete])
        
        notify.leaving(__class__.__name__, "transform")

        return X        


# =========================================================================== #
#                        6. ORDINAL PREPROCESSING                             #
# =========================================================================== #
class OrdinalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="most_frequent", ordinal=ordinal, 
        ordinal_map=ordinal_map, encoder=OrdinalMapEncoder()):
        self._strategy = strategy
        self._ordinal = ordinal
        self._ordinal_map = ordinal_map
        self._encoder = encoder

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
# =========================================================================== #
#                          7. TARGET TRANSFORMER                              #
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
#                         8. ORDINAL ENCODER                                  #
# =========================================================================== #
class OrdinalMapEncoder(BaseEstimator, TransformerMixin):
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
# =========================================================================== #
#                         9. ONEHOT ENCODER                                   #
# =========================================================================== #
class HotOneEncoder(BaseEstimator, TransformerMixin):
    """Accepts nominal features (only) and converts to One-Hot representation."""
    
    def __init__(self):        
        pass

    def fit(self, X, y=None, **fit_params):
        notify.entering(__class__.__name__, "fit")
        self.original_features_ = X.columns.tolist()
        self.ohe_ = OneHotEncoder(handle_unknown="ignore")
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
        
        for i in self.transformed_features_:
            for j in self.original_features_:
                if j in i:
                    self.to_original_[i] = j
                    break        

        notify.leaving(__class__.__name__, "transform")
        return self.X_ 
        
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

# =========================================================================== #
#                       10. ORDINAL ENCODER (SKL)                             #
# =========================================================================== #
class OrdinalEncoderSKL(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        from sklearn.preprocessing import OrdinalEncoder
        self._enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                                    unknown_value="unknown")
        self._enc.fit(X)
        return self
    
    def transform(self, X,  **transform_params):       
        notify.entering(__class__.__name__, "transform")
        notify.leaving(__class__.__name__, "transform")                  

        return self._enc.transform(X)

# =========================================================================== #
#                         11. MEAN ENCODER                                    #
# =========================================================================== #
class MeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, l1o=True, discrete=discrete):
        self._l1o = l1o
        self._discrete = discrete

    def _fit_column(self, X, y=None, **fit_params):
        fq = df.groupby('columnName').size()/len(df)    
        df.loc[:, "{}_freq_encode".format('columnName')] = df['columnName'].map(fq)   
        df = df.drop(['columnName'], axis = 1)          
        
        return self
    
    def transform(self, X,  **transform_params):       
        notify.entering(__class__.__name__, "transform")
        notify.leaving(__class__.__name__, "transform")                  

        return self._enc.transform(X)

# =========================================================================== #
#                          9. PREPROCESSOR                                    #
# =========================================================================== #
class PreProcessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):   
        # Clean data 
        cleaner = DataCleaner()
        X, y = cleaner.run(X,y)
        
        # Screen data of outliers and non-informative features
        screener = DataScreener()
        X, y = screener.run(X, y)        

        # Perform data augmentation
        augmentor = DataAugmentor()
        X, y = augmentor.run(X, y)      

        # Transform Target
        x4mr = TargetTransformer()
        x4mr.fit(y)                    
        self.y_ = x4mr.transform(y)          

        # Execute feature preprocessors
        preprocessors = [ContinuousPreprocessor(), 
                         CategoricalPreprocessor(), DiscretePreprocessor(),
                         OrdinalEncoder()]        
        for preprocessor in preprocessors:
            x4mr = preprocessor
            x4mr.fit(X, y)
            self.X_ = x4mr.transform(X)

        return self
    
    def transform(self, X):
        return self.X_, self.y_        


