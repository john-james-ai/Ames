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
from abc import ABC, abstractmethod
import datetime
import glob
import itertools
from joblib import dump, load
import os
import pickle
import uuid
# Manipulating, analyzing and processing data
from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats.stats import pearsonr, f_oneway
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from category_encoders import TargetEncoder

# Feature and model selection and evaluation
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
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
from utils import notify, Persist, comment

# Data Source
from data import AmesData


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# =========================================================================== #
#                                 COLUMNS                                     #
# =========================================================================== #
discrete =  ["PID", "Year_Built","Year_Remod_Add","Bsmt_Full_Bath","Bsmt_Half_Bath",
    "Full_Bath","Half_Bath","Bedroom_AbvGr","Kitchen_AbvGr","TotRms_AbvGrd",
    "Fireplaces","Garage_Cars","Mo_Sold","Year_Sold","Age", "Garage_Age", "Garage_Yr_Blt"]
continuous = ["Lot_Frontage","Lot_Area","Mas_Vnr_Area","BsmtFin_SF_1","BsmtFin_SF_2",
    "Bsmt_Unf_SF","Total_Bsmt_SF","First_Flr_SF","Second_Flr_SF","Low_Qual_Fin_SF",
    "Gr_Liv_Area","Garage_Area","Wood_Deck_SF","Open_Porch_SF","Enclosed_Porch",
    "Three_season_porch","Screen_Porch","Pool_Area","Misc_Val"]
numeric = discrete + continuous

n_nominal_levels = 191
nominal = ['MS_SubClass', 'MS_Zoning', 'Street', 'Alley', 'Land_Contour', 'Lot_Config', 'Neighborhood',
 'Condition_1', 'Condition_2', 'Bldg_Type', 'House_Style', 'Roof_Style', 'Roof_Matl',
 'Exterior_1st', 'Exterior_2nd', 'Mas_Vnr_Type', 'Foundation', 'Heating', 'Central_Air',
 'Garage_Type', 'Misc_Feature', 'Sale_Type', 'Sale_Condition']

ordinal = ['BsmtFin_Type_1', 'BsmtFin_Type_2', 'Bsmt_Cond', 'Bsmt_Exposure', 
'Bsmt_Qual', 'Electrical', 'Exter_Cond', 'Exter_Qual', 'Fence', 'Fireplace_Qu', 
'Functional', 'Garage_Cond', 'Garage_Finish', 'Garage_Qual', 'Heating_QC', 'Kitchen_Qual', 
'Land_Slope', 'Lot_Shape', 'Overall_Cond', 'Overall_Qual', 'Paved_Drive', 'Pool_QC', 'Utilities']

pre_features = ['PID', 'MS_SubClass', 'MS_Zoning', 'Lot_Frontage', 'Lot_Area', 'Street',
       'Alley', 'Lot_Shape', 'Land_Contour', 'Utilities', 'Lot_Config',
       'Land_Slope', 'Neighborhood', 'Condition_1', 'Condition_2', 'Bldg_Type',
       'House_Style', 'Overall_Qual', 'Overall_Cond', 'Year_Built',
       'Year_Remod_Add', 'Roof_Style', 'Roof_Matl', 'Exterior_1st',
       'Exterior_2nd', 'Mas_Vnr_Type', 'Mas_Vnr_Area', 'Exter_Qual',
       'Exter_Cond', 'Foundation', 'Bsmt_Qual', 'Bsmt_Cond', 'Bsmt_Exposure',
       'BsmtFin_Type_1', 'BsmtFin_SF_1', 'BsmtFin_Type_2', 'BsmtFin_SF_2',
       'Bsmt_Unf_SF', 'Total_Bsmt_SF', 'Heating', 'Heating_QC', 'Central_Air',
       'Electrical', 'First_Flr_SF', 'Second_Flr_SF', 'Low_Qual_Fin_SF',
       'Gr_Liv_Area', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath', 'Full_Bath',
       'Half_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr', 'Kitchen_Qual',
       'TotRms_AbvGrd', 'Functional', 'Fireplaces', 'Fireplace_Qu',
       'Garage_Type', 'Garage_Yr_Blt', 'Garage_Finish', 'Garage_Cars',
       'Garage_Area', 'Garage_Qual', 'Garage_Cond', 'Paved_Drive',
       'Wood_Deck_SF', 'Open_Porch_SF', 'Enclosed_Porch', 'Three_season_porch',
       'Screen_Porch', 'Pool_Area', 'Pool_QC', 'Fence', 'Misc_Feature',
       'Misc_Val', 'Mo_Sold', 'Year_Sold', 'Sale_Type', 'Sale_Condition',
       'Longitude', 'Latitude']
post_features = ['PID', 'MS_SubClass', 'MS_Zoning', 'Lot_Frontage', 'Lot_Area', 'Street',
       'Alley', 'Lot_Shape', 'Land_Contour', 'Utilities', 'Lot_Config',
       'Land_Slope', 'Neighborhood', 'Condition_1', 'Condition_2', 'Bldg_Type',
       'House_Style', 'Overall_Qual', 'Overall_Cond', 'Year_Built',
       'Year_Remod_Add', 'Roof_Style', 'Roof_Matl', 'Exterior_1st',
       'Exterior_2nd', 'Mas_Vnr_Type', 'Mas_Vnr_Area', 'Exter_Qual',
       'Exter_Cond', 'Foundation', 'Bsmt_Qual', 'Bsmt_Cond', 'Bsmt_Exposure',
       'BsmtFin_Type_1', 'BsmtFin_SF_1', 'BsmtFin_Type_2', 'BsmtFin_SF_2',
       'Bsmt_Unf_SF', 'Total_Bsmt_SF', 'Heating', 'Heating_QC', 'Central_Air',
       'Electrical', 'First_Flr_SF', 'Second_Flr_SF', 'Low_Qual_Fin_SF',
       'Gr_Liv_Area', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath', 'Full_Bath',
       'Half_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr', 'Kitchen_Qual',
       'TotRms_AbvGrd', 'Functional', 'Fireplaces', 'Fireplace_Qu',
       'Garage_Type', 'Garage_Yr_Blt', 'Garage_Finish', 'Garage_Cars',
       'Garage_Area', 'Garage_Qual', 'Garage_Cond', 'Paved_Drive',
       'Wood_Deck_SF', 'Open_Porch_SF', 'Enclosed_Porch', 'Three_season_porch',
       'Screen_Porch', 'Pool_Area', 'Pool_QC', 'Fence', 'Misc_Feature',
       'Misc_Val', 'Mo_Sold', 'Year_Sold', 'Sale_Type', 'Sale_Condition',
       'Age', 'Garage_Age']
# =========================================================================== #
#                              ORDINAL MAP                                    #
# =========================================================================== #
ordinal_map = {'BsmtFin_Type_1': {'ALQ': 5, 'BLQ': 4, 'GLQ': 6, 'LwQ': 2, 'No_Basement': 0, 'Rec': 3, 'Unf': 1},
 'BsmtFin_Type_2': {'ALQ': 5, 'BLQ': 4, 'GLQ': 6, 'LwQ': 2, 'No_Basement': 0, 'Rec': 3, 'Unf': 1},
 'Bsmt_Cond': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Basement': 0, 'Poor': 1, 'Typical': 3},
 'Bsmt_Exposure': {'Av': 3, 'Gd': 4, 'Mn': 2, 'No': 1, 'No_Basement': 0},
 'Bsmt_Qual': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Basement': 0, 'Poor': 1, 'Typical': 3},
 'Electrical': {'FuseA': 4, 'FuseF': 2, 'FuseP': 1, 'Mix': 0, 'SBrkr': 5, 'Unknown': 3},
 'Exter_Cond': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'Poor': 0, 'Typical': 2},
 'Exter_Qual': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'Poor': 0, 'Typical': 2},
 'Fence': {'Good_Privacy': 4, 'Good_Wood': 2, 'Minimum_Privacy': 3, 'Minimum_Wood_Wire': 1,'No_Fence': 0},
 'Fireplace_Qu': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Fireplace': 0, 'Poor': 1, 'Typical': 3},
 'Functional': {'Maj1': 3, 'Maj2': 2, 'Min1': 5, 'Min2': 6, 'Mod': 4, 'Sal': 0, 'Sev': 1, 'Typ': 7},
 'Garage_Cond': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Garage': 0, 'Poor': 1, 'Typical': 3},
 'Garage_Finish': {'Fin': 3, 'No_Garage': 0, 'RFn': 2, 'Unf': 1},
 'Garage_Qual': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Garage': 0, 'Poor': 1, 'Typical': 3},
 'Heating_QC': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'Poor': 0, 'Typical': 2},
 'Kitchen_Qual': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'Poor': 0, 'Typical': 2},
 'Land_Slope': {'Gtl': 0, 'Mod': 1, 'Sev': 2},
 'Lot_Shape': {'Irregular': 0, 'Moderately_Irregular': 1, 'Regular': 3, 'Slightly_Irregular': 2},
 'Overall_Cond': {'Above_Average': 5, 'Average': 4,'Below_Average': 3,'Excellent': 8,'Fair': 2,
                  'Good': 6,'Poor': 1,'Very_Excellent': 9,'Very_Good': 7,'Very_Poor': 0},
 'Overall_Qual': {'Above_Average': 5,'Average': 4,'Below_Average': 3,'Excellent': 8,'Fair': 2,
                  'Good': 6,'Poor': 1,'Very_Excellent': 9,'Very_Good': 7,'Very_Poor': 0},
 'Paved_Drive': {'Dirt_Gravel': 0, 'Partial_Pavement': 1, 'Paved': 2},
 'Pool_QC': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'No_Pool': 0, 'Typical': 2},
 'Utilities': {'AllPub': 2, 'NoSeWa': 0, 'NoSewr': 1}}

# =========================================================================== #
#                                ESTIMATORS                                   #
# =========================================================================== #
regressors = {}
regressors.update({"Linear Regression": LinearRegression()})
regressors.update({"Lasso": Lasso()})
regressors.update({"Ridge": Ridge()})
regressors.update({"ElasticNet": ElasticNet()})

ensembles = {}
ensembles.update({"AdaBoost": AdaBoostRegressor()})
ensembles.update({"Bagging": BaggingRegressor()})
ensembles.update({"Extra Trees": ExtraTreesRegressor()})
ensembles.update({"Gradient Boosting": GradientBoostingRegressor()})
ensembles.update({"Random Forest": RandomForestRegressor()})
ensembles.update({"Histogram Gradient Boosting": HistGradientBoostingRegressor()})


# =========================================================================== #
#                             HYPERPARAMETERS                                 #
# =========================================================================== #
# Parameter Grid
regressor_parameters = {}
regressor_parameters.update({"Linear Regression":{"estimator__normalize": [False]}})
regressor_parameters.update({"Lasso": {
    "estimator__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
    "estimator__n_jobs": [-1]}})
regressor_parameters.update({"Ridge":{
        "estimator__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
        "estimator__n_jobs": [-1]}})        
regressor_parameters.update({"ElasticNet":{
        "estimator__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
        "estimator__l1_ratio": np.arange(0.0,1.0,0.1),
        "estimator__n_jobs": [-1]}})        

ensemble_parameters = {}
ensemble_parameters.update({"AdaBoost": {
        "estimator__base_estimator": None,
        "estimator__n_estimators": [50,100],
        "estimator__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0]}})
ensemble_parameters.update({"Bagging": {
        "estimator__base_estimator": None,
        "estimator__n_estimators": [50,100],
        "estimator__max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "estimator__n_jobs": [-1]}}) 
ensemble_parameters.update({"Extra Trees": {        
        "estimator__n_estimators": [50,100],
        "estimator__max_depth": [2,3,4,5,6],
        "estimator__min_samples_split": [0.005, 0.01, 0.05, 0.10],
        "estimator__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
        "estimator__max_features": ["auto", "sqrt", "log2"],
        "estimator__n_jobs": [-1]}})         
ensemble_parameters.update({"Gradient Boosting": {        
        "estimator__learning_rate": [0.15,0.1,0.05,0.01,0.005,0.001],
        "estimator__n_estimators": [50,100],
        "estimator__max_depth": [2,3,4,5,6],
        "estimator__criterion": ["mse"],
        "estimator__min_samples_split": [0.005, 0.01, 0.05, 0.10],
        "estimator__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
        "estimator__max_features": ["auto", "sqrt", "log2"]}})                        
ensemble_parameters.update({"Random Forest": {        
        "estimator__n_estimators": [50,100],
        "estimator__max_depth": [2,3,4,5,6],
        "estimator__criterion": ["mse"],
        "estimator__min_samples_split": [0.005, 0.01, 0.05, 0.10],
        "estimator__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
        "estimator__max_features": ["auto", "sqrt", "log2"],
        "estimator__n_jobs": [-1]}})      
ensemble_parameters.update({"Histogram Gradient Boosting": {  
        "estimator__learning_rate": [0.15,0.1,0.05,0.01,0.005,0.001],              
        "estimator__max_depth": [2,3,4,5,6],        
        "estimator__min_samples_leaf": [0.005, 0.01, 0.05, 0.10]}})       



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

        # Remove longitude and latitude
        X = X.drop(columns=["Latitude", "Longitude", "PID"])

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
#                          4. TARGET TRANSFORMER                              #
# =========================================================================== #
class TargetTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self,y, **fit_params):
        return self
    
    def transform(self, y, **transform_params):
        return np.log(y)

    def inverse_transform(self, y, **transform_params):
        return np.exp(y)

# =========================================================================== #
#                      5. CONTINUOUS  PREPROCESSING                           #
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

        # Scale the features and standardize to zero mean unit variance
        scaler = StandardScaler()
        X[self._continuous] = scaler.fit_transform(X[self._continuous])        

        notify.leaving(__class__.__name__, "transform")
        
        return X

# =========================================================================== #
#                        6. DISCRETE PREPROCESSING                            #
# =========================================================================== #
class DiscretePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="most_frequent", discrete=discrete):
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

        # Scale the features and standardize to zero mean unit variance
        scaler = StandardScaler()
        X[self._discrete] = scaler.fit_transform(X[self._discrete])
        
        
        notify.leaving(__class__.__name__, "transform")

        return X        

# =========================================================================== #
#                        7. ORDINAL PREPROCESSING                             #
# =========================================================================== #
class OrdinalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="most_frequent", ordinal=ordinal, 
        ordinal_map=ordinal_map):
        self._strategy = strategy
        self._ordinal = ordinal
        self._ordinal_map = ordinal_map

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, y=None,  **transform_params):       
        notify.entering(__class__.__name__, "transform")
        categorical = list(X.select_dtypes(include=["object"]).columns)
        # Create imputer object
        imputer = SimpleImputer(strategy=self._strategy)
        
        # Perform imputation of categorical variables to most frequent
        X[self._ordinal] = imputer.fit_transform(X[self._ordinal])        

        # Map levels to ordinal mappings
        for variable, mappings in self._ordinal_map.items():
            for k,v in mappings.items():
                X[variable].replace({k:v}, inplace=True)        

        # Scale the features and standardize to zero mean unit variance
        scaler = StandardScaler()
        X[self._ordinal] = scaler.fit_transform(X[self._ordinal])

        notify.leaving(__class__.__name__, "transform")
        
        return X       
# =========================================================================== #
#                      8. TARGET LEAVE-ONE-OUT ENCODER                        #
# =========================================================================== #        
class TargetEncoderLOO(TargetEncoder):
    """Leave-one-out target encoder.
    Source: https://brendanhasz.github.io/2019/03/04/target-encoding
    """
    
    def __init__(self, cols=nominal):
        """Leave-one-out target encoding for categorical features.
        
        Parameters
        ----------
        cols : list of str
            Columns to target encode.
        """
        self.cols = cols
        

    def fit(self, X, y):
        """Fit leave-one-out target encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to target encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X
                         if str(X[col].dtype)=='object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Encode each element of each column
        self.sum_count = dict()
        for col in self.cols:
            self.sum_count[col] = dict()
            uniques = X[col].unique()
            for unique in uniques:
                ix = X[col]==unique
                count = X[X[col] == unique].shape[0]
                singleton = "N" if (count > 1) else "Y" 
                self.sum_count[col][unique] = \
                    (y[ix].sum(),ix.sum(), singleton)
            
        # Return the fit object
        return self

    
    def transform(self, X, y=None):
        """Perform the target encoding transformation.

        Uses leave-one-out target encoding for the training fold,
        and uses normal target encoding for the test fold.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        
        # Create output dataframe
        Xo = X.copy()

        # Use normal target encoding if this is test data
        if y is None:
            for col in self.sum_count.keys():
                vals = np.full(X.shape[0], np.nan)
                for cat, sum_count in self.sum_count[col].items():
                    vals[X[col]==cat] = sum_count[0]/sum_count[1]
                Xo[col] = vals

        # LOO target encode each column
        else:
            for col in self.sum_count.keys():
                vals = np.full(X.shape[0], np.nan)
                for cat, sum_count in self.sum_count[col].items():
                    ix = X[col]==cat     
                    if sum_count[2] == "Y":
                        vals[ix] = sum_count[0]/sum_count[1]
                    else:               
                        vals[ix] = (sum_count[0]-y[ix])/(sum_count[1]-1)
                Xo[col] = vals
            
        # Return encoded DataFrame
        return Xo
      
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)
# =========================================================================== #
#                        8. NOMINAL PREPROCESSING                             #
# =========================================================================== #
class NominalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, encoder=TargetEncoderLOO(nominal), nominal=nominal):                
        self._encoder = encoder
        self._nominal = nominal

    def fit(self, X, y=None, **fit_params):
        notify.entering(__class__.__name__, "fit")
        self._encoder.fit(X, y)
        notify.leaving(__class__.__name__, "fit")
        return self
    
    def transform(self, X, y=None,  **transform_params):            
        notify.entering(__class__.__name__, "transform")        
        notify.leaving(__class__.__name__, "transform")
        X = self._encoder.transform(X, y)

        # Scale the features and standardize to zero mean unit variance
        scaler = StandardScaler()
        X[self._nominal] = scaler.fit_transform(X[self._nominal])
        return X        
    
    def fit_transform(self, X,y=None):        
        self.fit(X,y)
        return self.transform(X,y)

# =========================================================================== #
#                            9. BASE FILTER                                   #
# =========================================================================== #  
class BaseFilter(BaseEstimator, TransformerMixin, ABC):
    def __init__(self):
        pass    
    
    def report(self, X, y=None):
        classname = self.__class__.__name__
        message = f"The following {len(self.features_removed_)} features were removed from the data.\n{self.features_removed_}."
        comment.regarding(classname, message)


# =========================================================================== #
#                         10. COLLINEARITY FILTER                             #
# =========================================================================== #  
class CollinearityFilter(BaseFilter):
    def __init__(self, features, threshold=0.7, alpha=0.05, numeric=numeric):
        self._threshold = threshold
        self._alpha = alpha
        self._features = features        
        self._numeric = numeric
        
    def _fit(self, X, y=None):
        notify.entering(__class__.__name__, "_fit") 
        correlations = pd.DataFrame()
        all_columns = X.columns.tolist()
        columns = list(set(X.columns.tolist()).intersection(self._numeric))
        
        # Perform pairwise correlation coefficient calculations
        for col_a, col_b in itertools.combinations(columns,2):            
            r, p = pearsonr(X[col_a], X[col_b])
            cols =  col_a + "__" + col_b
            d = {"Columns": cols, "A": col_a, "B": col_b,
                 "Correlation": r, "p-value": p}
            df = pd.DataFrame(data=d, index=[0])
            correlations = pd.concat((correlations, df), axis=0)

        # Now compute correlation between features and target.
        relevance = pd.DataFrame()
        for column in columns:
            r, p = pearsonr(X.loc[:,column], y)
            d = {"Feature": column, "Correlation": r, "p-value": p}
            df = pd.DataFrame(data=d, index=[0])            
            relevance = pd.concat((relevance,df), axis=0)

        # Obtain observations above correlation threshold and below alpha
        self.suspects_ = correlations[(correlations["Correlation"] >= self._threshold) & (correlations["p-value"] <= self._alpha)]
        if self.suspects_.shape[0] == 0:
            self.X_ = X
            return 
        
        # Iterate over suspects and determine column to remove based upon
        # correlation with target
        to_remove = []
        for index, row in self.suspects_.iterrows():
            a = np.abs(relevance[relevance["Feature"] == row["A"]]["Correlation"].values)
            b = np.abs(relevance[relevance["Feature"] == row["B"]]["Correlation"].values)
            if a > b:
                to_remove.append(row["B"])
            else:
                to_remove.append(row["A"])
        
        self.X_ = X.drop(columns=to_remove)
        self.features_removed_ += to_remove
        self._fit(self.X_,y)
        notify.leaving(__class__.__name__, "fit_") 

    def fit(self, X, y=None):
        notify.entering(__class__.__name__, "fit") 
        self.features_removed_ = []
        self._fit(X, y)        
        notify.leaving(__class__.__name__, "fit") 
        return self

    def transform(self, X, y=None):
        return self.X_

    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X,y)
# =========================================================================== #
#                            11. ANOVA FILTER                                 #
# =========================================================================== # 
class AnovaFilter(BaseFilter):
    """Eliminates predictors with equal between means w.r.t response."""
    def __init__(self, alpha=0.05, ordinal=ordinal, nominal=nominal):
        self._alpha = alpha
        self._ordinal = ordinal
        self._nominal = nominal

    def fit(self, X, y=None):
        notify.entering(__class__.__name__, "_fit") 
        results = pd.DataFrame()
        all_columns = X.columns.tolist()
        categorical = self._ordinal + self._nominal
        columns = list(set(X.columns.tolist()).intersection(categorical))  

        # Measure variance between predictor levels w.r.t. the response
        self.remaining_ = pd.DataFrame()
        self.features_removed_ = []
        for column in columns:
            f, p = f_oneway(X[column], y)
            if p > self._alpha:
                self.features_removed_.append(column)
            else:
                d = {"Feature": column, "F-statistic": f, "p-value": p}
                df = pd.DataFrame(data=d, index=[0])
                self.remaining_ = pd.concat((self.remaining_, df), axis=0)
                


        # Drop features
        self.X_ = X.drop(columns=self.features_removed_)
        notify.leaving(__class__.__name__, "_fit") 
        return self
    
    def transform(self, X, y=None):
        return self.X_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


# =========================================================================== #
#                         12. CORRELATION FILTER                              #
# =========================================================================== # 
class CorrelationFilter(BaseFilter):
    """Eliminates predictors that do not have a strong correlation with the response."""
    def __init__(self, threshold=0.7, alpha=0.05, continuous=continuous, discrete=discrete):
        self._threshold = threshold
        self._alpha = alpha
        self._continuous = continuous
        self._discrete = discrete

    def fit(self, X, y=None):
        notify.entering(__class__.__name__, "_fit") 
        results = pd.DataFrame()
        all_columns = X.columns.tolist()
        categorical = self._continuous + self._discrete
        columns = list(set(X.columns.tolist()).intersection(categorical))  

        # Measure variance between predictor levels w.r.t. the response
        self.remaining_ = pd.DataFrame()
        self.features_removed_ = []
        for column in columns:
            r, p = pearsonr(X[column], y)
            if (np.abs(r) <= self._threshold) & (p <= self._alpha):
                self.features_removed_.append(column)
            else:
                d = {"Feature": column, "Correlation": r, "p-value": p}
                df = pd.DataFrame(data=d, index=[0])
                self.remaining_ = pd.concat((self.remaining_, df), axis=0)                

        # Drop features
        self.X_ = X.drop(columns=self.features_removed_)
        notify.leaving(__class__.__name__, "_fit") 
        return self
    
    def transform(self, X, y=None):
        return self.X_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

# =========================================================================== #
#                              13. SCORING                                    #
# =========================================================================== #   
def RMSE(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)    
    return np.sqrt(mse)
rmse = make_scorer(RMSE, greater_is_better=False)
# =========================================================================== #
#                         14. MODEL EVALUATOR                                 #
# =========================================================================== #
class Evaluator:
    """Evaluates models, stores scores and returns the best model."""

    def __init__(self, estimators, param_grids, step=1, min_features_to_select=4,cv=5, 
                 scoring=rmse, n_jobs=2, refit=True, verbose=5, submission=1):
        self._estimators = estimators
        self._param_grids = param_grids
        self._step = step
        self._min_features_to_select = min_features_to_select
        self._cv = cv
        self._scoring = scoring
        self._n_jobs = n_jobs
        self._refit = refit
        self._verbose = verbose
        self._submission = submission    

    def select_features(self, X, y):
        self.feature_selection_scores_ = pd.DataFrame()
        self.selections_ = {}

        # Perform recursive feature elimination
        selector = RFECV(self.estimator_, self._step, self._cv)
        selector = selector.fit(X, y)
        
        # Extract selected features
        columns = X.columns.tolist()
        selected_features = columns[selector.support_]
        X_selected = X[selected_features]

        # Score the new dataset 
        score = self.estimator_.score(X_selected, y)        

        # Save the data        

        d = {"Estimator": self.estimator_.__class__.__name__,
             "Score": score, 
             "Num Features": selector.n_features_}        
        df = pd.DataFrame(data=d, index=[0])
        self.feature_selection_scores_ = pd.concat((self.scores_,df),axis=0)
        d = {}
        d["Selected Features"] =  selected_features
        d["Feature Ranks"] = selector.ranking_
        d["Score"] = score
        d["Num Features"] = selector.n_features_
        self.selections[self.estimator_.__class__.__name__] = d
        return X_selected, y

    def grid_search(self, X, y):
        """Performs a gridsearch for the optimal parameters"""
        self.model_id_ = uuid.uuid4()
        self.models_ = {}
        self.scores_ = pd.DataFrame()

        # Run Gridsearch
        self.model_ = GridSearchCV(estimator=self.estimator_, 
                            param_grid=self._param_grid,
                            scoring=self._scoring,
                            n_jobs=self._n_jobs,
                            cv=self._cv,
                            refit=self._refit,
                            verbose=self._verbose) 
        # Store model details in dectionary indexed by model id
        d = {"best_estimator_name": self.model_.best_estimator_.__class__.__name__,
             "best_estimator": self.model_.best_estimator_,
             "best_score": self.model_.best_score_,
             "best_params": self.model_.best_params_,
             "refit_time": self.model_.refit_time_}
        self.models_[self.model_id_] = d

        # Track scores in a dataframe for easy sorting later
        d = {"Model Id": self.model_id_,
             "Estimator": self.estimator_.__class__.__name__,
             "Num Features": X.shape[1],
             "Test Score": score}
        df = pd.DataFrame(data=d, index=[0])
        self.scores_ = pd.concat((self.scores_,df),axis=0)             
    
    def fit(self, X, y):
        """Selects the features and fits the model."""
        X, y = self.select_features(X,y)
        self.grid_search(X,y)
        self.model_.fit(X,y)

    def predict(self, X):
        """Predicts using the best model on new X."""       

        # Extract and drop PID from the test set.
        PID = X["PID"].values
        X.drop(columns="PID")
        
        # Obtain best scoring model
        self.scores_.sort_values(by="Test Score", ascending=False, inplace=True)
        model_id = self.scores_["Model Id"].iloc[0]
        self.best_model_ = self.models_[model_id]["best_estimator"]       

        # Make Predictions
        y_pred self.best_model_.predict(X)

        # Concatenate PID and y_pred
        d = {"PID", PID, "Sale_Price": np.exp(y_pred)}
        self.submission_ = pd.DataFrame(data=d)

    def submit(self):
        """Saves submission"""
        filename = "mysubmission" + self._submission + ".txt"
        self.submission_.to_csv(filename)

    def summary(self):
        print(tabulate.tabulate(self.scores_, header="keys"))

    def evaluate(self, X_train, y_train, X_test):
        for name, self.estimator_ in self._estimators.items():
            self.fit(X_train, y_train)
        self.predict(X_test)
        self.submit()

        self.fit(X_train, y_train)



# =========================================================================== #
#                         9. DATA PROCESSING PIPELINE                         #
# =========================================================================== #
class DataProcessor:
    """Performs processing of training and test data."""
    def __init__(self):
        pass

    def _fit_test(X):

        # Create new features
        X, y = FeatureEngineer().run(X, y)
        assert(X.isnull().sum().sum() == 0), f"{X.isnull().sum().sum()} null values after New Features"

        # Extract features from training set
        X = X[self.X_train_.columns]

        # Continuous     
        X = ContinuousPreprocessor(continuous=continuous).fit(X).transform(X)    
        assert(X.isnull().sum().sum() == 0), f"{X.isnull().sum().sum()} null values after Continuous"

        # Discrete     
        X = DiscretePreprocessor().fit(X).transform(X)
        assert(X.isnull().sum().sum() == 0), f"{X.isnull().sum().sum()} null values after Discrete"

        # Ordinal    
        X = OrdinalPreprocessor().fit(X).transform(X)
        assert(X.isnull().sum().sum() == 0), f"{X.isnull().sum().sum()} null values after Ordinal"

        # Nominal    
        X = NominalPreprocessor().fit(X, y).transform(X, y)        
        assert(X.isnull().sum().sum() == 0), f"{X.isnull().sum().sum()} null values after Nominal"
        
        return X

    def _fit_train(X,y=None):

        # Clean data
        X, y = DataCleaner().run(X, y)

        # Screen Data
        X, y = DataScreener().run(X, y)    
        
        # Create new features
        X, y = FeatureEngineer().run(X, y)
        assert(X.isnull().sum().sum() == 0), f"{X.isnull().sum().sum()} null values after New Features"

        # Transform Target        
        y = TargetTransformer().fit_transform(y)

        # Continuous     
        X = ContinuousPreprocessor(continuous=continuous).fit(X).transform(X)    
        assert(X.isnull().sum().sum() == 0), f"{X.isnull().sum().sum()} null values after Continuous"

        # Discrete     
        X = DiscretePreprocessor().fit(X).transform(X)
        assert(X.isnull().sum().sum() == 0), f"{X.isnull().sum().sum()} null values after Discrete"

        # Ordinal    
        X = OrdinalPreprocessor().fit(X).transform(X)
        assert(X.isnull().sum().sum() == 0), f"{X.isnull().sum().sum()} null values after Ordinal"

        # Nominal    
        X = NominalPreprocessor().fit(X, y).transform(X, y)        
        assert(X.isnull().sum().sum() == 0), f"{X.isnull().sum().sum()} null values after Nominal"
        
        # Collinearity Filter
        filta = CollinearityFilter(X.columns)
        X = filta.fit(X, y).transform(X, y)
        filta.report(X, y)

        # Correlation Filter
        filta = CorrelationFilter(threshold=0.7,  alpha=0.05, continuous=continuous, discrete=discrete)
        X = filta.fit(X, y).transform(X, y)
        filta.report(X, y)    

        # Anova Filter
        filta = AnovaFilter(alpha=0.05, ordinal=ordinal, nominal=nominal)
        X = filta.fit(X, y).transform(X, y)
        filta.report(X, y)    

        self.X_train_ = X
        self.y_train_ = y
        
        return X, y

    def transform(self, X, y=None):
        if y:
            return self.X_train_, self.y_train_
        else:
            self._fit_test(X)

    def fit(X,y=None):
        if y:
            self._fit_train(X,y)
        else:
            self._fit_test(X)        


def main():

    # Retain objects
    data = AmesData()
    evaluator = Evaluator 
    X_train, y_train, X_test = data.get(1)    

    # Preprocess Data
    processor = DataProcessor()
    processor.fit(X_train, y_train)
    X_train, y_train = processor.transform(X,y)
    processor.fit(X_test)
    X_test = preprocess_train(X_test)

    # Perform feature selection


    




if __name__ == "__main__":
    main()
#%%
