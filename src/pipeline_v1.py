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
import time
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
from category_encoders import TargetEncoder, LeaveOneOutEncoder

# Feature and model selection and evaluation
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.feature_selection import VarianceThreshold, f_regression
from sklearn.metrics import make_scorer, mean_squared_error
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
from sklearn.tree import DecisionTreeRegressor

# Visualizing data
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

# Utilities
from utils import notify, PersistEstimator, comment, print_dict, print_dict_keys


# Data Source
from data import AmesData


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('mode.chained_assignment', None)
# =========================================================================== #
#                                 COLUMNS                                     #
# =========================================================================== #
discrete =  ["Year_Built","Year_Remod_Add","Bsmt_Full_Bath","Bsmt_Half_Bath",
    "Full_Bath","Half_Bath","Bedroom_AbvGr","Kitchen_AbvGr","TotRms_AbvGrd",
    "Fireplaces","Garage_Cars","Mo_Sold","Year_Sold", "Garage_Yr_Blt"]
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
model_groups = {
    "Regressors": {
        "Linear Regression": {                
            "Estimator": LinearRegression(),
            "Parameters": {"normalize": [False],"n_jobs": [4],"copy_X": [True]}
            },
        "Lasso": {
            "Estimator": Lasso(),
            "Parameters": {
                "alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]}
        },
        "Ridge": {
            "Estimator": Ridge(),
            "Parameters": {
                "alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]}
        },
        "ElasticNet": {
            "Estimator": ElasticNet(),
            "Parameters": {
                "alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
                "l1_ratio": np.arange(0.0,1.0,0.1)}
        }
        
    },
    "Ensembles": {
        "Random Forest": {
            "Estimator": RandomForestRegressor(),
            "Parameters": {
                "n_estimators": [50,100],
                "max_depth": [2,3,4,5,6],
                "criterion": ["mse"],
                "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                "max_features": ["auto"],
                "n_jobs": [4]}
        },
        "AdaBoost": {
            "Estimator": AdaBoostRegressor(),
            "Parameters": {
                "base_estimator": [DecisionTreeRegressor()],
                "n_estimators": [50,100],
                "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0]}
        },
        "Bagging": {
            "Estimator": BaggingRegressor(),
            "Parameters": {
                "base_estimator": [DecisionTreeRegressor()],
                "n_estimators": [50,100],
                "max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "n_jobs": [4]}
        },
        "Extra Trees": {
            "Estimator": ExtraTreesRegressor(),
            "Parameters": {
                "n_estimators": [50,100],
                "max_depth": [2,3,4,5,6],
                "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                "max_features": ["auto"],
                "n_jobs": [4]}
        },
        "Gradient Boosting": {
            "Estimator": GradientBoostingRegressor(),
            "Parameters": {
                "learning_rate": [0.15,0.1,0.05,0.01,0.005,0.001],
                "n_estimators": [50,100],
                "max_depth": [2,3,4,5,6],
                "criterion": ["mse"],
                "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                "max_features": ["auto"]}
        },
        "Histogram Gradient Boosting": {
            "Estimator": HistGradientBoostingRegressor(),
            "Parameters": {
                "learning_rate": [0.15,0.1,0.05,0.01,0.005,0.001],
                "max_depth": [2,3,4,5,6],
                "min_samples_leaf": [0.005, 0.01, 0.05, 0.10]}
        }
    }
}
regressors = model_groups["Regressors"]
ensembles = model_groups["Ensembles"]
# =========================================================================== #
#                          0. BASE TRANSFORMER                                #
# =========================================================================== #

class BaseTransformer(ABC):
    def __init__(self):
        pass

    def fit(self, X, y=None):                

        X_old = pd.DataFrame(data=X["X"], columns=X["Features"])
        y_old = X["y"]
        self._fit(X_old, y_old)
        return self

    def transform(self, X, y=None):
        X_old = pd.DataFrame(data=X["X"], columns=X["Features"])
        y_old = X["y"]
        X_new = self._transform(X_old, y_old)
        assert(len(X_new.columns) == len(X_old.columns)), f"Old new columns mismatch"
        assert(X_new.isnull().sum().sum() == 0), f"Warning nulls in test after clean"
        X["X"] = X_new

        return X
# =========================================================================== #
#                          1. BASE SELECTOR                                   #
# =========================================================================== #
class BaseSelector(ABC):
    def __init__(self):
        pass

    def fit(self, X, y=None):                

        X_old = pd.DataFrame(data=X["X"], columns=X["Features"])
        y_old = X["y"]
        self._fit(X_old, y_old)
        return self

    def transform(self, X, y=None):
        X_old = pd.DataFrame(data=X["X"], columns=X["Features"])
        y_old = X["y"]
        X_new = self._transform(X_old, y_old)        
        assert(X_new.isnull().sum().sum() == 0), f"Warning nulls"
        X["X"] = X_new
        X["Features"] = X_new.columns

        return X

# =========================================================================== #
#                          2. DATA CLEANER                                    #
# =========================================================================== #
class DataCleaner(BaseSelector):
    def __init__(self):
        pass
    def _fit(self, X, y=None):
        return X, y

    def _transform(self, X, y=None):
        notify.entering(__class__.__name__, "transform")
        
        X["Garage_Yr_Blt"].replace(to_replace=2207, value=2007, inplace=True)


        X = X.drop(columns=["Latitude", "Longitude"])
        X = X.fillna(X.median())
        notify.leaving(__class__.__name__, "transform")
        return X

# =========================================================================== #
#                         3. FEATURE ENGINEERING                              #
# =========================================================================== #
class FeatureEngineer(BaseSelector):
    def __init__(self):
        pass

    def _fit(self, X, y=None):
        return X, y

    def _transform(self, X, y=None):
        notify.entering(__class__.__name__, "transform")
        # Add an age feature and remove year built
        X["Age"] = X["Year_Sold"] - X["Year_Built"]
        X["Age"].fillna(X["Age"].median())

        # Add age feature for garage.
        X["Garage_Age"] = X["Year_Sold"] - X["Garage_Yr_Blt"]
        X["Garage_Age"].fillna(value=0,inplace=True)       


        notify.leaving(__class__.__name__, "transform")
        return X

# =========================================================================== #
#                      4. CONTINUOUS  PREPROCESSING                           #
# =========================================================================== #
class ContinuousPreprocessor(BaseEstimator, TransformerMixin, BaseTransformer):
    def __init__(self, continuous=continuous):
        self._continuous = continuous

    def _fit(self, X, y=None, **fit_params):
        return self

    def _transform(self, X, y=None, **transform_params):
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
#                        5. DISCRETE PREPROCESSING                            #
# =========================================================================== #
class DiscretePreprocessor(BaseEstimator, TransformerMixin, BaseTransformer):
    def __init__(self, strategy="most_frequent", discrete=discrete):
        self._strategy = strategy
        self._discrete = discrete

    def _fit(self, X, y=None, **fit_params):
        return self

    def _transform(self, X, y=None,  **transform_params):
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
#                        6. ORDINAL PREPROCESSING                             #
# =========================================================================== #
class OrdinalPreprocessor(BaseEstimator, TransformerMixin, BaseTransformer):
    def __init__(self, strategy="most_frequent", ordinal=ordinal,
        ordinal_map=ordinal_map):
        self._strategy = strategy
        self._ordinal = ordinal
        self._ordinal_map = ordinal_map

    def _fit(self, X, y=None, **fit_params):
        return self

    def _transform(self, X, y=None,  **transform_params):
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
#                      7. TARGET LEAVE-ONE-OUT ENCODER                        #
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
class NominalPreprocessor(BaseEstimator, TransformerMixin, BaseTransformer):
    def __init__(self, encoder=TargetEncoderLOO(cols=nominal), nominal=nominal):
        self._encoder = encoder
        self._nominal = nominal

    def _fit(self, X, y=None, **fit_params):        
        notify.entering(__class__.__name__, "fit")
        notify.leaving(__class__.__name__, "fit")
        return self

    def _transform(self, X, y=None,  **transform_params):        
        notify.entering(__class__.__name__, "transform")
        notify.leaving(__class__.__name__, "transform")
        self._encoder.fit(X, y)
        X = self._encoder.transform(X, y)

        # Scale the features and standardize to zero mean unit variance
        scaler = StandardScaler()        
        X[self._nominal] = scaler.fit_transform(X[self._nominal])
        #X = X.fillna(X.mean())
        return X

    def fit_transform(self, X,y=None):
        self.fit(X,y)
        return self.transform(X,y)

# =========================================================================== #
#                            9. BASE FILTER                                   #
# =========================================================================== #
class BaseFilter(BaseSelector):
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
    def __init__(self, features, threshold=0.65, alpha=0.05, numeric=numeric):
        self._threshold = threshold
        self._alpha = alpha
        self._features = features
        self._numeric = numeric

    def _fit(self, X, y=None):
        return X, y

    def _transform(self, X, y=None):
        notify.entering(__class__.__name__, "_fit")
        self.features_removed_ = []
        correlations = pd.DataFrame()
        all_columns = X.columns.tolist()
        columns = list(set(X.columns.tolist()).intersection(self._numeric))

        # Perform pairwise correlation coefficient calculations
        for col_a, col_b in itertools.combinations(columns,2):
            r, p = pearsonr(X[col_a], X[col_b])
            cols =  col_a + "__" + col_b
            d = {"Columns": cols, "A": col_a, "B": col_b,"Correlation": r, "p-value": p}
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
        return self.X_
# =========================================================================== #
#                            11. ANOVA FILTER                                 #
# =========================================================================== #
class AnovaFilter(BaseFilter):
    """Eliminates predictors with equal between means w.r.t response."""
    def __init__(self, alpha=0.05, ordinal=ordinal, nominal=nominal):
        self._alpha = alpha
        self._ordinal = ordinal
        self._nominal = nominal

    def _fit(self, X, y=None):
        return X, y

    def _transform(self, X, y=None):
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
        return self.X_


# =========================================================================== #
#                         12. CORRELATION FILTER                              #
# =========================================================================== #
class CorrelationFilter(BaseFilter):
    """Eliminates predictors that do not have a strong correlation with the response."""
    def __init__(self, threshold=0.65, alpha=0.05, continuous=continuous, discrete=discrete):
        self._threshold = threshold
        self._alpha = alpha
        self._continuous = continuous
        self._discrete = discrete

    def _fit(self, X, y=None):
        return X, y

    def _transform(self, X, y=None):
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
        return self.X_



    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


# =========================================================================== #
#                             13. DATA LOBBY                                  #
# =========================================================================== #
class DataLobby:
    """Place where data checks out of the pipeline."""
    def __init__(self):
        pass

    def checkout(self, train, test):
        """ Does final checks and ensures train and test have same dimensions."""

        assert(train["X"].isnull().sum().sum() == 0), f"Warning TRAIN has nulls at checkout"
        assert("Features" in train), "Training Features vector is missing."
        assert("y" in train), "y_train vector is missing."
        assert(train["X"].shape[0] == len(train["y"])), f"Train X/y length mismatch. "

        test["X"] = test["X"][train["Features"]]
        test["Features"] = train["Features"]

        assert(test["X"].isnull().sum().sum() == 0), f"Warning TEST has nulls at checkout"
        assert("Features" in test), "Training Features vector is missing."
        assert("y" in test), "y_train vector is missing."
        assert(test["X"].shape[0] == len(test["y"])), f"Test X/y length mismatch. "

# =========================================================================== #
#                         14. MANUAL FEATURE SELECTION                        #
# =========================================================================== #
class FeatureSelector:
    def __init__(self):
        select_features = ["MS_Zoning",	"Lot_Frontage",	"Lot_Area",	"Neighborhood",	
            "Overall_Qual",	"Overall_Cond",	"Gr_Liv_Area",	"Bedroom_AbvGr",	
            "Kitchen_AbvGr",	"Kitchen_Qual",	"Fireplaces",	"Garage_Area",	
            "Age"]
        self._features = {}
        self._features[1] = select_features
    def get_set(self, id):                
        return self._features[id]


###############################################################################
#                  8. Custom pipeline object to use with RFECV                #
###############################################################################
# Select Features using RFECV
class PipelineRFE(Pipeline):
    # Source: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].feature_importances_
        return self
# =========================================================================== #
#                            13. SCORING                                      #
# =========================================================================== #
def RMSE(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)
rmse = make_scorer(RMSE, greater_is_better=False)

# =========================================================================== #
#                          14. EVALUATOR                                      #
# =========================================================================== #
class Evaluator(ABC):
    """Evaluates models, stores scores and returns the best model."""

    def __init__(self, estimator, param_grid, group, split, k, step=1,
                 min_features_to_select=10,cv=5, scoring=rmse, n_jobs=2,
                 refit=True, verbose=0):
        self._estimator = estimator
        self._param_grid = param_grid
        self._group = group
        self._split = split
        self._k = k
        self._step = step
        self._min_features_to_select = min_features_to_select
        self._cv = cv
        self._scoring = scoring
        self._n_jobs = n_jobs
        self._refit = refit
        self._verbose = verbose    

    def _feature_selection(self, X, y):
        """Manual feature selection."""
        selector = FeatureSelector()
        self.selected_features_ = [f for f in selector.get_set(1)]
        X = X[self.selected_features_]
        return X, y

    def _rfecv(self, X, y):
        # Perform recursive feature elimination
        # We use the estimator being evaluated, unless it doesn't
        # have a 'feature_importance_' or 'coef_' attribute. In 
        # those cases, we use the ElasticNet estimator. 
        estimator = self._estimator
        if ("Bagging" in self._estimator.__class__.__name__) or \
            ("Hist" in self._estimator.__class__.__name__):
            estimator = ElasticNet()

        selector = RFECV(estimator=estimator, step=self._step,
                         cv=self._cv, 
                         min_features_to_select=self._min_features_to_select)
        selector = selector.fit(X, y)

        self.selected_features_ = list(itertools.compress(X.columns,selector.support_))
        self.results_.update({"Features": self.selected_features_})
        X = X[self.selected_features_]

        return X, y

    def _gridsearch(self, X, y):
        """Performs the parameter gridsearch and returns best model."""
        self.model_ = GridSearchCV(estimator=self._estimator,
                            param_grid=self._param_grid,
                            scoring=self._scoring,
                            n_jobs=self._n_jobs,
                            cv=self._cv,
                            refit=self._refit,
                            verbose=self._verbose)
        self.model_.fit(X, y)

        # Save key CV Results on attributes
        self.best_score_ = np.abs(self.model_.best_score_)
        self.best_estimator_ = self.model_.best_estimator_
        self.best_params_ = self.model_.best_params_

    def _init_fit(self, X, y):
        """Starts clock, initialize variables and print initial header."""
        now = datetime.datetime.now()
        self.start_fit_ = time.time()

        print("\n")
        print("="*40)
        print(f"      Estimator: {self._estimator.__class__.__name__}")
        print("-"*40)
        print("         ", end="")
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        print("-"*40)

        self.model_id_ = uuid.uuid4().time_low
        self.performance_ = pd.DataFrame()

        self.results_ = {}
        self.results_.update({"Model Id": self.model_id_})

    def _fini(self, X, y):
        """ Wraps up the fit process and housekeeping."""
        self.end_fit_ = time.time()
        self.fit_time_elapsed_ = (self.end_fit_ - self.start_fit_)

        self.results_.update({"Best Score": self.best_score_})
        self.results_.update({"Best Estimator": self.best_estimator_})
        self.results_.update({"Best Parameters": self.best_params_})
        self.results_.update({"Selected Features": self.selected_features_})
        self.results_.update({"Fit Time": self.fit_time_elapsed_})

        # Stash score information for easy retrieval of best-of-best models.
        d = {"FileSplit": self._split,
             "Group": self._group,
             "Model Id": self.model_id_,
             "Estimator": self.best_estimator_.__class__.__name__,
             "Score": self.best_score_,
             "Fit Time": self.fit_time_elapsed_}
        self.performance_ = pd.DataFrame(data=d,index=[0])

        print(f"        Best CV Score: {np.round(self.best_score_,4)}")
        print(f"      Best Test Score: {np.round(self.test_score_,4)}")
        print(f"           # Features: {X.shape[1]}")
        print(f"             Fit Time: {np.round(self.fit_time_elapsed_,4)} seconds")
        print(f"            Test Time: {np.round(self.test_time_elapsed_,4)} seconds")
        print(f"      Best Parameters:")
        print_dict(self.best_params_)

    def _save_model(self):
        persist = PersistEstimator()
        persist.dump(self.best_estimator_, cv=self._split, eid=self.model_id_)

    def evaluate(self,train, test):
        """Performs feature selection and model training."""
        X = train["X"]
        y = train["y"]
        self._init_fit(X, y)

        X, y = self._feature_selection(X, y)

        self._gridsearch(X, y)        

        self.score(test)

        self._fini(X, y)

        self._save_model()

    def score(self, test):
        self.start_test_ = time.time()
        # Extract selected features
        X = test["X"][self.selected_features_]
        #Conduct predition and scoring and save its absolute value
        y_pred = self.best_estimator_.predict(X)
        self.test_score_ = np.abs(RMSE(test["y"], y_pred))

        self.end_test_ = time.time()        
        self.test_time_elapsed_ = (self.end_test_ - self.start_test_)

        # Save score and test time
        self.performance_["Test RMSE"] = self.test_score_
        self.performance_["Test Time"] = self.test_time_elapsed_
        # Format test results 
        d = {"PID": test["PID"], "Sale_Price": np.exp(y_pred)}
        test_results = pd.DataFrame(data=d)
        self.results_.update({"Test Results": test_results})

# =========================================================================== #
#                            15. SUBMISSION                                   #
# =========================================================================== #
class Submission:
    def __init__(self, results, performance):
        self._results = results
        self._performance = performance

    def compute_avg():
        """Predicts using the best model on new X."""

        # Make Predictions
        y_pred = self.model_.predict(test["X"])

        # Concatenate PID and y_pred
        d = {"PID": test["PID"], "Sale_Price": np.exp(y_pred)}

    def submit(self, submission_no):
        """Saves submission"""
        filename = "mysubmission" + submission_no + ".txt"
        self.submission_.to_csv(filename)

# =========================================================================== #
#                        16. DATA PROCESSING PIPELINE                         #
# =========================================================================== #
class DataProcessor:
    """Performs processing of training and test data."""
    def __init__(self):
        pass

    def fit(self, train, test):
        return self

    def transform(self, train, test):

        # Clean data
        cleaner = DataCleaner()
        train = cleaner.fit(train).transform(train)
        test = cleaner.fit(test).transform(test)

        # Create new features
        engineer = FeatureEngineer()
        train = engineer.fit(train).transform(train)
        test = engineer.fit(test).transform(test)

        # Continuous
        continuous_processor = ContinuousPreprocessor(continuous=continuous)
        train = continuous_processor.fit(train).transform(train)
        test = continuous_processor.fit(test).transform(test)

        # Discrete
        discrete_processor = DiscretePreprocessor()
        train = discrete_processor.fit(train).transform(train)
        test = discrete_processor.fit(test).transform(test)

        # Ordinal
        ordinal_processor = OrdinalPreprocessor()
        train = ordinal_processor.fit(train).transform(train)
        test = ordinal_processor.fit(test).transform(test)

        # Nominal
        nominal_processor = NominalPreprocessor()
        train = nominal_processor.fit(train).transform(train)
        test = nominal_processor.transform(test)

        # # Collinearity Filter
        # filta = CollinearityFilter(train["X"].columns)
        # train = filta.fit(train).transform(train)
        # filta.report(train)

        # # Correlation Filter
        # filta = CorrelationFilter(threshold=0.7,  alpha=0.05, continuous=continuous, discrete=discrete)
        # train = filta.fit(train).transform(train)
        # filta.report(train)

        # # Anova Filter
        # filta = AnovaFilter(alpha=0.05, ordinal=ordinal, nominal=nominal)
        # train = filta.fit(train).transform(train)
        # filta.report(train)

        # Harmonize test with train features
        lobby = DataLobby()
        lobby.checkout(train, test)

        return train, test



def main():

    scores_filename = "../reports/scores.csv"
    performance = pd.DataFrame()
    model_objects = {}
    # Retain objects
    data = AmesData()
    for i in range(1,3):
        print(f"Processing Cycle {i}")
        train, test = data.get(i)

        # Preprocess Data
        processor = DataProcessor()
        train, test = processor.fit(train, test).transform(train,test)

        # Perform Evaluation       
        for groups, models in model_groups.items():
            for model, components in models.items():
                estimator = components["Estimator"]
                params = components["Parameters"]                
                evaluator = Evaluator(estimator=estimator, param_grid=params, k=15, group="Regressors", split=i)
                evaluator.evaluate(train, test)                
                performance = pd.concat((performance, evaluator.performance_),axis=0)            
                performance.to_csv(scores_filename, index=False)


if __name__ == "__main__":
    main()
#%%
 