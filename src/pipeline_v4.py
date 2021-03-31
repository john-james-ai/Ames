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
import sys
# Manipulating, analyzing and processing data
from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats.stats import pearsonr, f_oneway, zscore
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from category_encoders import TargetEncoder, LeaveOneOutEncoder
from sklearn.neighbors import LocalOutlierFactor

# Statistical models
import statsmodels.api as sm


# Feature and model selection and evaluation
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.feature_selection import VarianceThreshold, f_regression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

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
cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
palette = sns.color_palette("dark:#124683")
sns.set_theme(palette=palette, style="whitegrid")


# Utilities
from utils import notify, PersistEstimator, comment, print_dict, print_dict_keys
from utils import print_list, diagnose


# Data Source
from data import AmesData


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('mode.chained_assignment', None)
random_state = 6589
# =========================================================================== #
#                               0. SCORING                                    #
# =========================================================================== #
def RMSE(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)
rmse = make_scorer(RMSE, greater_is_better=False)
# =========================================================================== #
#                               1. MODELS                                     #
# =========================================================================== #
baseline_group = {
    "Regressors": {
        "Linear Regression": {                
            "Estimator": LinearRegression(),
            "Parameters": {"normalize": [False],"n_jobs": [4],"copy_X": [True]}
            },
        "Lasso": {
            "Estimator": Lasso(),
            "Parameters": {
                "alpha": [1.0]}
        },
        "Ridge": {
            "Estimator": Ridge(),
            "Parameters": {
                "alpha": [1.0]}
        },
        "ElasticNet": {
            "Estimator": ElasticNet(),
            "Parameters": {
                "alpha": [1.0],
                "l1_ratio": np.arange(0.5)}
        }    
    },
    "Ensembles": {
        "Random Forest": {
            "Estimator": RandomForestRegressor(),
            "Parameters": {
                "n_estimators": [100],
                "criterion": ["mse"],
                "max_features": ["auto"],
                "n_jobs": [4]}
        },
        "AdaBoost": {
            "Estimator": AdaBoostRegressor(),
            "Parameters": {
                "base_estimator": [DecisionTreeRegressor()],
                "n_estimators": [100]}
        },
        "Extra Trees": {
            "Estimator": ExtraTreesRegressor(),
            "Parameters": {
                "n_estimators": [100],
                "max_features": ["auto"],
                "n_jobs": [4]}
        },
        "Gradient Boosting": {
            "Estimator": GradientBoostingRegressor(),
            "Parameters": {
                "n_estimators": [100],
                "criterion": ["friedman_mse"],
                "max_features": ["auto"]}
        }
    }
}
optimized_group = {
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
                "criterion": ["friedman_mse"],
                "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                "max_features": ["auto"]}
        }
    }
}


model_groups = {"Baseline": baseline_group, "Optimized": optimized_group}
# =========================================================================== #
#                         3. FEATURE METADATA                                 #
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
class FeatureMetadata:
    def __init__(self, discrete=discrete, continuous=continuous,
                    nominal=nominal, ordinal=ordinal):
        self._discrete = discrete
        self._continuous = continuous
        self._nominal = nominal
        self._ordinal = ordinal
        self.load_features()

    def load_features(self):
        self.fm_ = pd.DataFrame()
        columns = {"Continuous": self._continuous, "Discrete": self._discrete,
                    "Nominal": self._nominal, "Ordinal": self._ordinal}
        for ftype, features in columns.items():
            d = {"Feature": features, "Type": ftype, "Source": "Original", "Active": True, 
            "Signature": "FeatureMetadata: load_features"}
            df = pd.DataFrame(data=d)
            self.fm_ = pd.concat((self.fm_,df),axis=0)

    def get_feature(self, feature):
        return self.fm_[self.fm_["Type"] == coltype]

    def get_features(self, feature_type=None):
        """Returns all features or all features of the requested feature type."""
        if feature_type:
            return list(self.fm_[(self.fm_["Type"] == feature_type)]["Feature"].values)
        else:
            return list(self.fm_["Feature"].values)

    def get_categorical_features(self):
        """Returns a list of nominal and ordinal features."""
        nominal = list(self.fm_[(self.fm_["Type"] == "Nominal") & (self.fm_["Active"] == True)]["Feature"].values)
        ordinal = list(self.fm_[(self.fm_["Type"] == "Ordinal") & (self.fm_["Active"] == True)]["Feature"].values)
        return nominal + ordinal

    def get_numeric_features(self):
        """Returns a list of continuous and discrete features."""
        discrete = list(self.fm_[(self.fm_["Type"] == "Discrete") & (self.fm_["Active"] == True)]["Feature"].values)
        continuous = list(self.fm_[(self.fm_["Type"] == "Continuous") & (self.fm_["Active"] == True)]["Feature"].values)
        return discrete + continuous        


    def get_original_features(self, feature_type=None):
        """Returns original features or original features of the requested feature type."""
        if feature_type:
            return list(self.fm_[(self.fm_["Type"] == feature_type)& (self.fm_["Source"] == "Original")]["Feature"].values)
        else:
            return list(self.fm_[(self.fm_["Source"] == "Original")]["Feature"].values)

    def get_active_features(self, feature_type=None):
        """Returns original features or original features of the requested feature type."""
        if feature_type:
            return list(self.fm_[(self.fm_["Active"] == True) & (self.fm_["Type"] == feature_type)]["Feature"].values)
        else:
            return list(self.fm_[(self.fm_["Active"] == True)]["Feature"].values)

    def exclude_feature(self, feature):
        self.fm_.loc[self.fm_["Feature"]==feature, "Active"] = False
        self.fm_.loc[self.fm_["Feature"]==feature, "Signature"] = sys._getframe(1).f_code.co_name        

    def include_feature(self, feature):
        self.fm_.loc[self.fm_["Feature"]==feature, "Active"] = True
        self.fm_.loc[self.fm_["Feature"]==feature, "Signature"] = sys._getframe(1).f_code.co_name        

    def exclude_features(self,features):
        for feature in features:
            self.exclude_feature(feature)

    def include_features(self,features):
        for feature in features:
            self.include_feature(feature)                    

    def add_feature(self, feature, feature_type, active=True):
        if self.fm_[self.fm_["Feature"]==feature].shape[0] == 0:
            d = {"Feature": feature, "Type": feature_type, "Source": "Derived",
                "Active": active, "Signature": sys._getframe(1).f_code.co_name }
            df = pd.DataFrame(data=d, index=[0])
            self.fm_ = pd.concat((self.fm_,df),axis=0)      
    
    def print(self, feature=None, feature_type=None):
        if feature_type:
            print(tabulate(self.fm_[self.fm_["Type"]==feature_type],headers="keys", showindex=False))
        elif feature:
            print(tabulate(self.fm_[self.fm_["Type"]==feature],headers="keys", showindex=False))
        else:
            print(tabulate(self.fm_,headers="keys", showindex=False))

# =========================================================================== #
#                        4. ENCODERS: ORDINAL MAP                             #
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

class OrdinalMap(BaseEstimator, TransformerMixin):
    """ Transforms ordinal features using a one-to-one mapping."""
    def __init__(self, ordinal_map=ordinal_map):
        self._ordinal_map = ordinal_map
    
    def fit(self, X, y=None):
        """Formats inverse map."""
        return self

    def transform(self, X, y=None):
        features = []
        for feature,mappings in self._ordinal_map.items():
            features.append(feature)
            X[feature] = X[feature].replace(mappings)
        #print(X[features].head())
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
# =========================================================================== #
#                        5. NEIGHBORHOOD CLASSIFIER                           #
# =========================================================================== #
class NeighborhoodClassifier:
    """Classifies neighborhoods based upon average home sales price."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Summarize the dataset by Neighborhood        
        X["Sale_Price"] = y
        neighborhoods = X.groupby(["Neighborhood"])
        
        # Obtain the mean sales price by neighborhood
        self.neighborhood_stats_ = neighborhoods["Sale_Price"].describe()
        
        # Break the range of sales prices into 5 quantiles and assign one to each neighborhood
        self.neighborhood_stats_["Neighborhood_Class"] = pd.cut(self.neighborhood_stats_["mean"],5, 
                labels=[1,2,3,4,5], include_lowest=True)
        return self
    
    def transform(self, X, y=None):
        # Merge now each neighborhood has a class associated with it.
        X = X.merge(self.neighborhood_stats_["Neighborhood_Class"], how="left", on="Neighborhood")                
        return X

    def fit_transform(self, X, y=None):
        self.fit(X,y).transform(X)

        
# =========================================================================== #
#                        5. NEIGHBORHOOD QUALITY                              #
# =========================================================================== #
class NeighborhoodQualifier:
    """Creates a linear combination of various normalized quality measures"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.neighborhood_ = pd.DataFrame()
        # Compute Quality for each observation
        X["Quality"] = ((X["Overall_Qual"]/X["Overall_Qual"].mean()) +
                        (X["Overall_Cond"]/X["Overall_Cond"].mean()) + 
                        (X["Exter_Qual"]/X["Exter_Qual"].mean()) + 
                        (X["Exter_Cond"]/X["Exter_Cond"].mean()) + 
                        (X["Bsmt_Qual"]/X["Bsmt_Qual"].mean()) + 
                        (X["Bsmt_Cond"]/X["Bsmt_Cond"].mean()) + 
                        (X["Kitchen_Qual"]/X["Kitchen_Qual"].mean()) + 
                        (X["Bsmt_Cond"]/X["Bsmt_Cond"].mean()) +
                        (X["Functional"]/X["Functional"].mean()))
        
        # Obtain the mean sales price by neighborhood
        self.neighborhood_["Neighborhood_Quality"] = X.groupby(["Neighborhood"])["Quality"].mean().transform(
            lambda x: pd.cut(x,5,labels=[1,2,3,4,5], include_lowest=True))
        
        return self
    
    def transform(self, X, y=None):
        # Merge now each neighborhood has a class associated with it.
        X = X.merge(self.neighborhood_["Neighborhood_Quality"] , how="left", on="Neighborhood")                        
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)
        


# =========================================================================== #
#                          1. PREPROCESSOR                                    #
# =========================================================================== #
class Preprocessor:
    def __init__(self, discrete_encoder=None, ordinal_encoder=OrdinalMap(), 
            nominal_encoder=LeaveOneOutEncoder(drop_invariant=False, 
            return_df=True, sigma=0.1), feature_metadata=FeatureMetadata(), 
            neighborhood_classifier=NeighborhoodClassifier(),
            neighborhood_qualifier = NeighborhoodQualifier(),
            remove_redundancies=False,
            power_transform=True):        
        self._discrete_encoder = discrete_encoder
        self._ordinal_encoder = ordinal_encoder
        self._nominal_encoder = nominal_encoder
        self._feature_metadata = feature_metadata
        self._neighbhorhood_classifier = neighborhood_classifier
        self._neighbhorhood_qualifier = neighborhood_qualifier
        self._remove_redundancies = remove_redundancies
        self._power_transform = power_transform

    def fit(self, X, y=None):
        self.X_ = X
        self.y_ = y
        return self        

    def transform(self, X, y=None):
        self.fit(X,y)
        self.clean().detect_outliers().engineer()
        self.encode_discrete().encode_ordinal().encode_nominal()
        self.engineer2().transformer()       

        if y is not None and self._remove_redundancies:
            self.filter()
        self._get_active_features()
        self.X_ = self.X_[self.active_features_]
        return self.X_, self.y_

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X,y)

    def _check_data(self):
        
        # Check for nulls and na
        if self.X_.isnull().sum().sum() != 0:
            n_nulls = self.X_.isnull().sum().sum()
            print(f"\nWarning, {n_nulls} nulls found by {sys._getframe(1).f_code.co_name}")
            print(self.X_[self.X_.isnull().any(axis=1)])
        
        # Confirm lengths of X and y (if y is not None) 
        if self.y_ is not None:
            assert(self.X_.shape[0] == self.y_.shape[0]), \
            f"X has length {self.X_.shape[0]} and y has length {self.y_.shape[0]}. coming from {sys._getframe(1).f_code.co_name}."

    # ====================================================================== #
    #                          FEATURE NAMES                                 #
    # ====================================================================== #
    def _get_feature(self, feature):
        return self._feature_metadata.get_feature(feature)

    def _get_features(self):
        self.continuous_ = self._feature_metadata.get_features("Continuous")
        self.discrete_ = self._feature_metadata.get_features("Discrete")
        self.nominal_ = self._feature_metadata.get_features("Nominal")
        self.ordinal_ = self._feature_metadata.get_features("Ordinal")
        self.features_ = self._feature_metadata.get_features()

    def _get_original_features(self):
        self.continuous_ = self._feature_metadata.get_original_features("Continuous")
        self.discrete_ = self._feature_metadata.get_original_features("Discrete")
        self.nominal_ = self._feature_metadata.get_original_features("Nominal")
        self.ordinal_ = self._feature_metadata.get_original_features("Ordinal")        
        self.original_features_ = self._feature_metadata.get_original_features()

    def _get_active_features(self):
        self.continuous_ = self._feature_metadata.get_active_features("Continuous")
        self.discrete_ = self._feature_metadata.get_active_features("Discrete")
        self.nominal_ = self._feature_metadata.get_active_features("Nominal")
        self.ordinal_ = self._feature_metadata.get_active_features("Ordinal")                
        self.active_features_ = self._feature_metadata.get_active_features()
        
    def clean(self):
        # Transform the target
        self.y_ = None if self.y_ is None else np.log(self.y_)
        # Initiate imputers
        mean_imputer = SimpleImputer(strategy="mean") 
        median_imputer = SimpleImputer(strategy="median") 
        frequent_imputer = SimpleImputer(strategy="most_frequent") 
        # Get Feature Names
        self._get_original_features()
        # ------------------------------------------------------------------- #
        #                      Continuous Variables                           #
        # ------------------------------------------------------------------- #
        # Remove Nulls
        self.X_[self.continuous_].fillna(self.X_[self.continuous_].mean(), inplace = True)        
        self.X_["Garage_Yr_Blt"].fillna(self.X_["Garage_Yr_Blt"].mean(), inplace = True)        
        self._check_data()

        # Correct data entry errors
        self.X_["Garage_Yr_Blt"].replace(to_replace=2207, value=2007, inplace=True)

        # ------------------------------------------------------------------- #
        #                      Discrete Variables                             #
        # ------------------------------------------------------------------- #
        frequent_imputer.fit(self.X_[self.discrete_])        
        self.X_[self.discrete_] = frequent_imputer.transform(self.X_[self.discrete_])

        # ------------------------------------------------------------------- #
        #                      Ordinal Variables                              #
        # ------------------------------------------------------------------- #      
        frequent_imputer.fit(self.X_[self.ordinal_])        
        self.X_[self.ordinal_] = frequent_imputer.transform(self.X_[self.ordinal_])  

        # ------------------------------------------------------------------- #
        #                      Ordinal Variables                              #
        # ------------------------------------------------------------------- #      
        frequent_imputer.fit(self.X_[self.nominal_])        
        self.X_[self.nominal_] = frequent_imputer.transform(self.X_[self.nominal_])  

        #Check for again nulls
        self._check_data()
        
        return self

    def detect_outliers(self):
        features = self.continuous_ + self.discrete_
        X_numeric = self.X_[features]   
        X_outliers = X_numeric[(np.abs(zscore(X_numeric)) > 3).all(axis=1)]     

        if (X_outliers.shape[0]>0):
            print("\n")
            print("="*40)
            print("            Outlier Detection")
            print("-"*40)
            print(f"         Observations: {X_numeric.shape[0]}")
            print(f"             Features: {X_numeric.shape[1]}")
            print(f"  # Outliers Detected: {X_outliers.shape[0]}")        
            print("-"*40)
            if (X_outliers.shape[0]> 0):
                print(X_outliers)
        return self
    
    def engineer(self):
        """Create new features that increase predictive capability."""
        # ------------------------------------------------------------------- #
        #                        Age Related                                  #
        # ------------------------------------------------------------------- #      
        self.X_.drop(columns=["PID","Latitude", "Longitude"], inplace=True)
        self._feature_metadata.exclude_feature("PID")
        self._feature_metadata.exclude_feature("Latitude")
        self._feature_metadata.exclude_feature("Longitude")        

        # Age 
        self.X_["Age"] = self.X_["Year_Sold"] - self.X_["Year_Built"]
        self.X_["Age"].fillna(self.X_["Age"].median(),inplace=True)      
        self.X_.drop(columns="Year_Built", inplace=True)
        self._feature_metadata.add_feature("Age", "Discrete",True)
        self._feature_metadata.exclude_feature("Year_Built")
        self._check_data()

        # Garage Age 
        self.X_["Garage_Age"] = self.X_["Year_Sold"] - self.X_["Garage_Yr_Blt"]
        self.X_["Garage_Age"].fillna(self.X_["Garage_Age"].mean(),inplace=True)      
        self.X_.drop(columns="Garage_Yr_Blt", inplace=True)
        self._feature_metadata.add_feature("Garage_Age", "Discrete", True)
        self._feature_metadata.exclude_feature("Garage_Yr_Blt")
        self._check_data()

        # Age since remodeled
        self.X_["Age_Remod"] = self.X_["Year_Sold"] - self.X_["Year_Remod_Add"]
        self.X_["Age_Remod"].fillna(self.X_["Age_Remod"].median(),inplace=True)  
        self.X_.drop(columns="Year_Remod_Add", inplace=True)
        self._feature_metadata.add_feature("Age_Remod", "Discrete", True)
        self._feature_metadata.exclude_feature("Year_Remod_Add")
        self._check_data()

        # ------------------------------------------------------------------- #
        #                     Amenity Features                                #
        # ------------------------------------------------------------------- #               
        
        self.X_["Has_Garage"] =  np.where((self.X_["Garage_Finish"] != "No_Garage") & 
            (self.X_["Garage_Type"] != "No_Garage") & 
            (self.X_["Garage_Qual"] != "No_Garage") & 
            (self.X_["Garage_Cond"] != "No_Garage") ,True, False) 
        
        self.X_["Has_Pool"] = np.where((self.X_["Pool_QC"] != "No_Pool"),True, False)
        self.X_["Has_Basement"] = np.where((self.X_["BsmtFin_Type_1"] != "No_Basement") &
                                           (self.X_["BsmtFin_Type_2"] != "No_Basement") ,True, False)


        self.X_["Has_Fireplace"] = np.where((self.X_["Fireplace_Qu"] != "No_Fireplace"),True, False)
        self.X_["Has_Porch"] = self.X_["Open_Porch_SF"].values + \
                               self.X_["Enclosed_Porch"].values + \
                                self.X_["Three_season_porch"].values + \
                                    self.X_["Screen_Porch"].values == 0

              
        self.X_["Has_Garage"].replace(to_replace=[True, False], value=[1,0], inplace=True)
        self.X_["Has_Pool"].replace(to_replace=[True, False], value=[1,0], inplace=True)
        self.X_["Has_Basement"].replace(to_replace=[True, False], value=[1,0], inplace=True)
        self.X_["Has_Fireplace"].replace(to_replace=[True, False], value=[1,0], inplace=True)
        self.X_["Has_Porch"].replace(to_replace=[True, False], value=[1,0], inplace=True)

        self._feature_metadata.add_feature(feature="Has_Garage",feature_type="Discrete",active=True)
        self._feature_metadata.add_feature(feature="Has_Pool",feature_type="Discrete",active=True)
        self._feature_metadata.add_feature(feature="Has_Basement",feature_type="Discrete",active=True)
        self._feature_metadata.add_feature(feature="Has_Fireplace",feature_type="Discrete",active=True)
        self._feature_metadata.add_feature(feature="Has_Porch",feature_type="Discrete",active=True)
        
        self._check_data()


        # ------------------------------------------------------------------- #
        #        FINANCIAL DEMOGRAPHICS: HOUSING PRICE INDEX BY QUARTER       #
        # ------------------------------------------------------------------- #        
        self.X_.loc[self.X_["Mo_Sold"].isin([1,2,3]), "Qtr_Sold"] = str(1)
        self.X_.loc[self.X_["Mo_Sold"].isin([4,5,6]), "Qtr_Sold"] = str(2)
        self.X_.loc[self.X_["Mo_Sold"].isin([7,8,9]), "Qtr_Sold"] = str(3)
        self.X_.loc[self.X_["Mo_Sold"].isin([10,11,12]), "Qtr_Sold"] = str(4)      

        # Format Qtr-Sold Feature 
        self.X_["Year_Sold"] = self.X_["Year_Sold"].astype(int)
        self.X_["Qtr_Sold"] = self.X_["Year_Sold"].astype(str) + "-" + self.X_["Qtr_Sold"].astype(str)

        # Housing Price Index for 2006-2010. Merge with data
        d = {'Qtr_Sold': 
                    ['2006-1', '2006-2', '2006-3', '2006-4', '2007-1', '2007-2',
                    '2007-3', '2007-4', '2008-1', '2008-2', '2008-3', '2008-4',
                    '2009-1', '2009-2', '2009-3', '2009-4', '2010-1', '2010-2',
                    '2010-3', '2010-4'], 
            'HPI': [368.63, 372.4 , 375.47, 379.31, 380.72, 380.48, 376.22, 375.02,
                    372.29, 362.88, 351.46, 348.23, 350.79, 341.56, 332.61, 330.09,
                    326.14, 323.2 , 326.31, 323.96]} 
        hpi = pd.DataFrame(data=d)                                   
        self.X_ = pd.merge(self.X_, hpi, on="Qtr_Sold", how="inner")    

        # Add new fields to feature metadata    

        self._feature_metadata.add_feature(feature="Qtr_Sold",feature_type="Nominal",active=True)        
        self._feature_metadata.add_feature(feature="HPI",feature_type="Continuous",active=True)        
        self._check_data()        

        assert(self.X_["HPI"].mean()>300), "HPI merge problem in engineer."          

        # ------------------------------------------------------------------- #
        #                       NEIGHBORHOOD CLASS                            #
        # ------------------------------------------------------------------- #                
        # If y is not None, then we can add the neighborhood class to the 

        if "Neighborhood_Class" not in self.X_.columns.tolist():
            if self.y_ is not None:
                self._neighbhorhood_classifier.fit(self.X_, self.y_)
            self.X_ = self._neighbhorhood_classifier.transform(self.X_)

            # Add Class to the feature list
            self._feature_metadata.add_feature(feature="Neighborhood_Class", 
                feature_type="Discrete", active=True)

            # Refresh the list of features
            self._get_features()

        return self      

    def _encode(self, encoder, features):
        """ Generic encoding function that takes an encoder and a list of features."""        

        if self.y_ is not None:        
            encoder.fit(self.X_[features], self.y_)
        self.X_[features] = encoder.transform(self.X_[features])                            

    def encode_discrete(self):
        """Encodes discrete variables using 1-Hot, LOO encoding or no encoding."""    
        self._get_features()
        # Check for discrete variables in the dataset
        discrete = list(set(self.X_.columns.tolist()) & set(self.discrete_))        

        if len(discrete) > 0 and self._discrete_encoder is not None:
            self._encode(self._discrete_encoder, discrete)
            self._check_data()
        return self

    def encode_ordinal(self):
        """Ordinal encoding by map, leave one out, or one hot."""
        self._get_features()          
        ordinal = list(set(self.X_.columns.tolist()) & set(self.ordinal_))

        if len(ordinal) > 0 and self._ordinal_encoder is not None:
            self._encode(self._ordinal_encoder, ordinal)        
            self._check_data()
        return self

    def encode_nominal(self):
        """Nominal encoding leave one out, or one hot."""
        self._get_features()          
        nominal = list(set(self.X_.columns.tolist()) & set(self.nominal_))

        if len(nominal) > 0 and self._nominal_encoder is not None:
            self._encode(self._nominal_encoder, nominal)        
            self._check_data()
        return self

    def engineer2(self):
        # ------------------------------------------------------------------- #
        #                       NEIGHBORHOOD QUALITY                          #
        # ------------------------------------------------------------------- #                
        # If y is not None, then we can add the neighborhood quality 

        if "Neighborhood_Quality" not in self.X_.columns.tolist():
            if self.y_ is not None:
                self._neighbhorhood_qualifier.fit(self.X_, self.y_)
            self.X_ = self._neighbhorhood_qualifier.transform(self.X_)

            # Add Class to the feature list
            self._feature_metadata.add_feature(feature="Neighborhood_Quality", 
                feature_type="Discrete", active=True)

            # Refresh the list of features
            self._get_features()        
        return self


    def transformer(self,sigma=0.3):
        """Power transform continuous and leave-one-out target encode categorical."""
        # Get current feature names just in case      
        self._get_features()          

        # ------------------------------------------------------------------- #
        #                           Continuous                                #
        # ------------------------------------------------------------------- #
        # Power transformation to make feature distributions closer to Guassian
        if self._power_transform:
            power = PowerTransformer(method="yeo-johnson", standardize=False)        
            self.X_[self.continuous_] = power.fit_transform(self.X_[self.continuous_])
            self._check_data()

        # ------------------------------------------------------------------- #
        #                          Standardize                                #
        # ------------------------------------------------------------------- #        
        # Obtain active features for standardization and processing.
        self._get_active_features()
        self.X_ = self.X_[self.active_features_]

        standard = StandardScaler()
        standard.fit(self.X_)
        X = standard.transform(self.X_)
        self.X_ = pd.DataFrame(data=X, columns=self.active_features_)
        self._check_data()

        return self

    def _select_redundant_feature(self, a,b):
        features = [a,b]
        model = LinearRegression()
        model.fit(self.X_[features], self.y_)
        return features[np.argmin(abs(model.coef_))]   

    def filter(self, max_collinearity=0.7):
        
        features = self._feature_metadata.get_active_features()
        self.feature_correlations_ = pd.DataFrame()

        # Perform pairwise correlation coefficient calculations
        for col_a, col_b in itertools.combinations(features,2):
            r, p = pearsonr(self.X_[col_a], self.X_[col_b])
            cols =  col_a + "__" + col_b
            d = {"Columns": cols, "A": col_a, "B": col_b,"Correlation": abs(round(r,3)), "p-value": round(p,3)}
            df = pd.DataFrame(data=d, index=[0])
            self.feature_correlations_ = pd.concat((self.feature_correlations_, df), axis=0)

        # Select correlations above threshold
        redundancies = self.feature_correlations_[self.feature_correlations_["Correlation"]>max_collinearity]
        if redundancies.shape[0] > 0:
            features_to_remove = []
            print("\nFiltering Redundant Features")
            print(f"{redundancies.shape[0]} pairs of redundant features found.")            
            print(tabulate(redundancies, headers="keys",showindex=False))
            for idx, row in redundancies.iterrows():
                features_to_remove.append(self._select_redundant_feature(row["A"], row["B"]))
            self._feature_metadata.exclude_features(features_to_remove)        

            print("\nThe following features are excluded.")
            print_list(features_to_remove)
        return self

# =========================================================================== #
#                  2.0 FEATURE SELECTORS: FULL DATA SET                       #
# =========================================================================== #    
class FullSelector(BaseEstimator, TransformerMixin):
    """Simple class that  returns the full data set."""
    def __init__(self, estimator):
        self._estimator = estimator
        self._name = "Full DataSet"

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, x):
        self._name = x                 

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        self.selected_features_ = X.columns
        return X
    
    def fit_transform(self, X, y=None):
        return self.transform(X, y)

# =========================================================================== #
#                2.1 FEATURE SELECTORS: IMPORTANCE SELECTOR                   #
# =========================================================================== #    
class ImportanceSelector(BaseEstimator, TransformerMixin):
    """Returns a dataset with top N important features."""
    def __init__(self, estimator, top_n=10):
        self._estimator = estimator
        self._top_n = top_n
        self._name = f"{self._top_n} Most Important Features"  

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, x):
        self._name = x                   

    def _fit_regression_feature_importance(self, X, y=None):
        self._estimator.fit(X, y)
        importances = {"Feature": X.columns.tolist(), "Importance": abs(self._estimator.coef_)}                        
        self.importances_ = pd.DataFrame(data=importances)

    def _fit_tree_based_feature_importance(self, X, y=None):
        self._estimator.fit(X, y)
        importances = {"Feature": X.columns.tolist(), "Importance": self._estimator.feature_importances_}                           
        self.importances_ = pd.DataFrame(data=importances)        

    def fit(self, X, y=None):
        regressors = ["LinearRegression", "Lasso", "Ridge", "ElasticNet"]
        estimator = self._estimator.__class__.__name__
        if estimator in regressors:
            self._fit_regression_feature_importance(X, y)
        elif "GridSearchCV" == estimator:
            self._estimator = self._estimator.best_estimator_
            self.fit(X, y)
        else:
            self._fit_tree_based_feature_importance(X, y)
        return self

    def transform(self, X, y=None):
        importances = self.importances_.sort_values(by="Importance", ascending=False)
        top_importances = importances.head(self._top_n)
        self.selected_features_ = top_importances["Feature"].values
        return X[self.selected_features_]

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)

# =========================================================================== #
#                  2.2 FEATURE SELECTORS: RFECV SELECTOR                      #
# =========================================================================== #    
class RFECVSelector(BaseEstimator, TransformerMixin):
    """Returns a dataset with top N important features."""
    def __init__(self, estimator, step=1, min_features_to_select=20,cv=5,
                    scoring=rmse, n_jobs=4):
        self._estimator = estimator
        self._step = step
        self._min_features_to_select = min_features_to_select
        self._cv = cv
        self._scoring = scoring        
        self._n_jobs = n_jobs             
        self._name = "RFECV Features"

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, x):
        self._name = x     

    def fit(self, X, y=None):
        if self._estimator.__class__.__name__ == "GridSearchCV":
            self._estimator = self._estimator.best_estimator_
            self.fit(X,y)
        selector = RFECV(estimator=self._estimator, step=self._step,
                         min_features_to_select=self._min_features_to_select,
                         cv=self._cv, scoring=self._scoring, n_jobs=self._n_jobs)
        selector.fit(X,y)
        self.selected_features_ = list(itertools.compress(X.columns,selector.support_))
        return self

    def transform(self, X, y=None):
        return X[self.selected_features_]

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)
# =========================================================================== #
#                  2.3 FEATURE SELECTORS: CUSTOM SELECTOR                     #
# =========================================================================== #    
class CustomSelector(BaseEstimator, TransformerMixin):
    """Simple class that  returns the full data set."""
    def __init__(self, estimator):
        self._estimator = estimator
        self.baseline_features_ = ["Overall_Qual",	"Overall_Cond",	
        "Total_Bsmt_SF", "Gr_Liv_Area",	"Exter_Qual","Age",	"Exter_Cond",
        "Neighborhood_Class", "Has_Garage", "Has_Pool", "Has_Basement",
         "Has_Porch"]
        self.neighborhood_ = ["Age","Gr_Liv_Area", 
        "Neighborhood_Class", "Neighborhood_Quality", "Overall_Qual",	
        "Overall_Cond","Exter_Qual","Exter_Cond"]

        self.neighborhood_amenities = ["Age","Gr_Liv_Area", 
        "Neighborhood_Class", "Neighborhood_Quality", "Overall_Qual",	
        "Overall_Cond","Exter_Qual","Exter_Cond",
        "Has_Garage", "Has_Pool", "Has_Basement",
         "Has_Porch"]        

        self._name = "Neighborhood Class Feature Set"        

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, x):
        self._name = x           

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        self.selected_features_ = self.neighborhood_amenities        
        return X[self.selected_features_]
    
    def fit_transform(self, X, y=None):
        return self.transform(X, y)

selectors = {"Full": FullSelector, "Importance": ImportanceSelector, 
             "RFECV": RFECVSelector, "Custom": CustomSelector}
# =========================================================================== #
#                              3. MODEL                                       #
# =========================================================================== #
class Model:
    def __init__(self, estimator, parameters, selector, groupname,
                scoring=rmse, n_jobs=4, cv=5, refit=True):
        self._estimator = estimator
        self._parameters = parameters
        self._selector = selector   
        self._groupname = groupname
        self._scoring = scoring
        self._n_jobs = n_jobs
        self._cv = cv
        self._refit = refit
        self.model_id_ = uuid.uuid4().time_low            

    def _select_features(self, X, y):
        X = self._selector(self._estimator).fit_transform(X, y)
        self.select_features_ = X.columns.tolist()
        self.n_features_ = X.shape[1]
        return X

    def fit(self, X, y):        
        
        self.started_ = datetime.datetime.now()
        start = time.time()
        # Obtain features for the designated feature selector
        X = self._select_features(X,y)
        self.X_train_ = X

        # Find the best model by iterating over the parameter space.
        gscv = GridSearchCV(estimator=self._estimator, 
                            param_grid=self._parameters, 
                            scoring=self._scoring,
                            n_jobs=self._n_jobs,
                            cv=self._cv,
                            refit=self._refit)
        gscv.fit(X, y)

        # Stop the clock and load up attributes
        end = time.time()
        
        self.name_ = gscv.best_estimator_.__class__.__name__
        self.best_estimator_ = gscv.best_estimator_                         
        self.best_params_ = gscv.best_params_
        self.train_score_ = -gscv.best_score_
        self.training_time_ = round(end - start,3) 
        
        return self

    def predict(self, X):
        start = time.time()
        X = X[self.X_train_.columns]
        y_pred = self.best_estimator_.predict(X)
        end = time.time()
        self.prediction_time_ = round(end - start,3)
        return y_pred
        
    def score(self, X, y):
        PID = y["PID"]        
        y = np.log(y["Sale_Price"])                
        y_pred = self.predict(X)
        self.test_score_ = RMSE(y, y_pred)
        return self.test_score_

    def summary(self):
        name = self._selector.name
        print("\n")
        print("="*40)
        print(f"      Estimator: {self._estimator.__class__.__name__}")
        print(f"          Group: {self._groupname}")
        print(f"        DataSet: {name}")
        print("-"*40)
        print("         ", end="")
        print(self.started_.strftime("%Y-%m-%d %H:%M:%S"))
        print("-"*40)      
        print(f"      # Features: {self.n_features_}")
        print(f"     Train Score: {self.train_score_}")
        print(f"   Training Time: {self.training_time_}")
        print(f"      Test Score: {self.test_score_}")
        print(f" Prediction Time: {self.prediction_time_}")
        print("="*40)

        

# =========================================================================== #
#                           4. EVALUATOR                                      #
# =========================================================================== #
class Evaluator:
    """Captures performance results, reports, and returns best estimator."""
    def __init__(self, persist=True):
        self.models_ = {}
        self.results_ = pd.DataFrame()
        self._persist = persist


    def add_model(self, model, cv):        
        self.models_[model.model_id_] = model
        d = {"CV": cv, "Id": model.model_id_, "Estimator": model.name_,
            "# Features": model.n_features_, "Train Score": model.train_score_,
            "Test Score": model.test_score_, "Train Time": model.training_time_,
            "Predict Time": model.prediction_time_}
        df = pd.DataFrame(data=d, index=[0])
        self.results_ = pd.concat((self.results_,df), axis=0)
        if self._persist:
            self.save(model,cv)

    def detail(self):
        scores = self.results_.pivot(index="Estimator", columns="CV", values="Test Score")
        print("\n")
        print("="*40)
        print("        Scores by Cross-Validation Set")
        print("-"*40)
        print(scores)
        print("-"*40)
        return scores


    def summary(self):
        results = self.results_.groupby(by=["Estimator"]).mean()
        results.sort_values(by="Test Score", inplace=True)
        print("\n")
        print("="*40)
        print("        Performance Results by Algorithm")
        print("-"*40)
        print(results)
        print("-"*40)
        return results

    def save(self, model, cv):
        # Save model
        directory = "../models/"
        filename = directory +str(cv) + "_" + model.name_ + "_" + str(model.model_id_) +\
            "_score_" + str(model.test_score_) + ".joblib"
        dump(model, filename)

        # Save performance results
        cdate = datetime.datetime.now()
        date = cdate.strftime("%B") + "-" + str(cdate.strftime("%d")) + "-" + str(cdate.strftime("%Y"))         
        filename = directory + "performance_results_" + date + ".csv"
        self.results_.to_csv(filename, index=False)



# =========================================================================== #
#                            5. PIPELINE                                      #
# =========================================================================== #
class Pipeline:
    """Selects best algorithm for regression via cross-validation."""
    def __init__(self, preprocessor, model_groups=model_groups, 
                 selectors=selectors, data_loader=AmesData(),  
                 evaluator=Evaluator(persist=True)):

        self._model_groups = model_groups
        self._selectors = selectors
        self._data_loader = data_loader
        self._preprocessor = preprocessor      
        self._evaluator = evaluator


    def fit_ols(self):
        train= self._data_loader.get_train()            
        X_train = train["X"]
        y_train = train["y"]

        # Clean, screen, transform and encode the data            
        X_train, y_train = self._preprocessor.fit_transform(X_train, y_train)
        selector = CustomSelector(estimator=LinearRegression())        
        X_train = selector.fit_transform(X_train)
        print(X_train.describe())

        # # Linear Regression
        X = sm.add_constant(X_train)
        y = y_train
        model = sm.OLS(y, X)
        results = model.fit()
        print(results.summary())


    def fit_cv(self):
            
        for i in range(1,11):
            train, test = self._data_loader.get_cv(i)            
            X_train = train["X"]
            y_train = train["y"]
            X_test = test["X"]
            y_test = test["y"]

            # Clean, screen, transform and encode the data            
            X_train, y_train = self._preprocessor.fit_transform(X_train, y_train)
            X_test, _ = self._preprocessor.transform(X_test)  

            for groupname, model_group in self._model_groups.items():                
                for setname, model_set in model_group.items():                    
                    for name, components in model_set.items():
                        estimator = components["Estimator"]
                        parameters = components["Parameters"]                        
                        for selector in self._selectors.values():
                            model = Model(estimator=estimator, parameters=parameters, 
                                            selector=selector, groupname=groupname)
                            model.fit(X_train,y_train)
                            model.score(X_test, y_test)
                            model.summary()
                            self._evaluator.add_model(model,i)
        return self                        



# =========================================================================== #
#                            15. SUBMISSION                                   #
# =========================================================================== #
class Submission:
    def __init__(self, estimator, cv):
        self._estimator = estimator
        self._cv = cv

    def fit(self, X, y):
        """Retrain best model on entire training set."""
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

    def submit(self):
        """Saves submission"""
        filename = "mysubmission" + str(self._cv) + ".txt"
        self.submission_.to_csv(filename)


def main():
    preprocessor = Preprocessor(discrete_encoder=None, ordinal_encoder=OrdinalMap(),
                                nominal_encoder=LeaveOneOutEncoder(sigma=0.1,
                                drop_invariant=False, return_df=True),
                                remove_redundancies=False,
                                power_transform=False)
    pipe = Pipeline(preprocessor=preprocessor).fit_ols()


if __name__ == "__main__":
    main()
#%%
