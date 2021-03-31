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


# Feature and model selection and evaluation
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.feature_selection import VarianceThreshold, f_regression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

# Regression based estimators
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV

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
from utils import print_list


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
            print(self.fm_[self.fm_["Type"]==feature_type])
        elif feature:
            print(self.fm_[self.fm_["Feature"]==feature])
        else:
            print(self.fm_)
# =========================================================================== #
#                          1. PREPROCESSOR                                    #
# =========================================================================== #
class Preprocessor:
    def __init__(self, feature_metadata=FeatureMetadata(), 
                looe=LeaveOneOutEncoder(drop_invariant=False, return_df=True)):        
        self._feature_metadata = feature_metadata
        self._looe = looe

    def fit(self, X, y=None):
        self.X_ = X
        self.y_ = y
        return self        

    def transform(self, X, y=None):
        self.fit(X,y)
        if y is not None:
            self.clean().detect_outliers().engineer().transformer().filter()
        else: 
            self.clean().engineer().transformer()
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
        self._feature_metadata.include_feature("Age")
        self._feature_metadata.exclude_feature("Year_Built")
        self._check_data()

        # Garage Age 
        self.X_["Garage_Age"] = self.X_["Year_Sold"] - self.X_["Garage_Yr_Blt"]
        self.X_["Garage_Age"].fillna(self.X_["Garage_Age"].mean(),inplace=True)      
        self.X_.drop(columns="Garage_Yr_Blt", inplace=True)
        self._feature_metadata.include_feature("Garage_Age")
        self._feature_metadata.exclude_feature("Garage_Yr_Blt")
        self._check_data()

        # Age since remodeled
        self.X_["Age_Remod"] = self.X_["Year_Sold"] - self.X_["Year_Remod_Add"]
        self.X_["Age_Remod"].fillna(self.X_["Age_Remod"].median(),inplace=True)  
        self.X_.drop(columns="Year_Remod_Add", inplace=True)
        self._feature_metadata.include_feature("Age_Remod")
        self._feature_metadata.exclude_feature("Year_Remod_Add")
        self._check_data()

        # ------------------------------------------------------------------- #
        #                     Amenity Features                                #
        # ------------------------------------------------------------------- #      
        self.X_["Has_Garage"] =  "No_Garage" not in self.X_["Garage_Type"].values
        self.X_["Has_Pool"] = "No_Pool" not in self.X_["Pool_QC"].values 
        self.X_["Has_Basement"] = "No_Basement" not in self.X_["Bsmt_Qual"].values 
        self.X_["Has_Fireplace"] = "No_Fireplace" not in self.X_["Fireplace_Qu"].values 
        self.X_["Has_Porch"] = self.X_["Open_Porch_SF"].values + \
                               self.X_["Enclosed_Porch"].values + \
                                self.X_["Three_season_porch"].values + \
                                    self.X_["Screen_Porch"].values == 0

        self.X_["Has_Garage"].replace(to_replace=[True, False], value=["Y","N"], inplace=True)
        self.X_["Has_Pool"].replace(to_replace=[True, False], value=["Y","N"], inplace=True)
        self.X_["Has_Basement"].replace(to_replace=[True, False], value=["Y","N"], inplace=True)
        self.X_["Has_Fireplace"].replace(to_replace=[True, False], value=["Y","N"], inplace=True)
        self.X_["Has_Porch"].replace(to_replace=[True, False], value=["Y","N"], inplace=True)

        self._feature_metadata.add_feature(feature="Has_Garage",feature_type="Nominal",active=True)
        self._feature_metadata.add_feature(feature="Has_Pool",feature_type="Nominal",active=True)
        self._feature_metadata.add_feature(feature="Has_Basement",feature_type="Nominal",active=True)
        self._feature_metadata.add_feature(feature="Has_Fireplace",feature_type="Nominal",active=True)
        self._feature_metadata.add_feature(feature="Has_Porch",feature_type="Nominal",active=True)
        
        self._check_data()

        # ------------------------------------------------------------------- #
        #           School and Zip Code Information                           #
        # ------------------------------------------------------------------- #
        # filename = "../data/external/schools.csv"              
        # schools = pd.read_csv(filename)        
        # self.X_ = pd.merge(self.X_, schools, on="Neighborhood", how="inner")           

        # Add variables to metadata
        # self._feature_metadata.add_feature(feature="Zip",feature_type="Nominal",active=True)
        # self._feature_metadata.add_feature(feature="School_Title_1",feature_type="Nominal",active=True)
        # self._feature_metadata.add_feature(feature="School_Students",feature_type="Continuous",active=True)
        # self._feature_metadata.add_feature(feature="School_Teachers",feature_type="Continuous",active=True)
        # self._feature_metadata.add_feature(feature="School_Student_Teacher_Ratio",feature_type="Continuous",active=True)
        # self._feature_metadata.add_feature(feature="Free_or_Reduced_Lunch",feature_type="Continuous",active=True)

        # self._check_data()        

        # ------------------------------------------------------------------- #
        #                2008 Financial Crisis Sale                           #
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
        return self

    def transformer(self,sigma=0.3):
        """Power transform continuous and leave-one-out target encode categorical."""
        # Get current feature names just in case      
        self._get_features()          

        # ------------------------------------------------------------------- #
        #                           Continuous                                #
        # ------------------------------------------------------------------- #
        # Power transformation to make feature distributions closer to Guassian
        power = PowerTransformer(method="yeo-johnson", standardize=False)
        self.X_[self.continuous_] = power.fit_transform(self.X_[self.continuous_])
        self._check_data()

        # ------------------------------------------------------------------- #
        #                          Categorical                                #
        # ------------------------------------------------------------------- #        
        categorical = self.nominal_ + self.ordinal_

        if self.y_ is not None:        
            self._looe.fit(self.X_[categorical], self.y_)
        self.X_[categorical] = self._looe.transform(self.X_[categorical])
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
            d = {"Columns": cols, "A": col_a, "B": col_b,"Correlation": abs(r), "p-value": p}
            df = pd.DataFrame(data=d, index=[0])
            self.feature_correlations_ = pd.concat((self.feature_correlations_, df), axis=0)

        # Select correlations above threshold
        redundancies = self.feature_correlations_[self.feature_correlations_["Correlation"]>max_collinearity]
        if redundancies.shape[0] > 0:
            features_to_remove = []
            print("\nFiltering Redundant Features")
            print(f"{redundancies.shape[0]} pairs of redundant features found.")
            print(redundancies)
            for idx, row in redundancies.iterrows():
                features_to_remove.append(self._select_redundant_feature(row["A"], row["B"]))
            self._feature_metadata.exclude_features(features_to_remove)        

            print("\nThe following features are excluded.")
            print_list(features_to_remove,2)
        return self

# =========================================================================== #
#                  2.0 FEATURE SELECTORS: FULL DATA SET                       #
# =========================================================================== #    
class FullSelector(BaseEstimator, TransformerMixin):
    """Simple class that  returns the full data set."""
    def __init__(self, estimator):
        self._estimator = estimator

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

# =========================================================================== #
#                2.1 FEATURE SELECTORS: IMPORTANCE SELECTOR                   #
# =========================================================================== #    
class ImportanceSelector(BaseEstimator, TransformerMixin):
    """Returns a dataset with top N important features."""
    def __init__(self, estimator, top_n=10):
        self._estimator = estimator
        self._top_n = top_n

    def _fit_regression_feature_importance(self, X, y=None):
        self._estimator.fit(X, y)
        importances = {"Feature": X.columns.tolist(), "Importance": abs(model.coef_)}                        
        self.importances_ = pd.DataFrame(data=importances)

    def _fit_tree_based_feature_importance(self, X, y=None):
        self._estimator.fit(X, y)
        importances = {"Feature": X.columns.tolist(), "Importance": model.feature_importances_}                           
        self.importances_ = pd.DataFrame(data=importances)        

    def fit(self, X, y=None):
        regressors = ["LinearRegression", "LassoCV", "RidgeCV", "ElasticNetCV"]
        estimator = self._estimator.__class__.__name__
        if estimator in regressors:
            self._fit_regression_feature_importance(X, y)
        elif "GridSearchCV" == estimator:
            self._estimator = self._estimator.best_estimator_
            self.fit(X, y)
        else:
            self._fit_tree_based_feature_importance(X, y)
        return self

    def transform(self, X):
        importances = self.importances_.sort_values(by="Importance", ascending=False)
        top_importances = importances.head(self._top_n)
        top_features = top_importances["Feature"].values
        return X[top_features]

    def fit_transform(X, y=None):
        return self.fit(X,y).transform(X)

# =========================================================================== #
#                  2.2 FEATURE SELECTORS: RFECV SELECTOR                      #
# =========================================================================== #    
class RFECVSelector(BaseEstimator, TransformerMixin):
    """Returns a dataset with top N important features."""
    def __init__(self, estimator, step=1, min_features_to_select=5,cv=5,
                    scoring=rmse, n_jobs=4):
        self._estimator = estimator
        self._step = step
        self._min_features_to_select = min_features_to_select
        self._cv = cv
        self._scoring = scoring        
        self._n_jobs = n_jobs        

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

    def transform(self, X):
        return X[self.selected_features_]

    def fit_transform(X, y=None):
        return self.fit(X,y).transform(X)

selectors = {"Full": FullSelector, "Importance": ImportanceSelector, "RFECV": RFECVSelector}
# =========================================================================== #
#                              3. MODEL                                       #
# =========================================================================== #
class Model:
    def __init__(self, estimator, selector, **kwargs):
        self._estimator = estimator
        self._selector = selector   
        self.model_id_ = uuid.uuid4().time_low    

    def _start_training(self): 

        now = datetime.datetime.now()
        self.start_training_ = time.time()      
        
        print("\n")
        print("="*40)
        print(f"      Estimator: {self._estimator.__class__.__name__}")
        print("-"*40)
        print("         ", end="")
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        print("-"*40)      

    def _get_params(self):
        """Obtains parameter for LassoCV, RidgeCV, and ElasticNetCV."""
        if self._estimator.__class__.__name__ in ["LassoCV", "RidgeCV"]:
            return self._estimator.alpha_

        elif self._estimator.__class__.__name__ == "ElasticNetCV":
            return {"alpha_": self._estimator.alpha_, 
                    "l1_ratio":self._estimator.l1_ratio_}

        else:
            return None
    def _end_training(self, X, y):

        self.end_training_ = time.time()
        self.training_time_ = round(self.end_training_ - self.start_training_,3)            
        
        if self._estimator.__class__.__name__ ==  "GridSearchCV":
            self.name_ = self._estimator.best_estimator_.__class__.__name__
            self.best_estimator_ = self._estimator.best_estimator_                         
            self.best_params_ = self._estimator.best_params_
            self.train_score_ = self._estimator.best_score_
        else:            
            self.name_ = self._estimator.__class__.__name__
            self.best_estimator_ = self._estimator
            self.best_params_ = self._get_params()
            self.train_score_ = self._score(X, y)
        self.n_features_ = X.shape[1]
        print(f"      # Features: {self.n_features_}")
        print("-"*40)
        print(f"     Train Score: {self.train_score_}")
        

    def select_features(self, X, y):
        X = self._selector.fit(X.values, y).transform(X)
        self.select_features_ = X.columns.tolist()
        self.n_features_ = X.shape[1]
        print(f"                                        ")
        print(f"      # Features: {self.n_features_}")

    def fit(self, X, y):

        self._start_training()    
        self._estimator.fit(X.values,y)   
        self._end_training(X, y)
        
        return self

    def predict(self, X):
        start = time.time()
        y_pred = self._estimator.predict(X)
        end = time.time()
        self.prediction_time_ = round(end - start,3)
        return y_pred

    def _score(self, X, y):        
        y_pred = self.predict(X.values)
        self.train_score_ = RMSE(y, y_pred)
        return self.train_score_
        
    def score(self, X, y):
        PID = y["PID"]        
        y = np.log(y["Sale_Price"])
        y_pred = self.predict(X)
        self.test_score_ = RMSE(y, y_pred)
        print(f"      Test Score: {self.test_score_}")        
        print(f"   Training Time: {self.training_time_}")
        print(f"    Predict Time: {self.prediction_time_}")
        print("="*40)
        print("\n")
        return self.test_score_

        

# =========================================================================== #
#                           4. EVALUATOR                                      #
# =========================================================================== #
class Evaluator:
    """Captures performance results, reports, and returns best estimator."""
    def __init__(self, persist=False):
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
    def __init__(self, model_groups=model_groups, selectors=selectors, 
                    data_loader=AmesData(), preprocessor=Preprocessor(),                    
                    evaluator=Evaluator(persist=True)):

        self._model_groups = model_groups
        self._selectors = selectors
        self._data_loader = data_loader
        self._preprocessor = preprocessor        
        self._evaluator = evaluator


    def fit(self):
        max_alphas_lasso = []
        min_alphas_lasso = []
        max_alphas_enet = []
        min_alphas_enet = []
            
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
                print(groupname)                
                for setname, model_set in model_group.items():                    
                    print(setname)                    
                    for name, estimator in model_set.items():
                        
                        for selector in self._selectors.values():
                            model = Model(estimator=estimator, selector=selector)
                            model.fit(X_train,y_train)
                            model.score(X_test, y_test)
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
    pipe = Pipeline().fit()


if __name__ == "__main__":
    main()
#%%
