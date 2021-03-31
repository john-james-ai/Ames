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
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
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
from data import AmesData, FeatureMetadata


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('mode.chained_assignment', None)
random_state = 6589
# =========================================================================== #
#                                  MODELS                                     #
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
regressors = model_groups["Regressors"]
ensembles = model_groups["Ensembles"]

# =========================================================================== #
#                                ESTIMATORS                                   #
# =========================================================================== #
baseline_estimator_groups = {
    "Regressors": {
        "Linear Regression": LinearRegression(fit_intercept=True, normalize=False,
                            copy_X=True, n_jobs=4),
        "LassoCV": LassoCV(n_alphas=10, fit_intercept=True, normalize=False,
                    cv=5, n_jobs=4),
        "RidgeCV": RidgeCV(fit_intercept=True, normalize=False, scoring=rmse,
                    cv=5),
        "ElasticNetCV": ElasticNetCV(l1_ratio=[.5,.9], n_alphas=10,
                    fit_intercept=True, normalize=False, cv=5, copy_X=True,
                    n_jobs=4, random_state=random_state)
    },
    "Ensembles": {
        "Random Forest": RandomForestRegressor(n_estimators=100, criterion="mse",
                    n_jobs=4, random_state=random_state),
        "AdaBoost": AdaBoostRegressor(random_state=random_state),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, criterion="mse",
                    n_jobs=4),
        "Gradient Boosting": GradientBoostingRegressor(loss="ls", 
                    learning_rate=0.1)

    }
}
optimized_estimator_groups = {
    "Regressors": {
        "Linear Regression": LinearRegression(fit_intercept=True, normalize=False,
                            copy_X=True, n_jobs=4),
        "LassoCV": LassoCV(n_alphas=100, fit_intercept=True, normalize=False,
                    cv=5, n_jobs=4),
        "RidgeCV": RidgeCV(alphas=np.arange(0.5,10.5,0.5),fit_intercept=True, normalize=False, scoring=rmse,
                    cv=5),
        "ElasticNetCV": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], n_alphas=10,
                    fit_intercept=True, normalize=False, cv=5, copy_X=True,
                    n_jobs=4, random_state=random_state)
    },
    "Ensembles": {
        "Random Forest":
            GridSearchCV(estimator=RandomForestRegressor(),
                param_grid=model_groups["Ensembles"]["Random Forest"]["Parameters"],
                scoring=rmse, n_jobs=4,cv=5,refit=True),
        "AdaBoost":
            GridSearchCV(estimator=AdaBoostRegressor(),
                param_grid=model_groups["Ensembles"]["AdaBoost"]["Parameters"],
                scoring=rmse, n_jobs=4,cv=5,refit=True),
        "Extra Trees":
            GridSearchCV(estimator=ExtraTreesRegressor(),
                param_grid=model_groups["Ensembles"]["Extra Trees"]["Parameters"],
                scoring=rmse, n_jobs=4,cv=5,refit=True),         
        "Gradient Boosting":
            GridSearchCV(estimator=GradientBoostingRegressor(),
                param_grid=model_groups["Ensembles"]["Gradient Boosting"]["Parameters"],
                scoring=rmse, n_jobs=4,cv=5,refit=True)
    }    
}
# =========================================================================== #
#                            0. SCORING                                       #
# =========================================================================== #
def RMSE(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)
rmse = make_scorer(RMSE, greater_is_better=False)
# =========================================================================== #
#                          1. PREPROCESSOR                                    #
# =========================================================================== #
class Preprocessor:
    def __init__(self, X, y, feature_metadata=FeatureMetadata()):        
        self._feature_metadata = feature_metadata
        self.X_ = X
        self.y_ = y

    def _check_data(self):
        # Save current list of features
        self.features_ = self.X_.columns.tolist()
        # Check for nulls and na
        if self.X_.isnull().sum().sum() != 0:
            n_nulls = self.X_.isnull().sum().sum()
            print(f"\nWarning, {n_nulls} nulls found by {sys._getframe(1).f_code.co_name}")
            print(self.X_[self.X_.isnull().any(axis=1)])
        # Check lengths of input
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

    def _get_original_features(self):
        self.continuous_ = self._feature_metadata.get_original_features("Continuous")
        self.discrete_ = self._feature_metadata.get_original_features("Discrete")
        self.nominal_ = self._feature_metadata.get_original_features("Nominal")
        self.ordinal_ = self._feature_metadata.get_original_features("Ordinal")        

    def _get_active_features(self):
        self.continuous_ = self._feature_metadata.get_active_features("Continuous")
        self.discrete_ = self._feature_metadata.get_active_features("Discrete")
        self.nominal_ = self._feature_metadata.get_active_features("Nominal")
        self.ordinal_ = self._feature_metadata.get_active_features("Ordinal")                
        
    def clean(self):
        # Transform the target
        self.y_ = np.log(self.y_)       
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
        self.X_.drop(columns=["Latitude", "Longitude"], inplace=True)
        self._feature_metadata.exclude_feature("Latitude")
        self._feature_metadata.exclude_feature("Longitude")
        self._get_features() 

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

        self._feature_metadata.include_feature("Has_Garage")
        self._feature_metadata.include_feature("Has_Pool")
        self._feature_metadata.include_feature("Has_Basement")
        self._feature_metadata.include_feature("Has_Fireplace")
        self._feature_metadata.include_feature("Has_Porch")
        self._get_features()
        self._check_data()

        # ------------------------------------------------------------------- #
        #           School and Zip Code Information                           #
        # ------------------------------------------------------------------- #
        filename = "../data/external/schools.csv"              
        schools = pd.read_csv(filename)        
        self.X_ = pd.merge(self.X_, schools, on="Neighborhood", how="inner")        
        self._check_data()        

        # ------------------------------------------------------------------- #
        #                2008 Financial Crisis Sale                           #
        # ------------------------------------------------------------------- #        
        self.X_.loc[self.X_["Mo_Sold"].isin([1,2,3]), "Qtr_Sold"] = str(1)
        self.X_.loc[self.X_["Mo_Sold"].isin([4,5,6]), "Qtr_Sold"] = str(2)
        self.X_.loc[self.X_["Mo_Sold"].isin([7,8,9]), "Qtr_Sold"] = str(3)
        self.X_.loc[self.X_["Mo_Sold"].isin([10,11,12]), "Qtr_Sold"] = str(4)        

        self.X_["Year_Sold"] = self.X_["Year_Sold"].astype(int)
        self.X_["Qtr_Sold"] = self.X_["Year_Sold"].astype(str) + "-" + self.X_["Qtr_Sold"].astype(str)
                                
        filename = "../data/external/ushpi.csv"
        hpi = pd.read_csv(filename)                                   
        self.X_ = pd.merge(self.X_, hpi, on="Qtr_Sold", how="inner")        
        #self._check_data()        
        
        assert(self.X_["HPI"].mean()>300), "HPI merge problem in engineer."
        return self

    def transform(self,sigma=0.3):
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
        # Nominal 
        features = self._get_features()
        encoder = LeaveOneOutEncoder(return_df=True)
        encoder.fit(self.X_[self.nominal_], self.y_)
        self.X_[self.nominal_] = encoder.transform(self.X_[self.nominal_])
        self._check_data()

        # Ordinal
        encoder.fit(self.X_[self.ordinal_], self.y_)
        self.X_[self.ordinal_] = encoder.transform(self.X_[self.ordinal_])
        self._check_data()
        
        # ------------------------------------------------------------------- #
        #                          Standardize                                #
        # ------------------------------------------------------------------- #        
        standard = StandardScaler()
        standard.fit(self.X_)
        X = standard.transform(self.X_)
        self.X_ = pd.DataFrame(data=X, columns=self.features_)
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
            print_list(features_to_remove,1)
        return self

# =========================================================================== #
#                2.0 FEATURE SELECTORS: IMPORTANCE SELECTOR                   #
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
        regressors = ["LinearRegression", "Lasso", "Ridge", "ElasticNet"]
        estimator = self._estimator.__class__.__name__
        if estimator in regressors:
            self._fit_regression_feature_importance(X, y)
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
#                  2.1 FEATURE SELECTORS: RFECV SELECTOR                      #
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

# =========================================================================== #
#                              3. MODEL                                       #
# =========================================================================== #
class Model:
    def __init__(self, estimator, selector, **kwargs):
        self._estimator = estimator
        self._selector = selector   
        self.model_id_ = uuid.uuid4().time_low                 

    def select_features(self, X, y):
        X = self._selector.fit(X, y).transform(X)
        self.select_features_ = X.columns.tolist()

    def _extract_fit_data(self, X, y):
        if self._estimator.__class__.__name__ ==  "GridSearchCV":
            self.name_ = self._estimator.best_estimator_.__class__.__name__
            self.best_estimator_ = self._estimator.best_estimator_                         
            self.best_params_ = self._estimator.best_params_
            self.train_score_ = self._estimator.best_score_
        else:            
            self.name_ = self._estimator.__class__.__name__
            self.best_estimator_ = self._estimator
            self.best_params_ = self._estimator.get_params()
            self.train_score_ = self._score(X, y)
        self.n_features_ = X.shape[1]


    def fit(self, X, y):
        start = time.time()
        X = self.select_features(X, y)
        self._estimator.fit(X,y)
        end = time.time()
        self.training_time_ = round(end - start,3)
        self._extract_fit_data(X, y)
        return self

    def predict(self, X):
        start = time.time()
        y_pred = self._estimator.predict(X)
        end = time.time()
        self.prediction_time_ = round(end - start,3)
        return y_pred

    def _score(self, X, y):        
        y_pred = self.predict(X)
        self.train_score_ = RMSE(y, y_pred)
        return self.train_score_
        
    def score(self, X, y):        
        y_pred = self.predict(X)
        self.test_score_ = RMSE(y, y_pred)
        return self.test_score_

        

# =========================================================================== #
#                           4. EVALUATOR                                      #
# =========================================================================== #
class Evaluator:
    """Captures performance results, reports, and returns best estimator."""
    def __init__(self, persist=True):
        self.models_ = {}
        self.results_ = pd.DataFrame()


    def add_model(self, model, cv):        
        self.models_[model.model_id_] = model
        d = {"CV": cv, "Id": model.model_id_, "Estimator": model.name_,
            "# Features": model.n_features_, "Train Score": model.train_score_,
            "Test Score": model.test_score_, "Train Time": model.training_time_,
            "Predict Time": model.prediction_time_}
        df = pd.DataFrame(data=d, index=[0])
        self.results_ = pd.concat((self.results_,df), axis=0)

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


# =========================================================================== #
#                          14. EVALUATOR                                      #
# =========================================================================== #
class Evaluator(ABC):
    """Evaluates models, stores scores and returns the best model."""

    def __init__(self, estimator, param_grid, group, split, k, step=1,
                 min_features_to_select=10,cv=5, scoring=rmse, n_jobs=4,
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
        
        d = {"FileSplit": self._split,"Group": self._group,"Model Id": self.model_id_,
        "Estimator": self.best_estimator_.__class__.__name__,"Score": self.best_score_,
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


def test():
    data = AmesData()
    train = data.get_train()
    X = train["X"]
    y = train["y"]

    preprocessor = Preprocessor(X,y)
    preprocessor.clean().detect_outliers()
    preprocessor.engineer().transform().filter()
    X = preprocessor.X_
    y = preprocessor.y_

    selector = Selector(X,y)
    selector.plot()



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
    test()
#%%
