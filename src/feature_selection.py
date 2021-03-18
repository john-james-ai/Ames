# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \build_features.py                                                #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Monday, March 8th 2021, 11:46:37 pm                         #
# Last Modified : Tuesday, March 9th 2021, 10:50:21 pm                        #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
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
from scipy.stats import sem
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
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

# Visualizing data
import seaborn as sns
import matplotlib.pyplot as plt

# Utilities
from utils import Notify, Persistence

# Global Variables
from globals import random_state, discrete, continuous, numeric, n_nominal_levels
from globals import nominal, ordinal, ordinal_map
from metrics import rmse
from utils import onehotmap, notify
 
# =========================================================================== #
#                        NUMERIC FEATURE SELECTION                            #
# =========================================================================== #   
class SelectNumeric(BaseEstimator, TransformerMixin):
    """Performs numeric feature selection using RFECV."""
    def __init__(self, estimator, scoring=rmse, numeric=numeric):
        self._estimator = estimator
        self._scoring = scoring                
        self._numeric = numeric

    def fit(self, X, y, **fit_params):
        """ Performs feature selection for numeric features."""
        notify.entering(__class__.__name__, "fit")

        X_new = X[self._numeric]

        # Run RFECV on chosen estimator and fit the model
        selector = RFECV(estimator=self._estimator,min_features_to_select=1,
                       step=1, n_jobs=2, scoring=self._scoring)
        selector.fit(X_new, y)

        # Extract selected numeric features
        self.numeric_features_ = self._numeric[selector.support_]
        print(f"Numeric Features Selected: {self.numeric_features_}")

        # Drop numeric features and concatenate (add) selected features to the design matrix
        X_new = X.drop(self._numeric, axis=1)
        X_new = pd.concat((X_new, X[self.numeric_features_]), axis=1)

        # Save the estimator
        persistence=Persistence()
        persistence.dump(selector)
        
        notify.leaving(__class__.__name__, "fit")
        return X_new

    def transform(self, X, **transform_params):
        return X[self.numeric_features_]

# =========================================================================== #
#                           FEATURE IMPORTANCE                                #
# =========================================================================== #   
class FeatureRanker(BaseEstimator, TransformerMixin):
    """Returns a list of features in descending order of importance."""
    def __init__(self, nominal=nominal):
        self._nominal = nominal

    def fit(self, X, y, **fit_params):
        notify.entering(__class__.__name__, "fit")
        # Prepare Data
        X = X[self._nominal]        
        X = pd.get_dummies(X)
        groups = onehotmap(X.columns, self._nominal) # Returns original column for each dummy variable

        # Instantiate the decision tree and store results in dataframe
        tree = DecisionTreeRegressor().fit(X, y)        
        d = {"Original":  groups, "Feature": X.columns, "Importance": tree.feature_importances_}
        df = pd.DataFrame(data=d)
        
        # Aggregate, summarize and sort mean importance by original column name
        self.importance_ = df.groupby("Original").mean().reset_index()        
        self.importance_.sort_values(by=["Importance"], inplace=True, ascending=False)

        # Save the estimator
        persistence=Persistence()
        persistence.dump(tree)

        notify.leaving(__class__.__name__, "fit")    

    def transform(self, X, **transform_params):
        return X[self.importance_]


# =========================================================================== #
#                            FORWARD SELECTION                                #
# =========================================================================== #   
class ForwardSelector(BaseEstimator, TransformerMixin):
    """Performs numeric feature selection using RFECV."""
    def __init__(self, estimator, scoring=rmse, nominal=nominal):
        self._estimator = estimator
        self._scoring = scoring                
        self._nominal = nominal

    def fit(self, X, y, **fit_params):
        """Performs forward feature selection for categorical variables."""
        notify.entering(__class__.__name__, "fit")

        # Initialize n_features as length of nominal
        n_features = len(self._nominal)        
        average_scores = []

        # Compute variable importance using FeatureRanker
        ranker = FeatureRanker()
        ranker.fit(X,y)        
        
        # Add features in order of descending importance and to model and cross-validate 
        for i in range(n_features):
            X_new = X[ranker.importance_[0:i+1]].copy()
            scores = cross_validate(self._estimator, X_new, y, scoring=self._scoring)
            average_scores.append(np.mean(scores["test_score"]))
        
        # Compute number of features based upon one standard error rule
        std_error = sem(average_scores)
        mean_average_scores = np.mean(average_scores)
        threshold = mean_average_scores - std_error
        num_features = np.where(scores["test_score"] >= threshold)[0]
        X_nominal = X[ranker.importance_["Original"][0:num_features+1]]
        self.nominal_features_ = ranker.importance_["Original"][0:num_features+1]
        print(f"Nominal Features Selected: {self.nominal_features}")
        

        # Drop nominal features and concatenate selected features
        X_new = X.drop(self._nominal, axis=1)
        X_new = pd.concat((X_new, X_nominal), axis=1)
        notify.leaving(__class__.__name__, "forward")
        return X_new

    def fit(self, X, y=None, **fit_params):
        # Engage the selector
        notify.entering(__class__.__name__, "fit")
        if (self._feature_type == 'numeric'):
            X_new = self.__rfecv(X, y)
        else:
            X_new = self._forward(X, y)
        notify.leaving(__class__.__name__, "fit")        
        
        return self

    def transform(self, X, **transform_params):
        # Engage the selector
        notify.entering(__class__.__name__, "transform")
        if (self._feature_type == 'numeric'):
            X_new = self.__rfecv()
        else:
            X_new = self._forward()
        self._persistence.dump(self._selector)
        notify.leaving(__class__.__name__, "transform")
        return X_new
        

    def plot(self):
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10,12))
        x = range(self._min_features, len(self._feature_selector.grid_scores_)+self._min_features)
        y = self._feature_selector.grid_scores_
        d = {"x": x, "y":y}
        df = pd.DataFrame(d)
        ax = sns.lineplot(x=x, y=y, data=df)
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Cross Validation Score (RMSE)")
        plt.title("Recursive Feature Elimination via Cross-Validation")
        plt.tight_layout()
        plt.show()
