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
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, OrdinalEncoder

# Feature and model selection and evaluation
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_regression
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

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
from utils import Notify, PersistEstimator, PersistNumpy, PersistDataFrame, PersistDictionary

# Global Variables
from globals import discrete, continuous, numeric, n_nominal_levels
from globals import nominal, ordinal, ordinal_map, directories, regressors, ensembles 
from metrics import rmse
from data import AmesData
from data_processor import HotOneEncoder
from utils import onehotmap, notify, validate, convert, comment
# =========================================================================== #
#                            COLUMN SELECTOR                                  #
# =========================================================================== #  
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Returns all columns of the designated type."""
    def __init__(self, columns=nominal):
        self._columns = columns

    def fit(self, X, y, **fit_params):
        return self

    def transform(self, X, **transform_params):        
        return X[self._columns]

# =========================================================================== #
#             PIPELINE RECURSIVE FEATURE ELIMINATION WRAPPER                  #
# =========================================================================== #     
class PipelineRFE(Pipeline):
    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(self, X, y=None, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].feature_importances_

 
# =========================================================================== #
#                        NUMERIC FEATURE SELECTION                            #
# =========================================================================== #   
class NumericSelector(BaseEstimator, TransformerMixin):
    """Performs numeric feature selection using RFECV."""
    def __init__(self, estimator, scoring=rmse, numeric=numeric):
        self._estimator = estimator
        self._scoring = scoring                
        self._numeric = numeric

    def fit(self, X, y, **fit_params):
        """ Performs feature selection for numeric features."""
        notify.entering(__class__.__name__, "fit")
        classname = self.__class__.__name__
        methodname = "fit"

        # Run RFECV on chosen estimator and fit the model
        self.selector_ = RFECV(estimator=self._estimator,min_features_to_select=1,
                       step=1, n_jobs=2, scoring=self._scoring)
        self.selector_.fit(X, y)

        # Extract selected numeric features
        self.feature_names_ = X.columns
        self.features_selected_ = np.array(self._numeric)[self.selector_.support_].tolist()
        message = f"\nModel: {self._estimator.__class__.__name__} selected {len(self.features_selected_)} numeric features\n"
        comment.regarding(classname, message)
        
        notify.leaving(__class__.__name__, "fit")
        return self

    def transform(self, X, **transform_params):
        return X[self.features_selected_]

    def plot(self):
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10,12))        

        # Fit performance curve
        performance_curve = {"Number of Features": list(range(1, len(self.feature_names_)+1)),
                             "Root Mean Squared Error": self.selector_.grid_scores_}
        performance_curve = pd.DataFrame(performance_curve)

        # Create plot objects
        sns.lineplot(x="Number of Features", y="Root Mean Squared Error", data=performance_curve,
                     ax=ax)
        sns.regplot(x=performance_curve["Number of Features"],y=performance_curve["Root Mean Squared Error"],
                    ax=ax)
        
        # Title and axis labels
        plt.xlabel("Number of Features")
        plt.ylabel("Root Mean Squared Error")
        plt.title("Recursive Feature Elimination")
        plt.tight_layout()
        plt.show()            

# =========================================================================== #
#                           FEATURE IMPORTANCE                                #
# =========================================================================== #   
class FeatureImportance(BaseEstimator, TransformerMixin):
    """Returns pre-transformed numeric categorical features in descending order of importance."""
    def __init__(self, estimator, nominal=nominal):
        self._estimator = estimator
        self._nominal = nominal

    def fit(self, X, y):
        notify.entering(__class__.__name__, "fit")   
        classname = self.__class__.__name__

        # Prepare Data
        enc = OrdinalEncoder()
        enc.fit(X)
        X = enc.transform(X)

        # Obtain feature importances from the model
        model = self._estimator.fit(X, y)
        importances_ = model.feature_importances_.flatten()

        # Store in a dataframe for easy manipulation
        d = {"Feature": X.columns.tolist(), "Importance": importances}
        self.importances_ = pd.DataFrame(data=d)                  
        self.importance_.sort_values(by=np.abs(["Importance"]), inplace=True, ascending=False)

        # Report importances
        message = f"Feature Importances\n{self.importances_}"
        comment.regarding(classname, message)    

    def transform(self, X):
        return self.importances_          
        

# =========================================================================== #
#                       GROUPED FEATURE IMPORTANCE                            #
# =========================================================================== #   
class GroupedFeatureImportance(BaseEstimator, TransformerMixin):
    """Returns pre-transformed one-hot features in descending order of importance."""
    def __init__(self, estimator, nominal=nominal):
        self._estimator = estimator
        self._nominal = nominal

    def fit(self, X, y, **fit_params):
        notify.entering(__class__.__name__, "fit")   
        classname = self.__class__.__name__
        methodname = "fit"
        
        # Prepare Data
        enc = HotOneEncoder()
        enc.fit(X)
        X = enc.transform(X)
        original_features = enc.get_original(enc.transformed_features_)      

        # Instantiate the estimator and store results in dataframe
        model = self._estimator.fit(X, y)  
        importances = model.coef_.flatten()        

        # Store in a dataframe for easy manipulation
        d = {"Feature":  original_features, "Level": enc.transformed_features_, "Importance": importances}
        df = pd.DataFrame(data=d)        
        
        # Aggregate, summarize and sort mean importance by original column name
        #self.importance_ = df.groupby("Feature").mean().reset_index()        
        self.importance_ = df.groupby("Feature").mean()
        self.importance_.sort_values(by=np.abs(["Importance"]), inplace=True, ascending=False)        

        message = f"Feature Importances\n{self.importance_}"
        comment.regarding(classname, message)

        notify.leaving(__class__.__name__, "fit")    
        return self

    def transform(self, X, **transform_params):
        return self.importance_

    def plot(self):
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10,12))        
        ax = sns.lineplot(x="Importance", y="Feature", data=self.importance_)
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance by Decision Tree Regression")
        plt.tight_layout()
        plt.show()    


# =========================================================================== #
#                            FORWARD SELECTION                                #
# =========================================================================== #   
class ForwardSelection(BaseEstimator, TransformerMixin):
    """Performs nominal feature selection using forward selection."""
    def __init__(self, estimator, scoring=rmse, nominal=nominal):
        self._estimator = estimator
        self._scoring = scoring                
        self._nominal = nominal

    def _encode_bind(self, X_numeric, X_nominal, features):
        """Encodes nominal features and binds with numeric features."""
        enc = HotOneEncoder()
        enc.fit(X_nominal[features],y)            
        X_nominal_transformed = enc.transform(X_nominal[features])        
        X_new = np.concatenate((X_numeric.to_numpy(), X_nominal_transformed), axis=1)        
        return X_new

    def fit(self, X, y, **fit_params):
        """Performs forward feature selection for categorical variables."""
        notify.entering(__class__.__name__, "fit")
        classname = self.__class__.__name__
        methodname = "fit"        

        # Obtain ranked list of NOMINAL features by importance
        fi = FeatureImportance(self._estimator, nominal=self._nominal).fit(X[self._nominal],y)
        self.importances_ = fi.transform(X)      
        self.num_original_features_ = self.importances_.shape[0]

        # Separate numeric features, then add nominal features one by one in order of importance
        X_numeric = X.drop(columns=self._nominal, inplace=False)
        X_nominal = X[self._nominal]        
        
        # Add features in order of descending importance and to model and cross-validate 
        self.scores_ = []
        for i in range(len(ranked_features)):
            features = self.importances_[0:i+1].index
            X_new = self._encode_bind(X_numeric, X_nominal, features)          
            X_new, y = convert(classname, context, X_new, y) 
            scores = cross_validate(self._estimator, X_new, y, scoring=self._scoring)
            self.scores_.append(np.mean(scores["test_score"]))
        
        # Compute number of features based upon one standard error rule. Note that
        # scikit-learn always maximizes the objective function. If optimizing a loss function
        # scikit-learn flips the sign of the metric so that it maximizes a negative number.
        # Consequently, we need to think in terms of maximizing the negative rmse.
        std_error = sem(self.scores_)
        max_average_scores = np.max(self.scores_)
        threshold = max_average_scores - std_error
        self.scores_ = np.array(self.scores_)

        # Obtain selected features
        acceptable_num_features = np.where(self.scores_ >= threshold)        
        self.num_features_ = acceptable_num_features[0][0]        
        self.features_selected_ = ranked_features[np.arange(self.num_features_+1)]

        # Obtain discarded features
        self.num_discarded_features_ = self.num_original_features_ - self.features_selected_
        self.discarded_features_ = ranked_features[-self.num_discarded_features_]

        message = f"Model: {self._estimator.__class__.__name__} selected {self.num_features_} \
            features and discarded the following {self.num_discarded_features_} features.\
                \n{self.discarded_features_} features."
        comment.regarding(classname, message)

        # Create final dataset with numeric and nominal features selected
        X_new = self._encode_bind(X_numeric, X_nominal, self.features_selected_)
        self.encoded_features_selected_ = X_new.columns.tolist()

        notify.leaving(__class__.__name__, "forward")
        return self

    def transform(self, X, **transform_params):   
        """Transform final nominal features to one-hot-encoding."""     
        return X[self.encoded_features_selected_]
        
    def plot(self):
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10,12))
        x = range(self._min_features, self.num_features_ + self._min_features)
        y = self.scores_
        d = {"x": x, "y":y}
        df = pd.DataFrame(d)
        ax = sns.lineplot(x=x, y=y, data=df)
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Cross Validation Score (RMSE)")
        plt.title("Forward Selection")
        plt.tight_layout()
        plt.show()


# =========================================================================== #
#                           GROUPED FORWARD SELECTION                         #
# =========================================================================== #   
class GroupedForwardSelection(BaseEstimator, TransformerMixin):
    """Performs nominal feature selection using forward selection."""
    def __init__(self, estimator, scoring=rmse, nominal=nominal):
        self._estimator = estimator
        self._scoring = scoring                
        self._nominal = nominal

    def _encode_bind(self, X_numeric, X_nominal, features):
        """Encodes nominal features and binds with numeric features."""
        enc = HotOneEncoder()
        enc.fit(X_nominal[features],y)            
        X_nominal_transformed = enc.transform(X_nominal[features])        
        X_new = np.concatenate((X_numeric.to_numpy(), X_nominal_transformed), axis=1)        
        return X_new

    def fit(self, X, y, **fit_params):
        """Performs forward feature selection for categorical variables."""
        notify.entering(__class__.__name__, "fit")
        classname = self.__class__.__name__
        methodname = "fit"        

        # Obtain ranked list of NOMINAL features by importance
        fi = GroupedFeatureImportance(self._estimator, nominal=self._nominal).fit(X[self._nominal],y)
        self.importances_ = fi.transform(X)      
        self.num_original_features_ = self.importances_.shape[0]

        # Separate numeric features, then add nominal features one by one in order of importance
        X_numeric = X.drop(columns=self._nominal, inplace=False)
        X_nominal = X[self._nominal]        
        
        # Add features in order of descending importance and to model and cross-validate 
        self.scores_ = []
        for i in range(self.importances_.shape[0])):
            features = self.importances_[0:i+1].index
            X_new = self._encode_bind(X_numeric, X_nominal, features)          
            X_new, y = convert(classname, context, X_new, y) 
            scores = cross_validate(self._estimator, X_new, y, scoring=self._scoring)
            self.scores_.append(np.mean(scores["test_score"]))
        
        # Compute number of features based upon one standard error rule. Note that
        # scikit-learn always maximizes the objective function. If optimizing a loss function
        # scikit-learn flips the sign of the metric so that it maximizes a negative number.
        # Consequently, we need to think in terms of maximizing the negative rmse.
        std_error = sem(self.scores_)
        max_average_scores = np.max(self.scores_)
        threshold = max_average_scores - std_error
        self.scores_ = np.array(self.scores_)

        # Obtain selected features
        acceptable_num_features = np.where(self.scores_ >= threshold)        
        self.num_features_ = acceptable_num_features[0][0]        
        self.features_selected_ = self.importances_[np.arange(self.num_features_+1)]

        # Obtain discarded features
        self.num_discarded_features_ = self.num_original_features_ - self.features_selected_
        self.discarded_features_ = self.importances_[-self.num_discarded_features_]

        message = f"Model: {self._estimator.__class__.__name__} selected {self.num_features_} \
            features and discarded the following {self.num_discarded_features_} features.\
                \n{self.discarded_features_} features."
        comment.regarding(classname, message)

        # Create final dataset with numeric and nominal features selected
        X_new = self._encode_bind(X_numeric, X_nominal, self.features_selected_)
        self.encoded_features_selected_ = X_new.columns.tolist()

        notify.leaving(__class__.__name__, "forward")
        return self

    def transform(self, X, **transform_params):   
        """Transform final nominal features to one-hot-encoding."""     
        return X[self.encoded_features_selected_]
        
    def plot(self):
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10,12))
        x = range(self._min_features, self.num_features_ + self._min_features)
        y = self.scores_
        d = {"x": x, "y":y}
        df = pd.DataFrame(d)
        ax = sns.lineplot(x=x, y=y, data=df)
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Cross Validation Score (RMSE)")
        plt.title("Forward Selection")
        plt.tight_layout()
        plt.show()


# =========================================================================== #
#                         CATEGORICAL FEATURE SELECTOR                        #
# =========================================================================== #   
class CategoricalSelector(BaseEstimator, TransformerMixin):
    """Performs Categorical feature selection using forward selection."""
    def __init__(self, estimator, scoring=rmse, nominal=nominal, numeric=numeric):
        self._estimator = estimator
        self._scoring = scoring
        self._nominal = nominal
        self._numeric = numeric

    def fit(self, X, y=None):
        notify.entering(__class__.__name__, "fit")
        classname = self.__class__.__name__

        if hasattr(self._estimator, "coef_"):
            selector = GroupedForwardSelection(estimator=self._estimator, 
                                scoring=self._scoring,nominal=self._nominal)
        else:
            selector = ForwardSelection(estimator=self._estimator, 
                                scoring=self._scoring,nominal=self._nominal)

        selector.fit(X, y)
        notify.leaving(__class__.__name__, "fit")
        self.X_ = selector.transform(X)
        return self

    def transform(self, X):
        return self.X_

# =========================================================================== #
#                            FEATURE SELECTOR                                 #
# =========================================================================== #   
class FeatureSelector:
    """Performs feature selection for an estimator."""
    def __init__(self, estimator, scoring=rmse, nominal=nominal, 
                numeric=numeric):
        self._estimator = estimator
        self._scoring = scoring
        self._nominal = nominal
        self._numeric = numeric
        self._fitted = False

    def fit(self, X, y):

        # Split data into numeric and nominal features
        selector = ColumnSelector(columns=numeric)
        X_numeric = selector.fit_transform(X, y)
        selector = ColumnSelector(columns=nominal)
        X_nominal = selector.fit_transform(X, y)        

        # Numeric Feature Selection
        self.numeric_selector_ = NumericSelector(estimator=self._estimator)
        self.X_numeric = self.numeric_selector_.fit_transform(X_numeric, y)

        # Nominal Feature Selection
        self.nominal_selector_ = CategoricalSelector(estimator=self._estimator)
        self.X_ = self.nominal_selector_.fit_transform(X, y)        

        return self

    def transform(self, X, y, **fit_params):
        return self.X_

 
# =========================================================================== #
#                                  MAIN                                       #
# =========================================================================== #   
def main():
    # Obtain the data
    data = AmesData()
    X, y = data.get()        

    # Test Feature Selector
    selector = FeatureSelector(LinearRegression())
    selector.fit(X,y)
    selector.print()
    selector.plot()

    # # Perform Feature selection for regressors
    # selector = ModelFeatureSelector(regressors)
    # selector.fit(X,y)    

    # # Perform Feature selection for regressors
    # selector = ModelFeatureSelector(ensembles)
    # selector.fit(X,y)        

if __name__ == "__main__":    
    main()
#%%

