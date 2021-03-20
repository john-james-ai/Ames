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
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

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
from data_processor import NominalEncoder
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
        comment.regarding(classname, methodname, message)
        
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
    """Returns a list of features in descending order of importance."""
    def __init__(self, estimator, nominal=nominal):
        self._estimator = estimator
        self._nominal = nominal

    def fit(self, X, y, **fit_params):
        notify.entering(__class__.__name__, "fit")   
        classname = self.__class__.__name__
        methodname = "fit"
        
        # Prepare Data
        enc = NominalEncoder()
        enc.fit(X)
        X = enc.transform(X)
        original_features = enc.get_original(enc.transformed_features_)      

        # Validate data
        message = "Pending validation of data after nominal encoding"
        comment.regarding(classname, methodname, message)
        validate(X,y)    
        message = "Completed validation of data after nominal encoding"
        comment.regarding(classname, methodname, message)

        # Instantiate the estimator and store results in dataframe
        model = self._estimator.fit(X, y)  
        if hasattr(self._estimator,"coef_"):
            importances = model.coef_.flatten()        
        else:
            importances = model.feature_importances_.flatten()

        d = {"Feature":  original_features, "Level": enc.transformed_features_, "Importance": importances}
        df = pd.DataFrame(data=d)
        
        # Aggregate, summarize and sort mean importance by original column name
        self.importance_ = df.groupby("Feature").mean().reset_index()        
        self.importance_.sort_values(by=["Importance"], inplace=True, ascending=False)
        self.ranked_features_ = self.importance_["Feature"].values

        message = f"Feature Importances\n{self.importance_}"
        comment.regarding(classname, methodname, message)

        notify.leaving(__class__.__name__, "fit")    
        return self

    def transform(self, X, **transform_params):
        return self.ranked_features_

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
class ForwardSelector(BaseEstimator, TransformerMixin):
    """Performs nominal feature selection using forward selection."""
    def __init__(self, estimator, scoring=rmse, nominal=nominal):
        self._estimator = estimator
        self._scoring = scoring                
        self._nominal = nominal

    def fit(self, X, y, **fit_params):
        """Performs forward feature selection for categorical variables."""
        notify.entering(__class__.__name__, "fit")
        classname = self.__class__.__name__
        methodname = "fit"        

        # Obtain ranked list of NOMINAL features by importance
        fi = FeatureImportance(self._estimator, nominal=self._nominal).fit(X[self._nominal],y)
        ranked_features = fi.transform(X)        

        # Separate numeric features, then add nominal features one by one in order of importance
        X_numeric = X.drop(columns=self._nominal, inplace=False)
        X_nominal = X[self._nominal]

        # Create nominal encoder object 
        enc = NominalEncoder()
        
        # Add features in order of descending importance and to model and cross-validate 
        self.scores_ = []
        for i in range(len(ranked_features)):
            features = ranked_features[0:i+1]
            message = f"Currently processing features: {features}"
            comment.regarding(classname, methodname, message)
            X_nominal_transformed = enc.fit_transform(X_nominal[features],y)

            X_new = pd.concat((X_numeric, X_nominal_transformed), axis=1)
            X_new, y = convert(X_new, y)
            scores = cross_validate(self._estimator, X_new, y, scoring=self._scoring)
            self.scores_.append(np.mean(scores["test_score"]))
        
        # Compute number of features based upon one standard error rule
        std_error = sem(self.scores_)
        mean_average_scores = np.mean(self.scores_)
        threshold = mean_average_scores - std_error
        self.num_features_ = np.where(scores["test_score"] >= threshold)[0]
        
        self.features_selected_ = ranked_features[0:self.num_features_+1]
        message = f"Model: {self._estimator.__class__.__name__} selected {len(self.features_selected_)} features."
        comment.regarding(classname, methodname, message)

        notify.leaving(__class__.__name__, "forward")
        return self

    def transform(self, X, **transform_params):        
        return X[self.features_selected_]
        
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
#                            FEATURE SELECTOR                                 #
# =========================================================================== #   
class FeatureSelector:
    """Performs feature selection for an estimator."""
    def __init__(self, estimator, scoring=rmse, nominal=nominal, 
                numeric=numeric, persist=True):
        self._estimator = estimator
        self._scoring = scoring
        self._nominal = nominal
        self._numeric = numeric
        self._persist = persist
        self._fitted = False

    def _select_features(self, X, y):

        # Split data into numeric and nominal features
        selector = ColumnSelector(columns=numeric)
        X_numeric = selector.fit_transform(X, y)
        selector = ColumnSelector(columns=nominal)
        X_nominal = selector.fit_transform(X, y)        

        # Numeric Feature Selection
        self.numeric_selector_ = NumericSelector(estimator=self._estimator)
        X_numeric = self.numeric_selector_.fit_transform(X_numeric, y)

        # Nominal Feature Selection
        self.nominal_selector_ = ForwardSelector(estimator=self._estimator)
        X_nominal = self.nominal_selector_.fit_transform(X, y)        

        # Store features selected
        numeric_features = X_numeric.columns.tolist()
        nominal_features = X_nominal.columns.tolist()
        self.features_selected_ = numeric_features + nominal_features  
        self.n_features_ = len(self.features_selected_)              

        # Merge Numeric and Nominal Columns
        return pd.concat((X_numeric,X_nominal), axis=1)

    def _refit(self, X, y):
        # Fit the estimator on the selected features
        notify.entering(__class__.__name__, "refit")        
        classname = self.__class__.__name__
        methodname = "refit"                
        
        self._estimator.fit(X, y)

        # Store feature importances in terms of parameter weights
        parameters = ["Intercept"] + X.columns
        coef = [self.estimator_.intercept_] + list(self.estimator_.coef_)
        d = {"Estimator": self.estimator_.__class__.__name__, "Parameter": parameters, "|Estimates|": np.abs(coef)}
        self.parameters_ = pd.DataFrame(data=d)            
        self.parameters_.sort_values(by=["|Estimates|"], inplace=True, ascending=False)        

    def _save(self, X, y):
        # Persist the data
        train = pd.concat((X,y), axis=1)
        persist = PersistDataFrame(directories["data"]["training"])
        persist.dump(train, self._estimator.__class__.__name__)    

        # Persist the estimator 
        persist = PersistEstimator(directories["features"])    
        persist.dump(self.estimator_, self.__class__.__name__)                


    def fit(self, X, y, **fit_params):
        """Fits the selector and determines features."""
        notify.entering(__class__.__name__, "fit")        
        classname = self.__class__.__name__
        methodname = "fit"        

        self.X_ = self._select_features(X,y.values)
        self._refit(self.X_,y)
        if self._persist: self._save(self.X_, y.values)
        self._fitted = True
        notify.leaving(__class__.__name__, "fit")
        
    def print(self):
        if self._fitted:
            print("\n")
            print("="*40)
            print(tabulate(self.parameters_, headers="keys"))
            print("="*40)
        else:
            print("The selector has not yet been fitted.")

    def plot(self):
        if self._fitted:


            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(10,12))        
            ax = sns.lineplot(x="|Estimates|", y="Parameter", data=self.parameters_)
            plt.xlabel("Feature Importance")
            plt.ylabel("Feature")
            plt.title(f"{self.estimator_.__class__.__name__} Feature Importance")
            plt.tight_layout()
            plt.show()               


# =========================================================================== #
#                         MODEL FEATURE SELECTOR                              #
# =========================================================================== #   
class ModelFeatureSelector:
    """Performs feature selection for all estimators."""
    def __init__(self, estimators, scoring=rmse, nominal=nominal, 
                numeric=numeric, persist=True):
        self._estimators = estimators
        self._scoring = scoring
        self._nominal = nominal
        self._numeric = numeric
        self._persist = persist
        self._fitted = False

    def fit(self, X, y, **fit_params):
        """Fits the feature selector on all estimators"""
        notify.entering(__class__.__name__, "fit")        
        classname = self.__class__.__name__
        methodname = "fit"                
        self.selectors_ = {}
        self.model_features_ = pd.DataFrame()  # Estimator, feature count
        
        for name, estimator in estimators.items():            
            # Refresh data
            X_train = X
            y_train = y            

            message = f"Now Processing Model {estimator.__class__.__name__}."
            comment.regarding(classname, methodname, message)
        
            selector = FeatureSelector(estimator, self._scoring, self._nominal, 
                                       self._numeric, self._persist)
            selector.fit(X_train, y_train)
            self.model_features_ = pd.concat((self.model_features_,selector.parameters_), axis=0)
            self.selectors_[name] = selector
        self._fitted = True
        notify.leaving(__class__.__name__, "fit")        

    def get_selector(self, estimator_label=None):
        if estimator_label:
            return self.selectors_[estimator_label]
        else:
            return self.selectors_
    
    def print(self):
        if self._fitted:
            print("\n")
            print("="*40)
            print(tabulate(self.model_features_, headers="keys"))
            print("="*40)
        else:
            print("The selector has not yet been fitted.")        

# =========================================================================== #
#                                  MAIN                                       #
# =========================================================================== #   
def main():
    # Obtain the data
    data = AmesData()
    X, y = data.get(force=True)        

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

