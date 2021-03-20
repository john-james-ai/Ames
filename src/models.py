# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \model.py                                                         #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Thursday, March 18th 2021, 12:48:52 am                      #
# Last Modified : Thursday, March 18th 2021, 12:49:41 am                      #
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

# Global Variables
from globals import discrete, continuous, numeric, n_nominal_levels
from globals import nominal, ordinal, ordinal_map
from globals import regressors, regressor_parameters, ensembles, ensemble_parameters

# Local modules
from data import AmesData
from data_processor import NominalEncoder, OrdinalEncoder
from data_processor import ContinuousPreprocessor, DiscretePreprocessor, CategoricalPreprocessor
from data_processor import DataScreener, TargetTransformer
from feature_selection import FeatureSelector
from metrics import rmse
from utils import notify, Persist


# =========================================================================== #
#                           MODEL EVALUATER                                   #
# =========================================================================== #
class ModelEvaluator:
    """Evaluates, stores and reports model performance."""
    def __init__(self, X, y, nominal=nominal, numeric=numeric, scoring=rmse):
        self.X = X
        self.y = y
        self._nominal = nominal
        self._numeric = numeric
        self._scoring = scoring
        self._scores = pd.DataFrame()
    
    def evaluate(self, estimators, parameters):
        """Performs model training, tuning and evaluation."""
        for name, estimator in estimators.items():
            print(f"\nNow training {name}")
            print(f"Estimator\n{estimator}")

            X = self.X.copy()
            y = self.y.copy()

            # Create Pipeline steps
            steps = [("numeric_features", 
                      FeatureSelector(estimator=estimator, feature_type="numeric",
                                      scoring=self._scoring,
                                      nominal=self._nominal,
                                      numeric=self._numeric)),
                    ("categorical_features", 
                     FeatureSelector(estimator=estimator, feature_type="categorical",
                                      scoring=self._scoring,
                                      nominal=self._nominal,
                                      numeric=self._numeric)),
                    ("nominal_encoder", NominalEncoder(nominal=self._nominal)),                    
                    ("estimator", estimator)]

            # Update parameters to include feature selection parameters
            parameters[name].update({
                "numeric_features__estimator": [estimator],
                "numeric_features__feature_type": ["numeric"],
                "numeric_features__scoring": [self._scoring],
                "numeric_features__nominal": [self._nominal],
                "numeric_features__numeric": [self._numeric]})

            parameters[name].update({
                "categorical_features__estimator": [estimator],
                "categorical_features__feature_type": ["categorical"],
                "categorical_features__scoring": [self._scoring],
                "categorical_features__nominal": [self._nominal],
                "categorical_features__numeric": [self._numeric]})

            parameters[name].update({
                "nominal_encoder__nominal": self._nominal})

            # Obtain parameters for estimator
            param_grid = parameters[name]	

            # Create pipeline object
            pipeline = Pipeline(steps=steps)  
            print(f"Pipeline\n{pipeline}")         		

            # Initialize and fit GridSearchCV object.
            gscv = GridSearchCV(pipeline,param_grid, cv=5, n_jobs=2, scoring=self._scoring, verbose=1)
            gscv.fit(X, y.values)

            # Store model scores
            d = {"Estimator": gscv.best_estimator_.__class__.__name__, 
                 "Best Index": gscv.best_index_, 
                 "Best Score": gscv.best_score_}
            df = pd.DataFrame(data=d, index=[0])
            self._scores = pd.concat((self._scores, df), axis=0)

            # Save Gridsearch CV and best model. 
            persistence = Persistence()
            persistence.dump(gscv)
            persistence.dump(gscv.best_estimator_)

        self.print()

    def print(self):
        print(tabulate(self._scores, showindex=False))

# =========================================================================== #
#                                12. MAIN                                     #
# =========================================================================== #   
def main():
    # Obtain the data
    data = AmesData()
    X, y = data.get()   

    # Train, tune and evaluate regressors
    evaluator = ModelEvaluator(X,y)
    evaluator.evaluate(regressors,regressor_parameters)


if __name__ == "__main__":    
    main()
#%%
