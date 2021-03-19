# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \eda.py                                                           #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Tuesday, March 9th 2021, 11:06:05 pm                        #
# Last Modified : Tuesday, March 9th 2021, 11:06:05 pm                        #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
import os
import pandas as pd
import numpy as np

from data_processor import ContinuousPreprocessor, CategoricalPreprocessor
from data_processor import DiscretePreprocessor, OrdinalEncoder
from data_processor import DataScreener
from utils import notify
# =========================================================================== #
#                              AMES DATA                                      #
# =========================================================================== #   
class AmesData:
    """ Obtains processed data if exists, processes raw data otherwise."""
    def __init__(self):
        self._train_directory = "../data/train/"
        self._processed_directory = "../data/processed/"
        self._X_filename = "X_train.csv"
        self._y_filename = "y_train.csv"
        
    
    def process(self, X, y, **transform_params):
        """Screens, preprocesses and transforms the data."""
        notify.entering(__class__.__name__, "transform")
        
        # Screen data of outliers and non-informative features
        screener = DataScreener()
        screener.fit(X, y)
        X, y = screener.transform(X, y)

        # Execute feature preprocessors
        preprocessors = [ContinuousPreprocessor(), 
                         CategoricalPreprocessor(), DiscretePreprocessor(),
                         OrdinalEncoder()]        
        for preprocessor in preprocessors:
            x4mr = preprocessor
            x4mr.fit(X, y)
            X = x4mr.transform(X)

        # Transform Target
        x4mr = TargetTransformer()
        x4mr.fit(y)                    
        y = x4mr.transform(y)

        # Save data
        X_filepath = self._processed_directory + self._X_filename
        y_filepath = self._processed_directory + self._y_filename
        X.to_csv(X_filepath)
        y.to_csv(y_filepath)

        notify.leaving(__class__.__name__, "transform")        
        return X, y

    def get(self):
        """Obtains processed data if extant, otherwise, processes raw data"""
        X_filepath = self._processed_directory + self._X_filename
        if os.path.exists(X_filepath):
            y_filepath = self._processed_directory + self._y_filename
            X = pd.read_csv(X_filepath)
            y = pd.read_csv(y_filepath)
        else:
            X_filepath = self._train_directory + self._X_filename
            y_filepath = self._train_directory + self._y_filename
            X = pd.read_csv(X_filepath)
            y = pd.read_csv(y_filepath)
            X, y = self.process(X,y)
        
        return X, y
#%%
        




