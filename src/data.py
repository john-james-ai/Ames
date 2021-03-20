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
from data_processor import DiscretePreprocessor, OrdinalEncoder, TargetTransformer
from data_processor import DataScreener, DataAugmentor, DataCleaner
from utils import notify, validate, comment

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# =========================================================================== #
#                              AMES DATA                                      #
# =========================================================================== #   
class AmesData:
    """ Obtains processed data if exists, processes raw data otherwise."""
    def __init__(self):
        self._raw_directory = "../data/raw/"
        self._train_directory = "../data/training/"
        self._processed_directory = "../data/processed/"
        self._X_filename = "X_train.csv"
        self._y_filename = "y_train.csv"
        
    
    def process(self, X, y, **transform_params):
        """Screens, preprocesses and transforms the data."""
        notify.entering(__class__.__name__, "process")

        # Clean data 
        cleaner = DataCleaner()
        X, y = cleaner.run(X,y)
        
        # Screen data of outliers and non-informative features
        screener = DataScreener()
        X, y = screener.run(X, y)        

        # Perform data augmentation
        augmentor = DataAugmentor()
        X, y = augmentor.run(X, y)      

        # Transform Target
        x4mr = TargetTransformer()
        x4mr.fit(y)                    
        y = x4mr.transform(y)    

        # Validate data
        message = "Pending validation of data after cleaning, screening and augmentation"
        comment.regarding(__class__.__name__, "process", message)
        validate(X,y)    
        message = "Completed validation of data after cleaning, screening and augmentation"
        comment.regarding(__class__.__name__, "process", message)        

        # Execute feature preprocessors
        preprocessors = [ContinuousPreprocessor(), 
                         CategoricalPreprocessor(), DiscretePreprocessor(),
                         OrdinalEncoder()]        
        for preprocessor in preprocessors:
            x4mr = preprocessor
            x4mr.fit(X, y)
            X = x4mr.transform(X)

        # Validate data
        message = "Pending validation of data after preprocessing"
        comment.regarding(__class__.__name__, "process", message)
        validate(X,y)    
        message = "Completed validation of data after preprocessing"
        comment.regarding(__class__.__name__, "process", message)        

        # Save data
        X_filepath = self._processed_directory + self._X_filename
        y_filepath = self._processed_directory + self._y_filename
        X.to_csv(X_filepath)
        y.to_csv(y_filepath)

        notify.leaving(__class__.__name__, "process")        
        return X, y

    def get(self, force=False):
        """Obtains processed data if extant, otherwise, processes raw data"""
        X_filepath = self._processed_directory + self._X_filename
        if os.path.exists(X_filepath) and not force:
            y_filepath = self._processed_directory + self._y_filename
            X = pd.read_csv(X_filepath)
            y = pd.read_csv(y_filepath)
        else:
            X_filepath = self._raw_directory + self._X_filename
            y_filepath = self._raw_directory + self._y_filename
            X = pd.read_csv(X_filepath)
            y = pd.read_csv(y_filepath)
            X, y = self.process(X,y)
        
        return X, y
#%%
        




