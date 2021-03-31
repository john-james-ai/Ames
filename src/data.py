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
import sys
# =========================================================================== #
#                              AMES DATA                                      #
# =========================================================================== #   
class AmesData:
    """ Obtains processed data if exists, processes raw data otherwise."""
    def __init__(self):
        self._raw_directory = "../data/raw/"
        self._train_directory = "../data/training/"
        self._processed_directory = "../data/processed/"
        self._cv_directory = "../data/cv/"
        self._X_filename = "X_train.csv"
        self._y_filename = "y_train.csv"
        self._external_source = "../data/external/Ames_data.csv"
        self._index = [range(0, 10)]
        self._next_index = 0
    
    def get_train(self):
        df = pd.read_csv(self._external_source)        
        X_train = df.drop(columns=["Sale_Price"])
        y_train = df["Sale_Price"]
        train = {"X": X_train, "y": y_train}
        return train

    def get_cv(self, idx=None):
        """Obtains processed data if extant, otherwise, processes raw data"""      

        train_filepath = self._cv_directory + str(idx) + "_train.csv"
        test_filepath = self._cv_directory + str(idx) + "_test.csv"
        test_y_filepath = self._cv_directory + str(idx) + "_test_y.csv"
        
        raw_train = pd.read_csv(train_filepath)        
        raw_test = pd.read_csv(test_filepath)
        raw_test_y = pd.read_csv(test_y_filepath)        

        # Format train
        X_train = raw_train.drop(columns=["Sale_Price"])
        y_train = raw_train["Sale_Price"]
        train = {"X": X_train, "y": y_train}
        # Format test                
        test = {"X": raw_test, "y": raw_test_y}

        return train, test






def main():
    
    meta = FeatureMetadata()
    names = meta.get_column_names()
    print(len(names))
    names = meta.get_column_names("ordinal")
    print(len(names))
    meta.exclude_feature("Lot_Area", "data_screener")
    meta.print(feature="Lot_Area")
    meta.include_feature("Lot_Area", "testing")
    meta.print(feature="Lot_Area")
    
if __name__ == "__main__":
    main()        
#%%    
