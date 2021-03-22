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
        self._index = [np.random.randint(1, 11) for p in range(0, 10)]
        self._next_index = 0
        
    def get(self, idx=None):
        """Obtains processed data if extant, otherwise, processes raw data"""
        idx = self._index[self._next_index]
        train_filepath = self._cv_directory + str(idx) + "_train.csv"
        test_filepath = self._cv_directory + str(idx) + "_test.csv"
        
        train = pd.read_csv(train_filepath)
        X_train = train.drop(columns=["Sale_Price"])
        y_train = train["Sale_Price"]

        X_test = pd.read_csv(test_filepath)
        self._next_index = self._next_index + 1 if self._next_index < 10 else 0 

        assert(X_train.shape[0]==y_train.shape[0]), f"X_train and y_train mismatched lengths {X_train.shape[0]} and {y_train.shape[0]}, respectively."        

        return X_train, y_train, X_test

def main():
    data = AmesData()
    X_train, y_train, X_test = data.get()

if __name__ == "__main__":
    main()        
#%%    
