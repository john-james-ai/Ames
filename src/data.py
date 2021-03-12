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
from tabulate import tabulate
from globals import data_paths
from data_builder import DataBuilder
# --------------------------------------------------------------------------- #
class Ames:
    """Encapsulates all operations on the Ames Housing Database."""
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def build(self):
        builder = DataBuilder()
        builder.build_data()
        self.X_train = builder.X_train
        self.y_train = builder.y_train
        self.X_test = builder.X_test
        self.y_test = builder.y_test

    def get(self, train_only=True, stage="interim"):
        if train_only:
            if not self.X_train or not self.y_train:
                return self.read(train_only, stage)
            else:
                return self.X_train, self.y_train
        else:
            if not self.X_train or not self.y_train or \
                not not self.X_test or not self.y_test:
                return self.read(train_only, stage)
            else:
                return self.X_train, self.y_train, self.X_test, self.y_test

    def process(self, X, y):
        pass


    def read(self, train_only=True, stage="interim"):

        self.X_train = pd.read_csv(os.path.join(data_paths[stage], "X_train.csv"))
        self.y_train = pd.read_csv(os.path.join(data_paths[stage], "y_train.csv"))
        if train_only:
            return self.X_train, self.y_train
        else:
            self.X_test = pd.read_csv(os.path.join(data_paths[stage], "X_test.csv"))
            self.y_test = pd.read_csv(os.path.join(data_paths[stage], "y_test.csv"))
            return self.X_train, self.y_train, self.X_test, self.y_test

    def write(self, X, y, train=True, stage="processed"):
        if train:
            X_path = data_paths[stage] + "X_train.csv"
            y_path = data_paths[stage] + "y_train.csv"
        else:
            X_path = data_paths[stage] + "X_test.csv"
            y_path = data_paths[stage] + "y_test.csv"
        
        X.to_csv(X_path, index=False)            
        y.to_csv(y_path, index=False)            

def main():
    ames = Ames()
    ames.build()   

if __name__ == "__main__":
    main()
#%%
        




