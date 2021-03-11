# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \visualize.py                                                     #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Monday, March 8th 2021, 11:46:37 pm                         #
# Last Modified : Wednesday, March 10th 2021, 5:03:23 am                      #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from globals import ordinal_vars, nominal_vars, continuous_vars, discrete_vars, test_vars
from globals import data_paths

# --------------------------------------------------------------------------- #
class NominalVars:
    def __init__(self):
        self._X = None
        self._nominal_vars = nominal_vars
        self._theme = sns.set_theme(style="whitegrid")

    def fit(self, X, y=None):
        self._X = X
        self._y = y
        df = self._X[self._nominal_vars]
        self._counts = df.value_counts()

    def plot(self):
        fig, axes =plt.subplots(2,1, figsize=(12,10), sharex=False)
        axes = axes.flatten()        
        for ax, variable in zip(axes, self._nominal_vars):
            sns.countplot(y=variable, data=self._X, ax=ax)
            #ax.set_xticklabels(ax   .get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.tight_layout()  
        plt.show()        

def main():    
    path = data_paths['processed'] + "/X_train.csv"
    X = pd.read_csv(path)
    nominal = NominalVars()
    nominal.fit(X)
    nominal.plot()
    

if __name__ == "__main__":
    main()
#%%



