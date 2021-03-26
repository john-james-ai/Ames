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
sns.set(style="whitegrid")
#sns.set_palette("Blues_d")
from analysis import Analysis
# --------------------------------------------------------------------------- #
class Performance:
    def __init__(self, filepath="../reports/scores.csv"):
        self._filepath = filepath        
        self.read()

    def read(self):
        self.scores_ = pd.read_csv(self._filepath)        

    def eval(self):
        fig, ax = plt.subplots(figsize=(12,8))
        # Obtain grouped values
        scores=self.scores_.groupby(["Group", "Estimator"]).mean().reset_index()
        # Render test scores vis-a-vis the evaluation metric
        ax = sns.barplot(x="Estimator",y="Test RMSE",hue="Group", data=self.scores_)   
        # Print values 
        for index, row in scores.iterrows():
            ax.text(index,row["Test RMSE"], round(row["Test RMSE"],3), color='black', ha="center")                     
        #Drawing a horizontal line at point 1.25
        ax.axhline(y=0.125, color="red")
        # Title
        ax.set_title("Regression Algorithm Performance\nAmes Housing Prediction\nRoot Mean Squared Error of Log Price")
        #The plot is shown
        #plt.tight_layout()
        plt.show()        



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
    performance = Performance()
    performance.eval()    
    

if __name__ == "__main__":
    main()
#%%



