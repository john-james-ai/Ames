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
import scipy.stats as stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tabulate import tabulate
import pprint

from globals import nominal, ordinal_map
from data import AmesData
from tabulate import tabulate
import pprint
from pipeline_v1 import ordinal, continuous, discrete, nominal
# =========================================================================== #
#                             FEATURE METADATA                                #
# =========================================================================== #
def feature_metadata(X):
    features = pd.DataFrame()
    for feature in X.columns:
        ftype = "Ordinal" if feature in ordinal else \
            "Nominal" if feature in nominal else \
                "Continuous" if feature in continuous else \
                    "Discrete"
        d = {"Feature": feature, "Type": ftype, "Source": "Original", 
             "Active": True, "Signature":""}
        df = pd.DataFrame(data=d, index=[0])
        features = pd.concat((features,df), axis=0)    
    return features
# =========================================================================== #
#                        CATEGORICAL FEATURE METADATA                         #
# =========================================================================== #
def categorical_metadata(X):
    summary = pd.DataFrame()
    detail = pd.DataFrame()

    for var_type, variables in categoricals.items():        
        detail = pd.DataFrame()
        filename = var_type + ".csv"
        for variable in variables:
            # Summary information
            d = {"Type": var_type, "Variable": variable, "nLevels": X[variable].nunique()} 
            df = pd.DataFrame(data=d,index=[0])
            summary = pd.concat([summary,df], axis=0)
            # Detail
            df = pd.DataFrame(data=X.value_counts(subset=[variable])).reset_index()
            df.columns = ["Levels", "Counts"]
            df["Type"] = var_type
            df["Variable"] = variable
            df = df[["Type", "Variable", "Levels", "Counts"]]            
            detail = pd.concat([detail,df], axis=0)
        detail.to_csv(os.path.join(data_paths["metadata"],filename), index=False)

    summary.to_csv(os.path.join(data_paths["metadata"],"categorical_summary.csv"), index=False)    
    print(f"{detail.shape[0]} categorical variables and levels.")        
    print(summary)

def create_ordinal_map():
    filename = "ordered.csv"
    codes = pd.read_csv(os.path.join(data_paths["metadata"], filename))
    ordinal_map = {}
    levels = {}

    variables = codes.groupby("Variable")
    for variable, group in variables:
        levels = dict(zip(group["Levels"], group["Order"]))
        ordinal_map[variable] = levels
    pp = pprint.PrettyPrinter(compact=True, width=100)
    pp.pprint(ordinal_map)
    
def create_nominal_map():
    filename = "categorical_summary.csv"
    categoricals = pd.read_csv(os.path.join(data_paths["metadata"], filename))
    nominals = categoricals[categoricals["Type"]=="nominal"]["Variable"]
    pp = pprint.PrettyPrinter(compact=True, width=100)
    pp.pprint(list(nominals))





# =========================================================================== #
#                        CATEGORICAL FEATURE SELECTION                        #
# =========================================================================== #
def create_formula_univariate(feature):
    formula = "Sale_Price ~ C(" + feature + ")"
    return formula

def create_formula_multivariate(features=nominal):
    formula = "Sale_Price ~ "
    n_features = len(nominal)
    for i, feature in enumerate(nominal):
        formula += "C(" + feature + ")"
        if i < n_features - 1:
            formula += " + "    
    return formula
class Catalyst:
    """Categorical feature analysis."""
    def __init__(self, nominal=nominal):
        self._nominal = nominal
        self._anova = pd.DataFrame()
        self._importance = pd.DataFrame()
        self._importance_orig = pd.DataFrame()
        self._coef = pd.DataFrame()

    def anova(self, X, y):
        """Performs Anova testing on nominal categorical features."""
        X["Sale_Price"] = np.log(y["Sale_Price"].values)
        for feature in self._nominal:
            formula = create_formula_univariate(feature)
            model = ols(formula, data=X).fit()
            anova_table = sm.stats.anova_lm(model, typ=3)
            d = {"Feature": feature, "F": anova_table["F"][0], "PR(>F)": anova_table["PR(>F)"][0]}
            df = pd.DataFrame(data=d, index=[0])
            self._anova = pd.concat((self._anova, df), axis=0)
            
    def regression(self, X, y):
        """Performs OLS on all predictors and presents coefficients."""
        X["Sale_Price"] = np.log(y["Sale_Price"].values)
        formula = create_formula_multivariate()
        # X = pd.get_dummies(X[nominal])
        # model = sm.OLS(np.asarray(y["Sale_Price"].values), X)
        model = ols(formula, data=X).fit()        
        print(model.summary())

    def multicollinearity(self, X, y, threshold=5):
        """Recursively eliminates variables with highest VIF below threshold."""        
        X = pd.get_dummies(X[nominal])
        def calc_vif(X):
            vif = pd.DataFrame()
            vif["Feature"] = X.columns
            vif["VIF"] = [variance_inflation_factor(np.asarray(X.values), i) for i in range(X.shape[1])]
            return vif
        vif = calc_vif(X)
        while(max(vif["VIF"]) > threshold):
            vif = vif.sort_values(by="VIF", ascending=False)
            X = X.drop([vif["Feature"][0]], axis=1)
            vif = calc_vif(X)
        print(tabulate(vif, headers="keys", showindex=False))    

    def importance(self, X, y):
        # Prepare Data
        y["Sale_Price"] = np.log(y["Sale_Price"].values)
        X = X[nominal]        
        X = pd.get_dummies(X)
        groups = onehotmap(X.columns, nominal) # Returns original column for each dummy

        # Instantiate the decision tree and store results in dataframe
        tree = DecisionTreeRegressor().fit(X, y)        
        d = {"Original":  groups, "Feature": X.columns, "Importance": tree.feature_importances_}
        self._importance = pd.DataFrame(data=d)
        
        # Aggregate, summarize and sort mean importance by original column name
        self._importance_orig = self._importance.groupby("Original").mean().reset_index()        
        self._importance_orig.sort_values(by=["Importance"], inplace=True, ascending=False)
        print(tabulate(self._importance_orig, headers="keys", showindex=False))        

    def get_None_columns(self,X):
        print(X.columns[X.isna().any()].tolist())
        print(X["Garage_Yr_Blt"].describe()).T
        print(X.shape[0])
        print(len(X['Garage_Yr_Blt'].isna()))
        print(sum(X['Garage_Yr_Blt'].isna()))

# =========================================================================== #
#                             DESCRIPTIVE STATISTICS                          #
# =========================================================================== #    
def describe(X):
    print("\n\nNumeric Features")
    #df = X.describe(percentiles=[.5],include=[np.number]).T
    df = X.describe(percentiles=[0.5], include=[np.number]).apply(lambda s: s.apply(lambda x: format(x, 'g'))).T
    print(df)
    print("\n\nCategorical Features")
    df = X.describe(exclude=[np.number]).T    
    print(df)
    
def outliers(X):
    out_gr_liv_area = X[X["Gr_Liv_Area"]>4000].shape[0]
    out_garage_yr_blt = X[X["Garage_Yr_Blt"]>2010].shape[0]
    X = X[X["Gr_Liv_Area"]<=4000]
    out_garage_yr_blt2 = X[X["Garage_Yr_Blt"]>2010].shape[0]
    print(f"There are {out_gr_liv_area} homes with extremely large living areas ")
    print(f"There are {out_garage_yr_blt} homes with garages from the future ")
    print(f"There are {out_garage_yr_blt2} homes with garages from the future after removing large homes")

def find_NAs(X):
        print(X.columns[X.isna().any()].tolist())
        print(X["Garage_Yr_Blt"].describe().T)
        print(X.shape[0])
        print(len(X['Garage_Yr_Blt'].isna()))
        print(sum(X['Garage_Yr_Blt'].isna()))
        df = X['Garage_Yr_Blt'].isna()
        print(np.unique(df, return_counts=True))
        df['Garage'] = np.where(df, 'Garage', 'No Garage')
        print(np.unique(df['Garage'], return_counts=True))

def check_unique():
    filepath = "../data/external/Ames_data.csv"
    X = pd.read_csv(filepath)
    df = X.select_dtypes(include=[object])
    cols = {}
    counts = {}
    for column in df.columns:
        if column in ordinal_map.keys():
            dtype = "(Ordinal)"
            values = df[column].unique()
            print(f"Column: {column} {dtype}")
            for value in values:
                count = df[df[column]==value].shape[0]
                if value in ordinal_map[column].keys():
                    n_value = ordinal_map[column][value]                    
                    print(f"   {n_value}: {value}  Count: {count}")
                else:
                    print(f"   {value} Count: {count} Missing from ordinal map")
        else:
            dtype = "(Nominal)"
            print(f"Column: {column} {dtype}")
            values = df[column].unique()
            for value in values:
                count = df[df[column]==value].shape[0]
                print(f"   {value} Count: {count}")
        print("\n\n")
def main():
    filename = "../data/external/Ames_data.csv"
    X = pd.read_csv(filename)
    X.drop(columns="Sale_Price", inplace=True)
    metadata = feature_metadata(X)
    print(metadata)
    filename = "../data/metadata/feature_metadata.csv"
    metadata.to_csv(filename, index=False)
    

if __name__ == "__main__":
    main()
#%%
        




