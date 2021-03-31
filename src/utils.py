# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \utils.py                                                         #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Wednesday, March 17th 2021, 9:06:12 pm                      #
# Last Modified : Wednesday, March 17th 2021, 9:06:13 pm                      #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
from abc import ABC, abstractmethod
import datetime
import glob
from joblib import dump, load
import os
import numpy as np
import pandas as pd
import pickle
from tabulate import tabulate
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_X_y, check_array

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# =========================================================================== #
#                        VALIDATION AND CONVERSION                            #
# =========================================================================== #
def check_NaN(X):
    if isinstance(X, pd.DataFrame):
        X_nan = X[X.isna().any(axis=1)]
        assert(isinstance(X_nan, pd.DataFrame))
        if X_nan.shape[0] > 0:            
            cols_w_nans = X.columns[X.isna().any()].tolist()
            data_w_nans = X.filter(items=cols_w_nans)
            data_w_nans = data_w_nans[data_w_nans.isna()]
            n_cols_w_nans = len(cols_w_nans)
            print(f"\nThere are {X_nan.shape[0]} observations with NaN values")
            print(f"There are {n_cols_w_nans} columns with NaN values")
            print("Samples\n")
            print(data_w_nans.sample(10))            
            assert(X_nan.shape[0]==0), "The above rows have NaN values."
        

def validate(classname, context, X, y=None, dtype=None):
    """Validates, but does not convert X, y."""
    message = f"Context: {context}. X shape: {X.shape}. Validation initiated."
    comment.regarding(classname, message)    
    if isinstance(X, pd.DataFrame):
        check_NaN(X)
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame):
        y = y.values.ravel()
    if y.any():
        check_X_y(X,y, dtype=dtype)
    else:
        check_array(X, dtype=dtype)

    # message = f"Context: {context}. Validation complete!"
    # comment.regarding(classname, message)        

def convert(classname, context, X, y, dtype="numeric"):
    """Validates and converts X to 2-d array and y to 1d array."""
    message = f"Context: {context}. X shape: {X.shape}. Conversion initiated."
    comment.regarding(classname, message)

    if isinstance(X, pd.DataFrame):
        check_NaN(X)
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame):
        y = y.values.ravel()
    X, y = check_X_y(X,y, dtype=dtype)
    # message = f"Context: {context}. Validation complete!"
    # comment.regarding(classname, message)    
    return X, y

# =========================================================================== #
#                                 PERSIST                                     #
# =========================================================================== #
class Persist(ABC):
    """Base class for classes that manage model and model metadata  persistence."""
    def __init__(self, directory="../models/"):
        self._directory = directory
        self._index_filename = "index.npy"
        self._file_ext = ".joblib"        

    def _gen_id(self):
        if os.path.exists(self._index_filename):
            with open(self._index_filename, 'rb') as f:
                ids = np.load(f)
        else:
            ids = np.array([])
        new_id = np.random.randint(1000,9999)
        while (new_id in ids):
            new_id = np.random.randint(1000,9999)
        np.append(ids, new_id)
        return new_id

    @abstractmethod
    def _get_item_name(self, item):
        pass

    @abstractmethod
    def _get_file_ext(self, item):
        pass

    def search(self, index):
        """Returns the filename matching the index."""
        filenames = []
        search_string = self._directory + "*" + index + "*"

        for name in glob.glob(search_string):
            filenames.append(name)
        if len(filenames) == 0:
            print("No files matching index were found.")
        elif len(filenames) == 1:
            return filenames[0]
        else:
            return filenames
        

    def _create_filepath(self, item, cv, eid):
        """Creates a filename for an item."""
        cdate = datetime.datetime.now()
        name = (self._get_item_name(item) + "_") if self._get_item_name(item) else ""        
        cv = "cv" + str(cv) + "_"
        date = cdate.strftime("%B") + "-" + str(cdate.strftime("%d")) + "-" + str(cdate.strftime("%Y")) + "_"
        index = str(self._gen_id()) 
        ext = self._get_file_ext(item)        
        filename = self._directory + name + cv + date + index + str(eid) + ext
        return filename

    @abstractmethod
    def load(self, filename):
        pass

    @abstractmethod
    def dump(self, item, descriptor=None, cv=None, eid=None):
        pass

# =========================================================================== #
#                          PERSIST ESTIMATOR                                  #
# =========================================================================== #
class PersistEstimator(Persist):
    """ Responsible for serializing and deserializing estimators."""
    def __init__(self, directory="../models/"):
        super().__init__(directory)

    def _get_item_name(self, item):
        return item.__class__.__name__    

    def _get_file_ext(self, item):
        return ".joblib"

    def load(self, filename):
        return load(filename)

    def dump(self, item, cv, eid):
        filename = self._create_filepath(item, cv, eid)
        dump(item, filename)

# =========================================================================== #
#                          PERSIST NUMPY                                      #
# =========================================================================== #
class PersistNumpy(Persist):
    """ Responsible for persisting arrays."""
    def __init__(self, directory="../tests/"):
        super().__init__(directory)

    def _get_item_name(self, item):
        return ""

    def _get_file_ext(self, item):
        return ".npy"

    def load(self, filename):
        return np.load(filename)

    def dump(self, item, descriptor=None):
        filename = self._create_filepath(item, descriptor=descriptor)
        np.save(filename, item)        

# =========================================================================== #
#                        PERSIST DATAFRAME                                    #
# =========================================================================== #
class PersistDataFrame(Persist):
    """ Responsible for persisting dataframes."""
    def __init__(self, directory="../tests/"):
        super().__init__(directory)

    def _get_item_name(self, item):
        return ""

    def _get_file_ext(self, item):
        return ".pkl"

    def load(self, filename):
        return pd.read_pickle(filename)

    def dump(self, item, descriptor=None):
        filename = self._create_filepath(item, descriptor)
        item.to_pickle(filename)   

# =========================================================================== #
#                        PERSIST DICTIONARY                                   #
# =========================================================================== #
class PersistDictionary(Persist):
    """ Responsible for persisting dictionary."""
    def __init__(self, directory="../tests/"):
        super().__init__(directory)

    def _create_filepath(self, item, descriptor):
        return self._directory + descriptor + self._get_file_ext()


    def _get_item_name(self, item):
        return ""  

    def _get_file_ext(self, item=None):
        return ".joblib"

    def load(self, filename):
        return load(filename)

    def dump(self, item, descriptor=None):
        filename = self._create_filepath(item, descriptor)
        dump(item, filename)
 
        
# --------------------------------------------------------------------------- #
def onehotmap(features, nominal):
    groups = []
    for feature in features:        
        for col in nominal:
            if col in feature:
                np.append(groups,col)
                break
    return groups            
# --------------------------------------------------------------------------- #
def print_list(items, width=5):
    for i, elem in enumerate(items):
        print(f"            {str(elem)}")        
    print("\n")    

# --------------------------------------------------------------------------- #
def print_dict(d):
    for k, v in d.items():
        if isinstance(v,(list,np.ndarray)):
            print(f"         {k}:")
            for i in v:
                print(f"          {i}")
        else:
            print(f"         {k}: {v}")
# --------------------------------------------------------------------------- #
def print_dict_keys(d):
    for k, v in d.items():
        print(f"         {k}")
# --------------------------------------------------------------------------- #
# Creates formulas to be used with statsmodels ols function.        
def get_formulas(features):
    formulas = []
    for feature in features:
        formula = "Sale_Price ~ C(" + feature + ")"
        formulas.append(formula)
    return formulas        
# --------------------------------------------------------------------------- #
class Notify:
    def __init__(self, verbose=True):
        self._verbose = verbose

    def entering(self, classname, methodname=None):
        if self._verbose:
            print(f">>>> Entering {classname}: {methodname}")

    def leaving(self, classname, methodname=None):
        if self._verbose:
            print(f"<<<< Leaving {classname}: {methodname}")
notify = Notify(verbose=True)    

class Comment:
    def __init__(self, verbose=True):
        self._verbose = verbose

    def regarding(self, classname, message=None):
        if self._verbose:
            print(f"\nClass {classname}: {message}\n")

notify = Notify(verbose=True)    
comment = Comment(verbose=True)

# --------------------------------------------------------------------------- #
def test():
    from data import AmesData
    data = AmesData()
    X, y = data.get()
    X = X[["Year_Built", "Gr_Liv_Area"]]

    directory = "../tests/"
    
    # Test Estimator
    estimator = LinearRegression().fit(X,y)
    persist = PersistEstimator(directory)
    persist.dump(estimator, "test_estimator")       
    filename = persist.search("5611")
    assert(persist.load(filename).__class__.__name__ == "LinearRegression"), "Lost something with PersistEstimator"


    # Numpy array 
    y = np.array(y.values)
    persist = PersistNumpy(directory)
    persist.dump(y,  "test_numpy")
    filename = persist.search("9434")
    assert(persist.load(filename).all() == y.all()), "Numpy Persist ain't workin'"


    # DataFrame
    persist = PersistDataFrame(directory)
    persist.dump(X,"test_dataframe")
    filename = persist.search("9634")
    assert(persist.load(filename).shape == X.shape), "Persist DataFrame ain't workin'"
    
def diagnose(X):
    print("\n\n")
    print("="*40)
    print("            Starting Diagnosis")
    print("-"*40)
    print(f"Garage_Finish:{{{X['Garage_Finish'].value_counts()}}}")
    print(f"Garage_Type:{X['Garage_Type'].unique()}")
    print(f"Garage_Cars:{X['Garage_Cars'].unique()}")
    print(f"Garage_Qual:{X['Garage_Qual'].unique()}")
    print(f"Garage_Cond:{X['Garage_Cond'].unique()}")
    print("\n")
    print(f"Has_Garage:{X['Has_Garage'].unique()}")
    print(f"Has_Pool:{X['Has_Pool'].unique()}")
    print(f"Has_Basement:{X['Has_Basement'].unique()}")
    print(f"Has_Fireplace:{X['Has_Fireplace'].unique()}")
    print(f"Has_Porch:{X['Has_Porch'].unique()}")


    
    print("            Diagnosis Complete")
    print("="*40)

if __name__ == "__main__":        
    test()
#%%
