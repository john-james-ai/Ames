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
from sklearn.linear_model import LinearRegression

# =========================================================================== #
#                              PERSIST                                        #
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
        

    def _create_filepath(self, item, descriptor=None):
        """Creates a filename for an item."""
        cdate = datetime.datetime.now()
        name = (self._get_item_name(item) + "_") if self._get_item_name(item) else ""
        descriptor = (descriptor + "_") if descriptor else ""
        date = cdate.strftime("%B") + "-" + str(cdate.strftime("%d")) + "-" + str(cdate.strftime("%Y")) + "_"
        index = str(self._gen_id()) 
        ext = self._get_file_ext(item)        
        filename = self._directory + name + descriptor + date + index + ext
        return filename

    @abstractmethod
    def load(self, filename):
        pass

    @abstractmethod
    def dump(self, item, descriptor=None):
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

    def dump(self, item, descriptor=None):
        filename = self._create_filepath(item, descriptor)
        dump(item, filename)

# =========================================================================== #
#                          PERSIST NUMPY                                      #
# =========================================================================== #
class PersistNumpy(Persist):
    """ Responsible for persisting arrays."""
    def __init__(self, directory="../models/"):
        super().__init__(directory)

    def _get_item_name(self, item):
        return ""

    def _get_file_ext(self, item):
        return ".npy"

    def load(self, filename):
        return np.load(filename)

    def dump(self, item, descriptor=None):
        filename = self._create_filepath(item, descriptor)
        np.save(filename, item)        

# =========================================================================== #
#                        PERSIST DATAFRAME                                    #
# =========================================================================== #
class PersistDataFrame(Persist):
    """ Responsible for persisting dataframes."""
    def __init__(self, directory="../models/"):
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
class Notify:
    def __init__(self, verbose=True):
        self._verbose = verbose

    def entering(self, classname, methodname=None):
        if self._verbose:
            print(f">>>>Entering {classname}: {methodname}")

    def leaving(self, classname, methodname=None):
        if self._verbose:
            print(f"<<<<Leaving {classname}: {methodname}")
notify = Notify(verbose=True)    

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
    


if __name__ == "__main__":    
    test()
#%%