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
from abc import ABC, abstractmethod
import datetime
import glob
from joblib import dump, load
import os
import pickle
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
class Persist(ABC):
    """Base class for classes that manage model and model metadata  persistence."""
    def __init__(self):
        self._directory = "../models/"
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
        ids.append(new_id)
        return new_id

    @abstractmethod
    def _get_item_name(self, item):
        pass

    def search(self, index):
        

    def _create_filepath(self, item, descriptor):
        """Creates a filename for an item."""
        index = self._gen_id()
        cdate = datetime.datetime.now()
        month = cdate.strftime("%B")
        day = cdate.strftime("%d")
        year = cdate.strftime("%Y")
        name = _get_item_name()
        filename = self._directory    + name + "_" + descriptor + "_" + str(month) + \
                    "-" + str(day) + "-" + str(year) + \
                        "_" + str(index) +  self._file_ext
        return filename

    @abstractmethod
    def load(self, index):
        pass

    @abstractmethod
    def dump(self, object, descriptor=None):
        filename = self._create_filepath(item, )



    def load(self, name, day, index, month='March'):
        """Loads an item from file."""
        filename = self._get_filename(name, day, index, month)
        if os.path.exists(filename):
            with open(filename, "rb") as handler:
                item = pickle.load(handler)
            return item
        else:
            print(f"File {filename} does not exist.")


    def dump(self, item, descriptor):
        filename = self._set_filename(item, descriptor)
        with open(filename, "wb") as handler:
            pickle.dump(item,handler, protocol=pickle.HIGHEST_PROTOCOL)
# --------------------------------------------------------------------------- #
def onehotmap(features, nominal):
    groups = []
    for feature in features:        
        for col in nominal:
            if col in feature:
                groups.append(col)
                break
    return groups            