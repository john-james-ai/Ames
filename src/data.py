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
        X_test = raw_test
        y_test = raw_test_y
        test = {"X": X_test, "y": y_test}

        return train, test



# =========================================================================== #
#                              METADATA                                       #
# =========================================================================== # 
class FeatureMetadata:
    def __init__(self, filename="../data/metadata/feature_metadata.csv"):
        self._filename = filename
        self.fm_ = pd.read_csv(self._filename) # Feature Metadata

    def get_feature(self, feature):
        return self.fm_[self.fm_["Type"] == coltype]

    def get_features(self, feature_type=None):
        """Returns all features or all features of the requested feature type."""
        if feature_type:
            return list(self.fm_[(self.fm_["Type"] == feature_type)]["Feature"].values)
        else:
            return list(self.fm_["Feature"].values)

    def get_categorical_features(self):
        """Returns a list of nominal and ordinal features."""
        nominal = list(self.fm_[(self.fm_["Type"] == "Nominal") & (self.fm_["Active"] == True)]["Feature"].values)
        ordinal = list(self.fm_[(self.fm_["Type"] == "Ordinal") & (self.fm_["Active"] == True)]["Feature"].values)
        return nominal + ordinal

    def get_numeric_features(self):
        """Returns a list of continuous and discrete features."""
        discrete = list(self.fm_[(self.fm_["Type"] == "Discrete") & (self.fm_["Active"] == True)]["Feature"].values)
        continuous = list(self.fm_[(self.fm_["Type"] == "Continuous") & (self.fm_["Active"] == True)]["Feature"].values)
        return discrete + continuous        


    def get_original_features(self, feature_type=None):
        """Returns original features or original features of the requested feature type."""
        if feature_type:
            return list(self.fm_[(self.fm_["Type"] == feature_type)& (self.fm_["Source"] == "Original")]["Feature"].values)
        else:
            return list(self.fm_[(self.fm_["Source"] == "Original")]["Feature"].values)

    def get_active_features(self, feature_type=None):
        """Returns original features or original features of the requested feature type."""
        if feature_type:
            return list(self.fm_[(self.fm_["Active"] == True) & (self.fm_["Type"] == feature_type)]["Feature"].values)
        else:
            return list(self.fm_[(self.fm_["Active"] == True)]["Feature"].values)

    def exclude_feature(self, feature):
        self.fm_.loc[self.fm_["Feature"]==feature, "Active"] = False
        self.fm_.loc[self.fm_["Feature"]==feature, "Signature"] = sys._getframe(1).f_code.co_name
        self.save()

    def include_feature(self, feature):
        self.fm_.loc[self.fm_["Feature"]==feature, "Active"] = True
        self.fm_.loc[self.fm_["Feature"]==feature, "Signature"] = sys._getframe(1).f_code.co_name
        self.save()

    def exclude_features(self,features):
        for feature in features:
            self.exclude_feature(feature)

    def include_features(self,features):
        for feature in features:
            self.include_feature(feature)                    

    def add_feature(self, feature, feature_type, active=True):
        d = {"Feature": feature, "Type": feature_type, "Source": "Derived",
            "Active": active, "Signature": sys._getframe(1).f_code.co_name }
        df = pd.DataFrame(data=d, index=[0])
        self.fm_ = pd.concat((self.fm_,df),axis=0)
        self.save()

    def save(self):
        self.fm_.to_csv(self._filename, index=False)    
    
    def print(self, feature=None, feature_type=None):
        if feature_type:
            print(self.fm_[self.fm_["Type"]==feature_type])
        elif feature:
            print(self.fm_[self.fm_["Feature"]==feature])
        else:
            print(self.fm_)


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
