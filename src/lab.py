# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \lab.py                                                           #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Friday, March 12th 2021, 1:28:12 pm                         #
# Last Modified : Friday, March 12th 2021, 1:28:12 pm                         #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import seaborn as sns
from utils import print_dict
filename = "../data/external/schools.csv"              
schools = pd.read_csv(filename)     
d = {"Neighborhood": schools["Neighborhood"].values,
     "Zip": schools["Zip"].values,
     "School_Title_1": schools["School_Title_1"].values,
     "School_Students": schools["School_Students"].values,
     "School_Teachers": schools["School_Teachers"].values,
     "School_Student_Teacher_Ratio": schools["School_Student_Teacher_Ratio"].values,
     "Free_or_Reduced_Lunch": schools["Free_or_Reduced_Lunch"].values}
print_dict(d)     
