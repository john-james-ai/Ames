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
from sklearn.preprocessing import OneHotEncoder
from mymain import ordinal_map

x = [1,2,3,4,5,6,7,8,9,0]
y = [0,9,8,7,6,5,4,3,2,1]
df = pd.DataFrame(data=x, columns=['x'])
y = pd.DataFrame(data=y, columns=['y'])
idx = df[df['x'] < 5].index.tolist()
print(idx)
print(y.iloc[idx])

