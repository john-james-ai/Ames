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
cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
X, y = make_regression(n_samples=1000, n_features=1, noise=100, random_state=0)
sns.barplot(x=X, y=y, palette=cmap)
plt.show()
