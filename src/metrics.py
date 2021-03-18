# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Ames House Prediction Model                                       #
# File    : \metrics.py                                                       #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Ames/                            #
# --------------------------------------------------------------------------- #
# Created       : Thursday, March 18th 2021, 6:03:33 am                       #
# Last Modified : Thursday, March 18th 2021, 6:03:34 am                       #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
# =========================================================================== #
#                                SCORING                                      #
# =========================================================================== #   
def RMSE(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)    
    return np.sqrt(mse)
rmse = make_scorer(RMSE, greater_is_better=False)