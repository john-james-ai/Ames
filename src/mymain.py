# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Data Mining                                                       #
# File    : \mymain.py                                                        #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Data-Mining/                     #
# --------------------------------------------------------------------------- #
# Created       : Tuesday, March 9th 2021, 12:24:24 am                        #
# Last Modified : Tuesday, March 9th 2021, 12:24:24 am                        #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
# =========================================================================== #
#                               1. LIBRARIES                                  #
# =========================================================================== #
#%%
# Manipulating, analyzing and processing data
from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

# Feature and model selection and evaluation
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

# Regression based estimators
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

# Tree-based estimators
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# Visualizing data
import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================================== #
#                               2. GLOBALS                                    #
# =========================================================================== #
random_state = 9876
discrete =  ["Year_Built","Year_Remod_Add","Bsmt_Full_Bath","Bsmt_Half_Bath",
    "Full_Bath","Half_Bath","Bedroom_AbvGr","Kitchen_AbvGr","TotRms_AbvGrd",
    "Fireplaces","Garage_Cars","Mo_Sold","Year_Sold"],
continuous = ["Lot_Frontage","Lot_Area","Mas_Vnr_Area","BsmtFin_SF_1","BsmtFin_SF_2",
    "Bsmt_Unf_SF","Total_Bsmt_SF","First_Flr_SF","Second_Flr_SF","Low_Qual_Fin_SF",
    "Gr_Liv_Area","Garage_Area","Wood_Deck_SF","Open_Porch_SF","Enclosed_Porch",
    "Three_season_porch","Screen_Porch","Pool_Area","Misc_Val"]
n_nominal_levels = 189
nominals = ['MS_SubClass', 'MS_Zoning', 'Street', 'Alley', 'Land_Contour', 'Lot_Config', 'Neighborhood',
 'Condition_1', 'Condition_2', 'Bldg_Type', 'House_Style', 'Roof_Style', 'Roof_Matl',
 'Exterior_1st', 'Exterior_2nd', 'Mas_Vnr_Type', 'Foundation', 'Heating', 'Central_Air',
 'Garage_Type', 'Misc_Feature', 'Sale_Type', 'Sale_Condition']
ordinal_map = {'BsmtFin_Type_1': {'ALQ': 5, 'BLQ': 4, 'GLQ': 6, 'LwQ': 2, 'No_Basement': 0, 'Rec': 3, 'Unf': 1},
 'BsmtFin_Type_2': {'ALQ': 5, 'BLQ': 4, 'GLQ': 6, 'LwQ': 2, 'No_Basement': 0, 'Rec': 3, 'Unf': 1},
 'Bsmt_Cond': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Basement': 0, 'Poor': 1, 'Typical': 3},
 'Bsmt_Exposure': {'Av': 3, 'Gd': 4, 'Mn': 2, 'No': 1, 'No_Basement': 0},
 'Bsmt_Qual': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Basement': 0, 'Poor': 1, 'Typical': 3},
 'Electrical': {'FuseA': 4, 'FuseF': 2, 'FuseP': 1, 'Mix': 0, 'SBrkr': 5, 'Unknown': 3},
 'Exter_Cond': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'Poor': 0, 'Typical': 2},
 'Exter_Qual': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'Poor': 0, 'Typical': 2},
 'Fence': {'Good_Privacy': 4, 'Good_Wood': 2, 'Minimum_Privacy': 3, 'Minimum_Wood_Wire': 1,'No_Fence': 0},
 'Fireplace_Qu': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Fireplace': 0, 'Poor': 1, 'Typical': 3},
 'Functional': {'Maj1': 3, 'Maj2': 2, 'Min1': 5, 'Min2': 6, 'Mod': 4, 'Sal': 0, 'Sev': 1, 'Typ': 7},
 'Garage_Cond': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Garage': 0, 'Poor': 1, 'Typical': 3},
 'Garage_Finish': {'Fin': 3, 'No_Garage': 0, 'RFn': 2, 'Unf': 1},
 'Garage_Qual': {'Excellent': 5, 'Fair': 2, 'Good': 4, 'No_Garage': 0, 'Poor': 1, 'Typical': 3},
 'Heating_QC': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'Poor': 0, 'Typical': 2},
 'Kitchen_Qual': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'Poor': 0, 'Typical': 2},
 'Land_Slope': {'Gtl': 0, 'Mod': 1, 'Sev': 2},
 'Lot_Shape': {'Irregular': 0, 'Moderately_Irregular': 1, 'Regular': 3, 'Slightly_Irregular': 2},
 'Overall_Cond': {'Above_Average': 5, 'Average': 4,'Below_Average': 3,'Excellent': 8,'Fair': 2,
                  'Good': 6,'Poor': 1,'Very_Excellent': 9,'Very_Good': 7,'Very_Poor': 0},
 'Overall_Qual': {'Above_Average': 5,'Average': 4,'Below_Average': 3,'Excellent': 8,'Fair': 2,
                  'Good': 6,'Poor': 1,'Very_Excellent': 9,'Very_Good': 7,'Very_Poor': 0},
 'Paved_Drive': {'Dirt_Gravel': 0, 'Partial_Pavement': 1, 'Paved': 2},
 'Pool_QC': {'Excellent': 4, 'Fair': 1, 'Good': 3, 'No_Pool': 0, 'Typical': 2},
 'Utilities': {'AllPub': 2, 'NoSeWa': 0, 'NoSewr': 1}}

# =========================================================================== #
#                             3. DATA SOURCES                                 #
# =========================================================================== #
class AmesDataIO:
    """Ames data sources, including training and cross-validation data sets."""
    def __init__(self, filepath_train="../data/train/", filepath_cv="../data/cv/"):        
        self._filepath_train = filepath_train
        self._filepath_cv = filepath_cv

    def get_train(self):
        """Returns training set X, y data."""
        X_filepath = os.path.join(self._filepath_train, "X_train.csv")
        y_filepath = os.path.join(self._filepath_train, "y_train.csv")
        X = pd.read_csv(X_filepath)
        y = pd.read_csv(y_filepath)
        return X, y

    def get_split(self, s=np.random.randint(1,11,1)):
        """Returns training and test data for k-fold cross-validation."""
        train_filename = s + "_train.csv"
        test_filename = s + "_test.csv"
        test_y_filename = s + "_test_y.csv"
        # Train data
        train = pd.read_csv(os.path.join(self._filepath_cv,train_filename))
        X_train = train.loc[:, train.columns != "Sale_Price"]
        y_train = train.loc[:, train.columns == "Sale_Price"]

        # Test data
        X_test = pd.read_csv(os.path.join(self._filepath_cv,test_filename))
        y_test = pd.read_csv(os.path.join(self._filepath_cv,test_y_filename))

        return X_train, y_train, X_test, y_test

# =========================================================================== #
#                          4. DATA PREPROCESSING                              #
# =========================================================================== #
class ContinuousPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        continuous = list(X.select_dtypes(include=["float64"]).columns)
        # Create imputer and power transformer objects
        imputer = IterativeImputer()
        power = PowerTransformer(method="yeo-johnson", standardize=True)
        
        # Perform imputation of continuous variables
        X[continuous] = imputer.fit_transform(X[continuous])

        # Perform power transformations to make data closer to Guassian distribution
        X[continuous] = power.fit_transform(X[continuous])
        
        return X
# --------------------------------------------------------------------------- #
class CategoricalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        categorical = list(X.select_dtypes(include=["object"]).columns)
        # Create imputer object
        imputer = SimpleImputer(strategy="most_frequent")
        
        # Perform imputation of categorical variables to most frequent
        X[categorical] = imputer.fit_transform(X[categorical])
        
        return X        
# --------------------------------------------------------------------------- #
class DiscretePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        discrete = list(X.select_dtypes(include=["int"]).columns)
        # Create imputer and scaler objects
        imputer = SimpleImputer(strategy="most_frequent")
        scaler = StandardScaler()        
        
        # Perform imputation of categorical variables to most frequent
        X[discrete] = imputer.fit_transform(X[discrete])
        X[discrete] = scaler.fit_transform(X[discrete])
        
        return X        
        
def build_preprocessor(X, y=None):
    """Builds the data processing component of the pipeline."""
    continuous = list(X.select_dtypes(include=["float64"]).columns)
    categorical = list(X.select_dtypes(include=["object"]).columns)
    discrete = list(X.select_dtypes(include=["int"]).columns)
    features = list(X.columns)

    continuous_pipeline = make_pipeline(
        IterativeImputer(),
        PowerTransformer(method="yeo-johnson", standardize=True)
    )

    categorical_pipeline = make_pipeline(
        IterativeImputer(),
        OneHotEncoder()
    )

    discrete_pipeline = make_pipeline(
        IterativeImputer(),
        StandardScaler()
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', continuous_pipeline, continuous),
            ('categorical', categorical_pipeline, categorical),
            ('discrete', discrete_pipeline, discrete)
        ]
    )
    return preprocessor
# =========================================================================== #
#                          5. FEATURE ENGINEERING                             #
# =========================================================================== #    
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_map=ordinal_map):
        self._ordinal_map = ordinal_map

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        """Creation, removal, and encoding of features."""
        # Add an age feature and remove year built
        X["Age"] = X["Year_Sold"] - X["Year_Built"]
        X["Age"].fillna(X["Age"].median())
        X.drop(columns=["Year_Built"], inplace=True)

        # Remove longitude and latitude
        X = X.drop(columns=["Latitude", "Longitude"])

        ## Encode ordinal features
        for variable, mappings in self._ordinal_map:
            for k,v in mappings.items():
                X[variable].replace({k:v}, inplace=True)    

        return X
# --------------------------------------------------------------------------- #
class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_map=ordinal_map):
        self._ordinal_map = ordinal_map

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        for variable, mappings in self._ordinal_map:
            for k,v in mappings.items():
                X[variable].replace({k:v}, inplace=True)    

        return X        
# --------------------------------------------------------------------------- #
class NominalEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, nominals=nominals):
        self._nominals = nominals

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        """Converting nominal variables to one-hot representation."""
        n = X.shape[0]
        # Extract nominals from X
        nominals = pd.Series(self._nominals)
        features = X.columns
        nominal_features = features[features.isin(nominals)]
        X_nominals = X[nominal_features]   
        X.drop(nominal_features, axis=1, inplace=True)    
        n_other_features = X.shape[1]

        # Encode nominals and store in dataframe with feature names
        enc = OneHotEncoder()
        X_nominals = enc.fit_transform(X_nominals).toarray()
        X_nominals = pd.DataFrame(data=X_nominals)
        X_nominals.columns = enc.get_feature_names()

        # Concatenate X with X_nominals and validate    
        X = pd.concat([X, X_nominals], axis=1)
        expected_shape = (n,n_other_features+n_total_feature_levels)
        assert(X.shape == expected_shape), "Error in Encode Nominal. X shape doesn't match expected."

        return X

# =========================================================================== #
#                           6. PIPELINE RFE                                   #
# =========================================================================== #   
class PipelineRFE(Pipeline):
    # Source: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].feature_importances_
        return self

# =========================================================================== #
#                             7. SCORING                                      #
# =========================================================================== #   
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))
rmse = make_scorer(rmse, greater_is_better=False)    
# =========================================================================== #
#                         8. FEATURE SELECTION                                #
# =========================================================================== #   
class DataPipeline(BaseEstimator, TransformerMixin):
    """Performs feature selection using Recursive Feature Elimination with CV."""
    def __init__(self, estimator, cv=5, scoring=rmse, min_features=1):
        self._estimator = estimator
        self._cv = cv
        self._scoring = scoring
        self._min_features = min_features
        self._feature_selector = None

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, y, **transform_params):
        # Define the pipeline for RFECV
        preprocessor = build_preprocessor(X)
        steps = [("preprocessor", preprocessor),
                 ("features", FeatureEngineer()),
                 ("nominal", NominalEncoder()),
                 ("ordinal", OrdinalEncoder()),
                 ("estimator", self._estimator)]
        pipe = PipelineRFE(steps=steps)

        # Initialize the RFECV object
        self._feature_selector = RFECV(pipe, cv=self._cv, step=1, scoring=self._scoring)

        # Fit RFECV
        print(f"Inside DataPipeline transform and y shape is {y.shape}")
        self._feature_selector.fit(X,y)

        # Get selected features
        feature_names = X.columns
        self.selected_features = feature_names[self._feature_selector.support_].tolist()
        return X[self.selected_features]

    def plot(self):
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10,12))
        x = range(self._min_features, len(self._feature_selector.grid_scores_)+self._min_features)
        y = self._feature_selector.grid_scores_
        d = {"x": x, "y":y}
        df = pd.DataFrame(d)
        ax = sns.lineplot(x=x, y=y, data=df)
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Cross Validation Score (RMSE)")
        plt.title("Recursive Feature Elimination via Cross-Validation")
        plt.tight_layout()
        plt.show()



# =========================================================================== #
#                              10. ESTIMATORS                                 #
# =========================================================================== #
regressors = {}
regressors.update({"Linear Regression": LinearRegression()})
regressors.update({"Lasso": Lasso()})
regressors.update({"Ridge": Ridge()})
regressors.update({"ElasticNet": ElasticNet()})

ensembles = {}
ensembles.update({"AdaBoost": AdaBoostRegressor()})
ensembles.update({"Bagging": BaggingRegressor()})
ensembles.update({"Extra Trees": ExtraTreesRegressor()})
ensembles.update({"Gradient Boosting": GradientBoostingRegressor()})
ensembles.update({"Random Forest": RandomForestRegressor()})
ensembles.update({"Histogram Gradient Boosting": HistGradientBoostingRegressor()})


# =========================================================================== #
#                          11. HYPERPARAMETERS                                #
# =========================================================================== #
# Parameter Grid
regressor_parameters = {}
regressor_parameters.update({"Linear Regression":[]})
regressor_parameters.update({"Lasso": {
    "estimator__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
    "estimator__n_jobs": [-1]}})
regressor_parameters.update({"Ridge":{
        "estimator__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
        "estimator__n_jobs": [-1]}})        
regressor_parameters.update({"ElasticNet":{
        "estimator__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
        "estimator__l1_ratio": np.arange(0.0,1.0,0.1),
        "estimator__n_jobs": [-1]}})        

ensemble_parameters = {}
ensemble_parameters.update({"AdaBoost": {
        "estimator__base_estimator": None,
        "estimator__n_estimators": [50,100],
        "estimator__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0]}})
ensemble_parameters.update({"Bagging": {
        "estimator__base_estimator": None,
        "estimator__n_estimators": [50,100],
        "estimator__max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "estimator__n_jobs": [-1]}}) 
ensemble_parameters.update({"Extra Trees": {        
        "estimator__n_estimators": [50,100],
        "estimator__max_depth": [2,3,4,5,6],
        "estimator__min_samples_split": [0.005, 0.01, 0.05, 0.10],
        "estimator__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
        "estimator__max_features": ["auto", "sqrt", "log2"],
        "estimator__n_jobs": [-1]}})         
ensemble_parameters.update({"Gradient Boosting": {        
        "estimator__learning_rate": [0.15,0.1,0.05,0.01,0.005,0.001],
        "estimator__n_estimators": [50,100],
        "estimator__max_depth": [2,3,4,5,6],
        "estimator__criterion": ["mse"],
        "estimator__min_samples_split": [0.005, 0.01, 0.05, 0.10],
        "estimator__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
        "estimator__max_features": ["auto", "sqrt", "log2"]}})                        
ensemble_parameters.update({"Random Forest": {        
        "estimator__n_estimators": [50,100],
        "estimator__max_depth": [2,3,4,5,6],
        "estimator__criterion": ["mse"],
        "estimator__min_samples_split": [0.005, 0.01, 0.05, 0.10],
        "estimator__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
        "estimator__max_features": ["auto", "sqrt", "log2"],
        "estimator__n_jobs": [-1]}})      
ensemble_parameters.update({"Histogram Gradient Boosting": {  
        "estimator__learning_rate": [0.15,0.1,0.05,0.01,0.005,0.001],              
        "estimator__max_depth": [2,3,4,5,6],        
        "estimator__min_samples_leaf": [0.005, 0.01, 0.05, 0.10]}})                                        
# =========================================================================== #
#                           12. MODEL EVALUATION                              #
# =========================================================================== #
class ModelEvaluator:
    def __init__(self, data_pipeline, estimator, parameters):
        self._data_pipeline = data_pipeline
        self._estimator = estimator
        self._parameters = parameters        

    def fit(self, X, y=None):
        # Prepare data and perform feature selection
        self._data_pipeline.fit(X,y)
        X = self._data_pipeline.transform(X,y) 
        self._data_pipeline.plot()

        # Initialize GridSearchCV object and fit the search
        gscv = GridSearchCV(estimator=self._estimator,
                             param_grid=self._parameters)
        gscv.fit(X,y)

        # Get the best parameters and score
        self.best_params = gscv.best_params_
        self.best_score = gscv.best_score_

# --------------------------------------------------------------------------- #
#                                    MAIN                                     #
# --------------------------------------------------------------------------- #    
def main():
    # Obtain the data
    io = AmesDataIO()
    X, y = io.get_train()
    print(X.shape)
    print(y.shape)

    best_params = {}
    best_scores = {}

    # Train and evaluate regression models
    for name, estimator in regressors.items():
        # Instantiate, run and plot data pipeline
        data_pipeline = DataPipeline(estimator)
        data_pipeline.fit(X,y)
        X = data_pipeline.transform(X,y)
        data_pipeline.plot()

        # Obtain GridSearchCV parameters for estimator
        parameters = regressor_parameters[name]

        # Instantiate Model evaluator and perform tuning and evaluation
        evaluator = ModelEvaluator(data_pipeline, estimator, parameters)
        evaluator.fit(X,y)

        # Store parameters and scores 
        best_params[name] = evaluator.best_params
        best_scores[name] = evaluator.best_score

        # Print scores 
        print(f"Best score for {name}: {evaluator.best_score}")



if __name__ == "__main__":
    main()
#%%
