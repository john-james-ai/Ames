# Notes
## Data Preparation
The data were prepared in four stages:
1. Data Preprocessing
2. Feature Engineering
3. Data Transformation
4. Feature Selection
5. Outlier Detection and Treatment. 
7. One-Hot-Encoding

Before preprocessing, the data were split into training, validation and test sets. The data preparation pipeline was constructed and each set was prepared separately.

### Data Preprocessing
1. Removed all homes with GR LIVE AREA greater than 4000 square as per DeCock's recommendation. 
2. Remove all homes with Total_Bsmt_SF > 4000. Should remove 2 properties.
3. Remove latitude and longitude

Things to consider:
1. Cap some continous variables at 95% ile.
2. Remove extremely inbalanced categorical variables. See what RECV does first.

### Feature Engineering
1. Age: of home at time of sale by subtracting Year_Built from Year_Sold.
2. Age2 age since remodeling
3. Total_SQ = Total livable space. 
4. Create District variable which target encoding four quantiles based upon rank of sale price.
5. Crisis year if the property sold during 2008 crisis
6. Nominal variables:
   1. Has_Garage
   2. Has_Deck
   3. Has_Pool
   4. Has_Bsmt
7. Another nominal variable Two_Level

### Data Transformation
1.  Transformed target to Log Sales Priced.
2.  Transformed all ordinal variables to integers.
3.  Transformed all nominal variables using feature hashing.
4.  Continuous (Floating Point) variables were power transformed according to... [see Scikit Learn documentation]

### Pipeline
1. Preprocessing
   1. Data Cleaning
      1. Corrections
      2. Target Transformation
   2. Data Screening
   3. Feature Engineering
   4. Preprocessing
      1. Continuous Preprocessing
         1. Iterative Imputer
         2. Power Transformer (No Standard)
      2. Discrete Preprocessing
         1. SimpleImputer         
      3. Ordinal Preprocessing
         1. Simple Imputer (most-frequent)
         2. Ordinal Map Encoding
      4. Nominal Preprocessing
         1. Mean Encoding
      5. Standardize
   5. Feature Selection
   6. GridSearchCV
   7. Predict
   8. Rinse Repeat

Note: No need to keep track of PID. It will be in the test set.

# Todos
1. Make feature metadata resident in memory at beginning of script, 
2. Error when feature name is not found. this happens when deleting features.  Use compress whenever selecting features of a certain type to get the features of that type that are extant in the dataset.

## Hyperparameters
### Lasso 
Upper and lower alphas for lasso on the full dataset
0.1689913607363613
0.0001484184910276514
### Ridge
Upper and lower bound of alphas for ridgecv
0.3379827214727226
0.00029683698205530283   

# Drawing Board
New representation of data
