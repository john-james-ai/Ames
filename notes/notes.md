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
3. Remove latitude and longitude

Things to consider:
1. Cap some continous variables at 95% ile.
2. Remove extremely inbalanced categorical variables. See what RECV does first.

### Feature Engineering
1. Added Age of home at time of sale by subtracting Year_Built from Year_Sold.

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
   5. 
   