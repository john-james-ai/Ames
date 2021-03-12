# Notes
## Data Preparation
The data were prepared in four stages:
1. Data Preprocessing
2. Feature Engineering
3. Data Transformation
4. Feature Selection
5. Encoding

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
