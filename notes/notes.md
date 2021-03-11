# Notes
## Data Preparation
Only missing data was Garage_Yr_Blt with about 5% missing values. These observations are likely homes without garages. Removing it from further analysis.

Eliminate all sales except for the “normal” from the SALES CONDITION variable. Unless an instructor specifically wants to create an activity that investigates the difference between the various types of sales (foreclosures, new homes, family sales, etc.) the different conditions will simply serve to complicate the results and confuse the students. 

Removed all homes with GR LIVE AREA greater than 4000 square feet as per the following recommendation.
Remove all homes with a living area (GR LIVE AREA) above 1500 square feet. The purpose to the second step is to alleviate problems with non-homogeneous variance. As might be expected there is increasing variation with increasing price within the Ames housing market. This problem can be remedied by taking a transformation (square root) of the sales price but those wishing to keep the response in dollars can simply use the smaller homes as they tend to show more homogeneous variation. 