# Investing in lending club loans
### Description: 
The goal of the project is to identify borrowers who will NOT default on their loans with 95% certainty. In order to achieve that goal I built a machine learning model that classifies borrowers into two groups, those who will default on their loan and those who will pay it back in full. As well as the accuracy of the model to correctly predict the class to which the borrowers belong, it was important for the goal of the project to gain insights into what features, and how, of each borrower drive the probability of deafault. Because the data was imbalanced I decided to use area under the curve (AUC) to measure the model performance.

### Project Layout:
* [Data wrangling notebook](https://github.com/DarioMakaric/lending-club/blob/master/data_wrangling.ipynb)
* [Data modeling notebook](https://github.com/DarioMakaric/lending-club/blob/master/data_modeling.ipynb)

### Tools used:
* Data wrangling with PANDAS
* Data visualization with SEABORN and MATPLOTLIB, partial dependence plots with PDPBOX library 
* Statistical tests and cluster analysis with SCIPY
* Permutation importance with ELI5 library
* Bayesian hyperparameter optimization with SKOPT library
* Recursive feature selection and preprocessing with SKLEARN
* Gradient boosted trees with XGBOOST
## Data wrangling notebook summary
In this notebook I cleaned the data set downloaded from the lending club website. Dataset had a lot of columns with more than 50% of missing values. A lot of columns were dropped based on the number of missing values. Values in some columns were estimated based on their highly corralated columns. Categorical columns were either dropped if they contained more than 15 nominal values or they were one hot and label encoded. Dependent variable was picked from the dataset and then label encoded.

### Steps:
1. Removing data based on different criteria:
  * Remove columns that contain more than 50% missing values
  * Remove rows that contain 100% missing values
  * Remove columns that contain data leakage
  * Remove columns that contain a single unique value
  * Remove columns that contain redundant information
  * Remove columns that contain information that can be infered from other columns
 
2. Feature engineering:
  * Engineer datetime columns
  * Estimate missing values of highly correlated columns
  * Recategorize some nominal columns
  * Change dtypes of some object columns
  * Prepare the dependent variable
  
## Data modeling notebook
In this notebook first I used Bayesian hyperparameter optimization and then I trained an XGBoost model on the cleaned data. I recorded the area under the curve score and based on feature importance I recursively dropped half of the features. The score stayed the same with 50% less features. Then I used spearman rank correlation to drop one more column and keep the same model precision. Calculating feature importances on the new dataset gave me, as the most informative feature of a borrower's default chance, their annual icome.

![annual_inc](https://github.com/DarioMakaric/lending-club/blob/master/plots/annual_inc.png "Annual Income Partial Dependence Plot")

Partial dependence plot shows positive correlation between the borrower's income and the likelihood of them paying the loan back. I ran a t-test and got p-value < 0.001 concluding that on average we can expect that borrowers annual income is highly indicative of their chance to default. Borrowers who did not default made, on average $6,250.00 more compared to those who defaulted.

![int_rate_vs_annual_inc](https://github.com/DarioMakaric/lending-club/blob/master/plots/int_rate_vs_annual_inc.png "Annual Income vs Interest Rate PDP")

On the other hand 2D partial dependence plot shows that the higher the interest rate the more likely the borrower will default regardless of their income.

![age_cr_line_vs_annual_inc](https://github.com/DarioMakaric/lending-club/blob/master/plots/age_cr_line_vs_inc.png "Age Credit Line vs Annual Income PDP")

Age credit line is a feature which I constructed from the two features in the original dataset. It was the date of loan issue (which was removed as leaky feature) and the date of the first credit application. When we subtract the two and convert the dates to number of months we get the age_cr_line. It is interesting to notice that age of the credit line positively impacts the default rate but not as a monotonic function. It looks like there is a cluster of borrowers in the range 100-150 months and above average income who have a higher chance of defaulting on their loan. Those with low show have pretty much monotonic relationship between two variables. It would be interesting to create categories of borrowers based on their income and age credit line and see if it helps the model score better.

![small_business](https://github.com/DarioMakaric/lending-club/blob/master/plots/small_business.png "Small Business PDP")

It was surprising to see that the only one-hot encoded variable that show high importance was the one indicating whether the purpose of the loan was small business. I ran a t-test and recorded p-value < 0.001 indicating that on average if a borrower applied for a small business loan they were 52\% more likely to default.
