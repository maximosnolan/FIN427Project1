# Import libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import math
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline

# Change the number of rows and columns to display
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.precision', 4)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Define a path for import and export

path = '/Users/admin/fin427/project1/'

# Import data and create a quarter variable for alignment with quarterly financial information
returns = pd.read_excel(path + 'Excel03 Data 20230128.xlsx', sheet_name='ret06')
returns1 = returns
returns1['quarter'] = returns1['month'] + pd.offsets.QuarterEnd(0)
print(returns1.head(6))
print(returns1.columns)

# Import sales
sales = pd.read_excel(path + 'SP400_Sales_20230131.xlsx', sheet_name='Sales')
sales1 = sales
sales1 = sales1.dropna()
sales1 = sales1[sales1['Sales/Turnover (Net)'] > 0]
sales1['sales'] = sales1['Sales/Turnover (Net)']
sales1['quarter'] = sales1['Data Date'] + pd.offsets.QuarterEnd(0)
sales1['leadquarter'] = sales1['quarter'] + pd.offsets.QuarterEnd(1)
sales1['lnsales'] = np.log(sales1['sales']*1000)
univ_sales = sales1.sales.describe()
univ_lnsales = sales1.lnsales.describe()
sales1['cusip8'] = sales1['CUSIP'].str[0:8]
print(univ_sales)
print(univ_lnsales)
print(sales1.head(50))
print(sales1.columns)
print(sales1)

# Merge returns1 and sales1 dataframes
combined1 = returns1.merge(sales1, how='inner', left_on=['CUSIP', 'quarter'], right_on=['cusip8', 'leadquarter'])
combined1.rename(columns={'sales': 'lag1sales', 'lnsales': 'lag1lnsales'}, inplace=True)
print(combined1.head())
print(combined1.columns)
print(len(returns1.index))
print(len(sales1.index))
print(len(combined1.index))
print(combined1.lag1sales.describe(percentiles=[0.125, 0.875]))
print(combined1.lag1mc.describe(percentiles=[0.125, 0.875]))
print(combined1.lag1lnsales.describe(percentiles=[0.125, 0.875]))
print(combined1.lag1lnmc.describe(percentiles=[0.125, 0.875]))
print(combined1.retadj.describe(percentiles=[0.125, 0.875]))
print(combined1.bmret.describe(percentiles=[0.125, 0.875]))
print(combined1.abretadj.describe(percentiles=[0.125, 0.875]))

# Compute the correlation between lnsales and lag1lnmc
dfcorr = combined1[['lag1lnsales', 'lag1lnmc']].copy()
dfcorr.corr(method='pearson')

# OLS regression with two size variable, lag1lnmc and lag1lnsales, and industry indicator variables.
# There is code to run the regression with either the z-score transformed independent variables or the
# untransformed variables.
y1 = combined1['abretadj']
x = combined1[['lag1lnmc', 'lag1lnsales',
    'd_1100_Non_Energy_Minerals',
    'd_1200_Producer_Manufacturing',
    'd_1300_Electronic_Technology',
    'd_1400_Consumer_Durables',
    'd_2100_Energy_Minerals',
    'd_2200_Process_Industries',
    'd_2300_Health_Technology',
    'd_2400_Consumer_Non_Durables',
    'd_3100_Industrial_Services',
    'd_3200_Commercial_Services',
    'd_3250_Distribution_Services',
    'd_3300_Technology_Services',
    'd_3350_Health_Services',
    'd_3400_Consumer_Services',
    'd_3500_Retail_Trade',
    'd_4600_Transportation',
    'd_4700_Utilities',
    'd_4801_Banks',
    'd_4802_Finance_NEC',
    'd_4803_Insurance',
    'd_4885_Real_Estate_Dev',
    'd_4890_REIT',
    'd_4900_Communications']]
x1 = StandardScaler().fit_transform(x)
x = sm.add_constant(x)
x1 = sm.add_constant(x1)
model = sm.OLS(y1, x).fit()
# model = sm.OLS(y1, x1).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq  = model.rsquared
print(print_model)
print(f'R-squared: {model.rsquared:.4f}')
print(ols_coef)

# LASSO regression with transformation to z-scores.

y1 = combined1['abretadj']
x = combined1[['lag1lnmc', 'lag1lnsales',
    'd_1100_Non_Energy_Minerals',
    'd_1200_Producer_Manufacturing',
    'd_1300_Electronic_Technology',
    'd_1400_Consumer_Durables',
    'd_2100_Energy_Minerals',
    'd_2200_Process_Industries',
    'd_2300_Health_Technology',
    'd_2400_Consumer_Non_Durables',
    'd_3100_Industrial_Services',
    'd_3200_Commercial_Services',
    'd_3250_Distribution_Services',
    'd_3300_Technology_Services',
    'd_3350_Health_Services',
    'd_3400_Consumer_Services',
    'd_3500_Retail_Trade',
    'd_4600_Transportation',
    'd_4700_Utilities',
    'd_4801_Banks',
    'd_4802_Finance_NEC',
    'd_4803_Insurance',
    'd_4885_Real_Estate_Dev',
    'd_4890_REIT',
    'd_4900_Communications']]
x1 = StandardScaler().fit_transform(x)
x1 = sm.add_constant(x1)
lasso = Lasso(alpha = 0.001)
lasso.fit(x1, y1)
lasso_coef = lasso.fit(x1, y1).coef_
lasso_score = lasso.score(x1, y1)
print(f'Lasso score: {lasso_score: .4f}')
print(lasso.intercept_)
print(lasso_coef)
print(x.columns)
# print(pd.Series(lasso_coef, index = x.columns)) #This only works if the constant is excluded.

# OLS regression incorporating only those industries identified by LASSO as being material.
# Remove the industry indicator variables for the industries selected by LASSO as not being material.
# There is code to run the regression with either the z-score transformed independent variables or the
# untransformed variables.

y1 = combined1['abretadj']
x = combined1[['lag1lnmc','lag1lnsales',
   'd_1100_Non_Energy_Minerals',
    'd_1200_Producer_Manufacturing',
    'd_1300_Electronic_Technology',
    'd_1400_Consumer_Durables',
    'd_2100_Energy_Minerals',
    'd_2200_Process_Industries',
    'd_2300_Health_Technology',
    'd_2400_Consumer_Non_Durables',
    'd_3100_Industrial_Services',
    'd_3200_Commercial_Services',
    'd_3250_Distribution_Services',
    'd_3300_Technology_Services',
    'd_3350_Health_Services',
    'd_3400_Consumer_Services',
    'd_3500_Retail_Trade',
    'd_4600_Transportation',
    'd_4700_Utilities',
    'd_4801_Banks',
    'd_4802_Finance_NEC',
    'd_4803_Insurance',
    'd_4885_Real_Estate_Dev',
    'd_4890_REIT',
    'd_4900_Communications']]
x1 = StandardScaler().fit_transform(x)
x = sm.add_constant(x)
x1 = sm.add_constant(x1)
model = sm.OLS(y1, x1).fit()
# model = sm.OLS(y1, x1).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq  = model.rsquared
print(print_model)
print(f'R-squared: {model.rsquared:.4f}')
print(ols_coef)
