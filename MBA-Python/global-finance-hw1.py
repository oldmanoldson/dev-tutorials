import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

currency = pd.read_csv('C:/Users/600795/Documents/Personal/currency.csv', low_memory = False)

## Question 1:  Calculate annualized mean and std dev on returns for each G7 country
currency['USD_perc_change'] = np.nan
currency['UKS_perc_change'] = np.nan
currency['ITE_perc_change'] = np.nan
currency['JPY_perc_change'] = np.nan
currency['BDE_perc_change'] = np.nan
currency['FRE_perc_change'] = np.nan
currency['CAD_perc_change'] = np.nan

for i in range(1, len(currency)):
    currency.loc[i, 'USD_perc_change'] = (currency.loc[i, 'USD']/currency.loc[i-1, 'USD']) - 1
    currency.loc[i, 'UKS_perc_change'] = (currency.loc[i, 'UKS']/currency.loc[i-1, 'UKS']) - 1
    currency.loc[i, 'ITE_perc_change'] = (currency.loc[i, 'ITE']/currency.loc[i-1, 'ITE']) - 1
    currency.loc[i, 'JPY_perc_change'] = (currency.loc[i, 'JPY']/currency.loc[i-1, 'JPY']) - 1
    currency.loc[i, 'BDE_perc_change'] = (currency.loc[i, 'BDE']/currency.loc[i-1, 'BDE']) - 1
    currency.loc[i, 'FRE_perc_change'] = (currency.loc[i, 'FRE']/currency.loc[i-1, 'FRE']) - 1
    currency.loc[i, 'CAD_perc_change'] = (currency.loc[i, 'CAD']/currency.loc[i-1, 'CAD']) - 1

# Means
US_annualized_mean = 12*currency['USD_perc_change'].mean()
UK_annualized_mean = 12*currency['UKS_perc_change'].mean()
IT_annualized_mean = 12*currency['ITE_perc_change'].mean()
JP_annualized_mean = 12*currency['JPY_perc_change'].mean()
BD_annualized_mean = 12*currency['BDE_perc_change'].mean()
FR_annualized_mean = 12*currency['FRE_perc_change'].mean()
CA_annualized_mean = 12*currency['CAD_perc_change'].mean()

# Std Devs
US_annualized_stddev = sqrt(12)*currency['USD_perc_change'].std()
UK_annualized_stddev = sqrt(12)*currency['UKS_perc_change'].std()
IT_annualized_stddev = sqrt(12)*currency['ITE_perc_change'].std()
JP_annualized_stddev = sqrt(12)*currency['JPY_perc_change'].std()
BD_annualized_stddev = sqrt(12)*currency['BDE_perc_change'].std()
FR_annualized_stddev = sqrt(12)*currency['FRE_perc_change'].std()
CA_annualized_stddev = sqrt(12)*currency['CAD_perc_change'].std()

## Question 2:  Convert foreign currencies to US currency and calculate percent change
for i in range(0, len(currency)):
    currency.loc[i, 'USD_UKS_conv'] = currency.loc[i, 'UKS'] * currency.loc[i, 'USD_UKS']
    currency.loc[i, 'USD_ITE_conv'] = currency.loc[i, 'ITE'] * currency.loc[i, 'USD_ITE']
    currency.loc[i, 'USD_JPY_conv'] = currency.loc[i, 'JPY'] * currency.loc[i, 'USD_JPY']
    currency.loc[i, 'USD_BDE_conv'] = currency.loc[i, 'BDE'] * currency.loc[i, 'USD_BDE']
    currency.loc[i, 'USD_FRE_conv'] = currency.loc[i, 'FRE'] * currency.loc[i, 'USD_FRE']
    currency.loc[i, 'USD_CAD_conv'] = currency.loc[i, 'CAD'] * currency.loc[i, 'USD_CAD']

currency['USD_UKS_perc_change'] = np.nan
currency['USD_ITE_perc_change'] = np.nan
currency['USD_JPY_perc_change'] = np.nan
currency['USD_BDE_perc_change'] = np.nan
currency['USD_FRE_perc_change'] = np.nan
currency['USD_CAD_perc_change'] = np.nan

for i in range(1, len(currency)):
    currency.loc[i, 'USD_UKS_perc_change'] = (currency.loc[i, 'USD_UKS_conv']/currency.loc[i-1, 'USD_UKS_conv']) - 1
    currency.loc[i, 'USD_ITE_perc_change'] = (currency.loc[i, 'USD_ITE_conv']/currency.loc[i-1, 'USD_ITE_conv']) - 1
    currency.loc[i, 'USD_JPY_perc_change'] = (currency.loc[i, 'USD_JPY_conv']/currency.loc[i-1, 'USD_JPY_conv']) - 1
    currency.loc[i, 'USD_BDE_perc_change'] = (currency.loc[i, 'USD_BDE_conv']/currency.loc[i-1, 'USD_BDE_conv']) - 1
    currency.loc[i, 'USD_FRE_perc_change'] = (currency.loc[i, 'USD_FRE_conv']/currency.loc[i-1, 'USD_FRE_conv']) - 1
    currency.loc[i, 'USD_CAD_perc_change'] = (currency.loc[i, 'USD_CAD_conv']/currency.loc[i-1, 'USD_CAD_conv']) - 1

## Question 3:  Calculate annualized mean and Std. Devs for USD converted currencies
# Means
UK_annualized_mean_conv = 12*currency['USD_UKS_perc_change'].mean()
IT_annualized_mean_conv = 12*currency['USD_ITE_perc_change'].mean()
JP_annualized_mean_conv = 12*currency['USD_JPY_perc_change'].mean()
BD_annualized_mean_conv = 12*currency['USD_BDE_perc_change'].mean()
FR_annualized_mean_conv = 12*currency['USD_FRE_perc_change'].mean()
CA_annualized_mean_conv = 12*currency['USD_CAD_perc_change'].mean()

# Std Devs
UK_annualized_stddev_conv = sqrt(12)*currency['USD_UKS_perc_change'].std()
IT_annualized_stddev_conv = sqrt(12)*currency['USD_ITE_perc_change'].std()
JP_annualized_stddev_conv = sqrt(12)*currency['USD_JPY_perc_change'].std()
BD_annualized_stddev_conv = sqrt(12)*currency['USD_BDE_perc_change'].std()
FR_annualized_stddev_conv = sqrt(12)*currency['USD_FRE_perc_change'].std()
CA_annualized_stddev_conv = sqrt(12)*currency['USD_CAD_perc_change'].std()

## Question 4:  Calculate Skewness and Kurtosis
# Skewness
UKS_skew = currency['USD_UKS_perc_change'].skew()
ITE_skew = currency['USD_ITE_perc_change'].skew()
JPY_skew = currency['USD_JPY_perc_change'].skew()
BDE_skew = currency['USD_BDE_perc_change'].skew()
FRE_skew = currency['USD_FRE_perc_change'].skew()
CAD_skew = currency['USD_CAD_perc_change'].skew()

# Kurtosis
UKS_kurtosis = currency['USD_UKS_perc_change'].kurtosis()
ITE_kurtosis = currency['USD_ITE_perc_change'].kurtosis()
JPY_kurtosis = currency['USD_JPY_perc_change'].kurtosis()
BDE_kurtosis = currency['USD_BDE_perc_change'].kurtosis()
FRE_kurtosis = currency['USD_FRE_perc_change'].kurtosis()
CAD_kurtosis = currency['USD_CAD_perc_change'].kurtosis()

## Question 5:  Compute correlations across the G7 for regular currencies and USD converted currencies
# Regular Currencies
reg_currency = currency[['USD_perc_change', 'UKS_perc_change', 'ITE_perc_change', 'JPY_perc_change', 'BDE_perc_change', 'FRE_perc_change', 'CAD_perc_change']].copy()

reg_currency_corr = reg_currency.corr()

# USD Converted Currencies
USD_conv_currency = currency[['USD_perc_change', 'USD_UKS_perc_change', 'USD_ITE_perc_change', 'USD_JPY_perc_change', 'USD_BDE_perc_change', 'USD_FRE_perc_change', 'USD_CAD_perc_change']].copy()

USD_conv_currency_corr = USD_conv_currency.corr()

## Question 6:  60-month rolling correlation calculated and plotted for US and JP using regular currencies and converted currencies
usjp = currency[['USD_perc_change', 'JPY_perc_change', 'USD_JPY_perc_change']].copy()
usjp = usjp.dropna()
usjp['axis'] = 1

for i in range(2, len(usjp)):
    usjp.loc[i, 'axis'] = usjp.loc[i-1, 'axis'] + 1


###### Need to figure out why rolling std. dev isn't working correctly ######

# Regular Currencies
usjp['USD_rolling'] = usjp['USD_perc_change'].rolling(60, min_periods = 1).std()*sqrt(60)
usjp['JPY_rolling'] = usjp['JPY_perc_change'].rolling(60, min_periods = 1).std()*sqrt(60)

# USD Converted Currencies
usjp['USD_JPY_rolling'] = usjp['USD_JPY_perc_change'].rolling(60, min_periods = 1).std()*sqrt(60)

# Plot
ax = plt.gca()

usjp.plot(kind = 'line', x = 'axis', y = 'USD_rolling', ax = ax)
usjp.plot(kind = 'line', x = 'axis', y = 'JPY_rolling', ax = ax)
usjp.plot(kind = 'line', x = 'axis', y = 'USD_JPY_rolling', ax = ax)

plt.show()
