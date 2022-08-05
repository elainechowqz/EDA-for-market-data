import pandas_datareader as pdr
import datetime 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import the `api` model of `statsmodels` under alias `sm`
import statsmodels.api as sm



# Feb 15th 2019
# want to study some relations between S&P 500 companies from different sectors
# code based on Python for Finance online course run by DataCamp

# Communication Services: 'AT&T Inc.', 'T'
# Consumer Discretionary: 'Block H&R','HRB'
# Consumer Staple: 'Costco Wholesale Corp.', 'COST'
# Energy: 'Exxon Mobil Corp.', 'XOM'
# Financials: 'American Express Co', 'AXP'
# Health Care: 'CVS Health', 'CVS'
# Industrials: 'Boeing Company', 'BA'
# Information Technolog: 'Apple Inc.', 'AAPL'
# Materials: 'Newmont Mining Corporation', 'NEM'
# Real Estate": 'American Tower Corp.', 'AMT'
# Utilities: 'American Water Works Company Inc', 'AWK'

tickers = ['T', 'HRB', 'COST', 'XOM', 'AXP', 'CVS', 'BA', 'AAPL', 'NEM', 'AMT', 'AWK']

# all the data for the 11 representatives above during the year 2018

def get(tickers, startdate, enddate):
  def data(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map (data, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))
all_data = get(tickers, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 12, 31))

# Assign `Adj Close` to `daily_close`
#daily_close = all_data[['Adj Close']]

# Daily returns
#daily_pct_change = daily_close.pct_change()

# Replace NA values with 0
#daily_pct_change.fillna(0, inplace=True)

# Isolate the `Adj Close` values and transform the DataFrame
daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')

# Calculate the daily percentage change for `daily_close_px`
daily_pct_change = daily_close_px.pct_change()

#Replace NA values with 0
daily_pct_change.fillna(0, inplace=True)

cov_matrix = daily_pct_change.cov()
print(cov_matrix)

#print(daily_pct_change)

# Daily log returns
#daily_log_returns = np.log(daily_close.pct_change()+1)

# Print daily log returns
#print(daily_log_returns)

# Plot a scatter matrix with the `daily_pct_change` data 
pd.plotting.scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1,figsize=(12,12))

# Show the plot
plt.show()

# March 8th 2019
# want to study linear regression on some stocks data
# code based on Python for Finance online course run by Data Camp

# Ordinary Least Square (OLS) regression analysis

# Isolate the adjusted closing price
all_adj_close = all_data[['Adj Close']]

# Calculate the returns 
# shift function obtains adjusted close data for the previous day
all_returns = np.log(all_adj_close / all_adj_close.shift(1))

# Isolate the AAPL returns 
aapl_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AAPL']
aapl_returns.index = aapl_returns.index.droplevel('Ticker')

# Isolate the AXP returns
axp_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AXP']
axp_returns.index = axp_returns.index.droplevel('Ticker')

# Isolate the AWK returns 
awk_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AWK']
awk_returns.index = awk_returns.index.droplevel('Ticker')

# Build up a new DataFrame with AAPL and AXP returns
return_data1 = pd.concat([aapl_returns, axp_returns], axis=1)[1:]
return_data1.columns = ['AAPL', 'AXP']

# Add a constant 
X = sm.add_constant(return_data1['AAPL'])

# Construct the model
model_1 = sm.OLS(return_data1['AXP'],X).fit()

# Print the summary; performs some standard statistical calculations for the given regression function
print(model_1.summary())

# Build up a new DataFrame with AAPL and AWK returns
return_data2 = pd.concat([aapl_returns, awk_returns], axis=1)[1:]
return_data2.columns = ['AAPL', 'AWK']

# Add a constant 
Y = sm.add_constant(return_data2['AAPL'])

# Construct the model
model_2 = sm.OLS(return_data2['AWK'],Y).fit()

# Print the summary; performs some standard statistical calculations for the given regression function
print(model_2.summary())


# Plot returns of AAPL and MSFT
plt.plot(return_data1['AAPL'], return_data1['AXP'], 'r.')

# Add an axis to the plot
ax = plt.axis()

# Initialize `x`
x = np.linspace(ax[0], ax[1] + 0.01)

# Plot the regression line
plt.plot(x, model_1.params[0] + model_1.params[1] * x, 'b', lw=2)

# Customize the plot
plt.grid(True)
plt.axis('tight')
plt.xlabel('Apple Returns')
plt.ylabel('American Express returns')

# Show the plot
plt.show()

# Plot returns of AAPL and AWK
plt.plot(return_data2['AAPL'], return_data2['AWK'], 'g.')

# Add an axis to the plot
ax = plt.axis()

# Initialize `x`
x = np.linspace(ax[0], ax[1] + 0.01)

# Plot the regression line
plt.plot(x, model_2.params[0] + model_2.params[1] * x, 'y', lw=2)

# Customize the plot
plt.grid(True)
plt.axis('tight')
plt.xlabel('Apple Returns')
plt.ylabel('American Water Works Company returns')

# Show the plot
plt.show()

# Plot the rolling correlation for AXP and AAPL
return_data1['AXP'].rolling(window=42).corr(return_data1['AAPL']).plot()

# Show the plot
plt.show()

# Plot the rolling correlation for AWK and AAPL
return_data2['AWK'].rolling(window=42).corr(return_data2['AAPL']).plot()

# Show the plot
plt.show()


