import pandas_datareader as pdr
import datetime 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import the `api` model of `statsmodels` under alias `sm`
import statsmodels.api as sm



# March 20th 2019
# want to develop, test and improve a simple trading strategy by comparing two Simple Moving Averages (SMA)

# code based on Python for Finance online course run by DataCamp: https://www.datacamp.com/community/tutorials/finance-python-trading
# plan to follow along the "Building A Trading Strategy With Python" part

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


aapl = pdr.get_data_yahoo('AAPL', 
                          start=datetime.datetime(2006, 10, 1), 
                          end=datetime.datetime(2012, 1, 1))

# Initialize the short and long windows
short_window = 30
long_window = 100

# Initialize the `signals` DataFrame with the `signal` column
signals = pd.DataFrame(index=aapl.index)


# Create short simple moving average over the short window
signals['short_mavg'] = aapl['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

# Create long simple moving average over the long window
signals['long_mavg'] = aapl['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals according to long and short term SMA crosses
# https://www.youtube.com/watch?v=x4nO5yoarM0

signals['signal'] = np.sign(signals['short_mavg'] - signals['long_mavg'])

# crossing signals
signals['diff'] = signals['signal'] - signals['signal'].shift(1)

sig = signals[short_window + 1:]

sigdict = sig['diff'].to_dict()

turning_list = []

for k, v in sigdict.items():
    if v != 0:
        turning_list.append(k)

print(sig)


#plotting 


# Initialize the plot figure
fig = plt.figure()

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price in $')

# Plot the closing price
aapl['Close'].plot(ax=ax1, color='r', lw=2.)

# Plot the short and long moving averages
sig[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
# Plot the buy signals
ax1.plot(sig.loc[lambda sig: sig['diff'] == 2].index, 
         sig.short_mavg[sig['diff'] == 2.0],
         '^', markersize=10, color='m')
         
# Plot the sell signals
ax1.plot(sig.loc[lambda sig: sig['diff'] == -2.0].index, 
         sig.short_mavg[sig['diff'] == -2.0],
         'v', markersize=10, color='k')
         
# Show the plot
plt.show()


#Next projects:
#(0) proceed to the "Backtesting a Trading Strategy" section of this online course
#(1) apply the SMA strategy above on current data for ETFs
#(2) study the statistics for the time intervals between different crosses for short and long term SMAs, vary the time periods and stocks in different sectors as well as geographical locations

# building a simple portfolio with only AAPL stocks

port = pd.DataFrame(index=aapl.index)

# MVA crossing signals
port['crossing'] = (signals['diff'])/2

# stock prices
port['Close'] = aapl['Close']

portfolio = port[short_window + 1:]

# keeping track of cash changes
portfolio['cash'] = (100*portfolio['Close']*portfolio['crossing'])

portfolio['cumcash'] = portfolio['cash'].cumsum()

# keeping track of positions changes
portfolio['positions'] = (portfolio['crossing']*(-100))

portfolio['cumpositions'] = portfolio['positions'].cumsum()

# total equity 
portfolio['equity'] = portfolio['cumcash'] + portfolio['cumpositions']*portfolio['Close']

print(portfolio)







