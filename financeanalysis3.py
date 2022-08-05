import pandas_datareader as pdr
import datetime 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import the `api` model of `statsmodels` under alias `sm`
import statsmodels.api as sm

#June 26th 2019

#real estate in China: Invesco China Real Estate ETF(TAO)
#real estate in Canada: FTSE Canadian Capped REIT Index ETF (VRE)
#real estate in USA: Vanguard Real Estate ETF (VNQ)
#Casella Waste Systems, Inc. (CWST) (green tech)
#iShares Gold Trust (IAU), tracks price of physical gold
#Vanguard Canadian Aggregate Bond Index ETF (VAB.TO)

chhou = pdr.get_data_yahoo('TAO', datetime.datetime(2008, 1, 1), datetime.datetime(2019, 6, 26))

cahou = pdr.get_data_yahoo('VRE.TO', datetime.datetime(2013, 1, 1), datetime.datetime(2019, 6, 26))

ushou = pdr.get_data_yahoo('VNQ', datetime.datetime(2005, 9, 30), datetime.datetime(2019, 6, 26))

casella = pdr.get_data_yahoo('CWST', datetime.datetime(1997, 10, 12), datetime.datetime(2019, 6, 26))

igold = pdr.get_data_yahoo('IAU', datetime.datetime(2005, 1, 25), datetime.datetime(2019, 6, 26))

abond = pdr.get_data_yahoo('VAB.TO', datetime.datetime(2011, 11, 30), datetime.datetime(2019, 6, 26))

# real estate plots for China, Canada and USA
grach = chhou['Adj Close'].plot(x='Adj Close', rot=0)

graca = cahou['Adj Close'].plot(x='Adj Close', rot=0)

graus = ushou['Adj Close'].plot(x='Adj Close', rot=0)

# plot for Waste management company CWST

graw = casella['Adj Close'].plot(x='Adj Close', rot=0)

#plot for a gold ETF
grago = igold['Adj Close'].plot(x='Adj Close', rot=0)

#plot for aggregate bond ETF
grabon = abond['Adj Close'].plot(x='Adj Close', rot=0)


#describe statistics

chhou['Adj Close'].describe()

# how to draw a best-fitting line or curve here? Linear regression? Neural Networks?

# June 27th 2019

# recover the original integer indices of the dataframe

l = []
n = 0
for i in chhou.index:
    n = n + 1
    l.append(n)
    
chhouprice = pd.DataFrame({'day':l})
chhouprice.index = chhou.index
chhouprice['Adj Close'] = chhou['Adj Close']



X = sm.add_constant(chhouprice['day'])
Y = chhouprice['Adj Close']
model = sm.OLS(Y,X).fit()
print(model.summary())


# plot price and best-fitting line together

from numpy.polynomial.polynomial import polyfit

x = chhouprice['day']
y = chhouprice['Adj Close']

b,m = polyfit(x,y,1)

plt.plot(x, y, '.')
plt.plot(x, b + m * x, '-')
plt.show()









