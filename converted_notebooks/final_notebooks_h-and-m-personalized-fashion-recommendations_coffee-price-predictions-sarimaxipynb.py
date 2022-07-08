#!/usr/bin/env python
# coding: utf-8

# ## Importing important packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = (10,5)
import seaborn as sns
sns.set_style('darkgrid')

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings('ignore')


# ## Importing and Preprocessing dataset

# In[ ]:


df_raw = pd.read_csv('../input/daily-coffee-price/coffee.csv')
df_raw.Date = pd.to_datetime(df_raw.Date,yearfirst=True)
df_raw.set_index('Date',inplace=True)
df = df_raw.asfreq('b','ffill')


# ## Exploratory Data Analysis

# In[ ]:


fig,axes = plt.subplots(2,2,figsize=[15,7])
fig.suptitle('Coffee Price',size=24)
## resampling to daily freq (original data)
axes[0,0].plot(df.Close)
axes[0,0].set_title("Daily",size=16)

## resampling to monthly freq 
axes[0,1].plot(df.Close.resample('M').mean())
axes[0,1].set_title("Monthly",size=16)

## resmapling to quarterly freq 
axes[1,0].plot(df.Close.resample('Q').mean())
axes[1,0].set_title('Quarterly',size=16)

## resampling to annualy freq
axes[1,1].plot(df.Close.resample('A').mean())
axes[1,1].set_title('Annualy',size=16)

plt.tight_layout()
plt.show()


# In[ ]:


data = df.Close.resample('Q').mean()


# ## Decomposing

# In[ ]:


decompose_result = seasonal_decompose(data,model='additive')
## Systematic Components 
trend = decompose_result.trend
seasonal = decompose_result.seasonal

## Non-Systematic Components
residual = decompose_result.resid
decompose_result.plot();


# ## Stationarity

# In[ ]:


def plot_rolling_stats(series,window=1):
    ## calculating the rolling mean and rolling standard deviation
    rol_mean = series.rolling(window).mean()
    rol_std  = series.rolling(window).std()
    
    ## ploting the results along side the original data
    fig = plt.figure(figsize=(10,5))
    orig = plt.plot(series,color='blue',label='Original')
    mean = plt.plot(rol_mean,color='red',label='Rolling mean')
    std  = plt.plot(rol_std,color='black',label='Rolling std')
    
    plt.title('Rolling Mean/Standard Deviation',size=20)
    plt.legend(loc='best')
    plt.show(block=False)


# In[ ]:


def stationarity_check(series):
    print('Results of Dickey Fuller Test:')
    dftest = adfuller(series, autolag='AIC') 

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value',
                                             '#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        
    print(dfoutput)


# In[ ]:


## original data
plot_rolling_stats(data,4)
stationarity_check(data)


# In[ ]:


## regular differentiation
plot_rolling_stats(data.diff()[1:],4)
stationarity_check(data.diff()[1:])


# ## Autocorrelation and Partial Autocorrelation

# In[ ]:


fig = plt.figure(figsize=(14,5))
ax_1 = fig.add_subplot(121)
plot_pacf(data,lags=12,zero=False,ax=ax_1)

ax_2 = fig.add_subplot(122)
plot_acf(data,lags=12,zero=False,ax=ax_2);


# ## Modeling

# In[ ]:


size = 0.8 ## train size
train, test = data.iloc[:int(size*len(data))], data.iloc[int(size*len(data)):]

model = SARIMAX(train,order=(2,1,2),seasonal_order=(1,1,1,4)).fit(disp=-1)
model.summary()


# In[ ]:


model.plot_diagnostics(figsize=(10,8))
plt.show()


# ## Predictions

# In[ ]:


predictions = model.get_prediction(start='2000-03-31',end='2022-06-30')
conf = predictions.conf_int()
test_conf = conf.loc[test.index[0]:]
## ploting results
plt.plot(predictions.predicted_mean[1:],color='red',label='predictions')
plt.plot(train,color='blue',label='original')
plt.plot(test,color='green',label='test')
plt.fill_between(test_conf.index, test_conf.iloc[:,0], test_conf.iloc[:,1], color='gray', alpha=.2,label='95% confidence')
plt.title('Original vs Predictions',size=20)
plt.legend(loc='best');


# ## Accuracy Metrics

# In[ ]:


from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error
print(f"Mean Absolute Error: {mean_absolute_error(data[1:],predictions.predicted_mean[1:])}")
print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(data[1:],predictions.predicted_mean[1:])}")

