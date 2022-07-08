#!/usr/bin/env python
# coding: utf-8

# # G-Research Crypto Forecasting
# 
# G-ReSearch Crypto Forecasting is a time series data, which contains daily price of cryptocurrencies. Our task is to  forecast short term returns in 14 popular cryptocurrencies.
# 
# The purpose of this notebook is to share basic approach to time series data analysis. 

# ## Contents
# 1. [Basic EDA](#eda)
# 2. [Time series feature engineering](#fe)
# 3. [Modeling](#model)
#  + [Time series cross validation](#tscv)
#  + [Hyperparameter tuning with Optuna](#optuna)

# ## Load packages and data

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf, arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from scipy.stats.stats import pearsonr

import time
from datetime import datetime
from tqdm import tqdm


# In[ ]:


train = pd.read_csv('../input/g-research-crypto-forecasting/train.csv')
asset_details = pd.read_csv('../input/g-research-crypto-forecasting/asset_details.csv')
sup_train = pd.read_csv('../input/g-research-crypto-forecasting/supplemental_train.csv')

print(train.shape)
print(sup_train.shape)


# ## 1. Basic EDA <a class="anchor" id="eda"></a>
# This part contains simple exploration of G-Research data. Before we get into EDA, we need to change format of timestamp feature and create some new features.

# In[ ]:


# change format of timestamp variable
train['timestamp'] = pd.to_datetime(train['timestamp'], unit = 's')

# Diff: difference between close and open price of crypto
train['Diff'] = train['Close'] - train['Open']


# In[ ]:


# order by asset_id (ascending)
asset_details = asset_details.sort_values(by = 'Asset_ID').reset_index(drop = True)
asset_details


# ### Comparison of average trading volume of assets
# This graph shows average trading volume of each asset. **Bitcoin, Dodgecoin and Ethereum** are 3 most frequently traded assets among 14 cryptos.

# In[ ]:


assets = train.groupby('Asset_ID')['Count'].mean().reset_index()

colors = ['lightgrey']*14
colors[1] = colors[4] = colors[6] = '#3366ff'
assets_bar = go.Bar(x = assets['Asset_ID'], y = assets['Count'], marker_color = colors)
data = [assets_bar]
layout = go.Layout(title = 'Average trading volume of each asset')
fig = go.Figure(data = data, layout = layout)

fig.update_traces(marker_line_width = 1,marker_line_color = "black")
fig.update_layout(
    title = {
        'text': 'Average trading volume of each asset',
        'y':0.90,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'} , 
    
    xaxis = dict(
        tickvals = list(range(0, 14)),
        ticktext = asset_details.Asset_Name
    ),
    template = "plotly_white")

fig.update_xaxes(title_text = 'Asset')
fig.update_yaxes(title_text = 'Trading volume')

fig


# ### Candlestick chart of recent 1 year asset price
# This is candlestick chart of recent 1 year price of specific asset. I plotted chart of 3 most frequently traded assets, Bitcoin, Dogecoin and Ethereum. Overall trend of price of those 3 assets are quite similar.
# 
# Reference: <https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition/notebook#Building-your-prediction-model>

# In[ ]:


def candelstick_chart(data, id_, title):
    data = data[data['Asset_ID'] == id_].reset_index(drop = True)
    data = data.set_index('timestamp')
    data = data.iloc[-365:,:]  # recent 1 year
    data['MA5'] = data['Close'].rolling(window = 5, min_periods = 1).mean()
    data['MA30'] = data['Close'].rolling(window = 30, min_periods = 1).mean()
    data['MA120'] = data['Close'].rolling(window = 120, min_periods = 1).mean()
    
    candlestick = go.Figure(data = [go.Candlestick(x =data.index, 
                                               open = data[('Open')], 
                                               high = data[('High')], 
                                               low = data[('Low')], 
                                               close = data[('Close')])])
    candlestick.update_xaxes(title_text = 'Time')

    candlestick.update_layout(
    title = {
        'text': '{:} Candelstick Chart'.format(title),
        'y':0.90,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'} , 
    template="plotly_white")

    candlestick.update_yaxes(title_text = 'Price in USD', ticksuffix = '$')
    return candlestick


# In[ ]:


candelstick_chart(train, 1, 'Bitcoin')


# In[ ]:


candelstick_chart(train, 4, 'Dodgecoin')


# In[ ]:


candelstick_chart(train, 6, 'Ethereum')


# ### Check overall trend: Moving Average
# To check overall trend of asset price, we use **moving average(MA)** in general. In finance, a moving average (MA) is a stock indicator that is commonly used in technical analysis. The reason for calculating the moving average of a stock is to help smooth out the price data by creating a constantly updated average price.
# 
#  + Simple moving average(SMA): calculation that takes the arithmetic mean of a given set of prices over the specific number of days in the past
#  + Exponential moving average(EMA): weighted average that gives greater importance to the price of a stock in more recent days, making it an indicator that is more responsive to new information.
#  
#  
# In this part, I'll calculate SMA of close price of asset for 5(weekly), 30(monthly) and 120 days. MA5 and MA30 shows short-term trend and MA120 shows long-term trend of asset price.

# In[ ]:


def ma_chart(data, id_, title):
    data = data[data['Asset_ID'] == id_].reset_index(drop = True)
    data = data.set_index('timestamp')
    data = data.iloc[-365:,:]  # recent 1 year
    data['MA5'] = data['Close'].rolling(window = 5, min_periods = 1).mean()
    data['MA30'] = data['Close'].rolling(window = 30, min_periods = 1).mean()
    data['MA120'] = data['Close'].rolling(window = 120, min_periods = 1).mean()
    
    ma = go.Figure(data = [go.Scatter(x = data.index, y = data['Close'], mode='lines', 
                                     name = 'Close', line = dict(color = 'black', width = 2))])
    ma.update_xaxes(title_text = 'Time')

    ma.update_layout(
    title = {
        'text': '{:} Moving Average'.format(title),
        'y':0.90,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'} , 
    template="plotly_white")
    
    ma.add_trace(go.Scatter(x = data.index, y = data['MA5'], mode='lines', 
                                     name='MA5', line = dict(color = 'red', width = 2)))
    ma.add_trace(go.Scatter(x = data.index, y = data['MA30'], mode='lines', 
                                     name='MA30', line = dict(color = 'blue', width = 2)))
    ma.add_trace(go.Scatter(x = data.index, y = data['MA120'], mode='lines', 
                                     name='MA120', line = dict(color = 'orange', width = 2)))

    ma.update_yaxes(title_text = 'Price in USD', ticksuffix = '$')
    return ma


# In[ ]:


# Moving average chart of Bitcoin
ma_chart(train, 1, 'Bitcoin')


# ### Missing values
# There are some missing values in **Target** variable and **VWAP**. All missing values in VWAP variable belong to Asset 10, Maker. Since there are only 9 missings in this variable, we'll just remove those data. Missing in Target will not be included in modeling.

# In[ ]:


# Target - missing 3% 
train.isna().sum().sort_values(ascending = False)


# In[ ]:


# check missing in VWAP
train[train['VWAP'].isna()]


# In[ ]:


# remove NA values in VWAP
train = train[train['VWAP'].isna() == False]


# ## 2. Time series Feature Engineering <a class="anchor" id="fe"></a>
# We will use asset Bitcoin data only for feature engineering and modeling in this notebook. For other 13 assets, you can repeat same process.

# In[ ]:


# Bitcoin only
data = train[train['Asset_ID'] == 1]


# In[ ]:


# remove rows without target 
data = data[data['Target'].isna() == False]
data


# ### Additional features
# I created several additional features for modeling, including **time related features**. In time series analysis, creating time related features can significantly increase model performance. I used 2 mostly used technique for time series analysis: **moving average** and **lagged features**.
# 
# 1. Moving average 
#  + Above explanation with graph
# <br>
# 2. Lagged feature
#  + In time series analysis, future value is greatly affected by past values. Those past values are **lags** and we use thos lag features to enhance model performance.

# In[ ]:


# df should be particular asset
def get_feats(df):
    df['upper_shadow'] = df['High'] - np.maximum(df['Close'], df['Open'])
    df['lower_shadow'] = np.minimum(df['Close'], df['Open']) - df['Low']
    
    # average trading volume
    df['avg_volume'] = df['Volume'] / df['Count']
    
    # average price
    df['avg_price'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # difference between open and close price
    df.rename(columns = {'Diff': 'diff_open_close'})
    
    # moving average features - short term and long term
    df['ma_20'] = df['Close'].rolling(window = 20, min_periods = 1).mean()
    df['ma_120'] = df['Close'].rolling(window = 120, min_periods = 1).mean()
    
    # lagged features
    lags_ = [5, 20, 60, 120]
    for lag in lags_:
        df['lag_' + str(lag)] = df['Target'].shift(lag)
        
    df = df.fillna(0)
    
    return df


# In[ ]:


# lagged features
data = get_feats(data)


# #### FE function
# Below function creates time-series features of each assets. You can just enter asset_id in **asset_feats** function to get features for that particular asset.

# In[ ]:


# feature engineering for other assets 
def asset_feats(df, asset_id):
    data = df[df['Asset_ID'] == asset_id]
    data = data[data['Target'].isna() == False]
    data = get_feats(data)
    return data


# ## 3. Modeling <a class="anchor" id="model"></a>

# ### (1) Time Series Cross Validation <a class="anchor" id="tscv"></a>
# For time series modeling, I will use boosting regressors, LGBMRegressor and XGBRegressor, which show great performance in general. First step is to compare performance of two models with time series cross validation. In time series analysis, it's not recommended to apply KFold or Stratified KFold since observations in past influences current values. Instead, we use **Time series split** in this case. Below image shows how data is splitted if you apply time series split. Observations from the training set occur before their corresponding validation set. We will use 10-fold time series cross validation in this notebook. (Evaluation metric is Pearson Correlation Coefficient)
# <br>
# ![tcsv](https://miro.medium.com/max/1204/1*qvdnPF8ETV9mFdMT0Y_BBA.png)

# In[ ]:


# create X and y(target)
X = data.drop(['timestamp', 'Asset_ID'], axis = 1)
y = data['Target']


# In[ ]:


# 10-fold time series cross validation
def timecv_model(model, X, y):
    tfold = TimeSeriesSplit(n_splits = 10)
    pcc_list = []
    for _, (train_index, test_index) in tqdm(enumerate(tfold.split(X), start=1)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf = model.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pcc = pearsonr(pred, y_test) 
        pcc_list.append(pcc[0])
    
    return pcc_list

def cv_result(model, X, y):
    model_name = model.__class__.__name__
    pcc_ = timecv_model(model, X, y)
    for i, pcc in enumerate(pcc_):
        print(f'{i}th fold: {model_name} PCC: {pcc:.4f}')
    print(f'\n{model_name} average PCC: {np.mean(pcc_):.4f}')


# Below are results of 10-fold time series cross validation(TSCV) of LGBMRegressor and XGBRegressor. It takes much longer time in XGBRegressor. Parameters are randomly selected, just to compare general performance of two models. We will choose better model with TSCV and then conduct hyperparameter tuning.

# In[ ]:


lgb_model = LGBMRegressor(n_estimators = 1500,
                      max_depth = 10,
                      num_leaves = 20,
                      colsample_bytree = 0.8,
                      subsample = 0.7,
                      seed = 0)

cv_result(lgb_model, X, y)


# In[ ]:


#xgb_model = XGBRegressor(n_estimators = 1500,
                         #max_depth = 10,
                         #min_child_weight = 5,
                         #gamma = 0.1)

#cv_result(xgb_model, X, y)


# ### (2) Optuna
# Now, tune parameters to get optimal parameter for LGBMRegressor. There are many hyperparameter tuning techinques: GridSearchCV, RandomSearchCV, Bayesian Optimization..etc..
# 
# But I will try **Optuna** for hyperparameter tuning. Optuna is widely used in many data analysis platforms these days to get optimal parameters for model. Information about optuna is well explained [here](https://medium.com/@kalyaniavhale7/understanding-of-optuna-a-machine-learning-hyperparameter-optimization-framework-ed31ebb335b9). (I summarized explanations about optuna referring to this link.)

# In[ ]:


data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d')

print('Earliest date: ', min(data['timestamp']))
print('Lastest date: ', max(data['timestamp']))


# Bitcoin data starts from 2018-01-01 to 2021-09-20. I diveded full dataset into train and valid dataset first: Train data contains data before 2020-09-20 and validation data contains data after 2020-09-20. The ratio of train and valid size is apporixmately 73:27.

# In[ ]:


train = data[data['timestamp'] < '2020-09-20']
valid = data[data['timestamp'] >= '2020-09-20']

train.drop(['timestamp', 'Asset_ID'], axis = 1, inplace = True)
valid.drop(['timestamp', 'Asset_ID'], axis = 1, inplace = True)

X_train = train.drop(['Target'], axis = 1)
y_train = train['Target']
X_valid = valid.drop(['Target'], axis = 1)
y_valid = valid['Target']

print(X_train.shape)
print(X_valid.shape)


# ### Objective function <a class="anchor" id="optuna"></a>
# Our objective of hyperparameter tuning is to find parameter that maximizes or minimized output of objective function. If evaluation metric is logloss, we have to minimized objective function but if it is accuracy, we have to maximize objective function. During the optimization, Optuna repeatedly calls and evaluates the objective function with different parameters.
# 
# In objective function, we define parameter search space.
#  + categorical parameter: optuna.trial.Trial.suggest_categorical()
#  + integer parameter: optuna.trial.Trial.suggest_int()
#  + float parameter: optuna.trial.Trial.suggest_float()
#  
#  
#  
# In Optuna, we use the study object to manage optimization. Method create_study() returns a study object. A study object has useful properties for analyzing the optimization outcome. 
#  + create_study()

# In[ ]:


from optuna.samplers import TPESampler
import optuna

sampler = TPESampler(seed = 0)

def objective(trial):
    params = {
        'objective': 'regression',
        'verbose': -1,
        'max_depth': trial.suggest_int('max_depth',5, 20),
        'num_leaves': trial.suggest_int('num_leaves', 10, 40),
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2500),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1)}
    
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)],
         verbose = 0, early_stopping_rounds = 50)
    pred = model.predict(X_valid)
    pcc = pearsonr(pred, y_valid)[0]
    return pcc

study_model = optuna.create_study(direction = 'maximize', sampler = sampler)
study_model.optimize(objective, n_trials = 20) 


# In[ ]:


# select best trial and parameter
trial = study_model.best_trial
best_params = trial.params

print('Best params from optuna: \n', best_params)


# #### Plots from optuna
# Optuna provides visualization of result from hyperparmeter tuning.
# 
# 1. plot_optimization_history
#   + plot optimization history of all trials in a study
# <br>
# 2. plot_slice
#   + plot the parameter relationship as slice plot in a study.
# <br>  
# 3. plot_param_importances
#   + plot hyperparameter importances 

# In[ ]:


optuna.visualization.plot_optimization_history(study_model)


# In[ ]:


optuna.visualization.plot_slice(study_model)


# In[ ]:


optuna.visualization.plot_param_importances(study_model)


# ### (3) Prediction with selected optimal parameter

# In[ ]:


opt_model = LGBMRegressor(**best_params)

cv_result(opt_model, X, y)

