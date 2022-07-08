#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 
# # # LightGBM for prediction
# 
# LightGBM is a version of gradient boosting developed and maintained by Microsoft, it has become a goto algorithm in kaggle competitions . LightGBM algorithm has become bigger challenger along with CATboost algorithm to traditional dominate XBBoost Algorithm in gaggle competitions. 
# 
# 
# LightGBM Advantages
# * Faster training speed and higher efficiency.
# * Lower memory usage.
# * Better accuracy.
# * Support of parallel and GPU learning.
# * Capable of handling large-scale data.
# 
# LightGBM tree tends to grows leaf-wise  which will converge faster than depth wise ones trees. But they can be more prone to overfitting.
# 
# One significant advantage of LightGBM algorithm is the way it works seamlessly with categorical features, in general for categorical features we do one-hot encoding of the features which will make unbalanced trees which can impact the accuracy. With LightGBM algorithm we have categorical_feature attribute, can specify categorical features (without one-hot encoding) for the model. It will enable Categorical features to be encoded as non-negative integers less than Int32.MaxValue starting form zero. 

# # # **Data Description**
# 
# **train.csv**
# 
# timestamp: All timestamps are returned as second Unix timestamps (the number of seconds elapsed since 1970-01-01 00:00:00.000 UTC). Timestamps in this dataset are multiple of 60, indicating minute-by-minute data.
# 
# Asset_ID: The asset ID corresponding to one of the crytocurrencies (e.g. Asset_ID = 1 for Bitcoin). The mapping from Asset_ID to crypto asset is contained in asset_details.csv.
# 
# Count: Total number of trades in the time interval (last minute).
# 
# Open: Opening price of the time interval (in USD).
# 
# High: Highest price reached during time interval (in USD).
# 
# Low: Lowest price reached during time interval (in USD).
# 
# Close: Closing price of the time interval (in USD).
# 
# Volume: Quantity of asset bought or sold, displayed in base currency USD.
# 
# VWAP: The average price of the asset over the time interval, weighted by volume. VWAP is an aggregated form of trade data.
# 
# Target: Residual log-returns for the asset over a 15 minute horizon.
# supplemental_train.csv After the submission period is over this file's data will be replaced with cryptoasset prices from the submission period. In the Evaluation phase, the train, train supplement, and test set will be contiguous in time, apart from any missing data. The current copy, which is just filled approximately the right amount of data from train.csv is provided as a placeholder.
# 
# asset_details.csv Provides the real name and of the cryptoasset for each Asset_ID and the weight each cryptoasset receives in the metric. Weights are determined by the logarithm of each product's market cap (in USD), of the cryptocurrencies at a fixed point in time. Weights were assigned to give more relevance to cryptocurrencies with higher market volumes to ensure smaller cryptocurrencies do not disproportionately impact the models.
# 
# example_sample_submission.csv An example of the data that will be delivered by the time series API. The data is just copied from train.csv.
# 
# example_test.csv An example of the data that will be delivered by the time series API.

# In[ ]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV


# In[ ]:


### Loading datasets

path = "/kaggle/input/g-research-crypto-forecasting/"
df_train = pd.read_csv(path + "train.csv")
df_test = pd.read_csv(path + "example_test.csv")
df_asset_details = pd.read_csv(path + "asset_details.csv")
df_supp_train = pd.read_csv(path + "supplemental_train.csv")


# In[ ]:


df_train.head()


# In[ ]:


#df_asset_details.head().sort()
df_asset_details.sort_values(by='Asset_ID')


# In[ ]:


# Summary Statistics of overall coins.
df_train.describe()


# In[ ]:


df_train.isna().sum()


# In[ ]:


# Define plot space
fig, ax = plt.subplots(figsize=(10, 6))

# Create bar plot
ax.bar(df_asset_details['Asset_Name'], 
       df_asset_details['Weight'])
plt.xlabel('Asset Name')
plt.ylabel('Weight')
plt.xticks(rotation=70)
plt.show()


# In[ ]:


#setting date format
ds_train_copy = df_train
ds_train_copy['date'] = pd.to_datetime(ds_train_copy['timestamp'], unit='s')


# In[ ]:


bnc = ds_train_copy[ds_train_copy['Asset_ID']==0].set_index('timestamp')
btc = ds_train_copy[ds_train_copy['Asset_ID']==1].set_index('timestamp')
btcsh = ds_train_copy[ds_train_copy['Asset_ID']==2].set_index('timestamp')
car = ds_train_copy[ds_train_copy['Asset_ID']==3].set_index('timestamp')
dog = ds_train_copy[ds_train_copy['Asset_ID']==4].set_index('timestamp')
eos = ds_train_copy[ds_train_copy['Asset_ID']==5].set_index('timestamp')
eth = ds_train_copy[ds_train_copy['Asset_ID']==6].set_index('timestamp')
eth_csc = ds_train_copy[ds_train_copy['Asset_ID']==7].set_index('timestamp')
iot = ds_train_copy[ds_train_copy['Asset_ID']==8].set_index('timestamp')
ltc = ds_train_copy[ds_train_copy['Asset_ID']==9].set_index('timestamp')
mak = ds_train_copy[ds_train_copy['Asset_ID']==10].set_index('timestamp')
mon = ds_train_copy[ds_train_copy['Asset_ID']==11].set_index('timestamp')
ste = ds_train_copy[ds_train_copy['Asset_ID']==12].set_index('timestamp')
tro = ds_train_copy[ds_train_copy['Asset_ID']==13].set_index('timestamp')
eth.head()


# Check the train dataset start and end for each asset

# In[ ]:


beg_btcsh = btcsh.index[0].astype('datetime64[s]')
end_btcsh = btcsh.index[-1].astype('datetime64[s]')
beg_bnc = bnc.index[0].astype('datetime64[s]')
end_bnc = bnc.index[-1].astype('datetime64[s]')
beg_btc = btc.index[0].astype('datetime64[s]')
end_btc = btc.index[-1].astype('datetime64[s]')
beg_eos = eos.index[0].astype('datetime64[s]')
end_eos = eos.index[-1].astype('datetime64[s]')
beg_eth_csc = eth_csc.index[0].astype('datetime64[s]')
end_eth_csc = eth_csc.index[-1].astype('datetime64[s]')
beg_ltc = ltc.index[0].astype('datetime64[s]')
end_ltc = ltc.index[-1].astype('datetime64[s]')
beg_mon = mon.index[0].astype('datetime64[s]')
end_mon = mon.index[-1].astype('datetime64[s]')
beg_tro = tro.index[0].astype('datetime64[s]')
end_tro = tro.index[-1].astype('datetime64[s]')
beg_eth = eth.index[0].astype('datetime64[s]')
end_eth = eth.index[-1].astype('datetime64[s]')
beg_ste = ste.index[0].astype('datetime64[s]')
end_ste = ste.index[-1].astype('datetime64[s]')
beg_car = car.index[0].astype('datetime64[s]')
end_car = car.index[-1].astype('datetime64[s]')
beg_iot = iot.index[0].astype('datetime64[s]')
end_iot = iot.index[-1].astype('datetime64[s]')
beg_mak = mak.index[0].astype('datetime64[s]')
end_mak = mak.index[-1].astype('datetime64[s]')
beg_dog = dog.index[0].astype('datetime64[s]')
end_dog = dog.index[-1].astype('datetime64[s]')
print('Bitcoin Cash     :', beg_btcsh, 'to', end_btcsh)
print('Binance Coin     :', beg_bnc, 'to', end_bnc)
print('Bitcoin          :', beg_btc, 'to', end_btc)
print('EOS IO           :', beg_eos, 'to', end_eos)
print('Etherium Classic :', beg_eth_csc, 'to', end_eth_csc)
print('Ethereum         :', beg_eth, 'to', end_eth)
print('Lite Coin        :', beg_ltc, 'to', end_ltc)
print('Monero           :', beg_mon, 'to', end_mon)
print('TRON             :', beg_tro, 'to', end_tro)
print('Stellar          :', beg_ste, 'to', end_ste)
print('Cardano          :', beg_car, 'to', end_car)
print('IOTA             :', beg_iot, 'to', end_iot)
print('Maker            :', beg_mak, 'to', end_mak)
print('Dogecoin         :', beg_dog, 'to', end_dog)


# In[ ]:


btc=df_train[df_train['Asset_ID']==1].set_index('timestamp')
btc_mini = btc.iloc[-200:]


# Check the missing timestamp for each data

# In[ ]:


(eth.index[1:]-eth.index[:-1]).value_counts().head()


# Reindex all asset to remove the missing data in each timestamp

# In[ ]:


eth     = eth.reindex(range(eth.index[0],eth.index[-1]+60,60),method='pad')
btc     = btc.reindex(range(btc.index[0],btc.index[-1]+60,60),method='pad')
btcsh   = btcsh.reindex(range(btcsh.index[0],btcsh.index[-1]+60,60),method='pad')
bnc     = bnc.reindex(range(bnc.index[0],bnc.index[-1]+60,60),method='pad')
eos     = eos.reindex(range(eos.index[0],eos.index[-1]+60,60),method='pad')
eth_csc = eth_csc.reindex(range(eth_csc.index[0],eth_csc.index[-1]+60,60),method='pad')
ltc     = ltc.reindex(range(ltc.index[0],ltc.index[-1]+60,60),method='pad')
mon     = mon.reindex(range(mon.index[0],mon.index[-1]+60,60),method='pad')
tro     = tro.reindex(range(tro.index[0],tro.index[-1]+60,60),method='pad')
ste     = ste.reindex(range(ste.index[0],ste.index[-1]+60,60),method='pad')
car     = car.reindex(range(car.index[0],car.index[-1]+60,60),method='pad')
iot     = iot.reindex(range(iot.index[0],iot.index[-1]+60,60),method='pad')
mak     = mak.reindex(range(mak.index[0],mak.index[-1]+60,60),method='pad')
dog     = dog.reindex(range(dog.index[0],dog.index[-1]+60,60),method='pad')


# Show all close value for each asset from beginning until end

# In[ ]:


(eth.index[1:]-eth.index[:-1]).value_counts().head()


# In[ ]:


# Define plot space
fig, ax = plt.subplots(5, 3, figsize=(18, 22))

# Bitcoin Cash
ax[0, 0].plot(btcsh['Close'], label='BTCSH')
ax[0, 0].set_title('Bitcoin Cash')
ax[0, 1].plot(bnc['Close'], label='BNC')
ax[0, 1].set_title('Binance Coin')
ax[0, 2].plot(btc['Close'], label='BTC')
ax[0, 2].set_title('Bitcoin')
ax[1, 0].plot(eos['Close'], label='EOS')
ax[1, 0].set_title('EOS.IO')
ax[1, 1].plot(eth_csc['Close'], label='ETH_CSC')
ax[1, 1].set_title('Etherium Cash')
ax[1, 2].plot(ltc['Close'], label='LTC')
ax[1, 2].set_title('Lite Coin')
ax[2, 0].plot(mon['Close'], label='MON')
ax[2, 0].set_title('Monero')
ax[2, 1].plot(tro['Close'], label='TRO')
ax[2, 1].set_title('TRON')
ax[2, 2].plot(eth['Close'], label='ETH')
ax[2, 2].set_title('Etherium')
ax[3, 0].plot(ste['Close'], label='STE')
ax[3, 0].set_title('Stelar')
ax[3, 1].plot(car['Close'], label='CAR')
ax[3, 1].set_title('Cardano')
ax[3, 2].plot(iot['Close'], label='IOT')
ax[3, 2].set_title('IOTA')
ax[4, 0].plot(mak['Close'], label='MAK')
ax[4, 0].set_title('Maker')
ax[4, 1].plot(dog['Close'], label='DOG')
ax[4, 1].set_title('Dogecoin')
plt.show()


# In[ ]:


#Bit coin candle stick diagram
import plotly.graph_objects as go

fig = go.Figure(data=[go.Candlestick(x=btc_mini.index, open=btc_mini['Open'], high=btc_mini['High'], low=btc_mini['Low'], close=btc_mini['Close'])])
fig.show()


# In[ ]:


df_train.isna().sum()


# In[ ]:


eth.tail()


# In[ ]:


(eth.index[1:]-eth.index[:-1]).value_counts().head()


# In[ ]:


import matplotlib.pyplot as plt

# plot vwap time series for both chosen assets
f = plt.figure(figsize=(15,4))

# fill missing values for BTC
btc = btc.reindex(range(btc.index[0],btc.index[-1]+60,60),method='pad')

ax = f.add_subplot(121)
plt.plot(btc['Close'], label='BTC')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Bitcoin')

ax2 = f.add_subplot(122)
ax2.plot(eth['Close'], color='brown', label='ETH')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Ethereum')

plt.tight_layout()
plt.show()


# Coin Correlation for last 12000 Minutes

# In[ ]:


data =df_train[-12000:]
check = pd.DataFrame()
for i in data.Asset_ID.unique():
    check[i] = data[data.Asset_ID==i]['Target'].reset_index(drop=True) 
    
plt.figure(figsize=(10,8))
sns.heatmap(check.dropna().corr(), vmin=-1.0, vmax=1.0, annot=True, cmap='coolwarm', linewidths=0.1)
plt.show()


# In[ ]:


# creating the ratio

def hlco_ratio(df): 
    return (df['High'] - df['Low'])/(df['Close']-df['Open'])
def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])
def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

def get_features(df):
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['hlco_ratio'] = hlco_ratio(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    return df_feat


# In[ ]:


# train test split df_train into 80% train rows and 20% valid rows
train_data = df_train
# train_data = df_train.sample(frac = 0.8)
# valid_data = df_train.drop(train_data.index)

def get_Xy_and_model_for_asset(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    
    df = df.sample(frac=0.2)
    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    df_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_proc = df_proc.dropna(how="any")
    
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]   
    model = LGBMRegressor()
    model.fit(X, y)
    return X, y, model

Xs = {}
ys = {}
models = {}

for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
    X, y, model = get_Xy_and_model_for_asset(train_data, asset_id)       
    try:
        Xs[asset_id], ys[asset_id], models[asset_id] = X, y, model
    except: 
        Xs[asset_id], ys[asset_id], models[asset_id] = None, None, None 


# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = {
    # 'max_depth': range (2, 10, 1),
    'num_leaves': range(21, 161, 10),
    'learning_rate': [0.1, 0.01, 0.05]
}

new_models = {}
for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print("GridSearchCV for: " + asset_name)
    grid_search = GridSearchCV(
        estimator=get_Xy_and_model_for_asset(df_train, asset_id)[2], # bitcoin
        param_grid=parameters,
        n_jobs = -1,
        cv = 5,
        verbose=True
    )
    grid_search.fit(Xs[asset_id], ys[asset_id])
    new_models[asset_id] = grid_search.best_estimator_
    grid_search.best_estimator_


# In[ ]:


for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Tuned model for {asset_name:<1} (ID={asset_id:})")
    print(new_models[asset_id])


# In[ ]:


import gresearch_crypto
env = gresearch_crypto.make_env()
iter_test = env.iter_test()

for i, (df_test, df_pred) in enumerate(iter_test):
    for j , row in df_test.iterrows():        
        if new_models[row['Asset_ID']] is not None:
            try:
                model = new_models[row['Asset_ID']]
                x_test = get_features(row)
                y_pred = model.predict(pd.DataFrame([x_test]))[0]
                df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = y_pred
            except:
                df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = 0
                traceback.print_exc()
        else: 
            df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = 0  
    
    env.predict(df_pred)

