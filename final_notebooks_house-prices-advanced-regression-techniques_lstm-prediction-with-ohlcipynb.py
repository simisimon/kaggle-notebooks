#!/usr/bin/env python
# coding: utf-8

# # Import required libraries

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Reshape
from keras import backend as K
from keras import regularizers 
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from tensorflow.keras.losses import Loss


# In[ ]:


import sys, warnings, time, os, copy, gc, re, random
import pickle as pkl
warnings.filterwarnings('ignore')
from IPython.display import display
import matplotlib.pyplot as plt
# pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
# pd.set_option("display.max_colwidth", 10000)
import seaborn as sns
sns.set()
from pandas.io.json import json_normalize
from pprint import pprint
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()
from datetime import datetime, timedelta

# Pre-Processing
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

# Model
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, \
multiply, concatenate, Flatten, Activation, dot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from keras.callbacks import EarlyStopping
import pydot as pyd
from keras.utils.vis_utils import plot_model, model_to_dot
keras.utils.vis_utils.pydot = pyd


# # Load Dataset

# In[ ]:


stock_price_df = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")


# # A glance at the data

# In[ ]:


print('(rows, columns) =', stock_price_df.shape)
stock_price_df.tail()


# In[ ]:


stock_price_df['ExpectedDividend'] = stock_price_df['ExpectedDividend'].fillna(0)
stock_price_df['SupervisionFlag'] = stock_price_df['SupervisionFlag'].map({True: 1, False: 0})
stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
stock_price_df.info()


# # Import stock list
# 

# In[ ]:


stock_list = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")


# In[ ]:


stock_list = stock_list[['SecuritiesCode','NewMarketSegment','33SectorCode','17SectorCode','Universe0','Section/Products','NewIndexSeriesSize']]
stock_list = stock_list.replace(np.nan,'-')
stock_list['Universe0'] = np.where(stock_list['Universe0'], 1, 0)
stock_list = stock_list.drop_duplicates()
stock_list


# # Some Feature Engineering

# In[ ]:


def FE(stock_price_df):
    stock_price_df['BOP'] = (stock_price_df['Open']-stock_price_df['Close'])/(stock_price_df['High']-stock_price_df['Low'])
    stock_price_df['wp'] = (stock_price_df['Open']+stock_price_df['High']+stock_price_df['Low'])/3
    stock_price_df['TR'] = stock_price_df['High'] - stock_price_df['Low']
    # stock_price_df['AD'] = ta.AD(High, Low, Close, Volume)
    # stock_price_df['OBV']  = ta.OBV(Close, Volume)
    stock_price_df['OC'] = stock_price_df['Open'] * stock_price_df['Close']
    stock_price_df['HL'] = stock_price_df['High'] * stock_price_df['Low']
    stock_price_df['logC'] = np.log(stock_price_df['Close']+1)
    stock_price_df['OHLCstd'] = stock_price_df[['Open','Close','High','Low']].std(axis=1)
    stock_price_df['OHLCskew'] = stock_price_df[['Open','Close','High','Low']].skew(axis=1)
    stock_price_df['OHLCkur'] = stock_price_df[['Open','Close','High','Low']].kurtosis(axis=1)
    stock_price_df['Cpos'] = (stock_price_df['Close']-stock_price_df['Low'])/(stock_price_df['High']-stock_price_df['Low']) -0.5
    stock_price_df['bsforce'] = stock_price_df['Cpos'] * stock_price_df['Volume']
    stock_price_df['Opos'] = (stock_price_df['Open']-stock_price_df['Low'])/(stock_price_df['High']-stock_price_df['Low']) -0.5
    stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
    stock_price_df['weekday'] = stock_price_df['Date'].dt.weekday+1
    stock_price_df['Monday'] = np.where(stock_price_df['weekday']==1,1,0)
    stock_price_df['Tuesday'] = np.where(stock_price_df['weekday']==2,1,0)
    stock_price_df['Wednesday'] = np.where(stock_price_df['weekday']==3,1,0)
    stock_price_df['Thursday'] = np.where(stock_price_df['weekday']==4,1,0)
    stock_price_df['Friday'] = np.where(stock_price_df['weekday']==5,1,0)
    return stock_price_df
stock_price_df = FE(stock_price_df)
stock_price_df = pd.merge(stock_price_df,stock_list, on='SecuritiesCode')


# In[ ]:


subf = ['Open', 'High', 'Low', 'Close',
       'Volume', 'ExpectedDividend',
       'SupervisionFlag', 'BOP', 'wp', 'TR', 'OC', 'HL', 'logC',
       'OHLCstd', 'OHLCskew', 'OHLCkur', 'Cpos', 'bsforce', 'Opos']


# # standardize features according to daily stock data (at the same day only)

# In[ ]:


def daily_standardize(df,col):
    avg = df[[col,'Date']].groupby('Date').mean()
    avg.columns = ['avg']
    avg['Date'] = avg.index
    avg = avg.reset_index(drop=True)
    std = df[[col,'Date']].groupby('Date').std()
    std.columns = ['std']
    std['Date'] = std.index
    std = std.reset_index(drop=True)
    df = pd.merge(df, avg, on='Date')
    df = pd.merge(df,std,on='Date')
    df[col] = (df[col] - df['avg'])/df['std']
    df = df.drop(['avg','std'],axis=1)
    df[col] = df[col].fillna(0)
    return df


# In[ ]:


def daily_normalize(df,col):
    avg = df[[col,'Date']].groupby('Date').min()
    avg.columns = ['min']
    avg['Date'] = avg.index
    avg = avg.reset_index(drop=True)
    std = df[[col,'Date']].groupby('Date').max()
    std.columns = ['max']
    std['Date'] = std.index
    std = std.reset_index(drop=True)
    df = pd.merge(df, avg, on='Date')
    df = pd.merge(df,std,on='Date')
    df[col] = (df[col] - df['min'])/(df['max']-df['min'])
    df = df.drop(['min','max'],axis=1)
    df[col] = df[col].fillna(0)
    return df


# In[ ]:


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()


# In[ ]:


for i in subf:
    stock_price_df = daily_standardize(stock_price_df,i)


# In[ ]:


stock_price_df = stock_price_df.fillna(0)
stock_price_df.head()


# In[ ]:


stock_price_df.columns


# # standardize "Target" as well

# In[ ]:


stock_price_df = daily_normalize(stock_price_df,'Target')


# In[ ]:


stock_price_df['Target'].head()


# In[ ]:


listSC = stock_price_df['SecuritiesCode'].unique().tolist()
X = []
Y = []
TIME_WINDOW = 15

for sc in tqdm(listSC):
    dfTemp = stock_price_df[stock_price_df['SecuritiesCode'] == sc]
    # dfTemp = dfTemp.interpolate()
    dfTemp = dfTemp.dropna(how='any')
    iterN = (dfTemp.shape[0] - TIME_WINDOW + 1)
    
    for i in range(iterN):
        x = dfTemp['Close'].iloc[i:(i+TIME_WINDOW)].to_numpy().reshape([TIME_WINDOW, 1])
        y = dfTemp['Target'].iat[(i+TIME_WINDOW-1)].reshape([1, 1])

        X.append(x)
        Y.append(y)

X = np.array(X)
Y = np.array(Y) 


# In[ ]:


np.shape(X), np.shape(Y)


# In[ ]:


def model(dfx, dfy, n_hidden=50):
    epc = 3
    batch_size=4096
    input_train = Input(shape=(dfx.shape[1], dfx.shape[2]))
    output_train = Input(shape=(dfy.shape[1], dfy.shape[2]))

    encoder_last_h1, encoder_last_h2, encoder_last_c = LSTM(
    n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2,
    return_sequences=False, return_state=True)(input_train)

    encoder_last_h1 = BatchNormalization(momentum=0.6)(encoder_last_h1)
    encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)
    
    decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)
    decoder = LSTM(n_hidden, activation='elu', dropout=0.2,
    recurrent_dropout=0.2, return_state=False,
    return_sequences=True)(decoder, initial_state=[encoder_last_h1, encoder_last_c])

    out = TimeDistributed(Dense(output_train.shape[2], activation='relu'))(decoder)

    model = Model(inputs=input_train, outputs=out)
    opt = Adam(lr=0.01, clipnorm=1)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    print(model.summary())
    es = EarlyStopping(monitor='val_loss', mode='min', patience=30)
    history = model.fit(dfx, dfy, validation_split=0.2, epochs=epc, verbose=1,
                        callbacks=[es], batch_size=batch_size)
    train_mae = history.history['mae']
    valid_mae = history.history['val_mae']
    
    return model


# In[ ]:


a=model(X,Y)
a.save(f'lstm_model_naive.h5')


# In[ ]:


def prepare_data(df, listSC):
    '''Pre-processing and prepare data with target shape securitiesCode y securitiesCode.'''
    
    X = []
    for sc in tqdm(listSC):
        dfTemp = df[df['SecuritiesCode'] == sc]
        #dfTemp['Close'] = dfTemp['Close'].apply(np.log1p)
       # dfTemp[['Close','Open','High','Low','BOP','Cpos','wp','weekday']] = dfTemp[['Close','Open','High','Low','BOP','Cpos','wp','weekday']].values.reshape([10, 8])
       # dfTemp = dfTemp.interpolate()
        dfTemp = dfTemp.fillna(0)
        x = dfTemp['Close'].to_numpy().reshape([TIME_WINDOW, 1])
        X.append(x)
        
    X = np.array(X)
#    assert X.shape == (2000, TIME_WINDOW, 1), "Shape of X is not correct."
    return X


# In[ ]:


def predict_rank(X, model, sample_prediction, listSC):
    '''Predict Target value and make a ranking. Return submittion df.'''
    
    Y = model.predict(X)
    print(Y)
    dfY = pd.DataFrame(Y.reshape(-1, 1))
    dfSC = pd.DataFrame(listSC)
    dfTemp = pd.concat([dfSC, dfY], axis=1)
    dfTemp.columns = ['SecuritiesCode', 'prediction']
    dfTemp['Rank'] = dfTemp["prediction"].rank(ascending=False,method="first") -1
    dfTemp['Rank'] = dfTemp['Rank'].astype(int)
    dfTemp = dfTemp.drop('prediction', axis=1)
    sample_prediction = sample_prediction.drop('Rank', axis=1)
    dfSub = sample_prediction.merge(dfTemp, on='SecuritiesCode', how='left')

 #   assert dfSub.shape == (2000, 3), "Shape of dfSub is not correct."
    return dfSub    


# In[ ]:


import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test files
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    
    prices['ExpectedDividend'] = prices['ExpectedDividend'].fillna(0)
    prices['SupervisionFlag'] = prices['SupervisionFlag'].map({True: 1, False: 0})
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = pd.merge(prices,stock_list, on='SecuritiesCode')
    prices = FE(prices)
    prices[subf] = StandardScaler().fit_transform(prices[subf])
    prices = prices.fillna(0)
    stock_price_df = pd.concat([stock_price_df, prices], axis=0).reset_index(drop=True)
    dfTarget = stock_price_df.tail(TIME_WINDOW*2000)
    X = prepare_data(dfTarget, listSC)
    dfSub = predict_rank(X, a, sample_prediction, listSC)
    print(dfSub)
    env.predict(dfSub)   # register your predictions


# In[ ]:


np.shape(X)


# In[ ]:


sum(sum(np.isinf(X)))


# In[ ]:


len(np.unique(dfSub['Rank']))

