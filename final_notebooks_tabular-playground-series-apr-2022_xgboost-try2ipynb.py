#!/usr/bin/env python
# coding: utf-8

# JPX is hosting this competition and is supported by AI technology company AlpacaJapan Co.,Ltd.
# 
# This competition will compare your models against real future returns after the training phase is complete. The competition will involve building portfolios from the stocks eligible for predictions (around 2,000 stocks). Specifically, each participant ranks the stocks from highest to lowest expected returns and is evaluated on the difference in returns between the top and bottom 200 stocks.
# 
# In this notebook I will be doing some basic EDA and some feature creation to feed into high depth XGBoost decission trees.

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
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm import tqdm
import jpx_tokyo_market_prediction
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
import gc
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Loading the training files, suppliment files and the file contais list of all the stocks.

# In[ ]:


stock_list = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
financials = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/financials.csv")
options = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/options.csv")
sprices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/secondary_stock_prices.csv")
supplemental_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
supplemental_sprices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/secondary_stock_prices.csv")
testprices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/example_test_files/sample_submission.csv")
teststockprices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/example_test_files/stock_prices.csv")


# Check the basic information about the files loaded into dataframes

# In[ ]:


supplemental_prices.info()


# In[ ]:


supplemental_prices['Date'].unique()


# In[ ]:


prices.info()


# In[ ]:


stock_list.info()


# In[ ]:


stock_list.head(2)


# Append the secondary prices and suppliment prices

# In[ ]:


prices=prices.append(sprices,ignore_index=True)
prices=prices.append(supplemental_prices,ignore_index=True)
prices=prices.append(supplemental_sprices,ignore_index=True)
prices=prices.drop(['RowId','ExpectedDividend'],axis=1)
prices=prices.dropna()


# In[ ]:


prices.tail(5)


# In[ ]:


plt.scatter(prices['Date'],prices['Target'])


# Separate train and test data sets

# In[ ]:


prices['DateValue']=prices['Date'].str.replace('-','')
xprices=prices[prices['DateValue']<'20220401']
xprices=xprices.drop(['DateValue'],axis=1)
yprices=prices[prices['DateValue']>='20220401']
yprices=yprices.drop(['DateValue'],axis=1)


# In[ ]:


yprices.count()


# In[ ]:


gc.collect()


# pop the target values into another dataframe

# In[ ]:


y_train=xprices.pop('Target')
y_test=yprices.pop('Target')
X_train=xprices
X_test=yprices


# In[ ]:


del prices
del xprices
del yprices


# feature engineering

# In[ ]:


def featuring(train):
    dfa=pd.DataFrame()
    for code in train['SecuritiesCode'].unique():
        df=train[train['SecuritiesCode']==code]

        df=df.sort_values(by=['Date'], ascending=True)
        df['RA_20'] = df.Close.rolling(5, min_periods=1).mean()
        df['RA_40'] = df.Close.rolling(10, min_periods=1).mean()
        df['RA_60'] = df.Close.rolling(15, min_periods=1).mean()
        #df['RA_80'] = df.Close.rolling(20, min_periods=1).mean()
        #df['RA_100'] = df.Close.rolling(30, min_periods=1).mean()
        dfa=dfa.append(df)
    dfa['year']=pd.to_numeric(dfa['Date'].str[0:4]).astype(float)
    dfa['month']=pd.to_numeric(dfa['Date'].str[5:7]).astype(float)
    dfa['day']=pd.to_numeric(dfa['Date'].str[8:10]).astype(float)
    dfa['delta']=pd.to_numeric(dfa['High']-dfa['Low']).astype(float)
    dfa['change']=pd.to_numeric(dfa['Close']-dfa['Open']).astype(float)
    dfa=dfa[['Date','SecuritiesCode','delta','change','RA_20','RA_40','RA_60','year','month','day']]
    train=train.merge(dfa,how='left',on=['Date','SecuritiesCode'],suffixes=('', 'b')).set_axis(train.index)
    train=train.drop(['Date'],axis=1)
    #train=train.merge(stock_list, how='inner',on='SecuritiesCode',suffixes=('', 'b')).set_axis(train.index)
    #train=train.drop(['EffectiveDate','Name','33SectorName','17SectorName','NewIndexSeriesSize','TradeDate','Closeb'],axis=1)
    #dfa=dfa.join(stock_list,how='left',on='SecuritiesCode',rsuffix='b')
    #dfa=dfa.drop(['SecuritiesCodeb','Name', 'NewMarketSegment','33SectorCode','33SectorName','17SectorCode','17SectorName','NewIndexSeriesSizeCode', 'NewIndexSeriesSize',
    #   'TradeDate','Closeb','Universe0'],axis=1)
    #dfa['Section']=label_encoder.fit_transform(dfa['Section/Products'])
    #dfa=dfa.drop(['Section/Products'],axis=1)
    #dfa.sort_index(inplace=True)
    return train


# In[ ]:


X_train=featuring(X_train)
X_test=featuring(X_test)
gc.collect()


# XGBoost regressor

# In[ ]:


model = xgb.XGBRegressor(
    n_estimators=800,
    max_depth=16,
    learning_rate=0.01,
    subsample=0.5,
    colsample_bytree=0.75,
    missing=-999,
    random_state=2020,
    tree_method='gpu_hist' # THE MAGICAL PARAMETER
    )
model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_test, y_test)], verbose=1)
gc.collect()


# predict the target value for test data

# In[ ]:


env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()


# In[ ]:


for (df_test, options, financials, trades, secondary_prices, df_pred) in iter_test:
    df_test=df_test.drop(['RowId','ExpectedDividend'],axis=1)
    print(df_test.info())
    print(df_test.head(10))
    x_test = featuring(df_test)

    y_pred = model.predict(x_test)
    df_pred['Target'] = y_pred
    df_pred = df_pred.sort_values(by = "Target", ascending = False)
    df_pred['Rank'] = np.arange(len(df_pred.index))
    df_pred = df_pred.sort_values(by = "SecuritiesCode", ascending = True)
    df_pred.drop(["Target"], axis = 1)
    submission = df_pred[["Date", "SecuritiesCode", "Rank"]]    
    env.predict(submission)

