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


# In[ ]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


def prep_prices(price):
    from decimal import ROUND_HALF_UP, Decimal
    pcols = ["Open","High","Low","Close"]
    price.ExpectedDividend.fillna(0,inplace=True)
    def qround(x):
        return float(Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
    def adjust_prices(df):
        df = df.sort_values("Date", ascending=False)
        df.loc[:, "CumAdjust"] = df["AdjustmentFactor"].cumprod()
        # generate adjusted prices
        for p in pcols:
            df.loc[:, p] = (df["CumAdjust"] * df[p]).apply(qround)
        df.loc[:, "Volume"] = df["Volume"] / df["CumAdjust"]
        return df
    # generate Adjusted
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(adjust_prices).reset_index(drop=True)
    price = price.sort_values("RowId")
    if 'Target' in price.columns:
        price.dropna(subset=["Open","High","Low","Close",'Volume','Target'],inplace=True)
    return price


# Data Preparation

# In[ ]:


# Load stock_prices.csv
stock_prices_test = prep_prices(pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv'))
#y_stock_prices.head()
stock_prices_train = prep_prices(pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv'))


# Modeling

# In[ ]:


features = ['Open', 'High', 'Low', 'Close', 'Volume']

X_train = stock_prices_train[features].dropna()
y_train = stock_prices_train['Target'].dropna()
X_test = stock_prices_test[features].dropna()
y_test = stock_prices_test['Target'].dropna()

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# In[ ]:


clf = LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)

clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)
df_pred = stock_prices_test
df_pred['Target'] = y_pred
df_pred['Rank'] = df_pred.groupby('Date')['Target'].rank(ascending=False, method='first')
df_pred = df_pred.groupby('Date', as_index=False).apply(lambda x: x.sort_values('Rank')).reset_index()
submission = df_pred[["Date", "SecuritiesCode", "Rank"]]
submission


# In[ ]:


import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()    # an iterator which loops over the test files
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    prices = prep_prices(prices)
    X_test = prices[features].fillna(0)
    X_test = scaler.transform(X_test)
    
    y_pred = clf.predict(X_test)
    df_pred = prices
    df_pred['Target'] = y_pred
    df_pred['Rank'] = df_pred.groupby('Date')['Target'].rank(ascending=False, method='first').map(lambda x: int(x)-1)
    #df_pred = df_pred.groupby('Date', as_index=False).apply(lambda x: x.sort_values('Rank')).reset_index()
    submission = df_pred[["Date", "SecuritiesCode", "Rank"]]
    print(submission)
    env.predict(submission)
    

