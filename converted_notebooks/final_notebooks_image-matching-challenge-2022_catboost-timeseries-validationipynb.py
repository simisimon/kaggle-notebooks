#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
pd.options.display.max_columns = 50

from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import shap

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


SEED = 42
PATH = '../input/m5-accuracy-goes-again-csc-22/'
CAL_DTYPES={
    "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16", "month": "int16", "year": "int16", 
    "snap_CA": 'bool', 'snap_TX': 'bool', 'snap_WI': 'bool'
}
PRICE_DTYPES = {
    "store_id": "category", "item_id": "category", 
    "wm_yr_wk": "int16", "sell_price":"float32"
}


# In[ ]:


TEST_START = 1914
TEST_SIZE = 28
DATE_SPLIT = datetime(2016, 4, 25)
DATE_END = DATE_SPLIT + timedelta(TEST_SIZE)


# # Preprocess

# In[ ]:


def create_dt(is_train=True, first_day=TEST_START-366):
    prices = pd.read_csv(os.path.join(PATH, "prices_sell.csv"), dtype=PRICE_DTYPES)
    cal = pd.read_csv(os.path.join(PATH, "calendar.csv"), dtype=CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    
    numcols = [f"d_{day}" for day in range(first_day, TEST_START)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol: 'int16' for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv(os.path.join(PATH, "sales_train_competition.csv"), usecols=catcols+numcols, dtype=dtype)
    if not is_train:
        for day in range(TEST_START, TEST_START+TEST_SIZE):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                 id_vars=catcols,
                 value_vars=[col for col in dt.columns if col.startswith("d_")],
                 var_name="d",
                 value_name="sales")
    dt = dt.merge(cal, on="d", copy=False)
    dt = dt.merge(prices, how='left', on=["store_id", "item_id", "wm_yr_wk"], copy=False)
    
    # fillNA
    dt['sell_price'] = dt['sell_price'].fillna(1e6)
    for col in dt.columns:
        if col.startswith('event_'):
            dt[col] = dt[col].fillna('').astype('category')

    dt = dt.sort_values(['date', 'id'])
    return dt


# In[ ]:


def create_feats(df):
    df['snap'] = np.where(df['state_id'] == 'CA', 
                          df['snap_CA'], 
                          np.where(df['state_id'] == 'TX',
                                   df['snap_TX'],
                                   df['snap_WI']))
    
    # date features
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
    }
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in df.columns:
            df[date_feat_name] = df[date_feat_name].astype("int16")
        else:
            df[date_feat_name] = getattr(df["date"].dt, date_feat_func).astype("int16")
        
def create_lag_feats(df):
    # sales lags
    lags = [28, 56]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df.groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        df[f"rmean_{TEST_SIZE}_{win}"] = df.groupby("id")[f'lag_{TEST_SIZE}'].transform(lambda x : x.rolling(win).mean())
        
    # sell_price diffs
    lags = [1, 7]
    for lag in lags:
        df[f'sell_price_lag_{lag}'] = df.groupby("id")["sell_price"].diff(lag)
    


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nFIRST_DAY = 1913 - 28 * 6\ndf = create_dt(is_train=False, first_day=FIRST_DAY)\ndf.shape\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ncreate_feats(df)\ncreate_lag_feats(df)\nprint(df.shape)\ndf.tail(2)\n')


# # Train

# In[ ]:


# lag_cols = [col for col in df if col.startswith(('lag_', 'rmean_'))]
cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "sales","d", "wm_yr_wk", "weekday", 'year'] + ['snap_CA', 'snap_TX', 'snap_WI']
train_cols = df.columns[~df.columns.isin(useless_cols)]

X = df[train_cols]
y = df["sales"]


# In[ ]:


TRAIN_SIZE = 28
DATE_START = DATE_SPLIT - timedelta(TRAIN_SIZE)

cb_params = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'cat_features': cat_feats,
    'verbose': 200,
    'random_seed': SEED,
    'task_type': 'GPU',
    'border_count': 32,
}
model = CatBoostRegressor(**cb_params)


# In[ ]:


def make_prediction(X, shift_days, is_train=True, plot_shap=False):
    X_train, X_valid, date_start, date_split, date_end = get_relevant_timeframes(X, shift_days)
    y_train = y.loc[X_train.index]
    y_valid = y.loc[X_valid.index]

    print(f'Training on data {shift_days} days ago:')
    if is_train:
        model.fit(X_train, y_train, 
                  eval_set=(X_valid, y_valid),
                  use_best_model=False
                 )
    else:
        model.fit(X_train, y_train)
    
    if plot_shap:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train)
    
    y_pred = model.predict(X_valid)
    y_pred = y_pred.clip(0.) # sales >= 0
    score = None
    if is_train:
        score = rmse(y_valid, y_pred)
        print(f'RMSE {shift_days} days ago = {score:.4f}\n')
    return y_pred, score

def get_relevant_timeframes(X, shift_days):
    delta = timedelta(-shift_days)
    date_start = DATE_START + delta
    date_split = DATE_SPLIT + delta
    date_end = DATE_END + delta
    X_timeframe_train = X.loc[(X.date >= date_start) & (X.date < date_split)].drop(columns='date')
    X_timeframe_valid = X.loc[(X.date >= date_split) & (X.date < date_end)].drop(columns='date')
    return X_timeframe_train, X_timeframe_valid, date_start, date_split, date_end

def rmse(y, y_pred):
    return mean_squared_error(y, y_pred, squared=False)


# In[ ]:


n_folds = 3
shifts_days = [TEST_SIZE*(i+1) for i in reversed(range(n_folds))]
val_scores = []
for n_days in shifts_days:
    _, score = make_prediction(X, n_days)
    val_scores.append(score)
    
print(f'Scores: {val_scores}')
print(f'{np.mean(val_scores)} +- {np.std(val_scores)}')


# # Test

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ny_pred, _ = make_prediction(X, 0, is_train=False, plot_shap=True)\ndf.loc[df.date >= DATE_SPLIT, 'sales'] = y_pred\n")


# In[ ]:


df_sub = df[df['d'].isin([f"d_{i}" for i in range(TEST_START, TEST_START+TEST_SIZE)])]
df_sub.sales.describe()


# In[ ]:


df_compare = df[df['d'].isin([f"d_{i}" for i in range(TEST_START-TEST_SIZE, TEST_START)])]
df_compare.sales.describe()


# In[ ]:


sample_sub = pd.read_csv(os.path.join(PATH, 'sample_submission (2).csv'))
sample_sub.head(2)


# In[ ]:


df_sub = pd.pivot_table(df_sub, index='id', columns='d', values='sales').reset_index().set_index('id').reindex(sample_sub['id']).reset_index()
df_sub.to_csv('submission_catboost.csv', index=False)
df_sub.head(2)


# In[ ]:




