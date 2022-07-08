#!/usr/bin/env python
# coding: utf-8

# I have updated this notebook to modify the wrmsse function  at 29th Mar.  
# New wrmsse function for LGBM metric calculate wrmsse only for last 28 days to consider non-zero demand period.  
# Please refer comment section. I have commented the detail of my fixing.
# (note:I have also remove some variable to reduce the run-time and changed 'objective' in lgbm to 'poisson'.)
# 
# This kernel is:  
# - Based on [Very fst Model](https://www.kaggle.com/ragnar123/very-fst-model). Thanks [@ragnar123](https://www.kaggle.com/ragnar123).  
# - Based on [m5-baseline](https://www.kaggle.com/harupy/m5-baseline). Thank [@harupy](https://www.kaggle.com/harupy).  
# to explain the detail of these great notebook by Japanese especially for beginner.  
# 
# Additionaly, I have added an relatively efficient evaluation of WRSSE for LGBM metric to these kernel.

# ## module import

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import dask.dataframe as dd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import lightgbm as lgb
#import dask_xgboost as xgb
#import dask.dataframe as dd
from sklearn import preprocessing, metrics
from sklearn.preprocessing import LabelEncoder
import gc
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## functionの定義

# reduce_mem_usageは、データのメモリを減らすためにデータ型を変更する関数です。  
# ('reduce_mem_usage' is a functin which reduce memory usage by changing data type.)
# https://qiita.com/hiroyuki_kageyama/items/02865616811022f79754　を参照ください。

# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns毎に処理
        col_type = df[col].dtypes
        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# PandasのdataFrameをきれいに表示する関数
# (This function is to diplay a head of Pandas DataFrame.)

# In[ ]:


import IPython

def display(*dfs, head=True):
    for df in dfs:
        IPython.display.display(df.head() if head else df)


# ## 特徴量作成

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndata = pd.read_pickle("../input/make-merged-dataset/m5_all_data_merged.pickle")\n#data = data[data[\'id\']=="HOBBIES_1_001_CA_1_validation"].head(600)\ndata.head(3)\n')


# In[ ]:


# サンプルの特徴量作成関数
# 既存の列であるsub_id1,2,7,8,9,10,12は作成不要。
def sample_fe(data):    

    ##################################################
    # sub_id3,4,5,6 曜日、週、月、年
    ##################################################
    # 日付に関する特徴量を作成する元となるdate列を指定
    dt_col = "date"
    # 上記作成元の列をdatetime型に変換
    data[dt_col] = pd.to_datetime(data[dt_col])
    # 作成したい日付関連の特徴量リストを指定
    attrs = [
        "year",
        "month",
        "week",
        "dayofweek",
    ]
    # 上記で指定したリストの特徴量を作成
    for attr in tqdm(attrs):
        dtype = np.int16 if attr == "year" else np.int8
        data[attr] = getattr(data[dt_col].dt, attr).astype(dtype)
    print("Phase1 : sub_id=3,4,5,6 日付関連特徴量 finished")

        
    ##################################################
    # sub_id=11 価格増減率(今週の商品価格/先週の商品価格)
    ##################################################
    # 先週の価格列を作成　※価格は週ごとに出ている
    tmp_data = data[['id','wm_yr_wk','sell_price']]
    tmp_data = tmp_data.drop_duplicates()
    tmp_data.columns = ['id','wm_yr_wk','lastweek_sell_price']
    tmp_data['wm_yr_wk'] = tmp_data['wm_yr_wk']+1
    data = pd.merge(data, tmp_data, on=['id','wm_yr_wk'], how='left', copy=False)
    del tmp_data
    gc.collect()


    # 今週の価格/先週の価格で、価格の増減率を計算
    data["price_volatility_w1"] = data["sell_price"]/data["lastweek_sell_price"]
    print("Phase2 : sub_id=11 価格増減率 finished")
    

    ##################################################    
    # sub_id=13 直近の売上平均
    ##################################################
    # 売上平均を取りたい日数リストを作成
    windows = [7]
    # 上記作成したリストに基づいて、売上の移動平均を取る
    for window in tqdm(windows):
        data['rolling_demand_mean_'+str(window)] = data.groupby(['id'])['demand'].transform(lambda x: x.rolling(window=window).mean()).astype(np.float16)
    print("Phase3 : sub_id=13 直近の売上平均 finished")

        
    ##################################################
    # sub_id=14 前年同時期の売上平均
    ##################################################
    # sub_id=13で作成した移動平均を365日shiftし、値を取得
    for window in tqdm(windows):
        data['last_year_rolling_demand_mean'] = data.groupby(['id'],as_index=False)['rolling_demand_mean_'+str(window)].shift(365).astype(np.float16)
    print("Phase4 : sub_id=14 前年同時期の売上平均 finished")
        
    ##################################################        
    # sub_id=15 売上のシフト
    ##################################################
    # 作成したいラグのリストを作成
    # lags = [1,2,3,6,12,24,36]
    lags = [1,2]
    # 上記で作成したリストを用いて、ラグを作成
    for lag in tqdm(lags):
        data['demand_lag_'+str(lag)] = data.groupby(['id'],as_index=False)['demand'].shift(lag).astype(np.float16)
    print("Phase5 : sub_id=15 売上のシフト finished")
        
    # 特徴量を追加したデータフレームを返す
    return data


# In[ ]:


get_ipython().run_cell_magic('time', '', 'data = sample_fe(data)\ndata = reduce_mem_usage(data)\n')

data.head(400)
# In[ ]:


# 不要列の削除
data.drop(['lastweek_sell_price'], inplace = True, axis = 1)
gc.collect()


# In[ ]:


display(data.head())


# ## 出力

# In[ ]:


data.to_pickle("m5_add_features_data_sample.pickle")

