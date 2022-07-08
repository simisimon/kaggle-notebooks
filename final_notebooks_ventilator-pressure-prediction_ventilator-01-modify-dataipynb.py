#!/usr/bin/env python
# coding: utf-8

# # reference
# 
# 

# # Specify library

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
#from joypy import joyplot for matplotlib
import tensorflow as tf
from tqdm import tqdm
import optuna

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize, StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold, KFold

tqdm.pandas()


# # Settings

# In[ ]:


save_modified_data = True


# # Read data

# In[ ]:


path = '../input/ventilator-pressure-prediction'
train = pd.read_csv(f"{path}/train.csv")
test = pd.read_csv(f"{path}/test.csv")
submission = pd.read_csv(f'{path}/sample_submission.csv')

train = train.astype({'time_step': float, 'pressure': float, 'u_in' : float})
test = test.astype({'time_step': float, 'u_in' : float})


# # Try and Error

# # Utilitys

# In[ ]:


def data_clean(df):
    ## pickup ignore breath id
    ignore_breath_ids = set()
    
    time_step_diff_limit = 0.04
    for k, grp in tqdm(df.groupby("breath_id")):
        
        ## ignore non liner time_step data
        diff_se = grp["time_step"].diff()
        diff_chk = diff_se[diff_se > time_step_diff_limit]
        if len(diff_chk) != 0:
            ignore_breath_ids.add(k)
            
        ## ignor negative pressure data
        #mi = grp["pressure"].min()
        #if mi < 0:
        #    ignore_breath_ids.add(k)
            
        ## ignore len(u_out == 0) =< 28 
        u_out_0_len = len(grp[grp["u_out"] == 0])
        if u_out_0_len < 29:
            ignore_breath_ids.add(k)
            
        ## ignore pressure max == 64.8209917386395
        ma = grp["pressure"].max()
        if ma == 64.8209917386395:
            ignore_breath_ids.add(k)
    
    df = df[~df["breath_id"].isin(np.array(list(ignore_breath_ids)))]
    return df

def change_type(df):
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    return df

def add_features(df):
    df['u_in_cumsum'] = df.groupby('breath_id')["u_in"].cumsum()
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    #df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    #df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    #df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    #df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df = df.fillna(0)
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)
    return df

def tf_tpu_or_gpu_or_cpu():
    tpu = None
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        return "tpu"

    elif tf.test.is_gpu_available():
        strategy = tf.distribute.get_strategy()
        print('Running on GPU')
        return "gpu"

    else:
        strategy = tf.distribute.get_strategy()
        print('Running on CPU')
        return "cpu"


# # pickup u_out == 0

# In[ ]:


def u_out_0_df(df):
    grp_len = int(32)
    new_df = pd.DataFrame()
    for k, grp in tqdm(df.groupby("breath_id")):
        tmp_df = grp[grp["u_out"] == 0]
        rowno  = tmp_df.shape[0]
        for i in range(grp_len - rowno):
            row_df = tmp_df.tail(1).copy()
            time_diff = tmp_df.tail(2).diff().tail(1)["time_step"]
            row_df["time_step"] = row_df["time_step"] + time_diff * (i + 1)
            row_df["id"] = row_df["id"] + int(1)
            tmp_df = tmp_df.append(row_df,ignore_index=True)
        new_df = new_df.append(tmp_df,ignore_index=True)
    return new_df


# # apply utilitys for data

# In[ ]:


print("**Info : Data clean of train.")
train = data_clean(train)
print("**Info : pick up u_out == 0 of train.")
train = u_out_0_df(train)
train.to_csv("./train_u_out_0.csv")
print("**Info : add features of train.")
train = add_features(train)

print("**Info : pick up u_out == 0 of test.")
test = u_out_0_df(test)
test.to_csv("./test_u_out_0.csv")
print("**Info : add features of test.")
test = add_features(test)


# # Save modified train/test data

# In[ ]:


train.to_csv("./train_mod.csv")
test.to_csv("./test_mod.csv")


# # Visualize original data u_in histgram for each time step id

# In[ ]:


train = pd.read_csv(f"{path}/train.csv")
train["time_step_id"] = list(range(1,81,1)) * int(len(train)/80)
range_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,70,75,80,85,90,95,100]
bins_name = ['~5', '~10', '~15','~20','~25', '~30', '~35', '~40','~45', '~50', '~55', '~60','~65','~70','~75','~80','~85','~90','~95','~100']
train["u_in_range"] = pd.cut(train["u_in"],bins=range_bins,labels=bins_name)
tmp_df = pd.DataFrame()
for k,grp in train.groupby("time_step_id"):
    tmp_df = tmp_df.append(grp["u_in_range"].value_counts())
tmp_df = tmp_df.reset_index(drop=True)
tmp_df.columns = tmp_df.columns.astype(str)
tmp_df = tmp_df.reindex(columns=bins_name)
ax = sns.heatmap(tmp_df)
ax.set_xlabel("u_in_range", fontsize = 20)
ax.set_ylabel("time_step_id", fontsize = 20)
plt.show()


# # Visualize original data pressure histgram for each time step id

# In[ ]:


train = pd.read_csv(f"{path}/train.csv")
train["time_step_id"] = list(range(1,81,1)) * int(len(train)/80)
range_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
bins_name = ['~5', '~10', '~15','~20','~25', '~30', '~35', '~40','~45', '~50', '~55', '~60', '~65']
train["pressure_range"] = pd.cut(train["pressure"],bins=range_bins,labels=bins_name)
tmp_df = pd.DataFrame()
for k,grp in train.groupby("time_step_id"):
    tmp_df = tmp_df.append(grp["pressure_range"].value_counts())
tmp_df = tmp_df.reset_index(drop=True)
tmp_df.columns = tmp_df.columns.astype(str)
tmp_df = tmp_df.reindex(columns=bins_name)
ax = sns.heatmap(tmp_df)
ax.set_xlabel("pressure_range", fontsize = 20)
ax.set_ylabel("time_step_id", fontsize = 20)
plt.show()


# # Visualize u_in hist with StandardScaler() for each time step id

# In[ ]:


train = pd.read_csv(f"{path}/train.csv")
train = u_out_0_df(train)
columns = train.columns
#scaler = RobustScaler()
scaler = StandardScaler()
train = scaler.fit_transform(train)
train = pd.DataFrame(train,columns= columns)
train["time_step_id"] = list(range(1,33,1)) * int(len(train)/32)
range_bins = [-1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8]
bins_name = ['~-1.6', '~-1.4', '~-1.2','~-1.0','~-0.8','~-0.6','~-0.4','~-0.2','~0','~0.2','~0.4','~0.6','~0.8','~1.0','~1.2','~1.4','~1.6','~1.8']
#train["pressure_range"] = pd.cut(train["pressur"],bins=range_bins,labels=bins_name)
train["u_in_range"] = pd.cut(train["u_in"],bins=range_bins,labels=bins_name)
tmp_df = pd.DataFrame()
for k,grp in train.groupby("time_step_id"):
    tmp_df = tmp_df.append(grp["u_in_range"].value_counts())
tmp_df = tmp_df.reset_index(drop=True)
tmp_df.columns = tmp_df.columns.astype(str)
tmp_df = tmp_df.reindex(columns=bins_name)
ax = sns.heatmap(tmp_df)
ax.set_xlabel("u_in_range", fontsize = 20)
ax.set_ylabel("time_step_id", fontsize = 20)
plt.show()


# # Visualize u_in hist with RobustScaler() for each time step id

# In[ ]:


train = pd.read_csv(f"{path}/train.csv")
train = u_out_0_df(train)
columns = train.columns
scaler = RobustScaler()
#scaler = StandardScaler()
train = scaler.fit_transform(train)
train = pd.DataFrame(train,columns= columns)
train["time_step_id"] = list(range(1,33,1)) * int(len(train)/32)
range_bins = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
range_bins = [-9, -8, -7, -6, -5, -4, -3, -2,-1,0,1,2,3,4,5,6,7,8,9,10]
bins_name = ['~-8', '~-7','~-6','~-5','~-4','~-3','~-2','~-1','~0','~1','~2','~3','~4','~5','~6','~7','~8','~9','10']
#train["pressure_range"] = pd.cut(train["pressure"],bins=range_bins,labels=bins_name)
train["u_in_range"] = pd.cut(train["u_in"],bins=range_bins,labels=bins_name)
tmp_df = pd.DataFrame()
for k,grp in train.groupby("time_step_id"):
    #tmp_df = tmp_df.append(grp["pressure_range"].value_counts())
    tmp_df = tmp_df.append(grp["u_in_range"].value_counts())
tmp_df = tmp_df.reset_index(drop=True)
tmp_df.columns = tmp_df.columns.astype(str)
tmp_df = tmp_df.reindex(columns=bins_name)
ax = sns.heatmap(tmp_df)
ax.set_xlabel("u_in_range", fontsize = 20)
ax.set_ylabel("time_step_id", fontsize = 20)
plt.show()

