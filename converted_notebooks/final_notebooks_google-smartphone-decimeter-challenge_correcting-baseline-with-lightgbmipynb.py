#!/usr/bin/env python
# coding: utf-8

# # Correcting baseline with lightgbm 

# ## Additive functions

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import itertools
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
import optuna

import plotly.express as px
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler as MMScaler, StandardScaler as SSScaler
from sklearn.preprocessing import OneHotEncoder

import os
from torchvision.io import read_image

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

#print(os.listdir())

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist

def compute_dist_hav(df, cols):
    df = df.copy()
    #cols = ['latDeg', 'lngDeg']
    prev_cols = [col+'_prev' for col in cols]
    
    df.loc[:, ['dist']] = np.nan
    
    for phone in tqdm(df['phone'].unique()):
        ser = df[df['phone'] == phone]
        
        ser.loc[:, prev_cols] = ser.loc[:, cols].shift(1).values
        #ser['latDeg_prev'] = ser['latDeg'].shift(1)
        #ser['lngDeg_prev'] = ser['lngDeg'].shift(1)
        
        #display(ser)
        f_ind = ser.index[0]
        ser.loc[f_ind, prev_cols] = ser.loc[f_ind, cols].values
        #ser.loc[f_ind, 'latDeg_prev'] = ser.loc[f_ind, 'latDeg']
        #ser.loc[f_ind, 'lngDeg_prev'] = ser.loc[f_ind, 'lngDeg']
        
        df.loc[(df['phone'] == phone), prev_cols] = ser[prev_cols]
        #df.loc[(df['phone'] == phone), ['latDeg_prev', 'lngDeg_prev']] = ser[['latDeg_prev', 'lngDeg_prev']]
        
    df['dist'] = calc_haversine(df[cols[0]], df[cols[1]], df[prev_cols[0]], df[prev_cols[1]])
    return df

def get_deltas(t):
    t = t.copy()
    
    t.loc[:, ['d_latDeg', 'd_lngDeg']] = np.nan
    #display(t)
    
    for phone in tqdm(t['phone'].unique()):
        ser = t[t['phone'] == phone]
        
        ser['d_latDeg'] = ser['latDeg'] - ser['latDeg'].shift(1)
        ser['d_lngDeg'] = ser['lngDeg'] - ser['lngDeg'].shift(1)
        ser = ser.fillna(0)
        
        #display(t.loc[t['phone'] == phone])
        t.loc[(t['phone'] == phone), ['d_latDeg', 'd_lngDeg']] = ser[['d_latDeg', 'd_lngDeg']]
    
    return t

def get_moving_average(t, w=5):
    t = t.copy()
    t2 = pd.DataFrame([])
    for phone in t['phone'].unique():
        t3 = t[t['phone'] == phone].rolling(w).mean().shift(-w//2)
        t3 = t3.interpolate(method='linear', limit_direction='both', axis=0)

        if t2.shape[0] == 0:
            t2 = t3
        else:
            t2 = pd.concat([t2, t3], axis=0)

    t = t.loc[:, [col for col in t if col not in t2.columns]]
    t = pd.concat([t, t2], axis=1)
    
    return t

def weighted_average(df, data_col, weight_col, by_col):
    df['_data_times_weight'] = df[data_col] * df[weight_col]
    df['_weight_where_notnull'] = df[weight_col] * pd.notnull(df[data_col])
    g = df.groupby(by_col)
    result = g['_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
    del df['_data_times_weight'], df['_weight_where_notnull']
    return result


# In[ ]:


def get_score(df_to_score, ground_truth):
    df_to_score = df_to_score.copy()
    
    df_to_score.sort_values(by=['phone', 'time'], inplace=True)
    ground_truth.sort_values(by=['phone', 'time'], inplace=True)

    df_to_score['t_latDeg'] = ground_truth['latDeg'].values
    df_to_score['t_lngDeg'] = ground_truth['lngDeg'].values
    
    meter_score, score = _check_score(df_to_score)
    return meter_score, score

def _check_score(input_df, silent=True):
    output_df = input_df.copy()
    
    output_df['meter'] = calc_haversine(
            input_df.latDeg, input_df.lngDeg, input_df.t_latDeg, input_df.t_lngDeg
        )

    meter_score = output_df['meter'].mean()
    
    if not silent:
        print(f'error meter: {meter_score}')

    scores_50 = []
    scores_95 = []
    for phone in output_df['phone'].unique():
        _index = output_df['phone']==phone
        p_50 = np.percentile(output_df.loc[_index, 'meter'], 50)
        p_95 = np.percentile(output_df.loc[_index, 'meter'], 95)
        scores_50.append(p_50)
        scores_95.append(p_95)
    
    scores = scores_50 + scores_95
    
    score_50 = sum(scores_50) / len(scores_50)
    score_95 = sum(scores_95) / len(scores_95)
    score = sum(scores) / len(scores)
    
    if not silent:
        print(f'score 50: {score_50}')
        print(f'score 95: {score_95}')
        print(f'score   : {score}')
    
    return meter_score, score


# In[ ]:


# В некоторых местах преобразование не точно на 1 секунду. Но в рамках этой истории - это не важно
def utc_to_gps(time):#+ timedelta(seconds=1092121243.0 - (35 - 19))
    return time - 315964800000 + 18000

def gps_to_utc(time):
    return time + 315964800000 - 18000


# In[ ]:


def visualize_trafic(df, center, zoom=15):
    fig = px.scatter_mapbox(df,
                            
                            # Here, plotly gets, (x,y) coordinates
                            lat="latDeg",
                            lon="lngDeg",
                            
                            #Here, plotly detects color of series
                            color="phoneName",
                            labels="phoneName",
                            
                            zoom=zoom,
                            center=center,
                            height=600,
                            width=800)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()
    
def visualize_collection(df):
    target_df = df
    lat_center = target_df['latDeg'].mean()
    lng_center = target_df['lngDeg'].mean()
    center = {"lat":lat_center, "lon":lng_center}
    
    visualize_trafic(target_df, center)


# In[ ]:


def get_lag_features(df_train, columns, count=15):
    """ Get lag features ONLY for separated collections+phones"""
    new_feat_df = dict()
    
    t_size = df_train.shape[0]
    for col in columns:
        t = df_train[col]
        for i in range(1, count + 1):
            new_feat_df[f'd_next_{col}_{i}'] = t - t.shift(i).fillna(method='bfill')
        for i in range(1, count + 1):
            new_feat_df[f'd_prev_{col}_{i}'] = t - t.shift(-i).fillna(method='ffill')
            
    new_feat_df = pd.DataFrame(new_feat_df)
    
    return pd.concat( [df_train.reset_index(drop=True), new_feat_df.reset_index(drop=True)], axis=1)


def get_collections_list(df):
    """ Separate dataframe on collections"""
    df_list = []
    for phone in df['phone'].unique():
        df_list.append( df[df['phone'] == phone] )
        
    return df_list


# ## Prepare input

# In[ ]:


TRAIN_INPUT_DIR = '/kaggle/input/gps-datasets/' 
train_fname = 'train_clean.csv'
#TRAIN_INPUT_DIR = '/kaggle/input/data-derived/' 
#train_fname = 'df_train_derived.csv'
train_base = pd.read_csv(TRAIN_INPUT_DIR + train_fname)
train_base.head(2)


# In[ ]:


train_base.drop(['d_latDeg', 'd_lngDeg', 'dist',
                 'latDeg_prev', 'lngDeg_prev'], axis=1, inplace=True)


# In[ ]:


p = pathlib.Path('../input/google-smartphone-decimeter-challenge')
gt_files = list(p.glob('train/*/*/ground_truth.csv'))

gts = []
for gt_file in gt_files:
    gts.append(pd.read_csv(gt_file))
    
ground_truth = pd.concat(gts)
ground_truth['phone'] = ground_truth['collectionName'].astype(str) + '_' + ground_truth['phoneName']
ground_truth['time'] = pd.to_datetime(gps_to_utc(ground_truth['millisSinceGpsEpoch'])//1000, unit='s')
ground_truth = compute_dist_hav(ground_truth, ['latDeg', 'lngDeg'])


# In[ ]:


train_base.sort_values(by=['phone', 'time'], inplace=True)
ground_truth.sort_values(by=['phone', 'time'], inplace=True)

train_base.reset_index(drop=True, inplace=True)
ground_truth.reset_index(drop=True, inplace=True)


# In[ ]:


train_base_list = get_collections_list(train_base)
lag_columns = ['latDeg', 'lngDeg']
lag_count = 10
lag_dfs = [get_lag_features(df, lag_columns, lag_count) for df in train_base_list]
train_base = pd.concat(lag_dfs, axis=0)
train_base.reset_index(drop=True, inplace=True)


# In[ ]:


GT = pd.DataFrame({
    'target_d_lat' : (train_base['latDeg'].to_numpy() - ground_truth['latDeg'].to_numpy()),
    'target_d_lng' : (train_base['lngDeg'].to_numpy() - ground_truth['lngDeg'].to_numpy())
})


# In[ ]:


train_cols = ['latDeg', 'lngDeg'] + list(filter(lambda w : w.startswith('d_'), train_base.columns))


# In[ ]:


enc = OneHotEncoder()
enc.fit(train_base.phoneName.unique().reshape(-1, 1))
phone_feat = enc.transform(train_base.phoneName.values.reshape(-1, 1))
phone_cols = [f'oh{i}' for i in range(phone_feat.shape[1])]
phone_feat_df = pd.DataFrame( phone_feat.toarray(), columns=phone_cols )
train_base = pd.concat([train_base, phone_feat_df], axis=1)
train_cols += phone_cols


# ## Optuna parameters search

# In[ ]:


def train_evaluate(param, trial):
    
    kfold = GroupKFold(n_splits=min(5, max(2, len(np.unique(groups))) ))
    scores = []
    
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(X, y, groups=groups)):
        all_y_pred = []
        
        for ind in range(2):
            
            X_train = X[trn_idx]
            X_val = X[val_idx]
            y_train = y[trn_idx, ind]
            y_val = y[val_idx, ind] 
            
            dtrain = lgb.Dataset(X_train, y_train)
            
            dtest = lgb.Dataset(X_val, y_val)
            gbm = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=dtest, verbose_eval=500)
            
            y_pred = gbm.predict(X_val)
            
            all_y_pred.append( y_pred )

        gt_val = ground_truth.iloc[val_idx]
        ans_base = train_base.iloc[val_idx].copy()
        
        all_y_pred = np.array(all_y_pred).T
        ans_base[['latDeg', 'lngDeg']] -= all_y_pred
        meter_score, score = get_score(ans_base, gt_val)
        
        scores.append(score)
    
    return np.mean(scores)


def objective(trial):
    
    param = {'num_leaves': trial.suggest_int('num_leaves', 24, 1024),
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mae',
            #'eval_metric': 'mae', 
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'early_stopping_rounds': 200,
              #'device' : 'gpu',
             'verbosity' : -1,
             #'verbose_eval': -1
            }
    
    return train_evaluate(param, trial)


# In[ ]:


def find_params():
    X = train_base[train_cols].values
    y = GT.values
    groups = train_base.collectionName.values
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    
    print('Number of finished trials: {}'.format(len(study.trials)))
    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


# In[ ]:


# uncomment to find best params
# find_params()


# ## Cross validation

# In[ ]:


params = {
    
    'num_leaves': 99,
    'feature_fraction': 0.7544601013793489,
    'bagging_fraction': 0.9508442472495611,
    'bagging_freq': 5,
    'min_child_samples': 51,
    'lambda_l1': 0.00019899635179289397,
    'lambda_l2': 0.2466948417767759,
    
    'early_stopping_rounds': 200,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'verbosity' : -1,
}


# In[ ]:


def coord_cv_oof(params, groups):
    n_splits = min(5, max(2, len(np.unique(groups))))
    kfold = GroupKFold(n_splits=n_splits)
    
    scores = []
    oofs = train_base.copy()
    
    models_lat = []
    models_lng = []
    
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(X, y, groups=groups)):
        print(fold_id, '/', n_splits)
        
        X_train = X[trn_idx]
        X_val = X[val_idx]
        all_y_pred = []
        for ind in range(2):
            
            y_train = y[trn_idx, ind]
            y_val = y[val_idx, ind]
            
            dtrain = lgb.Dataset(X_train, y_train)
            
            dtest = lgb.Dataset(X_val, y_val)
            model = lgb.train(params[ind], dtrain, num_boost_round=1000, valid_sets=dtest, verbose_eval=500)
            y_pred = model.predict(X_val)
            
            all_y_pred.append( y_pred )
            
            if ind == 0:
                models_lat.append(model)
            else:
                models_lng.append(model)
        
        gt_val = ground_truth.iloc[val_idx]
        ans_base = train_base.iloc[val_idx].copy()
        all_y_pred = np.array(all_y_pred).T
        ans_base[['latDeg', 'lngDeg']] -= all_y_pred
        #ans_base[['latDeg_bl', 'lngDeg_bl']] -= all_y_pred
        meter_score, score = get_score(ans_base, gt_val)
        scores.append( score )
        
        # There was Danil's bug with iloc
        oofs.loc[val_idx, ['d_latDeg', 'd_lngDeg']] = all_y_pred

        print(f'Fold {fold_id}: {scores[-1]}')
        
    score = np.mean(scores)
    
    print(params)
    print('Validation:', score)
    print('-' * 60)
    
    return score, scores, oofs, models_lat, models_lng


# In[ ]:


def coord_cv_oof_zone(params, train_base, ground_truth, X, y, groups):
    n_splits = min(5, max(2, len(np.unique(groups))))
    kfold = GroupKFold(n_splits=n_splits)
    
    scores = []
    oofs = train_base.copy()
    
    models_lat = []
    models_lng = []
    
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(X, y, groups=groups)):
        print(fold_id, '/', n_splits)
        
        X_train = X[trn_idx]
        X_val = X[val_idx]
        all_y_pred = []
        for ind in range(2):
            
            y_train = y[trn_idx, ind]
            y_val = y[val_idx, ind]
            
            model = lgb.LGBMRegressor(**params)
            model = model.fit(X_train, 
                            y_train,
                            eval_metric=params['metric'])

            y_pred = model.predict(X_val, num_iteration = model.best_iteration_)
            all_y_pred.append( y_pred )
            
            if ind == 0:
                models_lat.append(model)
            else:
                models_lng.append(model)
        
        gt_val = ground_truth.iloc[val_idx]
        ans_base = train_base.iloc[val_idx].copy()
        all_y_pred = np.array(all_y_pred).T
        ans_base[['latDeg', 'lngDeg']] -= all_y_pred
        meter_score, score = get_score(ans_base, gt_val)
        scores.append( score )
        
        oofs.iloc[val_idx].loc[:, ['d_latDeg', 'd_lngDeg']] = all_y_pred

        print(f'Fold {fold_id}: {scores[-1]}')
        
    score = np.mean(scores)
    
    print(params)
    print('Validation:', score)
    print('-' * 60)
    
    return scores, oofs, models_lat, models_lng


# In[ ]:


def simple_crossval_scoring():
    results = []
    for road_type_num in range(3):
        road_ind = train_base['road'] == road_names[road_type_num]
        X = train_base[road_ind][train_cols].values
        y = GT[road_ind].values
        groups = train_base[road_ind].collectionName.values
        print('type:', road_names[road_type_num])
        results.append(
            coord_cv_oof_zone(params_zone[road_type_num], train_base[road_ind], ground_truth[road_ind], X, y, groups)
        )

    return np.mean( np.mean([res[0] for res in results], axis=0) ), results
    # result - 3.478843607529768
    
# # Use simple cross validation for predicting oof
# score, results = simple_crossval_scoring()
# oof_full = pd.concat([res[1] for res in results], axis=0).reset_index(drop=True)
# print(f'val score: {score}, train score: {get_score(oof_full, ground_truth)}')


# In[ ]:


def validate_one_zone(params, road_type_num, train_base_input, ground_truth_input):
    
    road_ind = train_base_input['road'] == road_names[road_type_num]
    train_base = train_base_input[road_ind]
    ground_truth = ground_truth_input[road_ind]
    
    X = train_base[train_cols].values
    y = GT[road_ind].values
    groups = train_base.collectionName.values
    param = params[road_type_num]
    
    kfold = GroupKFold(n_splits=min(5, max(2, len(np.unique(groups))) ))
    scores = []
    
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(X, y, groups=groups)):
        print(f'Fold {fold_id} ')
        all_y_pred = []
        
        for ind in range(2):
            
            X_train = X[trn_idx]
            X_val = X[val_idx]
            y_train = y[trn_idx, ind]
            y_val = y[val_idx, ind] 
            
            dtrain = lgb.Dataset(X_train, y_train)
            
            #dtest = lgb.Dataset(X_val, y_val)
            #gbm = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=dtest, verbose_eval=500)
            
            gbm = lgb.train(param, dtrain, num_boost_round=1000)
            y_pred = gbm.predict(X_val)
            
            all_y_pred.append( y_pred )
            

        gt_val = ground_truth.iloc[val_idx]
        ans_base = train_base.iloc[val_idx].copy()
        all_y_pred = np.array(all_y_pred).T
        ans_base[['latDeg', 'lngDeg']] -= all_y_pred
        meter_score, score = get_score(ans_base, gt_val)
        
        scores.append(score)
        
        print(scores[-1])
    
    return np.mean(scores)


# In[ ]:


def coord_cv_oof_zone_real(params, train_base, ground_truth):
    
    results = []
    n_splits = 3 #min(3, max(2, len(np.unique(groups))))
    n_roads = 1
    val_idx_list = [] # check correctness
    
    train_base.reset_index(drop=True, inplace=True)
    ground_truth.reset_index(drop=True, inplace=True)
    
    oofs = train_base.copy()
    
    for road_type_num in range(n_roads):
        road_ind = train_base['road'] == road_names[road_type_num]
        X = train_base[road_ind][train_cols].values
        y = GT[road_ind].values
        groups = train_base[road_ind].collectionName.values
    
        kfold = GroupKFold(n_splits=n_splits)

        scores = []
        results_split = [None] * n_splits
        
        for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(X, y, groups=groups)):
            print(fold_id, '/', n_splits)

            X_train = X[trn_idx]
            X_val = X[val_idx]
            all_y_pred = []
            models = []
            
            for ind in range(2):

                y_train = y[trn_idx, ind]
                y_val = y[val_idx, ind]
                dtrain = lgb.Dataset(X_train, y_train)
                dtest = lgb.Dataset(X_val, y_val)

                model = lgb.train(params[road_type_num], dtrain, num_boost_round=1000, valid_sets=dtest, verbose_eval=500)
                #model = lgb.train(params[road_type_num], dtrain, num_boost_round=1000)
                y_pred = model.predict(X_val)

                all_y_pred.append( y_pred )
                models.append(model)
            
            global_ind_val = train_base.index[road_ind][val_idx].values 
            # may be global_ind_val = train_base.index[road_ind].values[val_idx], didn't check
            
            gt_val = ground_truth.iloc[global_ind_val].copy()
            ans_base = train_base.iloc[global_ind_val].copy()
            all_y_pred = np.array(all_y_pred).T
            ans_base[['latDeg', 'lngDeg']] -= all_y_pred
            
            oofs.loc[global_ind_val, ['d_latDeg', 'd_lngDeg']] = all_y_pred
            val_idx_list += global_ind_val.tolist()
            
            results_split[fold_id] = (ans_base, gt_val, models[0], models[1])
        
        results.append(results_split)
        
    scores = []
    for i in range(n_splits):
        ans_base = pd.concat([results[road_type][i][0] for road_type in range(n_roads)], axis=0)
        gt_val = pd.concat([results[road_type][i][1] for road_type in range(n_roads)], axis=0)
        meter_score, score = get_score(ans_base, gt_val)
        print(ans_base.shape, gt_val.shape)
        scores.append( score )
        
    score = np.mean(scores)
    
    print(params)
    print('Validation:', score)
    print('-' * 60)
    
    return score, scores, results, oofs, val_idx_list


# In[ ]:


X = train_base[train_cols].values
y = GT.values
groups = train_base.collectionName.values

score, scores, oofs, models_lat, models_lng = coord_cv_oof([params, params], groups)
score


# In[ ]:


oofs_copy = oofs.copy()
oofs_copy.loc[:, ['latDeg']] = oofs_copy['latDeg'] - oofs_copy['d_latDeg']
oofs_copy.loc[:, ['lngDeg']] = oofs_copy['lngDeg'] - oofs_copy['d_lngDeg']
print(get_score(oofs_copy, ground_truth))
oofs_copy.to_csv('oofs_4p303.csv', index=False)


# In[ ]:


get_score(train_base, ground_truth)


# ## Test prediction

# In[ ]:


TEST_INPUT_DIR = '/kaggle/input/google-smartphone-decimeter-challenge/'
test_fname = 'baseline_locations_test.csv'

test_base = pd.read_csv(TEST_INPUT_DIR + test_fname)


# In[ ]:


test_base_list = get_collections_list(test_base)

lag_dfs = [get_lag_features(df, lag_columns, lag_count) for df in test_base_list]
test_base = pd.concat(lag_dfs, axis=0)
test_base.reset_index(drop=True, inplace=True)


# In[ ]:


phone_feat_test = enc.transform(test_base.phoneName.values.reshape(-1, 1))
phone_feat_test_df = pd.DataFrame( phone_feat_test.toarray(), columns=phone_cols )
test_base = pd.concat([test_base, phone_feat_test_df], axis=1)


# In[ ]:


test_base_deltas = get_deltas(test_base)[train_cols]
X_test = test_base_deltas[train_cols].values
all_y_test = []
for model_lat, model_lng in zip(models_lat, models_lng):
    y_test = np.array([model_lat.predict(X_test), model_lng.predict(X_test)]).T
    all_y_test.append(y_test)


# In[ ]:


all_y_test = np.array(all_y_test)
all_y_test.shape


# In[ ]:


y_test = np.median(all_y_test, axis=0)
y_test


# In[ ]:


test_base[['latDeg', 'lngDeg']] -= y_test
test_base.head(2)


# In[ ]:


sub_num=40
test_base[['phone', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']].to_csv(f'subm{sub_num}.csv', index=False)
test_base.to_csv(f'baseline_locations_test_upd_{sub_num}.csv', index=False)


# ## Visualization

# In[ ]:


def plot_change():
    #TRAIN_INPUT_DIR = '/kaggle/input/gps-datasets/' 
    #train_fname = 'train_clean.csv'
    INPUT_DIR = '/kaggle/input/gps-datasets/test_subm_boost.csv'
    test_base_best = pd.read_csv(INPUT_DIR)
    test_base_best.head(2)
    
    cname = 'MTV-1'
    pname = 'Pixel4'

    ind1 = test_base_best.collectionName.apply(lambda w : cname in w).values & test_base_best.phoneName.apply(lambda w : pname == w).values
    collection1 = test_base_best[ind1]

    ind2 = test_base.collectionName.apply(lambda w : cname in w).values & test_base.phoneName.apply(lambda w : pname == w).values
    collection2 = test_base[ind2].copy()

    collection2['phoneName'] = collection2['phoneName'].apply(lambda w : w + '_new')
    visualize_collection( pd.concat([collection1, collection2]) )

