#!/usr/bin/env python
# coding: utf-8

# This notebook is the binary lightgbm inference baseline.
# Another notebook is for training data generation [[training_data_for_binary_lgb_baseline](https://www.kaggle.com/code/guoyonfan/training-data-for-binary-lgb-baseline-0-834)].

# In[ ]:


## Imports
import warnings
warnings.filterwarnings('ignore')

import os
import gc
import time
import random
import Levenshtein
import difflib
import multiprocessing
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm.auto import tqdm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# In[ ]:


TRAIN_FEATURES = ['kdist',
                'kneighbors',
                'kdist_country',
                'kneighbors_country',
                'name_sim',
                'name_gesh',
                'name_leven',
                'name_jaro',
                'name_lcs',
                'name_len_diff',
                'name_nleven',
                'name_nlcsk',
                'name_nlcs',
                'address_sim',
                'address_gesh',
                'address_leven',
                'address_jaro',
                'address_lcs',
                'address_len_diff',
                'address_nleven',
                'address_nlcsk',
                'address_nlcs',
                'city_gesh',
                'city_leven',
                'city_jaro',
                'city_lcs',
                'city_len_diff',
                'city_nleven',
                'city_nlcsk',
                'city_nlcs',
                'state_sim',
                'state_gesh',
                'state_leven',
                'state_jaro',
                'state_lcs',
                'state_len_diff',
                'state_nleven',
                'state_nlcsk',
                'state_nlcs',
                'zip_gesh',
                'zip_leven',
                'zip_jaro',
                'zip_lcs',
                'url_sim',
                'url_gesh',
                'url_leven',
                'url_jaro',
                'url_lcs',
                'url_len_diff',
                'url_nleven',
                'url_nlcsk',
                'url_nlcs',
                'phone_gesh',
                'phone_leven',
                'phone_jaro',
                'phone_lcs',
                'categories_sim',
                'categories_gesh',
                'categories_leven',
                'categories_jaro',
                'categories_lcs',
                'categories_len_diff',
                'categories_nleven',
                'categories_nlcsk',
                'categories_nlcs',
                'country_sim',
                'country_gesh',
                'country_leven',
                'country_nleven',
                'kdist_diff',
                'kneighbors_mean',]


# In[ ]:


## Parameters
NUM_NEIGHBOR = 20
SEED = 2022
THRESHOLD = 0.5
NUM_SPLIT = 5
feat_columns = ['name', 'address', 'city', 
            'state', 'zip', 'url', 
           'phone', 'categories', 'country']
vec_columns = ['name', 'categories', 'address', 
               'state', 'url', 'country']

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
seed_everything(SEED)


# In[ ]:


get_ipython().run_line_magic('load_ext', 'Cython')


# In[ ]:


get_ipython().run_cell_magic('cython', '', 'def LCS(str S, str T):\n    cdef int i, j\n    cdef list dp = [[0] * (len(T) + 1) for _ in range(len(S) + 1)]\n    for i in range(len(S)):\n        for j in range(len(T)):\n            dp[i + 1][j + 1] = max(dp[i][j] + (S[i] == T[j]), dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1])\n    return dp[len(S)][len(T)]\n')


# In[ ]:


def post_process(df):
    id2match = dict(zip(df['id'].values, df['matches'].str.split()))

    for base, match in df[['id', 'matches']].values:
        match = match.split()
        if len(match) == 1:        
            continue

        for m in match:
            if base not in id2match[m]:
                id2match[m].append(base)
    df['matches'] = df['id'].map(id2match).map(' '.join)
    return df 


# In[ ]:


def recall_knn(df, Neighbors = 10):
    print('Start knn grouped by country')
    train_df_country = []
    for country, country_df in tqdm(df.groupby('country')):
        country_df = country_df.reset_index(drop = True)

        neighbors = min(len(country_df), Neighbors)
        knn = KNeighborsRegressor(n_neighbors = neighbors,
                                    metric = 'haversine',
                                    n_jobs = -1)
        knn.fit(country_df[['latitude','longitude']], country_df.index)
        dists, nears = knn.kneighbors(country_df[['latitude', 'longitude']], 
                                        return_distance = True)

        for k in range(neighbors):            
            cur_df = country_df[['id']]
            cur_df['match_id'] = country_df['id'].values[nears[:, k]]
            cur_df['kdist_country'] = dists[:, k]
            cur_df['kneighbors_country'] = k
            
            train_df_country.append(cur_df)
    train_df_country = pd.concat(train_df_country)
    
    print('Start knn')
    train_df = []
    knn = NearestNeighbors(n_neighbors = Neighbors)
    knn.fit(df[['latitude','longitude']], df.index)
    dists, nears = knn.kneighbors(df[['latitude','longitude']])
    
    for k in range(Neighbors):            
        cur_df = df[['id']]
        cur_df['match_id'] = df['id'].values[nears[:, k]]
        cur_df['kdist'] = dists[:, k]
        cur_df['kneighbors'] = k
        train_df.append(cur_df)
    
    train_df = pd.concat(train_df)
    train_df = train_df.merge(train_df_country,
                                 on = ['id', 'match_id'],
                                 how = 'outer')
    del train_df_country
    
    return train_df


# In[ ]:


def add_features(df):    
    for col in tqdm(feat_columns):       
        if col in vec_columns:
            tv_fit = tfidf_d[col]
            indexs = [id2index_d[i] for i in df['id']]
            match_indexs = [id2index_d[i] for i in df['match_id']]                    
            df[f'{col}_sim'] = np.array(tv_fit[indexs].multiply(tv_fit[match_indexs]).sum(axis = 1)).ravel()
        
        col_values = data.loc[df['id']][col].values.astype(str)
        matcol_values = data.loc[df['match_id']][col].values.astype(str)
        
        geshs = []
        levens = []
        jaros = []
        lcss = []
        for s, match_s in zip(col_values, matcol_values):
            if s != 'nan' and match_s != 'nan':                    
                geshs.append(difflib.SequenceMatcher(None, s, match_s).ratio())
                levens.append(Levenshtein.distance(s, match_s))
                jaros.append(Levenshtein.jaro_winkler(s, match_s))
                lcss.append(LCS(str(s), str(match_s)))
            else:
                geshs.append(np.nan)
                levens.append(np.nan)
                jaros.append(np.nan)
                lcss.append(np.nan)
        
        df[f'{col}_gesh'] = geshs
        df[f'{col}_leven'] = levens
        df[f'{col}_jaro'] = jaros
        df[f'{col}_lcs'] = lcss
        
        if col not in ['phone', 'zip']:
            df[f'{col}_len'] = list(map(len, col_values))
            df[f'match_{col}_len'] = list(map(len, matcol_values)) 
            df[f'{col}_len_diff'] = np.abs(df[f'{col}_len'] - df[f'match_{col}_len'])
            df[f'{col}_nleven'] = df[f'{col}_leven'] / \
                                    df[[f'{col}_len', f'match_{col}_len']].max(axis = 1)
            
            df[f'{col}_nlcsk'] = df[f'{col}_lcs'] / df[f'match_{col}_len']
            df[f'{col}_nlcs'] = df[f'{col}_lcs'] / df[f'{col}_len']
            
            df = df.drop(f'{col}_len', axis = 1)
            df = df.drop(f'match_{col}_len', axis = 1)
            gc.collect()
            
    return df


# In[ ]:


## Dada process
data = pd.read_csv('../input/foursquare-location-matching/test.csv')

if len(data) < 20:
    data = pd.read_csv('../input/foursquare-location-matching/train.csv',
                      nrows = 100)
    data = data.drop('point_of_interest', axis = 1)
    
id2index_d = dict(zip(data['id'].values, data.index))

tfidf_d = {}
for col in vec_columns:
    tfidf = TfidfVectorizer()
    tv_fit = tfidf.fit_transform(data[col].fillna('nan'))
    tfidf_d[col] = tv_fit

out_df = pd.DataFrame()
out_df['id'] = data['id'].unique().tolist()
out_df['match_id'] = out_df['id']

test_data = recall_knn(data, NUM_NEIGHBOR)
data = data.set_index('id')

print('Num of unique id: %s' % test_data['id'].nunique())
print('Num of test data: %s' % len(test_data))
print(test_data.sample(5))


# In[ ]:


## Model load
lgb_model_path = '../input/binary-lgb-baseline/lgb_baseline.lgb'
lgb_model = lgb.Booster(model_file = lgb_model_path)


# In[ ]:


## Prediction
count = 0
start_row = 0
pred_df = pd.DataFrame()
unique_id = test_data['id'].unique().tolist()
num_split_id = len(unique_id) // NUM_SPLIT
for k in range(1, NUM_SPLIT + 1):
    print('Current split: %s' % k)
    end_row = start_row + num_split_id
    if k < NUM_SPLIT:
        cur_id = unique_id[start_row : end_row]
        cur_data = test_data[test_data['id'].isin(cur_id)]
    else:
        cur_id = unique_id[start_row: ]
        cur_data = test_data[test_data['id'].isin(cur_id)]
    
    # add features & model prediction
    cur_data = add_features(cur_data)
    cur_data['kdist_diff'] = (cur_data['kdist'] - cur_data['kdist_country']) /\
                                cur_data['kdist_country']
    cur_data['kneighbors_mean'] = cur_data[['kneighbors', 'kneighbors_country']].mean(axis = 1)
    cur_data['pred'] = lgb_model.predict(cur_data[TRAIN_FEATURES])
    cur_pred_df = cur_data[cur_data['pred'] > THRESHOLD][['id', 'match_id']]
    pred_df = pd.concat([pred_df, cur_pred_df])
    
    start_row = end_row
    count += len(cur_data)

    del cur_data, cur_pred_df
    gc.collect()
print(count)


# In[ ]:


## Submission    
out_df = pd.concat([out_df, pred_df])
out_df = out_df.groupby('id')['match_id'].\
                        apply(list).reset_index()
out_df['matches'] = out_df['match_id'].apply(lambda x: ' '.join(set(x)))
out_df = post_process(out_df)
print('Unique id: %s' % len(out_df))
print(out_df.head())

out_df[['id', 'matches']].to_csv('submission.csv', index = False)

