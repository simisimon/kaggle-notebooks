#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import gc
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train = pd.read_csv('../input/whichbookshouldiread/train_ratings.csv')
test = pd.read_csv('../input/whichbookshouldiread/test_ratings.csv')
users = pd.read_csv('../input/whichbookshouldiread/users.csv')
books = pd.read_csv('../input/whichbookshouldiread/books.csv')

ind = books.year.isin(['DK Publishing Inc', 'Gallimard'])
books.loc[ind, "publisher"] = books.loc[ind, "year"]
books.loc[ind, "year"] = books.loc[ind, "author"]
books['year'] = books['year'].astype(int)
books.loc[ind, "author"] = ["Michael Teitelbaum", "Jean-Marie Gustave Le ClÃ?Â©zio","James Buckley"]
books.loc[ind, "title"] = ["DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)", "Peuple du ciel, suivi de \'Les Bergers","DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"]

train.shape, test.shape, users.shape, books.shape


# In[ ]:


train['istest'] = 0
test['istest'] = 1
train = pd.concat( (train, test) )
del test
gc.collect()
train


# In[ ]:


train = train.merge(users, on='user_id', how='left')
train = train.merge(books, on='book_id', how='left')
train.shape


# In[ ]:


train['title'] = train['title'].apply(lambda x: x.lower())
train['author'] = train['author'].fillna('nonspecified').apply(lambda x: x.lower())
train['publisher'] = train['publisher'].fillna('nonspecified').apply(lambda x: x.lower())

train['title_nchars'] = train.title.apply(lambda x: len(x))
train['title_nwords'] = train.title.apply(lambda x: len(x.split(' ')))
train['author_nchars'] = train.author.apply(lambda x: len(x))
train['author_nwords'] = train.author.apply(lambda x: len(x.split(' ')))


train['author'] = train['author'].factorize()[0]
train['publisher'] = train['publisher'].factorize()[0]
train['province'] = train['province'].factorize()[0]
train['city'] = train['city'].factorize()[0]
train.loc[train.age>=244, 'age'] = 24
train.loc[train.age>=100, 'age'] = np.nan
train['mean_age'] = train.groupby(['title', 'country'])['age'].transform('mean')
train['mean_age'] = train['mean_age'].fillna( train.age.mean() )
train.loc[train.age.isnull() ,'age'] = train.loc[train.age.isnull() ,'mean_age']

train['dif_age'] = train['mean_age'] - train['age']
gc.collect()
train


# In[ ]:


dt = train.groupby(['title','country'])['id'].agg('count').reset_index()
dt = dt.sort_values('id', ascending=False)
dt['rank'] = dt.groupby('title')['id'].cumcount()
dt = dt.loc[dt['rank']==0].reset_index(drop=True)
del dt['id'], dt['rank']
dt.rename(columns={'country': 'top_country'}, inplace=True)
train = train.merge(dt, on='title', how='left')
train.loc[train.country.isnull() ,'country'] = train.loc[train.country.isnull() ,'top_country']
train['country'] = train['country'].factorize()[0]
del train['top_country']
train


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.feature_extraction.text import TfidfVectorizer\n\nvectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=2 )\nvectorizer.fit(train.title.unique())\nX = vectorizer.transform(train.title)\nX.shape\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.decomposition import TruncatedSVD\n\nsvd = TruncatedSVD(n_components=32)\nEMB = svd.fit_transform(X)\nEMB.shape\n')


# In[ ]:


for i in range(EMB.shape[1]):
    train[f'svd{i}'] = EMB[:,i]
del EMB, X, svd, vectorizer
train['title'] = train['title'].factorize()[0]
gc.collect()
train.head()


# In[ ]:


for i in range(32):
    train[f'svd_mean{i}'] = train.groupby('user_id')[f'svd{i}'].transform('mean')
    train[f'svd_dif{i}'] = train[f'svd_mean{i}'] - train[f'svd{i}']
train.shape


# In[ ]:


train['user_count'] = train.groupby('user_id')['id'].transform('count')
train['book_count'] = train.groupby('book_id')['id'].transform('count')
train['country_count'] = train.groupby('country')['id'].transform('count')
train['user_publisher_count'] = train.groupby(['user_id','publisher'])['id'].transform('count')
train['user_author_count'] = train.groupby(['user_id','author'])['id'].transform('count')
train['age'] = train['age']//4
train.shape


# In[ ]:


train['book_id'] = train['book_id'].factorize()[0].astype('int32')
train['user_id'] = train['user_id'].factorize()[0].astype('int32')
gc.collect()
train.head()


# In[ ]:


from sklearn.model_selection import StratifiedKFold

NFOLDS = 20
train['fold'] = 0
skf = StratifiedKFold(n_splits=NFOLDS, random_state=2022, shuffle=True)
for fold, (train_index, test_index) in enumerate(skf.split(train, train.rating.fillna(0))):
    train.loc[test_index,'fold'] = fold
train.loc[train.istest == 1, 'fold'] = -1

train.groupby('fold')['rating'].agg('mean')


# In[ ]:


dt = train.groupby('user_id')['book_id', 'author', 'publisher'].agg(list).reset_index()
dt['text'] = dt.apply(lambda x:  ' '.join(['b'+str(b) for b in x['book_id']] + ['a'+str(b) for b in x['author']] + ['p'+str(b) for b in x['publisher']]) ,axis=1)
dt


# In[ ]:


get_ipython().run_cell_magic('time', '', 'vectorizer = TfidfVectorizer(ngram_range=(1,6), max_df=0.9, min_df=2 )\nX = vectorizer.fit_transform(dt.text)\nX.shape\n')


# In[ ]:


svd = TruncatedSVD(n_components=256)
EMB = svd.fit_transform(X)
EMB.shape


# In[ ]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters=100, random_state=2022).fit(EMB)
np.unique(km.labels_, return_counts=True)


# In[ ]:


dt['km'] = km.labels_
dt


# In[ ]:


train = train.merge(dt[['user_id', 'km']], on='user_id', how='left')
train


# In[ ]:


from sklearn.metrics import f1_score

# Target Encoding with smoothing
def TE(tr, vl, feat=[], L=1):
    M = tr['rating'].mean()
    dt = tr.groupby(feat)['rating'].agg(['mean','count']).reset_index()
    dt['ll'] = ((dt['mean']*dt['count'])+(M*L)) / (dt['count']+L)
    del dt['mean'], dt['count']
    vl = vl[feat].merge(dt, on=feat, how='left').reset_index(drop=True)
    return vl


# In[ ]:


for fold in range(NFOLDS):
    print(fold)
    indtra = (train.fold!=fold)&(train.istest==0)
    indval = (train.fold==fold)&(train.istest==0)
    indtra2 = (train.istest==0)
    indval2 = (train.istest==1)
    
    for feat in [
        ['title'],
        ['author'],
        ['title'],
        ['age'],
        ['city'],
        ['province'],
        ['country'],
        ['author'],
        ['year'],
        ['publisher'],
        ['user_id'],
        ['book_id'],
        ['km'],
        ['book_id', 'country'],
        ['book_id', 'age'],
        ['country', 'age'],
        ['user_id', 'author'],
        ['user_id', 'publisher'],
        ['user_id', 'year'],
        ['km', 'author'],
        ['km', 'publisher'],
        ['km', 'year'],
    ]:
        fname = f"te_{'_'.join(feat)}"
        if fname not in list(train.columns):
            train[fname] = np.nan
        
        vl = TE( train.loc[indtra].copy(), train.loc[indval].copy(), feat=feat, L=1 )
        train.loc[indval, fname] = vl['ll'].values
        
        if fold == 0:
            vl = TE( train.loc[indtra2].copy(), train.loc[indval2].copy(), feat=feat, L=1 )
            train.loc[indval2, fname] = vl['ll'].values

gc.collect()
train.head()


# In[ ]:


# Target Encoding with smoothing
def TE_max(tr, vl, feat=[], L=1):
    M = tr['rating'].max()
    dt = tr.groupby(feat)['rating'].agg(['max','count']).reset_index()
    dt['ll'] = ((dt['max']*dt['count'])+(M*L)) / (dt['count']+L)
    del dt['max'], dt['count']
    vl = vl[feat].merge(dt, on=feat, how='left').reset_index(drop=True)
    return vl

for fold in range(NFOLDS):
    print(fold)
    indtra = (train.fold!=fold)&(train.istest==0)
    indval = (train.fold==fold)&(train.istest==0)
    indtra2 = (train.istest==0)
    indval2 = (train.istest==1)
    
    for feat in [
        ['user_id'],
        ['user_id', 'author'],
        ['user_id', 'publisher'],
        ['user_id', 'year'],
    ]:
        fname = f"te_max_{'_'.join(feat)}"
        if fname not in list(train.columns):
            train[fname] = np.nan
        
        vl = TE_max( train.loc[indtra].copy(), train.loc[indval].copy(), feat=feat, L=1 )
        train.loc[indval, fname] = vl['ll'].values
        
        if fold == 0:
            vl = TE_max( train.loc[indtra2].copy(), train.loc[indval2].copy(), feat=feat, L=1 )
            train.loc[indval2, fname] = vl['ll'].values

gc.collect()
train.head()


# In[ ]:


# Target Encoding with smoothing
def TE_min(tr, vl, feat=[], L=1):
    M = tr['rating'].min()
    dt = tr.groupby(feat)['rating'].agg(['min','count']).reset_index()
    dt['ll'] = ((dt['min']*dt['count'])+(M*L)) / (dt['count']+L)
    del dt['min'], dt['count']
    vl = vl[feat].merge(dt, on=feat, how='left').reset_index(drop=True)
    return vl

for fold in range(NFOLDS):
    print(fold)
    indtra = (train.fold!=fold)&(train.istest==0)
    indval = (train.fold==fold)&(train.istest==0)
    indtra2 = (train.istest==0)
    indval2 = (train.istest==1)
    
    for feat in [
        ['user_id'],
        ['user_id', 'author'],
        ['user_id', 'publisher'],
        ['user_id', 'year'],
    ]:
        fname = f"te_min_{'_'.join(feat)}"
        if fname not in list(train.columns):
            train[fname] = np.nan
        
        vl = TE_min( train.loc[indtra].copy(), train.loc[indval].copy(), feat=feat, L=1 )
        train.loc[indval, fname] = vl['ll'].values
        
        if fold == 0:
            vl = TE_min( train.loc[indtra2].copy(), train.loc[indval2].copy(), feat=feat, L=1 )
            train.loc[indval2, fname] = vl['ll'].values

gc.collect()
train.head()


# In[ ]:


# Target Encoding with smoothing
def TE_std(tr, vl, feat=[], L=1):
    M = tr['rating'].std()
    dt = tr.groupby(feat)['rating'].agg(['std','count']).reset_index()
    dt['ll'] = ((dt['std']*dt['count'])+(M*L)) / (dt['count']+L)
    del dt['std'], dt['count']
    vl = vl[feat].merge(dt, on=feat, how='left').reset_index(drop=True)
    return vl

for fold in range(NFOLDS):
    print(fold)
    indtra = (train.fold!=fold)&(train.istest==0)
    indval = (train.fold==fold)&(train.istest==0)
    indtra2 = (train.istest==0)
    indval2 = (train.istest==1)
    
    for feat in [
        ['user_id'],
        ['user_id', 'author'],
        ['user_id', 'publisher'],
        ['user_id', 'year'],
    ]:
        fname = f"te_std_{'_'.join(feat)}"
        if fname not in list(train.columns):
            train[fname] = np.nan
        
        vl = TE_std( train.loc[indtra].copy(), train.loc[indval].copy(), feat=feat, L=1 )
        train.loc[indval, fname] = vl['ll'].values
        
        if fold == 0:
            vl = TE_std( train.loc[indtra2].copy(), train.loc[indval2].copy(), feat=feat, L=1 )
            train.loc[indval2, fname] = vl['ll'].values


gc.collect()
train.head()


# In[ ]:


FEATURES = [
    col for col in train.columns if col not in 
        ['id','user_id','book_id','rating','istest','fold']
]
FEATURES


# In[ ]:


train['pred'] = 0
TEST = []
for fold in range(NFOLDS):
    tra = train.loc[ (train.fold!=fold)&(train.istest==0) ].reset_index(drop=True)
    val = train.loc[ (train.fold==fold)|(train.istest==1) ].reset_index(drop=True)
    vl = train.loc[ (train.fold==fold)&(train.istest==0) ].reset_index(drop=True)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 10,
        'num_leaves': 63,
        'learning_rate': 0.02,
        'feature_fraction': 0.60,
        'bagging_fraction': 0.90,
        "min_data_in_leaf": 50,
        'bagging_freq': 1,
        'verbose': -1,
        'num_threads': -1,
        'max_bin': 255,
    }
    
    lgb_train = lgb.Dataset(tra[FEATURES], tra['rating']-1)
    lgb_valid = lgb.Dataset(vl[FEATURES], vl['rating']-1)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_valid,
                    callbacks=[lgb.early_stopping(stopping_rounds=50)],
                    verbose_eval=50,
                   )
    train.loc[ (train.fold==fold)|(train.istest==1), 'pred'] = gbm.predict(val[FEATURES]).argmax(1) + 1
    TEST.append( train.loc[train.istest==1, ['id','pred']].copy() )
    
    del gbm, lgb_train, lgb_valid, tra, val, vl
    gc.collect()


# In[ ]:


val = train.loc[train.istest==0].reset_index(drop=True)
gc.collect()

print(val.shape)
f1_score(val['rating'], val['pred'], average='micro')


# In[ ]:


del train
gc.collect()


# In[ ]:


sub = pd.concat(TEST).reset_index(drop=True)

del TEST
gc.collect()

sub.head()


# In[ ]:


sub = sub.groupby('id')['pred'].agg('mean').reset_index()
sub['rating'] = sub['pred'].round().astype('int')
sub[['id', 'rating']].to_csv('submission-giba.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




