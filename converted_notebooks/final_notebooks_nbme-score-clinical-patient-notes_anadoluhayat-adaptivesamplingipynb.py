#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgbm
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
import json
import gc
pd.options.display.max_rows=300

import os


# In[ ]:


sample = pd.read_csv("../input/anadolu-hayat-emeklilik-datathon-coderspace/samplesubmission.csv")
test = pd.read_csv("../input/anadolu-hayat-emeklilik-datathon-coderspace/test.csv")
train = pd.read_csv("../input/anadolu-hayat-emeklilik-datathon-coderspace/train.csv")


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


# combine data sets for convenience in feature engineering
dt = pd.concat([train, test])
# Change column names for convenience
cols = dt.columns
cols =list(map(str.lower, cols))

dt.columns = cols


# In[ ]:


dt.rename(columns = {'subat_odenen_tu':'subat_odenen_tutar'}, inplace = True)


# In[ ]:


# baslangic tarihi related
dt["year"] = pd.to_datetime(dt["baslangic_tarihi"]).dt.year
dt["month"] = pd.to_datetime(dt["baslangic_tarihi"]).dt.month


# In[ ]:


y = dt.artis_durumu
dt.dropna(axis=1, how='any', inplace = True)
dt['artis_durumu'] = y
del y


# In[ ]:


features = dt.loc[:, (dt.dtypes == int) | (dt.dtypes == float)].columns.tolist()
features.pop(0) # policy_id
features.pop(33) # artis_durumu, target


# In[ ]:


train = dt.loc[dt['artis_durumu'].isnull() == False]
test = dt.loc[dt['artis_durumu'].isnull() == True]


# In[ ]:


cutoff = 0.5 # can be decided dynamically in the fitting part
num_of_selected = 10000


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train,
                                                    train['artis_durumu'],
                                                    test_size=0.2,
                                                    stratify = train['artis_durumu'],
                                                    random_state=0)
# set data sets
X_train['artis_durumu'] = y_train
adaptive_data = X_train[X_train['artis_durumu'] == 1].copy()
remaining_data = X_train[X_train['artis_durumu'] == 0].copy()


# In[ ]:


test_perf = []
for i in range(15):
    print(i)
    lgbm_fit = lgbm.LGBMClassifier(boosting_type='gbdt', 
                                   objective='binary', 
                                   metric='roc_auc'
                                  )
    lgbm_fit.fit(adaptive_data[features],adaptive_data['artis_durumu'])
    
    prob = lgbm_fit.predict_proba(remaining_data[features])
    prob = pd.DataFrame(prob)[1]
    
    # test performance
    test_prob = lgbm_fit.predict_proba(X_test[features])
    test_prob = pd.DataFrame(test_prob)[1]
    test_auc = round(roc_auc_score(y_test,test_prob),2)
    test_perf = test_perf + test_auc
    print(test_auc)
    
    # selection of new data points
    remaining_data['prob'] = prob
    remaining_data = remaining_data.sort_values('prob', ascending = True) # less convinced
    selected_ids = remaining_data.iloc[range(num_of_selected)]['policy_id']

    # update adaptive data by adding selected ids
    adaptive_data = pd.concat([adaptive_data,
                               remaining_data[remaining_data['policy_id'].isin(selected_ids)== True]])
    
    remaining_data = remaining_data[remaining_data['policy_id'].isin(selected_ids) == False]
    print(adaptive_data.shape)
  


# In[ ]:


# predictions for test set
subm_pred = lgbm_fit.predict(test[features])


# In[ ]:


sample['ARTIS_DURUMU'] = np.where(subm_pred<1,0,1)
sample[['ARTIS_DURUMU']].value_counts(normalize = True)


# In[ ]:


sample.to_csv("AdaptiveSampling_basicdeneme2.csv",index =False)


# In[ ]:




