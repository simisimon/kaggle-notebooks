#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Logistic Regression without parameter tuning or feature enhancement
# Gets an average AUC score of .9877
# Not bad for first Kaggle comp attempt with minimal effort!!! :)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # EDA

# In[ ]:


def prep_data(df):
    df = df.fillna(0)
    return df

def parse_audio_embedding(df):
    df_subs = []
    for i in range(len(df)):
        # got the idea of using iteration on audio embedding from Tee Ming Yi, thanks!
        df_sub = pd.DataFrame([x for x in df.audio_embedding.iloc[i]])
        df_sub['vid_id'] = df.vid_id.iloc[i]
        df_subs.append(df_sub)
    df = pd.concat(df_subs)
    return df.groupby('vid_id').mean()

def get_data(filename):
    df1 = pd.read_json(filename)
    df2 = parse_audio_embedding(df1)
    df = df1.merge(df2, how='inner', on='vid_id')
    df = prep_data(df)
    print("Records: {}".format(len(df)))
    return df

def get_x(df):
    #x = df.drop(['audio_embedding', 'is_turkey', 'vid_id'], axis=1).values
    x = df.drop(['audio_embedding', 'end_time_seconds_youtube_clip', 'is_turkey','start_time_seconds_youtube_clip', 'vid_id'], axis=1).values
    return x

def get_y(df):
    y = df['is_turkey'].values
    return y


# In[ ]:


df_train = get_data('../input/train.json')
df_test = get_data('../input/test.json')


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# In[ ]:


df_train.columns


# In[ ]:


df_train.head()


# In[ ]:


df_train.hist(column='is_turkey')


# # Logistic Regression

# In[ ]:


from sklearn.model_selection import train_test_split

X = get_x(df_train)
y = get_y(df_train)
X_train, X_test, y_train, y_test = train_test_split(X, y)

print('Train: {}'.format(X_train.shape))
print('Test: {}'.format(X_test.shape))  


# In[ ]:


import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression   

model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)      
print('AUC: {}'.format(roc_auc_score(y_test, y_pred)))
print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))


# # Cross Validation

# In[ ]:


from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
kf.get_n_splits(X, y)
results, i = [], 0
for train_index, test_index in kf.split(X,y):
    model = LogisticRegression(solver='lbfgs')
    model.fit(X[train_index], y[train_index])    
    y_pred = model.predict(X_test)      
    results.append(roc_auc_score(y_test, y_pred))
    print("Fold {} AUC score: {}".format(i+1, results[i]))
    i+=1
print("Average AUC score:", np.mean(results))


# # ROC Curve

# In[ ]:


tpr, fpr, thresholds = metrics.roc_curve(y_pred, y_test)

plt.plot(tpr, fpr)
plt.title("ROC of last Prediction")
plt.show()

