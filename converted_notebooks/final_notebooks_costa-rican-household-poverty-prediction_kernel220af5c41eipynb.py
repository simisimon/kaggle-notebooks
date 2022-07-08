#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/train.csv')


# In[ ]:


data.info()


# In[ ]:


data.columns[data.dtypes==object]


# In[ ]:


data['dependency'].unique()


# In[ ]:


data[(data['dependency']=='no') & (data['SQBdependency']!=0)]


# In[ ]:


data[(data['dependency']=='yes') & (data['SQBdependency']!=1)]


# In[ ]:


data[(data['dependency']=='3') & (data['SQBdependency']!=9)]


# In[ ]:


data['dependency']=np.sqrt(data['SQBdependency'])


# In[ ]:


data['edjefe'].unique()


# In[ ]:


data['edjefa'].unique()


# In[ ]:


data['SQBedjefe'].unique()


# In[ ]:


data[['edjefe', 'edjefa', 'SQBedjefe']][:20]


# In[ ]:


data[['edjefe', 'edjefa', 'SQBedjefe']][data['edjefe']=='yes']


# In[ ]:


data[(data['edjefe']=='yes') & (data['edjefa']!='no')]


# In[ ]:


data[(data['edjefa']=='yes') & (data['parentesco1']==1)][['edjefe', 'edjefa', 'parentesco1', 'escolari']]


# In[ ]:


data[data['edjefe']=='yes'][['edjefe', 'edjefa','age', 'escolari', 'parentesco1','male', 'female', 'idhogar']]


# In[ ]:


data[(data['edjefe']=='no') & (data['edjefa']=='no')][['edjefe', 'edjefa', 'age', 'escolari', 'female', 'male', 'Id', 'parentesco1', 'idhogar']]


# In[ ]:


data[(data['edjefe']=='yes') & data['parentesco1']==1][['escolari']]


# In[ ]:


conditions = [
    (data['edjefe']=='no') & (data['edjefa']=='no'), #both no
    (data['edjefe']=='yes') & (data['edjefa']=='no'), # yes and no
    (data['edjefe']=='no') & (data['edjefa']=='yes'), #no and yes 
    (data['edjefe']!='no') & (data['edjefe']!='yes') & (data['edjefa']=='no'), # number and no
    (data['edjefe']=='no') & (data['edjefa']!='no') # no and number
]
choices = [0, 1, 1, data['edjefe'], data['edjefa']]
data['edjefx']=np.select(conditions, choices)
data['edjefx']=data['edjefx'].astype(int)
data[['edjefe', 'edjefa', 'edjefx']][:15]


# In[ ]:


data.describe()


# In[ ]:


data.columns[data.isna().sum()!=0]


# In[ ]:


data[data['meaneduc'].isnull()]


# In[ ]:


data[data['meaneduc'].isnull()][['Id','idhogar','edjefe','edjefa', 'hogar_adul', 'hogar_mayor', 'hogar_nin', 'age', 'escolari']]


# In[ ]:


print(len(data[data['idhogar']==data.iloc[1291]['idhogar']]))
print(len(data[data['idhogar']==data.iloc[1840]['idhogar']]))
print(len(data[data['idhogar']==data.iloc[2049]['idhogar']]))


# In[ ]:


meaneduc_nan=data[data['meaneduc'].isnull()][['Id','idhogar','escolari']]


# In[ ]:


me=meaneduc_nan.groupby('idhogar')['escolari'].mean().reset_index()


# In[ ]:


me


# In[ ]:


for row in meaneduc_nan.iterrows():
    idx=row[0]
    idhogar=row[1]['idhogar']
    m=me[me['idhogar']==idhogar]['escolari'].tolist()[0]
    data.at[idx, 'meaneduc']=m
    data.at[idx, 'SQBmeaned']=m*m
    


# In[ ]:


data['v2a1'].isnull().sum()


# In[ ]:


norent=data[data['v2a1'].isnull()]
print("Owns his house:", norent[norent['tipovivi1']==1]['Id'].count())
print("Owns his house paying installments", norent[norent['tipovivi2']==1]['Id'].count())
print("Rented ", norent[norent['tipovivi3']==1]['Id'].count())
print("Precarious ", norent[norent['tipovivi4']==1]['Id'].count())
print("Other ", norent[norent['tipovivi5']==1]['Id'].count())
print("Total ", 6860)


# In[ ]:


data['v2a1']=data['v2a1'].fillna(0)


# In[ ]:


data['v18q1'].isna().sum()


# In[ ]:


tabletnan=data[data['v18q1'].isnull()]
tabletnan[tabletnan['v18q']==0]['Id'].count()


# In[ ]:


data['v18q1'].unique()


# In[ ]:


data['v18q1']=data['v18q1'].fillna(0)


# In[ ]:


data['rez_esc'].isnull().sum()


# In[ ]:


data['rez_esc'].describe()


# In[ ]:


data['rez_esc'].unique()


# In[ ]:


data[data['rez_esc']>1][['age', 'escolari', 'rez_esc']][:20]


# In[ ]:


rez_esc_nan=data[data['rez_esc'].isnull()]
rez_esc_nan[(rez_esc_nan['age']<18) & rez_esc_nan['escolari']>0][['age', 'escolari']]


# In[ ]:


data['rez_esc']=data['rez_esc'].fillna(0)


# In[ ]:


d={}
weird=[]
for row in data.iterrows():
    idhogar=row[1]['idhogar']
    target=row[1]['Target']
    if idhogar in d:
        if d[idhogar]!=target:
            weird.append(idhogar)
    else:
        d[idhogar]=target


# In[ ]:


len(set(weird))


# In[ ]:


data[data['idhogar']==weird[2]][['idhogar','parentesco1', 'Target']]


# In[ ]:


for i in set(weird):
    hhold=data[data['idhogar']==i][['idhogar', 'parentesco1', 'Target']]
    target=hhold[hhold['parentesco1']==1]['Target'].tolist()[0]
    for row in hhold.iterrows():
        idx=row[0]
        if row[1]['parentesco1']!=1:
            data.at[idx, 'Target']=target


# In[ ]:


data[data['idhogar']==weird[1]][['idhogar','parentesco1', 'Target']]


# In[ ]:


def data_cleaning(data):
    data['dependency']=np.sqrt(data['SQBdependency'])
    data['rez_esc']=data['rez_esc'].fillna(0)
    data['v18q1']=data['v18q1'].fillna(0)
    data['v2a1']=data['v2a1'].fillna(0)
    
    conditions = [
    (data['edjefe']=='no') & (data['edjefa']=='no'), #both no
    (data['edjefe']=='yes') & (data['edjefa']=='no'), # yes and no
    (data['edjefe']=='no') & (data['edjefa']=='yes'), #no and yes 
    (data['edjefe']!='no') & (data['edjefe']!='yes') & (data['edjefa']=='no'), # number and no
    (data['edjefe']=='no') & (data['edjefa']!='no') # no and number
    ]
    choices = [0, 1, 1, data['edjefe'], data['edjefa']]
    data['edjefx']=np.select(conditions, choices)
    data['edjefx']=data['edjefx'].astype(int)
    data.drop(['edjefe', 'edjefa'], axis=1, inplace=True)
    
    meaneduc_nan=data[data['meaneduc'].isnull()][['Id','idhogar','escolari']]
    me=meaneduc_nan.groupby('idhogar')['escolari'].mean().reset_index()
    for row in meaneduc_nan.iterrows():
        idx=row[0]
        idhogar=row[1]['idhogar']
        m=me[me['idhogar']==idhogar]['escolari'].tolist()[0]
        data.at[idx, 'meaneduc']=m
        data.at[idx, 'SQBmeaned']=m*m
        
    return data


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


data['Target'].hist()


# In[ ]:


data_undersampled=data.drop(data.query('Target == 4').sample(frac=.75).index)


# In[ ]:


data_undersampled['Target'].hist()


# In[ ]:


X=data_undersampled.drop(['Id', 'idhogar', 'Target', 'edjefe', 'edjefa'], axis=1)
y=data_undersampled['Target']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


clf = RandomForestClassifier()
params={'n_estimators': list(range(40,61, 1))}
gs = GridSearchCV(clf, params, cv=5)


# In[ ]:


gs.fit(X_train, y_train)


# In[ ]:


preds=gs.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, preds))


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, preds))


# In[ ]:


print(gs.best_params_)
print(gs.best_score_)
print(gs.best_estimator_)


# In[ ]:


cvres = gs.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(mean_score), params)


# In[ ]:


test_data=pd.read_csv('../input/test.csv')


# In[ ]:


test_data=data_cleaning(test_data)


# In[ ]:


ids=test_data['Id']
test_data.drop(['Id', 'idhogar'], axis=1, inplace=True)


# In[ ]:


test_predictions=gs.predict(test_data)


# In[ ]:


test_predictions[:5]


# In[ ]:


submit=pd.DataFrame({'Id': ids, 'Target': test_predictions})


# In[ ]:


submit.to_csv('submit.csv', index=False)


# In[ ]:




