#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from matplotlib import ticker
import seaborn as sns
import warnings

# options
warnings.filterwarnings('ignore')


# # Data

# In[ ]:


train = pd.read_csv('../input/devday22-competition-datascience/train.csv')
test = pd.read_csv('../input/devday22-competition-datascience/test.csv')
submission = pd.read_csv('../input/devday22-competition-datascience/sample_submission.csv')


# # Exploratory Data Analysis (Basic EDA)

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


submission.head()


# * training data continous features distribution.

# In[ ]:


plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(10, 10), facecolor='#f6f5f5')
gs = fig.add_gridspec(7, 3)
gs.update(wspace=0.3, hspace=0.3)
background_color = "#dcdada"

run_no = 0
for row in range(0, 7):
    for col in range(0, 3):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1  

features_num = list([col for col in train.select_dtypes(exclude=object).columns if col not in ['ID', 'mutation']])

run_no = 0
for col in features_num:
    sns.kdeplot(ax=locals()["ax"+str(run_no)], x=train[col], zorder=2, alpha=1, linewidth=1, color='#FF355D')
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].set_ylabel('')
    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight='bold')
    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5)
    locals()["ax"+str(run_no)].xaxis.offsetText.set_fontsize(4)
    locals()["ax"+str(run_no)].yaxis.offsetText.set_fontsize(4)
    run_no += 1

plt.show()


# training data target label distribution.

# In[ ]:


plt.rcParams['figure.dpi'] = 100
fig = plt.figure(figsize=(5, 5), facecolor='#f6f5f5')
background_color = "#dcdada"
fig = sns.catplot(x="mutation",data=train, kind="count")


# # Training

# In[ ]:


def encoder(x_train, x_test):
    le = LabelEncoder()
    r = le.fit_transform(x_train)
    r2 = le.transform(x_test)
    return r, r2


# In[ ]:


features_cat = [col for col in train.columns if col in train.select_dtypes(include=object).columns]
features = [col for col in train.columns if col not in ['ID', 'mutation']]
print(f'total features: {len(features)}')
print(features)


# In[ ]:


X, y = train.loc[:, features], train.loc[:, 'mutation']
print(X.shape)
print(y.shape)


# In[ ]:


# encoding categorical features
for col in features_cat:
    X.loc[:, col], test.loc[:, col] = encoder(X.loc[:, col], test.loc[:, col])


# In[ ]:


test.head()


# In[ ]:


X.head()


# In[ ]:


# initializing classifier
clf = GaussianNB()
# training
clf.fit(X, y)
# 
print(clf)


# # Submission and Prediction

# * **predict_proba** is used instead of the typical **predict** method, this is due to the evaluation metric being the *area under the Roc curve*.

# In[ ]:


pred = clf.predict_proba(test[features])
pred = pred[:, 1]


# * for the conversion of dataframe to csv, it is necessary to specify **index=False**, if not specified, the submission will fail. 
# * The following format is **absolutely necessary.**

# In[ ]:


submission['mutation'] = pred
submission.to_csv('submission.csv', index=False)


# # Goodluck, Have Fun! 😄
