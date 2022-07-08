#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Import libraries here
import warnings
from glob import glob

import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import seaborn as sns
import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore", category=FutureWarning)


# ## EDA

# In[ ]:


df = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')


# In[ ]:


df.head()


# In[ ]:


cols = df.select_dtypes([np.number]).columns


# In[ ]:


cols


# In[ ]:


fig, axs = plt.subplots(6,figsize=(30,20))
for idx, name in enumerate(cols):
    df[cols[idx]].plot(kind='box', vert=False, ax=axs[idx])
    axs[idx].set_title(name, fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:


# Index(['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], dtype='object')
# Age < 70
# RoomService < 10000
# FoodCourt < 20000
# ShoppingMall < 10000
# Spa < 15000
# VRDeck < 15000
df = df[df['RoomService']<10000]
df = df[df['FoodCourt']<20000]
df = df[df['ShoppingMall']<10000]
df = df[df['Spa']<15000]
df = df[df['VRDeck']<15000]


# In[ ]:


fig, axs = plt.subplots(6,figsize=(30,20))
for idx, name in enumerate(cols):
    df[cols[idx]].plot(kind='box', vert=False, ax=axs[idx])
    axs[idx].set_title(name, fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:


min_age = min(df['Age'])
max_age = max(df['Age'])

print(f"min: {min_age}")
print(f"max: {max_age}")


# In[ ]:


df['Age'] = pd.cut(df['Age'], [min_age-1, 20, 40, 60, max_age+1], labels=['<20','20-40','40-60','>60'], retbins=False, right=False)


# In[ ]:


df['Age']


# ## code session

# In[ ]:


def wrangle(filename):
    df = pd.read_csv(filename)
    
    # drop unique variable
    df.drop(columns=['PassengerId'], inplace=True)
    
    # split cabin
    # Split "dech/num/side" column
    df[["deck", "num", "side"]] = df["Cabin"].str.split("/", expand=True).astype(str)
    df.drop(columns=['Cabin'], inplace=True)
    
    # dropna 
    thresh = len(df) * .5
    df.dropna(axis=1, thresh=thresh, inplace=True)
    return df


# In[ ]:


def submission_wrangle(filename):
    df = pd.read_csv(filename)
    
    # drop unique variable
    df.drop(columns=['PassengerId'], inplace=True)
    
    # split cabin
    # Split "dech/num/side" column
    df[["deck", "num", "side"]] = df["Cabin"].str.split("/", expand=True).astype(str)
    df.drop(columns=['Cabin'], inplace=True)
    return df


# In[ ]:


train_df = wrangle('/kaggle/input/spaceship-titanic/train.csv')
test_df = wrangle('/kaggle/input/spaceship-titanic/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


sns.heatmap(train_df.corr())
# seem everything fine, doesn't have any Multicollinearity relationship


# In[ ]:


X = train_df.drop(columns=['Transported'])
y = train_df['Transported']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


len(X_train)


# In[ ]:


# basedline model 
print(f'prior prob: {y_train.mean()}')


# In[ ]:


# Build Model
model = make_pipeline(OneHotEncoder(use_cat_names=True), 
                      SimpleImputer(), 
                      GradientBoostingClassifier(n_estimators=200,max_features='auto',max_depth=8,min_samples_leaf=20,verbose=True)) 
# Fit model
model.fit(X_train,y_train)


# In[ ]:


y_pred = model.predict(X_train)
print(y_pred)
print(classification_report(y_train, y_pred))


# In[ ]:


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[ ]:


X_test


# In[ ]:


X_submission = submission_wrangle('/kaggle/input/spaceship-titanic/test.csv')


# In[ ]:


X_submission.head()


# In[ ]:


y_submission = model.predict(X_submission)


# In[ ]:


y_submission


# In[ ]:


submission = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')


# In[ ]:


submission['Transported'] = y_submission


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




