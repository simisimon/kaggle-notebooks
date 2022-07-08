#!/usr/bin/env python
# coding: utf-8

# In[266]:


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


# # Data exploration
# will go through each files and explore what features can be consider for modeling.
# 1. train_user file
# 2. sessions
# 3. countries
# 4. age_gender_bkts

# In[267]:


#loading all the files and checking their shapes respectively
train_df = pd.read_csv('../input/train_users_2.csv',parse_dates=['date_account_created'])
test_df = pd.read_csv('../input/test_users.csv',parse_dates=['date_account_created'])
age_gender_df = pd.read_csv('../input/age_gender_bkts.csv')
countries_df = pd.read_csv('../input/countries.csv')
session_df = pd.read_csv('../input/sessions.csv')
train_df.shape,test_df.shape,age_gender_df.shape,countries_df.shape,session_df.shape


# ## 1. EDA on train data frame

# In[268]:


# lets see first 5 rows
train_df.head()


# In[269]:


#lets inspect the columns
train_df.info()


# In[270]:


#first find out any missing values
(train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)


# as age, date_first_booking are having almost 50 % missing values.
# * we could consider removing data_first_booking colum as we cant impute it's missing values with  any of the methods like(mean, 90%, etc..)
# * about age colum we try out possibilities of imputing with (avg, 90%)

# In[271]:


#lets see value counts for each of the categorical columns to  see if any of the values hardcoded in some common formate
category_columns = list(train_df.columns[train_df.dtypes == np.object].values)
category_columns.remove('date_first_booking') #doesnt make any sense to  see how many unique dates are present
category_columns.remove('country_destination') #target column 
category_columns.remove('id') #because it's unique one
print(category_columns)


# In[272]:


#lets print values counts for above columns
for a in category_columns:
    print(train_df[a].value_counts(),'value coutns',a)


# we can see except gender all values makes sense,
# 
# So we should consider doing some kind of imputation to the gender column.
# 
# And we can get dummy variables for all the category columns

# In[273]:


#lets analysis for remainng columns
remaining_columns = list(train_df.columns[train_df.dtypes != np.object].values)
remaining_columns

#we can convert timestamp_first_active to date one
train_df.timestamp_first_active = train_df.timestamp_first_active.apply(lambda x:pd.to_datetime(str(x)))


# In[274]:


import seaborn as sns


# In[275]:


train_df.age.apply(lambda x : x>120).sum()


# In[276]:


sns.barplot(data=train_df,x='age',y='country_destination')


# In[277]:


train_df.first_browser.value_counts().sort_values(ascending=False)/train_df.first_browser.count()


# In[278]:


sns.countplot(data=train_df,x='signup_method')


# In[279]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20, 6))
sns.countplot(data=train_df,x='affiliate_provider')


# In[280]:


sns.countplot(data=train_df,x='language')


# In[281]:


plt.figure(figsize=(16, 6))
sns.countplot(data=train_df,x='affiliate_channel')


# In[282]:


plt.figure(figsize=(16, 6))
sns.countplot(data=train_df,x='first_device_type')


# In[283]:


sns.countplot(data=train_df,x='country_destination')


# In[287]:


# category_columns = list(train_df.columns[train_df.dtypes == np.object].values)


# In[288]:


##Han


# ## Handling missing values

# In[289]:


train_df.columns


# In[290]:


# train_df['gender']= train_df.gender.replace('-unknown-',np.nan)


# In[291]:


train_df.gender.value_counts()


# In[292]:


train_df.isnull().sum().sort_values(ascending=False)


# In[293]:


# train_df['first_browser']= train_df.first_browser.replace('-unknown-',np.nan)


# In[294]:


#handling Age column
train_df.loc[train_df.age>120,'age'] = np.nan
train_df.age.fillna(train_df.age.mean(),inplace=True)


# In[295]:


train_df.isnull().sum().sort_values(ascending=False)


# In[296]:


train_df.age.isnull().sum()


# In[297]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ## Session data exploration : 
# 

# In[298]:


session_df.isnull().sum()


# In[299]:


session_df.head()


# In[300]:


session_df.info()


# In[301]:


total_seconds = session_df.groupby('user_id')['secs_elapsed'].sum()


# In[302]:


average_seconds = session_df.groupby('user_id')['secs_elapsed'].mean()


# In[303]:


total_sessions = session_df.groupby('user_id')['secs_elapsed'].count()


# In[304]:


unique_sessions = session_df.groupby('user_id')['secs_elapsed'].nunique()


# In[305]:


total_seconds.head(),total_sessions.head(),average_seconds.head(),unique_sessions.head()


# In[306]:


num_short_sessions = session_df[session_df['secs_elapsed'] <= 300].groupby('user_id')['secs_elapsed'].count()
num_long_sessions = session_df[session_df['secs_elapsed'] >= 2000].groupby('user_id')['secs_elapsed'].count()


# In[307]:


num_long_sessions.head(),num_short_sessions.head()


# In[308]:


num_devices = session_df.groupby('user_id')['device_type'].nunique()


# In[309]:


def session_id_features(df):
    df['total_seconds'] = df['id'].apply(lambda x:total_seconds[x] if x in total_seconds else 0)
    df['average_seconds'] = df['id'].apply(lambda x:average_seconds[x] if x in average_seconds else 0)
    df['total_sessions'] = df['id'].apply(lambda x:total_sessions[x] if x in  total_sessions else 0)
    df['unique_sessions'] = df['id'].apply(lambda x:unique_sessions[x] if x in  unique_sessions else 0)
    df['num_short_sessions'] = df['id'].apply(lambda x:num_short_sessions[x] if x in num_short_sessions else 0)
    df['num_long_sessions'] = df['id'].apply(lambda x:num_long_sessions[x] if x in num_long_sessions else 0)
    df['num_devices'] = df['id'].apply(lambda x:num_devices[x] if x in num_devices else 0)
    return df


# In[310]:


train_df.shape


# In[311]:


train_df = session_id_features(train_df)


# In[312]:


train_df[train_df.average_seconds.isnull()]
train_df.average_seconds.fillna(0,inplace=True)


# In[313]:


train_df.isnull().sum().sort_values(ascending=False)


# In[314]:


def language(df):
    df['language'] = df['language'].apply(lambda x:'foreign' if x!='en' else x)
    return df


# In[315]:


def affiliate_provider(df):
    df['affiliate_provider'] = df['affiliate_provider'].apply(lambda x:'rest' if x not in 
                                                              ['direct','google','other'] else x)
    return df


# In[316]:


def browser(df):
    df['first_browser'] = df['first_browser'].apply(lambda x: "Mobile_Safari" if x=='Mobile Safari' else x)
    major_browser = ['Chrome','Safari','Firefox','IE','Mobile_Safari']
    df['first_browser'] = df['first_browser'].apply(lambda x : 'Other' if x not in major_browser else x)
    return df


# In[317]:


def classify_device(x):
    if x.find('Desktop') != -1:
        return  "Desktop"
    elif x.find("Tablet") != -1 or x.find('iPad') != -1:
        return "Tablet"
    elif x.find('Phone') != -1:
        return  "Phone"
    else:
        return "Unknown"


# In[318]:


train_df = language(train_df)
train_df = affiliate_provider(train_df)
train_df = browser(train_df)


# In[319]:


train_df.shape


# In[320]:


train_df.drop(['date_first_booking','date_account_created','timestamp_first_active'],axis=1,inplace=True)
train_df = pd.get_dummies(train_df,columns=category_columns,drop_first=True)


# In[321]:


X_train,X_test,y_train,y_test = train_test_split(train_df.drop(['country_destination','id'],axis=1),train_df['country_destination'],test_size=0.3,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[322]:


lg = LogisticRegression()
lg.fit(X_train,y_train)
lg.score(X_test,y_test)


# In[323]:


lg.score(X_train,y_train)


# In[324]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()


# In[ ]:


gbc.fit(X_train,y_train)


# In[ ]:


gbc.score(X_train,y_train)


# In[ ]:




