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


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor


# # Data Acquring

# In[ ]:


df=pd.read_csv("../input/motorcycle-dataset/BIKE DETAILS.csv")


# In[ ]:


df


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# # Data Cleaning
# **** Missing values

# In[ ]:


df['ex_showroom_price'].fillna(np.round(df['ex_showroom_price'].mean(),2),inplace=True)


# In[ ]:


np.round(df['ex_showroom_price'],2)


# # DATA PREPROCESSING
# ** EDA

# In[ ]:


top_bike_company = df['name'].value_counts().head(10)
plt.figure(figsize=(10, 8))
sns.barplot(x = top_bike_company, y = top_bike_company.index)
plt.ylabel('Name of bike')
plt.title('Top 10 bike company')
plt.xlabel('Count')


# In[ ]:


sns.distplot(df['selling_price'].value_counts())


# In[ ]:


sns.countplot(x=df['year'].head(10))


# In[ ]:


sns.countplot(x=df['seller_type'], data=df)


# # Handling the Categorical values 

# In[ ]:


df=pd.get_dummies(df,columns=['owner','seller_type'],drop_first=True)
'''
or
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(drop='first')
enc=pd.DataFrame(encoder.fit_transform(df[['owner']]).toarray())
'''


# In[ ]:


def bike_model(model_name, excl_honda_hero=False):
    model_list = []
    if excl_honda_hero:
        for i in df['name']:
            if model_name in i and 'Hero' not in i:
                model_list.append(i)
        return model_list
    else:
        for i in df['name']:
            if model_name in i:
                model_list.append(i)
        return model_list


# In[ ]:


royal_enfield = bike_model('Royal Enfield')
honda = bike_model('Honda',excl_honda_hero=True)
bajaj = bike_model('Bajaj')
yamaha = bike_model('Yamaha')
suzuki = bike_model('Suzuki')
hero = bike_model('Hero')
tvs = bike_model('TVS')
ktm = bike_model('KTM')


# In[ ]:


def brand(i):
    if i in royal_enfield:
        return 'Royal Enfield'
    elif i in honda:
        return 'Honda'
    elif i in bajaj:
        return 'Bajaj'
    elif i in yamaha:
        return 'Yamaha'
    elif i in hero:
        return 'Hero'    
    elif i in tvs:
        return 'TVS'    
    elif i in suzuki:
        return 'Suzuki'  
    elif i in ktm:
        return 'KTM' 
    else:
        return 'Other'


# In[ ]:


df['brand'] = df['name'].apply(lambda x:brand(x))
df.head()


# Name Column is a categorical Variable so we will drop this Column

# In[ ]:


df=df.drop('name',axis='columns')


# In[ ]:


df= pd.get_dummies(df, columns=['brand'], drop_first=True)


# In[ ]:


x=df.drop('selling_price',axis='columns')
y=df['selling_price']


# ## Train Test Split

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# # Data Modeling

# In[ ]:


lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)


# In[ ]:


y_pred_test=lin_reg.predict(x_test)
y_pred_train=lin_reg.predict(x_train)


# In[ ]:


train_res=y_train-y_pred_train
test_res=y_test-y_pred_test


# In[ ]:


train_res


# In[ ]:


test_res


# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[ ]:


mse=mean_squared_error(y_test,y_pred_test)
mse


# In[ ]:


mse1=mean_squared_error(y_train,y_pred_train)
mse1


# In[ ]:


rmse=np.sqrt(mse)
rmse


# In[ ]:


rmse=np.sqrt(mse1)
rmse


# In[ ]:


lin_reg.score(x_train,y_train)


# In[ ]:


lin_reg.score(x_test,y_test)


# In[ ]:


plt.scatter(y_test,test_res)


# In[ ]:


sns.distplot(test_res,kde=True)


# In[ ]:


plt.scatter(y_pred_test,test_res,c='r')
plt.axhline(y=0,color='blue')


# In[ ]:


import statsmodels.formula.api as smf
model1=smf.ols('y~x',data=df).fit()
model1.summary()


# In[ ]:




