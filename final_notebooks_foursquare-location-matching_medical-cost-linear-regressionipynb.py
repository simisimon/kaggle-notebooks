#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries
# 

# In[ ]:


# importing necassry libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# # Reading the data

# In[ ]:


# loading the data
df = pd.read_csv('../input/insurance/insurance.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# # Exploratory Data Analysis

# In[ ]:


# checking for missing values
df.isnull().sum()


# In[ ]:


# looking for unique values
df['region'].nunique()


# In[ ]:


# comparing values per region
df['region'].value_counts()


# In[ ]:


df['age'].nunique()


# In[ ]:


# most populated ages
df['age'].value_counts().head(5)


# # Data Visualization

# In[ ]:


plt.figure(figsize=(12,8))
sns.pairplot(df, hue='region')


# In[ ]:


plt.figure(figsize=(12,8))
sns.pairplot(df, hue='sex')


# In[ ]:


plt.figure(figsize=(12,8))
sns.pairplot(df, hue='smoker')


# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:


sns.histplot(df['charges'],kde=True)


# In[ ]:


sns.countplot(x= 'smoker', data=df, hue='sex', palette = 'Set1')


# In[ ]:


sns.countplot(x='sex', data=df, palette = 'Set1')


# In[ ]:


sns.countplot(x='smoker', data=df, palette = 'Set1')


# In[ ]:


sns.countplot(x= 'region', data=df, palette = 'Set1')


# In[ ]:


sns.countplot(x= 'region', data=df, hue='sex', palette = 'Set1')


# In[ ]:


sns.countplot(x= 'region', data=df, hue='smoker', palette = 'Set1')


# In[ ]:


sns.boxplot(x="region", y="charges", data=df, palette = 'Set1')


# In[ ]:


# Converting Categorical Data to Numerical Data
region = pd.get_dummies(df['region'],drop_first = False)
df = pd.concat([df,region],axis = 1)
df.info()


# In[ ]:


df.head()


# In[ ]:


# Converting Categorical Data to Numerical Data
smoke = pd.get_dummies(df['smoker'],drop_first = True)
df = pd.concat([df,smoke],axis = 1)


# In[ ]:


df.head()


# In[ ]:


# Converting Categorical Data to Numerical Data
sex = pd.get_dummies(df['sex'],drop_first = True)
df = pd.concat([df,sex],axis = 1)
df.info()


# In[ ]:


df.head()


# In[ ]:


# droping categorical columns
df = df.drop(['sex','smoker','region'], axis = 1)
df.head()


# In[ ]:


plt.figure(figsize=(12,10))
sns.lmplot(x='age',y='charges',data=df)


# # 

# # Train Test Split

# In[ ]:


X = df.drop(['charges'],axis=1)
y = df['charges']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training the Model

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train, y_train)


# ## Model Evaluation

# In[ ]:


print(lm.intercept_)


# In[ ]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# ## Predictions from our Model

# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:


sns.histplot((y_test-predictions),kde=True, bins=50)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import r2_score

print('R2 Score for Linear Regression on test data: {}'.format( np.round(r2_score(y_test, predictions), 3)))
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

