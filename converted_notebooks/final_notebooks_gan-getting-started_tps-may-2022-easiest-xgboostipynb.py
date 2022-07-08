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


# # **IMPORTING LIBRARY**

# In[ ]:


#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # **READING THE DATASETS**

# In[ ]:


#reading the train dataset

train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')


# # **EDA**

# In[ ]:


#1st five records of the train set

train.head()


# > **This is a Classification Problem where the target variable is either 0 or 1, so let us count the occurance of each of these**

# In[ ]:


#count of the target variable

train['target'].value_counts()


# In[ ]:


#displaying the number of unique values for each columns

train.nunique().sort_values(ascending=True)


# >  f_27 has the most number of unique values

# In[ ]:


#number of rows and columns

train.shape


# > **There are 900000 rows with 33 columns (with an id column and 31 features)**

# In[ ]:


#dropping the id column

train.drop(columns='id',inplace=True)


# In[ ]:


#datatypes for the columns

train.dtypes


# In[ ]:


#columns of train set

train.columns


# In[ ]:


#displaying info about the train set

train.info()


# In[ ]:


#statistical details of the train set

train.describe()


# > **There are no missing values in our train set**

# In[ ]:


#no missing values

train.isnull().sum()


# # **DATA VISUALIZATION**

# **PROFILE REPORT**

# In[ ]:


#importing library Profile Report from Pandas Profiling library

from pandas_profiling import ProfileReport


# In[ ]:


#Performing Profile Report on the dataset to get the entire statistical report of the dataset in a detailed manner

ProfileReport(train)


# **CORRELATION**

# In[ ]:


#displaying the correlation matrix

train.corr()


# > Dispalying the train set correlations in the form of visualization

# **HEATMAP**

# In[ ]:


#displaying the correlation in our train set

fig,ax = plt.subplots(figsize=(30,30))
sns.heatmap(train.corr(),cmap='RdYlBu',cbar=False, annot=True)
plt.show()


# **PIE CHART**

# In[ ]:


#pie chart

import plotly.express as px

px.pie(train,names='target',title='Target')


#    **COUNT PLOT**

# In[ ]:


#count plot

plt.figure(figsize=(10,5))
sns.countplot(x=train['target'],data=train)


# **PAIR PLOT**

# # **FEATURE SELECTION**

# In[ ]:


train.nunique()


# > **since we have a large number of duplicates for the f_27 column,hence it is better to drop that column for our model building**

# In[ ]:


#dropping f_27 column

train.drop(columns='f_27',inplace=True)


# In[ ]:


train.shape


# In[ ]:


#splitting the features and label (target variable)

x = train.drop(columns='target')
y = train['target']


# In[ ]:


#features

x


# In[ ]:


#label 

y


# # **STANDARDIZATION**

# **Scaling the values (such that the mean of the values becomes 0 and standard deviation become 1) so that all the values are in the same range so that the model can understand and build the relations better in a lower scale**

# In[ ]:


#Min Max Scaling - importing libraries

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


#model building

scaler = MinMaxScaler()


# In[ ]:


#model fitting

arr = scaler.fit_transform(x)


# In[ ]:


#displaying the scaled values

x_scaled = pd.DataFrame(arr)
x_scaled


# In[ ]:


#statistical details of the scaled dataset

x_scaled.describe()


# # **VIF**

# In[ ]:


#Importing VIF from statsmodel library

from statsmodels.stats.outliers_influence import variance_inflation_factor   #importing Variance Inflation Factor


# In[ ]:


#displaying the array of values

arr


# In[ ]:


#number of rows and columns of the array

arr.shape


# In[ ]:


# VIF on the array values

[variance_inflation_factor(arr,i) for i in range(arr.shape[1])]


# In[ ]:


#Creating a dataframe to store the VIF values of the features

df2 = pd.DataFrame()
df2['vif'] = [variance_inflation_factor(arr,i) for i in range(arr.shape[1])] 
df2


# In[ ]:


#Displaying the VIF values for each feature in the  dataset

df2['features'] = x.columns
df2


# # **TRAIN TEST SPLIT**

# **IMPORT LIBRARY**

# In[ ]:


#import library from sklearn 

from sklearn.model_selection import train_test_split


# **TRAIN TEST SPLIT**

# In[ ]:


#train test split

x_train, x_test, y_train, y_test =  train_test_split(x_scaled,y,test_size=0.20,random_state=101)


# # **LOGISTIC REGRESSION**

# **IMPORTING LIIBRARY**

# In[ ]:


#import library

from sklearn.linear_model import LogisticRegression


# **MODEL BUILDING**

# In[ ]:


#model building

log = LogisticRegression(random_state=42)


# **MODEL FITTING**

# In[ ]:


#model fitting

log.fit(x_train,y_train)


# **MODEL PREDICTION**

# In[ ]:


#model prediction

log.predict(x_test)


# In[ ]:


#model prediction on test dataset

log.predict(test)


# **ACCURACY SCORE**

# In[ ]:


#accuracy score

log.score(x_test,y_test)


# > **Since our model's accuracy is vonly 61%, let us now use XgBoost classifier to increase our model accuracy**

# # **XGBOOST**

# > Using XgBoost Classifier- importing the library from sklearn library

# **IMPORTING LIBRARIES**

# In[ ]:


#importing from sklearn libraries

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score


# **MODEL BUILDING**

# In[ ]:


#model building

model = xgb.XGBClassifier(objective='binary:logistic')


# **MODEL FITTING**

# In[ ]:


#model fitting

model.fit(x_train, y_train)


# **MODEL ACCURACY**

# In[ ]:


# model training accuracy

y_pred = model.predict(x_train)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_train,predictions)
accuracy


# In[ ]:


# model test accuracy

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test,predictions)
accuracy


# > **Let us perform Hyper Parameter tuning to increase our model accuracy**

# **HYPER PARAMTER TUNING**

# In[ ]:


#hyper parameter tuning

from sklearn.model_selection import GridSearchCV


# In[ ]:


#param

param_grid={
   
    ' learning_rate':[1,0.5,0.1,0.01,0.001],
    'max_depth': [3,5,10,20],
    'n_estimators':[10,50,100,200]
    
}


# In[ ]:


#model building

grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),param_grid, verbose=3)


# In[ ]:


#model fitting

grid.fit(x_train,y_train)


# **BEST PARAMETERS**

# In[ ]:


# To  find the parameters that give maximum accuracy
grid.best_params_


# > **Out hyper parameter tuning has given the best params - {' learning_rate': 1, 'max_depth': 3, 'n_estimators': 10}. Hence, retraining our model with the best params**

# **MODEL RETRAINING**

# In[ ]:


# Create new model using the same parameters

new_model=XGBClassifier(learning_rate= 1, max_depth= 3, n_estimators= 10)
new_model.fit(x_train, y_train)


# **NEW MODEL ACCURACY**

# In[ ]:


#new model prediction

y_pred_new = new_model.predict(x_test)
predictions_new = [round(value) for value in y_pred_new]


# In[ ]:


#new model accuracy

accuracy_new = accuracy_score(y_test,predictions_new)
accuracy_new


# > **Our XgBoost model accuracy has now successfully been increased to 69% *(from 61% accuracy using Logistic regression model)***

# **PLEASE UPVOTE IF YOU FOUND THE CONTENT HELPFUL :)**
