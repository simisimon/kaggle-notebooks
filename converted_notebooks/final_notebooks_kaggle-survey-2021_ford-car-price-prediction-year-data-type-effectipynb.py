#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv("/kaggle/input/ford-car-price-prediction/ford.csv")
data.head()


# In[ ]:


print(data.isnull().sum())#No missing value
data.info()


# In[ ]:


#converting year as object type'
data['year'] = data['year'].astype('object')
data.info()


# In[ ]:


#lets seperate Numerical and categorical data
numeric_data = data.select_dtypes('number')
categorical_data = data.select_dtypes('object')


# In[ ]:


print(numeric_data.columns)
print(categorical_data.columns)


# In[ ]:


numeric_data.describe()


# Value of standard deviation for mileage is very high, so we need to check on that

# In[ ]:


plt.boxplot(data["mileage"], vert=False)
plt.xlabel("Mileage")
plt.title("Boxplot: Mileage");


# In[ ]:


##Removing outliers from mileage
low, high = data["mileage"].quantile([0.05,0.90])
mask_ma = data["mileage"].between(low,high)
data = data[mask_ma]


# In[ ]:


print(low ,high)


# In[ ]:


plt.boxplot(x  = data['mileage'] , vert = False)
plt.xlabel("Mileage")
plt.title("Boxplot : Mileage");


# In[ ]:


data.info()# nearly 2.7K records are removed


# In[ ]:


data.describe()


# In[ ]:


corr = data.select_dtypes("number").drop(columns="price").corr() 


# In[ ]:


sns.heatmap(corr , cmap = 'YlGnBu',annot = True)


# after looking this visualisation or above "corr" table we can say that their is no multicolinearity in our feature columns,

# In[ ]:


for i in categorical_data.columns:
    print(i ,"has",data[i].nunique(), "unique_value")


# In[ ]:


get_ipython().system('pip install dython')


# In[ ]:


from dython import nominal 
nominal.associations(data , figsize = (10,10) , mark_columns = True)


# In[ ]:


data.drop(['transmission' , 'fuelType'] ,axis = 1 ,inplace = True)


# In[ ]:


data.info()


# In[ ]:


feature = ['model' , 'year' , 'mileage','tax','mpg' ,'engineSize']
target = ['price']
X = data[feature]
y= data[target]


# In[ ]:


X = pd.get_dummies(X)


# In[ ]:


X.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y ,test_size =0.2 ,random_state = 2)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# **Model Development**

# In[ ]:


v = y_train


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100)
rf.fit(X_train, y_train.values.ravel())


# In[ ]:


# Make predictions for the test set
y_pred_test = rf.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred_test)


# *****Taking year as int******

# In[ ]:


data_int_year = data.copy()
data_int_year['year'] = data_int_year['year'].astype('int')
data_int_year.info()


# In[ ]:


feature_year_int = ['model' , 'year' , 'mileage','tax','mpg' ,'engineSize']
target_year_int = ['price']
X_year_int = data_int_year[feature_year_int]
y_year_int = data_int_year[target_year_int]


# In[ ]:


X_year_int = pd.get_dummies(X_year_int)


# In[ ]:


X_year_int_train , X_year_int_test , y_year_int_train ,y_year_int_test = train_test_split(X_year_int , y_year_int , test_size = 0.25)


# In[ ]:


X_year_int_train = scaler.fit_transform(X_year_int_train)
X_year_int_test = scaler.fit_transform(X_year_int_test)


# In[ ]:


rf2 = RandomForestRegressor(n_estimators = 100)
rf2.fit(X_year_int_train, y_year_int_train.values.ravel())


# In[ ]:


y_year_int_pred = rf2.predict(X_year_int_test)


# In[ ]:


r2_score(y_year_int_test , y_year_int_pred)


# In[ ]:




