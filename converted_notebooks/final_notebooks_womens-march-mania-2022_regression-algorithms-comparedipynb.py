#!/usr/bin/env python
# coding: utf-8

# **This notebook compares the performance of common regression algorithms for the same dataset.
# 1. Decision Trees
# 2. Random Forest
# 3. XGBoost**

# In[ ]:


### importing the required libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[ ]:


## importing the data and selecting the independent and independent variables 

train_data = pd.read_csv('../input/train.csv')
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = train_data.SalePrice
X = train_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])


# In[ ]:


## splitting the data for training and testing and cleaning it using imputation 

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.fit_transform(test_X)


# In[ ]:


## making predictions using the Decision Tree algorithm 

decision_model = DecisionTreeRegressor()  
decision_model.fit(train_X, train_y) 
predicted_decision_trees = decision_model.predict(test_X)
print ("Mean Absolute Error using Decision Tress :", mean_absolute_error(test_y, predicted_decision_trees))


# In[ ]:


## making predictions using the Random Forest algorithm 

forest_model = RandomForestRegressor(n_estimators=100, max_depth=10)
forest_model.fit(train_X, train_y )
predicted_random_forest = forest_model.predict(test_X)
print("Mean Absolute Error using Random Forest:", mean_absolute_error(test_y, predicted_random_forest))


# In[ ]:


## making predictions using the XGBoost algorithm 

xg_model = XGBRegressor(n_estimators=100)
xg_model.fit(train_X, train_y)
predicted_XGBoost = xg_model.predict(test_X)
print("Mean Absolute Error using XGBoost: ", mean_absolute_error(test_y, predicted_XGBoost))


# <strong>As evident from the above metrics, we can conclude that using a combination of Decision Trees (Random Forest) can prove to be very useful in bringing down the errors (increaing accuracy). Also, further improvement in the results can be made using some kind of boosting algorithm.</strong>
