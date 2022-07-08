#!/usr/bin/env python
# coding: utf-8

# **Importing the required libraries**

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# **Storing the dataset into a variable**

# In[ ]:


data1 = pd.read_csv('../input/electric-motor-temperature/measures_v2.csv')
data1


# **Checking the dataset whether it contain any Null values**

# In[ ]:


data1.info()# gives the information about the data 


# In[ ]:


data1.describe() # Gives the statistical information about the data


# **Preprocessing**

# In[ ]:


def preprocess_inputs(df):
    Y = df['pm'].copy()                 #Y stores the pm column as it is the temperature column
    X = df.drop('pm', axis=1).copy()    #X stores all other columns except pm
    scaler = RobustScaler()             # RobustScaler acts as a StandardScaler but it will be eliminating the outliers
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, Y


# In[ ]:


X, Y = preprocess_inputs(data1)  # calling the preprocessing function


# 

# **Train and Test Split**

# In[ ]:


#splitting the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)


# **Model Creation**

# In[ ]:


model =KNeighborsRegressor() #KNeighborsRegressor gives better output
model.fit(X_train, Y_train)  #model fitting


# In[ ]:


#printing the model score
print(model.score(X_test, Y_test))


# **The final score is 0.99 approx**

# 
