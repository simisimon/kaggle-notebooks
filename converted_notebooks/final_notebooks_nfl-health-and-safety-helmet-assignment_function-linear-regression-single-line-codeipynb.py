#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Create custom Function for linear regression
# The Function will return prediction in a single line code
import pandas as pd
x_train1 = pd.read_csv('../input/regression-datasets/X_train1_reg.csv')
x_train2 = pd.read_csv('../input/regression-datasets/X_train2_reg.csv')
y_train1 = pd.read_csv('../input/regression-datasets/y_train1_reg.csv')
y_train2 = pd.read_csv('../input/regression-datasets/y_train2_reg.csv')

x_train1.shape,x_train2.shape,y_train1.shape,y_train2.shape


# In[ ]:


# This is our function
def lin_reg(x_train,y_train,x_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    lm=LinearRegression()
    cv_mae=-cross_val_score(lm,
                        x_train,y_train,
                        cv=10,
                        scoring='neg_root_mean_squared_error')
    cv_mae.mean()
    cv_mae.std()
    lm.fit(x_train,y_train)
    test_pred=lm.predict(x_test)
    return(test_pred)


# In[ ]:


# Test the function 
test_pred = lin_reg(x_train1,y_train1,x_train2) # Single Line code
test_pred[0:5]


# In[ ]:


# Test Performance
from sklearn.metrics import mean_squared_error
mean_squared_error(test_pred,y_train2)

