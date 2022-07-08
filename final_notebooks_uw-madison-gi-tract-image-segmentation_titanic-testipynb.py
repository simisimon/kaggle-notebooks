#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# In[ ]:


df_train=pd.read_csv("../input/titanic/train.csv")
df_test=pd.read_csv("../input/titanic/test.csv")


# In[ ]:


df_train.head()


# In[ ]:


df_train["Sex_01"]=np.where(df_train["Sex"]=="male", 1, 0)
df_test["Sex_01"]=np.where(df_test["Sex"]=="male", 1, 0)

df_train["Age"]=df_train["Age"].fillna(df_train["Age"].mean())
df_test["Age"]=df_test["Age"].fillna(df_test["Age"].mean())
df_train["Fare"]=df_train["Fare"].fillna(df_train["Fare"].mean())
df_test["Fare"]=df_test["Fare"].fillna(df_test["Fare"].mean())


# In[ ]:


X=df_train[["Sex_01", "Pclass", "Fare", "Age"]]
Y=df_train["Survived"]


# In[ ]:


model=LogisticRegression().fit(X,Y)


# In[ ]:


model.coef_


# In[ ]:


X_test=df_test[["Sex_01", "Pclass", "Fare", "Age"]]
pred=model.predict(X_test)


# In[ ]:


pred


# In[ ]:


df_test["pred"]=pred
df_test


# In[ ]:




