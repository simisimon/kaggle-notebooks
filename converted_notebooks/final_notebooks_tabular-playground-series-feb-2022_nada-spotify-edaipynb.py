#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install pycaret


# In[ ]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import KNNImputer


# In[ ]:


#from pycaret.classification import *


# In[ ]:


df = pd.read_csv("../input/spotify-recommendation/data.csv")


# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:


#sns.pairplot(df,hue="liked")


# In[ ]:


y = df["liked"]
X = df.drop(["liked"],axis=1)


# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='liblinear')
cross_val_score(clf,X,y,scoring="accuracy",cv=7).mean()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


# In[ ]:


accuracy_score(y_test,predictions)

