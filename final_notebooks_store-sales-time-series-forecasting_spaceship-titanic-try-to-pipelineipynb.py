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


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier


# In[ ]:


X_raw= pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
X_raw_test= pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")

y= X_raw["Transported"]
X_raw= X_raw.drop(["Transported"],axis=1)

X_raw_test.head()


# **defining functions for preprocessing data of training and test set**

# In[ ]:


def transform_col_cabin(X_raw):
    #Transformation von "Cabin" Spalte in drei separate Spalten
    
    list_index=[]
    list_deck=[]
    list_num=[]
    list_side=[]
    for i in range(len(X_raw.index.values)):
        splitted= str(X_raw["Cabin"][i]).split("/")
        if splitted == float("NaN"):
            list_deck.append(float("NaN"))
            list_num.append(float("NaN"))
            list_side.append(float("NaN"))
            continue
        elif len(splitted) != 3:
            list_deck.append(float("NaN"))
            list_num.append(float("NaN"))
            list_side.append(float("NaN"))
            continue
        else:
            list_deck.append(splitted[0])
            list_num.append(int(splitted[1]))
            list_side.append(splitted[2])

    new_cabin_columns= pd.DataFrame({"deck" : list_deck, "num" : list_num, "side": list_side},index = X_raw.index.values.tolist() )
    df = pd.concat([X_raw, new_cabin_columns], axis= 1)
    return df


# In[ ]:


def remove_specified_cols(X_raw_transformed):

    cols_drop=["Cabin","Name"]
    X_prepared= X_raw_transformed.drop(cols_drop, axis=1)
    
    return (X_prepared)


# In[ ]:


# differ and define col types
def define_col_types(X):
    cols_cat= [cname for cname in X.columns if X[cname].dtype == "object"]
    cols_num= [cname for cname in X.columns if X[cname].dtype in ["float64"]]
    return (cols_num, cols_cat)


# **define Estimators for pipeline**

# In[ ]:


class Restructure(BaseEstimator, TransformerMixin):
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X= transform_col_cabin(X)
        X_prepared = remove_specified_cols(X)
        return X_prepared


# In[ ]:


class Imputer(BaseEstimator, TransformerMixin):
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        
        numerical_cols, categorical_cols = define_col_types(X)
        
        num_imputer = SimpleImputer()
        cat_imputer = SimpleImputer(strategy="most_frequent")
        
        imputed_X_num = pd.DataFrame(num_imputer.fit_transform(X[numerical_cols]))
        imputed_X_cat = pd.DataFrame(cat_imputer.fit_transform(X[categorical_cols]))
        
        imputed_X_num.columns = X[numerical_cols].columns
        imputed_X_cat.columns = X[categorical_cols].columns
        
        return imputed_X_num.join(imputed_X_cat, how='outer')


# In[ ]:


first_pipeline=Pipeline([("Restructure",Restructure()),
                  ("Imputer", Imputer())])

X_OH_ready=first_pipeline.fit_transform(X_raw)
# I created the following variable, so that I have access from outside the transform function within the next class FeatureEncoder
X_test_OH_ready=first_pipeline.fit_transform(X_raw_test)


# In[ ]:


class FeatureEncoder(BaseEstimator, TransformerMixin):
    
    def fit(self,X,y=None):
        return self
      
    def transform(self,X):
        
    # if I would try:  def transform(self,X,X_test_OH_ready):  and invoke this function afterwards I will get
    # TypeError: transform() missing 1 required positional argument: 'X_test_OH_ready'
    # although I invoked by pipeline.fit_transform(X, X_test_OH_ready)
        
        numerical_cols, categorical_cols = define_col_types(X)
        
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        OH_cat_cols= pd.DataFrame(OH_encoder.fit_transform(X[categorical_cols]))
        OH_cat_cols_test= pd.DataFrame(OH_encoder.transform(X_test_OH_ready[categorical_cols]))
        
        OH_cat_cols.index = X[categorical_cols].index
        OH_cat_cols_test.index = X_test_OH_ready[categorical_cols].index
        num_X = X.drop(categorical_cols, axis=1)
        num_X_test = X_test_OH_ready.drop(categorical_cols, axis=1)
        OH_X = pd.concat([num_X, OH_cat_cols], axis=1)
        OH_X_test = pd.concat([num_X_test, OH_cat_cols_test], axis=1)
        
        return (OH_X, OH_X_test)


# In[ ]:


second_pipeline=Pipeline([("FeatureEncoder",FeatureEncoder())])

OH_X, OH_X_test =second_pipeline.fit_transform(X_OH_ready)


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(OH_X, y, test_size=0.25, random_state=0)


# In[ ]:


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
preds= model.predict(X_valid)
print(accuracy_score(preds, y_valid))


# # Preparing Test Data for Submission

# In[ ]:


preds= model.predict(OH_X_test)

ids = X_raw_test['PassengerId']

df = {
    "PassengerId":[],
    "Transported":[]
}

for _id, pred in zip(ids,preds):
    df["PassengerId"].append(_id)
    df["Transported"].append(pred.astype(bool))

df= pd.DataFrame(df)


df.to_csv("Submission.csv",index=False)


# In[ ]:





# In[ ]:




