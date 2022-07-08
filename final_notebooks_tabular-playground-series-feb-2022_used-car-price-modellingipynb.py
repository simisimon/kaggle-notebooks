#!/usr/bin/env python
# coding: utf-8

# ## Import Library

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


# ## Import Data

# In[ ]:


df1 = pd.read_csv("../input/used-car-dataset-ford-and-mercedes/audi.csv")
df1["brand"] = "Audi"
df2 = pd.read_csv("../input/used-car-dataset-ford-and-mercedes/bmw.csv")
df2["brand"] = "BMW"
df3 = pd.read_csv("../input/used-car-dataset-ford-and-mercedes/ford.csv")
df3["brand"] = "Ford"
df4 = pd.read_csv("../input/used-car-dataset-ford-and-mercedes/hyundi.csv")
df4["brand"] = "Hyundi"
df4["tax"] = df4["tax(£)"]*1.07
df5 = pd.read_csv("../input/used-car-dataset-ford-and-mercedes/merc.csv")
df5["brand"] = "Merc"
df6 = pd.read_csv("../input/used-car-dataset-ford-and-mercedes/toyota.csv")
df6["brand"] = "Toyota"
df7 = pd.read_csv("../input/used-car-dataset-ford-and-mercedes/vw.csv")
df7["brand"] = "VW"


# ## Praproses Data

# In[ ]:


#Concate Setiap Tabel
df  = pd.concat([df1, df2, df3, df4, df5, df6, df7])


# In[ ]:


#Create brand_model column
df["brand_model"] = df["brand"] + " " + df["model"]


# In[ ]:


#Delete Some Column
df=df.drop(columns=["tax(£)","brand","model"]).reset_index()
df=df.drop(columns="index")


# In[ ]:


df


# In[ ]:


index_df = df.index
df["id"] = index_df


# In[ ]:


#One Hot Encoding
df_transmission = pd.get_dummies(df.transmission, prefix='transmission')
df_transmission["id1"] = df.index
df = df.drop(["transmission"],axis=1)

df_fuelType = pd.get_dummies(df.fuelType, prefix='fuelType')
df_fuelType["id2"] = df.index
df = df.drop(["fuelType"],axis=1)

df_brand_model = pd.get_dummies(df.brand_model, prefix='brand_model')
df_brand_model["id3"] = df.index
df = df.drop(["brand_model"],axis=1)


# In[ ]:


df=pd.merge(df, df_transmission, left_on='id', right_on='id1', how='left').drop('id1', axis=1)
df=pd.merge(df, df_fuelType, left_on='id', right_on='id2', how='left').drop('id2', axis=1)
df=pd.merge(df, df_brand_model, left_on='id', right_on='id3', how='left').drop('id3', axis=1)


# In[ ]:


df


# In[ ]:


df= df.drop(columns="id")
df


# ## Modeling and Validating

# In[ ]:


X = df.drop(columns="price")
y = df["price"]


# In[ ]:


from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# In[ ]:


#Parameter Tuning
def hyperParameterTuning(X_train, y_train):
    param_tuning = {
        'tree_method': ['gpu_hist'],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }

    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_


# In[ ]:


# hyperParameterTuning(X, y)


# In[ ]:


#Cross-Validation
model = XGBRegressor()
scoring = {'MSE' : make_scorer(mean_squared_error),
          'MAE' : make_scorer(mean_absolute_error)}
kfold = KFold(n_splits=5, random_state=1234, shuffle = True)
results_reg = cross_validate(estimator=model,X=X,
                                          y=y,
                                          cv=kfold,
                                          scoring=scoring)
results_reg


# In[ ]:


b_lin = np.sqrt(results_reg.get('test_MSE'))
print('mean_RMSE : ', "%.15f" % (b_lin.mean()))


# In[ ]:


#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[ ]:


model = model.fit(X_train, y_train)

# Memprediksi nilai y dari X_test
y_predict = model.predict(X_test)


# In[ ]:


# Aktual vs Prediksi 
fig, ax1 = plt.subplots(figsize=(8,6))
plt.scatter(y_predict,y_test,color='blue')
plt.plot(y_test,y_test,color='red')
plt.title('XGboost Regression Result')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# ## Saving Model

# In[ ]:


model.fit(X,y)
model.save_model("model_sklearn.json")


# In[ ]:


#Menyimpan Model
import joblib
joblib.dump(value=model, filename="model.pkl")


# ## Next Action
# 
# * Register Model To Azure ML
# * Deploy Model
# * Backend
# * Frontend
# * Deploy Web
