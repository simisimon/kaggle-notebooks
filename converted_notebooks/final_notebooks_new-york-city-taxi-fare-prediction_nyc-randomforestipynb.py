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


def rmse(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / y_true.size)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


np.random.seed(seed=1234)

skiprows = np.random.rand(55 * 10 ** 7) > 0.02
skiprows[0] = False

df_ = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", skiprows=lambda x: skiprows[x])

df_.head()


# In[ ]:


lon_min, lon_max = -75, -72
lat_min, lat_max = 40, 43


# In[ ]:


df = df_.copy()

df = df.drop(columns=["key"])

df["date"] = df["pickup_datetime"].apply(lambda x: x.split()[0])
df["time"] = df["pickup_datetime"].apply(lambda x: x.split()[1])
df = df.drop("pickup_datetime", axis=1)

df["year"] = df["date"].apply(lambda x: int(x.split("-")[0]))
df["month"] = df["date"].apply(lambda x: int(x.split("-")[1]))
df["day"] = df["date"].apply(lambda x: int(x.split("-")[2]))
df = df.drop("date", axis=1)
df["time"] = df["time"].apply(lambda x: int(x[:2]) * 60 + int(x[3:5]))

df["befor_shock"] = ((df["year"] <= 2011) | ((df["year"] <= 2012) & (df["month"] <= 8))).apply(int)

df = df[
    (~df["dropoff_longitude"].isnull()) &
    (~df["dropoff_latitude"].isnull())
]

df = df[
    (lon_min < df["pickup_longitude"]) &
    (df["pickup_longitude"] < lon_max) &
    (lat_min < df["pickup_latitude"]) &
    (df["pickup_latitude"] < lat_max) &

    (lon_min < df["dropoff_longitude"]) &
    (df["dropoff_longitude"] < lon_max) &
    (lat_min < df["dropoff_latitude"]) &
    (df["dropoff_latitude"] < lat_max)
]

df.head()


# In[ ]:


X = np.array(df.drop(
    columns=[
        "fare_amount",
    ]
))
y = np.array(df["fare_amount"])


# In[ ]:


np.random.seed(seed=1234)

train_rows = np.random.rand(y.size) > 0.2

X_train, y_train = X[train_rows, :], y[train_rows]
X_valid, y_valid = X[~train_rows, :], y[~train_rows]


# In[ ]:


model = RandomForestRegressor(max_depth=30, n_estimators=100, n_jobs=-1)

model.fit(X_train, y_train)

y_valid_pred = model.predict(X_valid)

rmse(y_valid, y_valid_pred)


# In[ ]:


test = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

test.head()


# In[ ]:


test["date"] = test["pickup_datetime"].apply(lambda x: x.split()[0])
test["time"] = test["pickup_datetime"].apply(lambda x: x.split()[1])
test = test.drop("pickup_datetime", axis=1)

test["year"] = test["date"].apply(lambda x: int(x.split("-")[0]))
test["month"] = test["date"].apply(lambda x: int(x.split("-")[1]))
test["day"] = test["date"].apply(lambda x: int(x.split("-")[2]))
test = test.drop("date", axis=1)
test["time"] = test["time"].apply(lambda x: int(x[:2]) * 60 + int(x[3:5]))

test["before_shock"] = ((test["year"] <= 2011) | ((test["year"] <= 2012) & (test["month"] <= 8))).apply(int)


test.head()


# In[ ]:


X_test = np.array(test.drop(columns=[
    "key",
]))
y_pred = model.predict(X_test)
test["fare_amount"] = y_pred
submission= test[["key", "fare_amount"]]

submission.to_csv("./submission.csv", index=False)


# In[ ]:




