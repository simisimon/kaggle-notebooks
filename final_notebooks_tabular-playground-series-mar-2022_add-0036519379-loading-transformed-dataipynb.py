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


# # Loading the transformed and partially preprocessed data

# > Dodati output iz druge biljeznice kao novi input (pod Add data)

# In[ ]:


df = pd.read_csv("../input/add-0036519379-data-transformation/df_transformed.csv") #čitanje podataka
print(df.shape)


# # XGB model

# Treniranje modela. Ovdje je proizvoljno odabran XGB.

# In[ ]:


X_train = df[df.columns[~df.columns.isin(['NEXT_WINNER'])]]
Y_train = df['NEXT_WINNER']


# In[ ]:


from sklearn.utils import shuffle
X_train, Y_train = shuffle(X_train, Y_train)


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


# In[ ]:


cv = TimeSeriesSplit(n_splits=5)
xgbr = XGBClassifier()
#xgbr.fit(X_train, Y_train)
XGB_accuracies = cross_val_score(estimator = xgbr, X = X_train, y = Y_train, cv = cv)
tupl = ('XGBClassifier', XGB_accuracies[0], XGB_accuracies[1], XGB_accuracies[2], XGB_accuracies[3], XGB_accuracies[4], XGB_accuracies.mean())
tupl

