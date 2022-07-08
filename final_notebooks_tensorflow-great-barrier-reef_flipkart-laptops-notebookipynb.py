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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import Ridge

Atr = pd.read_csv("../input/flipkart-india-laptops-under-75000/Flipkart_India_Laptops_under_75000.csv")
print(Atr.info())
print(Atr.describe())

num_feat = [x for x in Atr.columns if Atr[x].dtype != object]
cat_feat = [x for x in Atr.columns if Atr[x].dtype == object]
bool_feat = ['ASUS','DELL','HP','Lenovo','Other brand','Other OS','Windows 10','Windows 11','Touchscreen']
for i in bool_feat:
    num_feat.remove(i)

Atr.drop(columns = ['Laptop_Name','Processor'],inplace = True)
cat_feat.remove('Laptop_Name')
cat_feat.remove('Processor')
print(num_feat,cat_feat)
print(Atr.head())
sns.distplot(Atr.Price)
plt.show()
for i in cat_feat:
    if i not in ['Price']:
        I = pd.crosstab(Atr[i],Atr['Price'])
        I.div(I.sum(1).astype(float),axis = 0).plot(kind='bar',stacked = True)
        plt.show()

for i in num_feat:
    if i not in ['Popularity_Rank','RAM Capacity (in GB)','SSD (in GB)','HDD (in GB)']:
        Atr[i] = np.sqrt(np.array(Atr[i]).reshape(-1,1))
        sns.distplot(Atr[i])
        plt.show()
for i in ['RAM Capacity (in GB)','SSD (in GB)','HDD (in GB)']:
    sns.countplot(Atr[i])
    plt.show()
for i in bool_feat:
    sns.countplot(Atr[i])
    plt.show()

sns.heatmap(Atr.corr().abs(), annot = False)
plt.show()


# In[ ]:


y = Atr.loc[:,'Price']
X = Atr.drop(columns = 'Price')
model = Ridge()
range_ = np.linspace(0.001,0.02,20)
grid = [{'alpha': range_}]
GSCV = GridSearchCV(estimator = model,param_grid = grid,cv = KFold(n_splits = 10))
l = []
for i in range(1,50):
    X_train,X_test,Y_train,Y_test = train_test_split(X,y,random_state = i)
    GSCV.fit(X_train,Y_train)
    score = GSCV.score(X_test,Y_test)
    l.append(score)
print(sum(l)/len(l))

