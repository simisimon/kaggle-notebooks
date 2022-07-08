#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("/kaggle/input/iris/Iris.csv")


# In[ ]:


df.sample(5)


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


np.unique(df["Species"])


# In[ ]:


df.sample(5)


# In[ ]:


def clean_st(species):
    
    if species == 'Iris-setosa':
        species ="setosa"
    elif species == 'Iris-versicolor':
        species = "versicolor"
    elif species == 'Iris-virginica':
        species = "virginica"
    return species


# In[ ]:


df["Species"] = df["Species"].apply(lambda x: clean_st(x))
np.unique(df["Species"]) # to identify all unique values in a column of dataframe or array


# In[ ]:


df.sample(5)


# In[ ]:


df["Species"].value_counts()


# In[ ]:


encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])


# In[ ]:


df.head()


# In[ ]:


df = df[["SepalLengthCm","PetalLengthCm","Species"]]


# In[ ]:


df.sample(5)


# In[ ]:


df.shape


# In[ ]:


X = df.iloc[:,0:2]
y = df.iloc[:,-1]


# In[ ]:


X


# In[ ]:


y


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=21)


# In[ ]:


X_train.shape


# In[ ]:


clf = LogisticRegression(multi_class="multinomial")


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


print(accuracy_score(y_test,y_pred))


# In[ ]:


pd.DataFrame(confusion_matrix(y_test,y_pred))


# In[ ]:


query = np.array([[5.1,1.4],[6.0,5.4]])
clf.predict(query)


# In[ ]:


query = np.array([[5.1,1.4],[2.0,3.4]])
clf.predict_proba(query)


# In[ ]:


from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X.values, y.values, clf, legend=2)

# Adding axes annotations
plt.xlabel('sepal length [cm]')
plt.xlabel('petal length [cm]')
plt.title('Softmax on Iris')

plt.show()


# In[ ]:




