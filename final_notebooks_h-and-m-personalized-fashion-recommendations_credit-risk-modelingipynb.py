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


import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay


# In[ ]:


df = pd.read_csv("../input/german-credit-data-with-risk/german_credit_data.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# In[ ]:


print("Null columns:", df.columns[df.isnull().any()].to_list())


# In[ ]:


df["Age"].plot.hist(edgecolor="w")
plt.show()


# In[ ]:


sns.countplot(data=df, x="Risk")
plt.show()


# In the dataset, we have 700 good loans and 300 bad loans.

# In[ ]:


pd.crosstab(df["Risk"], df["Sex"]).plot.bar(rot=0)
plt.xlabel("Loans")
plt.ylabel("Count of good and bad loans")
plt.show()


# In both good and bad loans, male count is more than female count

# In[ ]:


pd.crosstab(df["Risk"], df["Job"]).plot.bar(rot=0)
plt.show()


# In[ ]:


pd.crosstab(df["Risk"], df["Housing"]).plot.bar(rot=0)
plt.show()


# In[ ]:


sns.boxplot(data=df, x="Risk", y="Duration")
plt.ylabel("Duration(in month)")
plt.xlabel("Loan")
plt.show()


# Loans which are more in duration results in default 

# In[ ]:


sns.boxplot(data=df, x="Risk", y="Credit amount")
plt.ylabel("Credit amount(in DM)")
plt.xlabel("Loan")
plt.show()


# Bad loans have high credit amount as compared to good loans and it is also varying more

# In[ ]:


df["Saving accounts"].value_counts()


# In[ ]:


df["Checking account"].value_counts()


# In[ ]:


# Dropping "Unnamed: 0" column, since it is not useful in predicting 
df.drop(columns="Unnamed: 0", inplace=True)


# In[ ]:


# Dropping checking account since it has about 50 %null values and it is difficult to impute
df.drop(columns="Checking account", inplace=True)


# In[ ]:


# Filling null value in saving accounts with "little" as it is more in count out of others
df["Saving accounts"] = df["Saving accounts"].fillna(df["Saving accounts"].mode()[0])


# In[ ]:


df["Purpose"].unique()


# In[ ]:


df.head()


# In[ ]:


df["Risk"] = df["Risk"].map({"good":1,"bad":0})


# In[ ]:


cat_df = pd.get_dummies(df.select_dtypes("object"), drop_first=True)
num_df = df.select_dtypes(exclude="object")
new_df = pd.concat([cat_df,num_df], axis=1)


# In[ ]:


new_df.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(new_df.drop(columns="Risk"), df["Risk"], test_size=0.2, random_state=42)


# In[ ]:


model_xgb = xgb.XGBClassifier(eta=0.02, n_estimators=75, reg_lambda=100, random_state=42)
model_xgb.fit(X_train, y_train)


# In[ ]:


print("Training scores:", model_xgb.score(X_train, y_train))
print("Test scores:", model_xgb.score(X_test, y_test))


# In[ ]:


print("Classification report:\n")
print(classification_report(y_test, model_xgb.predict(X_test)))


# In[ ]:


cm = confusion_matrix(y_test, model_xgb.predict(X_test))
ConfusionMatrixDisplay(cm, display_labels=["bad","good"]).plot()
plt.show()


# In[ ]:


RocCurveDisplay.from_estimator(model_xgb, X_test, y_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.show()


# In[ ]:




