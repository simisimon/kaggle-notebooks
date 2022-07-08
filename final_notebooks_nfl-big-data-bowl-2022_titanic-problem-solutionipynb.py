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


data=pd.read_csv("/kaggle/input/titanic/train.csv")
data


# In[ ]:


data1=pd.read_csv("/kaggle/input/titanic/test.csv")
data1


# In[ ]:


data1.isnull().sum()


# In[ ]:


data.isnull().sum()


# In[ ]:


data1.drop(['Cabin','Name'],axis=1, inplace=True)
data1


# In[ ]:


data.drop(['Cabin','Name'],axis=1, inplace=True)
data


# In[ ]:


data1['Age'] = data1['Age'].fillna(data1['Age'].mode()[0])
data1['Fare'] = data1['Fare'].fillna(data1['Fare'].mean())
data1.head()


# In[ ]:


data['Age'] = data['Age'].fillna(data['Age'].mode()[0])
data.head()


# In[ ]:


data1.isnull().sum()


# In[ ]:


data.isnull().sum()


# In[ ]:


df6=pd.get_dummies(data1['Sex'])
df6


# In[ ]:


df=pd.get_dummies(data['Sex'])
df


# In[ ]:


data1=data1.drop(labels=['Sex','Ticket'], axis=1 )
data1


# In[ ]:


data=data.drop(labels=['Sex','Ticket'], axis=1 )
data


# In[ ]:


horizontal_concat1 = pd.concat([data1, df6], axis=1)
horizontal_concat1


# In[ ]:


horizontal_concat = pd.concat([data, df], axis=1)
horizontal_concat


# In[ ]:


df7=horizontal_concat1.dropna(axis=0)
df7.head()
df7.shape


# In[ ]:


df1=horizontal_concat.dropna(axis=0)
df1.head()


# In[ ]:


df1.shape


# In[ ]:


df8=df7.drop('Embarked',axis=1)


# In[ ]:


df2 = df1[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'female', 'male']]


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
df8['embarked']=le1.fit(df7['Embarked']).transform(df7['Embarked'])
df8.head()


# In[ ]:


x_test=df8

pt=data1['PassengerId']
pt


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df2['embarked']=le.fit(df1['Embarked']).transform(df1['Embarked'])
df2.head()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (15,8))
plt.title("Heatmap displaying the correlations between all columns", fontsize = 20)
sns.heatmap(df2.corr(), annot=True, cmap="mako")


# In[ ]:


df3=df2.drop('Survived',axis=1)
y=df2['Survived']
y


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaled1=sc.fit(df8).transform(df8)
df_scaled1 = pd.DataFrame(scaled1, columns=df8.columns)
df_scaled1.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaled=sc.fit(df3).transform(df3)
df_scaled = pd.DataFrame(scaled, columns=df3.columns)
df_scaled.head()


# In[ ]:


x=df_scaled
x


# In[ ]:


from sklearn.svm import SVC
clf=SVC(kernel='linear',random_state=51)
model=clf.fit(x,y)
y_pred1=model.predict(x_test)
y_pred1
x_test.shape


# In[ ]:


from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test,y_pred1))


# In[ ]:


from sklearn.linear_model import LogisticRegression
clf1=LogisticRegression()
model=clf1.fit(x,y)
y_pred=model.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test,y_pred))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier 
clf2=KNeighborsClassifier()
model=clf2.fit(x,y)
y_pred=model.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test,y_pred))


# In[ ]:


data4=pd.DataFrame([pt,y_pred1])
data4


# In[ ]:


k = pd.DataFrame(y_pred1,columns=['Survived'])
result = pd.concat([pt,k],axis=1)
result


# In[ ]:


result.to_csv('submission.csv', index=False)


# In[ ]:




