#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('max_columns', None)

from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


# In[ ]:





# In[ ]:


train0 = pd.read_csv('../input/titanic/train.csv')
test0 = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train0


# # Cleaning

# In[ ]:


target = train0['Survived']
train1 = train0.drop('Survived', axis=1)
test1 = test0.copy()


# First we need to remove unnessery columns such as PassengerId, Name, Ticket, Cabin, Embarked
# and get numeric values sex column

# In[ ]:


train2 = train1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
test2 = test1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)


# Now we check Nan values

# In[ ]:


train2.isna().sum(), test2.isna().sum()


# In[ ]:


knn = KNNImputer()
test3 = test2.copy()
train3 = train2.copy()

train4 = pd.get_dummies(train3)
test4 = pd.get_dummies(test3)

transformed_train = knn.fit_transform(train4)
transformed_test = knn.transform(test4)

train5 = pd.DataFrame(transformed_train, columns=train4.columns, index=train4.index)
test5 = pd.DataFrame(transformed_test, columns=test4.columns, index=test4.index)


# In[ ]:


train5.isna().sum(), test5.isna().sum()


# # Feature engineering
# We are going to use unsupervised learning to create new feature namely KMeans

# In[ ]:


train6 = train5.copy()
test6 = test5.copy()

train6['Sex'] = train6['Sex_male']
test6['Sex'] = test6['Sex_male']

train6 = train6.drop(['Sex_male', 'Sex_female'], axis=1)
test6 = test6.drop(['Sex_male', 'Sex_female'], axis=1)


# We are checking optimal ammount of clusters 

# In[ ]:


n = 10
sse = []
for i in range(1,n):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(train6[['Age', 'Fare']])
    sse.append(kmeans.inertia_)
    

plt.style.use("fivethirtyeight")
plt.plot(range(1, n), sse)
plt.xticks(range(1, n))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=8, random_state=0)
train_kmeans = kmeans.fit(train6[['Age', 'Fare']])
train6['clusters'] = train_kmeans.labels_

test_kmeans = kmeans.fit(test6[['Age', 'Fare']])
test6['clusters'] = test_kmeans.labels_


# Now lets check cluster distribution 

# In[ ]:


fig, ax = plt.subplots(figsize=(10,7))
for cluster in np.unique(train_kmeans.labels_):
    plt.scatter(train6["Age"][train6["clusters"] == cluster], train6["Fare"][train6["clusters"] == cluster], 
             s=100, edgecolor='green',linestyle='--')


# In[ ]:


train_final = train6.drop(['Fare'], axis=1)
test_final = test6.drop(['Fare'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(train_final, target, test_size=0.2, random_state=42)


# In[ ]:


train_final


# In[ ]:


test_final


# In[ ]:


models = {
    "svc": SVC(),
    "gnb": GaussianNB(),
    "dtc": DecisionTreeClassifier(),
    "knc": KNeighborsClassifier(),
    "lr": LogisticRegression(max_iter=200),
    "lda": LinearDiscriminantAnalysis(),
    "rfc": RandomForestClassifier(),
}


# In[ ]:


for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    result = accuracy_score(y_test, pred)
    print(f'{name}: {result}')


# In[ ]:


model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

pred = model.predict(test_final)
submission = pd.concat([test0['PassengerId'], pd.DataFrame(pred, columns=['Survived'])], axis=1)
submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




