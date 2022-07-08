#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# # 1.read the data

# In[ ]:


data = pd.read_csv('../input/classified-data/Classified Data')


# In[ ]:


data.sample(2)


# # 2.preprocessing

# In[ ]:


data.info()


# In[ ]:


data.isna().sum()


# In[ ]:


from sklearn.preprocessing import StandardScaler

st_scaler = StandardScaler()
st_scaler.fit(data.drop('TARGET CLASS', axis = 1))
features = st_scaler.fit_transform(data.drop('TARGET CLASS', axis = 1))


# In[ ]:


features


# In[ ]:


features_df = pd.DataFrame(features, columns = data.columns[:-1])
features_df.head()


# In[ ]:


features_df.shape


# # 3.Corr func

# In[ ]:


corr = data.corr()
fig = px.imshow(corr, text_auto=True)
fig.show()


# # 4.Select Model

# In[ ]:


X = features
y = data['TARGET CLASS']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[ ]:


knn_model = KNeighborsClassifier(n_neighbors = 2)
knn_model.fit(x_train, y_train)
KNeighborsClassifier(n_neighbors=2)
y_pred = knn_model.predict(x_test)


# In[ ]:


print(confusion_matrix(y_test, y_pred))


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


model_acc = knn_model.score(x_test, y_test)
print('KNN Model Accuracy:', model_acc * 100, '%')


# In[ ]:


error_rate = []

for i in range(1, 40):
    knn_model = KNeighborsClassifier(n_neighbors = i)
    knn_model.fit(x_train, y_train)
    knn_y_pred = knn_model.predict(x_test)
    error_rate.append(np.mean(knn_y_pred != y_test))


# In[ ]:


knn_model = KNeighborsClassifier(n_neighbors = 1)

knn_model.fit(x_train, y_train)
k_pred = knn_model.predict(x_test)

print('WITH K = 1')
print('\n')
print(confusion_matrix(y_test, k_pred))
print('\n')
print(classification_report(y_test, k_pred))


# In[ ]:


one_k_train_accuracy = knn_model.score(x_train, y_train)
print('K = 1 Training Accuracy:', one_k_train_accuracy * 100, '%')

one_k_test_accuracy = knn_model.score(x_test, y_test)
print('K = 1 Test Accuracy:', one_k_test_accuracy * 100, '%')


# In[ ]:


knn_model = KNeighborsClassifier(n_neighbors = 24)

knn_model.fit(x_train, y_train)
k_pred = knn_model.predict(x_test)

print('WITH K = 24')
print('\n')
print(confusion_matrix(y_test, k_pred))
print('\n')
print(classification_report(y_test, k_pred))


# In[ ]:


twentyfour_k_train_accuracy = knn_model.score(x_train, y_train)
print('K = 10 Training Accuracy:', twentyfour_k_train_accuracy * 100, '%')

twentyfour_k_test_accuracy = knn_model.score(x_test, y_test)
print('K = 24 Test Accuracy:', twentyfour_k_test_accuracy * 100, '%')

