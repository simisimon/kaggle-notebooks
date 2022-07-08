#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train=pd.read_csv("../input/leaf-classification/train.csv.zip")
test =pd.read_csv("../input/leaf-classification/test.csv.zip")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


species = list(set(train.species))


# In[ ]:


species.sort()


# In[ ]:


map_num = np.arange(len(species))


# In[ ]:


print(len(species))
print(len(map_num))


# In[ ]:


X=train.drop(['id','species'],axis=1).values
y=train.loc[:,['species']].values.reshape(-1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_std,y,test_size=0.3,random_state=0)


# In[ ]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)


# In[ ]:


svc.score(X_test,y_test)


# In[ ]:


species_mapping = {species:map_num for (species,map_num) in zip(species,map_num) }
train.species = train.species.map(species_mapping)


# In[ ]:


import cv2


# In[ ]:


import os


# In[ ]:


train_plot = train.drop('id',axis = 1)
train_plot.head()


# In[ ]:


y = train_plot.loc[:,['species']].values.reshape(-1)


# In[ ]:


y.shape


# In[ ]:


X = train_plot.drop('species',axis=1).values


# In[ ]:


X.shape


# In[ ]:


from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)


# In[ ]:


tsne = manifold.TSNE(n_components=2, learning_rate=50,init='pca', random_state=0)
X_tsne = tsne.fit_transform(X_std)
plt.scatter(X_tsne[:,0],X_tsne[:,1],c=y,cmap='jet')
plt.colorbar()
plt.title('t-SNE')
plt.savefig('t-SNE Leaf')
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
X_tsne_train,X_tsne_test,y_train,y_test = train_test_split(X_tsne,y,test_size=0.3,random_state=0)


# In[ ]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(X_tsne_train,y_train)


# In[ ]:


svc.score(X_tsne_test,y_test)


# In[ ]:


train.head()


# In[ ]:


Y_train=train.iloc[:,1]
X_train=train.iloc[:,2:]


# In[ ]:


X_test=test.iloc[:,1:]


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


#X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
#logreg = linear_model.LogisticRegression(C=1000000)
#logreg.fit(X_train, Y_train)


# In[ ]:


pred=logreg.predict(X_train)


# In[ ]:


knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)


# In[ ]:


Y_predict=knn.predict(X_train)


# In[ ]:


clf = KNeighborsClassifier(3)
clf.fit(X_train, Y_train)
test_predictions = clf.predict_proba(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy=accuracy_score (Y_train, Y_predict)
accuracyy=accuracy_score (Y_train, pred)
print("KN Accuracy is {0:.2f}%".format(accuracy*100))
print("Logistic Accuracy is {0:.2f}%".format(accuracyy*100))


# In[ ]:


output=list(np.unique(Y_predict))
X_test.shape


# In[ ]:


submission = pd.DataFrame(test_predictions, columns=output)
submission.insert(0, 'id', test["id"])


# In[ ]:


submission.to_csv(path_or_buf="submission.csv",header=True)

