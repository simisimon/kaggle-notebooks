#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn import pipeline,ensemble,preprocessing, feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
#from sklearn import tree
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.linear_model import SGDClassifier
#from sklearn.linear_model import LogisticRegression


# In[ ]:


train=pd.read_json('../input/train.json')


# In[ ]:


train.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[ ]:


train['cuisine'].value_counts().plot(kind='bar')


# In[ ]:


counters={}
for cuisine in train['cuisine'].unique():
    counters[cuisine]= Counter()
    indices= (train['cuisine']==cuisine)
    for ingredients in train[indices]['ingredients']:
        counters[cuisine].update(ingredients)


# In[ ]:


top10= pd.DataFrame([[items[0] for items in counters[cuisine].most_common(10)] for cuisine in counters],
                   index=[cuisine for cuisine in counters],
                   columns=['top{}'.format(i) for i in range(1,11)])
top10


# In[ ]:


train.ingredients=train.ingredients.apply("".join)


# In[ ]:


train.head()


# In[ ]:


train.ingredients.str.contains('garlic cloves')


# In[ ]:


indices=train.ingredients.str.contains('garlic cloves')
train[indices]['cuisine'].value_counts().plot(kind='bar',
                                             title= 'Dientes de ajo hallados por cocina')


# In[ ]:


unique= np.unique(top10.values.ravel())
unique
fig, axes= plt.subplots(8,8, figsize=(20,20))
for ingredient, ax_index in zip(unique, range(64)):
    indices=train.ingredients.str.contains(ingredient)
    relative_freq= (train[indices]['cuisine'].value_counts()/train['cuisine'].value_counts())
    relative_freq.plot(kind='bar', ax=axes.ravel()[ax_index], fontsize=8, title=ingredient)


# In[ ]:


clf=pipeline.Pipeline([
        ('tfidf_vectorizer', feature_extraction.text.TfidfVectorizer(lowercase=True)),
        ('clf', LinearSVC(random_state=0))
    ])


# In[ ]:


# step 1: testing
X_train,X_test,y_train,y_test=train_test_split(train.ingredients,train.cuisine, test_size=0.2)


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


y_pred=clf.predict(X_test)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


# step 2: real training
test=pd.read_json('../input/test.json')


# In[ ]:


test.ingredients=test.ingredients.apply(' '.join)


# In[ ]:


test.head()


# In[ ]:


clf.fit(train.ingredients,train.cuisine)


# In[ ]:


pred=clf.predict(test.ingredients)


# In[ ]:


df=pd.DataFrame({'id':test.id,'cuisine':pred})


# In[ ]:


df.to_csv('LinearSVC.csv', columns=['id','cuisine'],index=False)

