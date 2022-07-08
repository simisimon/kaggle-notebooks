#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# load the dataset
train_df=pd.read_csv('../input/bbc-fulltext-and-category/bbc-text.csv')


# In[ ]:


train_df.head(10)


# In[ ]:


train_df.shape


# In[ ]:


train_df['category'].unique()


# In[ ]:


train_df['category'].value_counts()


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(train_df['category'],palette="mako")


# In[ ]:


# sns.set(rc={'figure.figsize':(10,10)})
# sns.countplot(train_df['category'])


# In[ ]:


train_df.isnull().sum()


# In[ ]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[ ]:


ps = PorterStemmer()
wordnet = WordNetLemmatizer()
corpus = []


# In[ ]:


train_df['text']


# In[ ]:


stopwords.words('english')


# In[ ]:


for i in range(len(train_df['text'])):
    review = re.sub('[^a-z A-Z]',' ',train_df['text'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]    
    review = ' '.join(review)
    corpus.append(review)


# In[ ]:


corpus


# In[ ]:


for i in range(len(train_df['text'])):
    each_text = re.sub('[^a-z A-Z]',' ',train_df['text'][i])
    each_text = each_text.lower()
    words_list = each_text.split()
    words_list = [wordnet.lemmatize(word) for word in words_list if word not in set(stopwords.words('english'))]     
    train_df['text'][i] = ' '.join(words_list)


# In[ ]:


train_df['text']


# **BOW**

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
X


# **TFIDF**

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf = TfidfVectorizer()
X = Tfidf.fit_transform(corpus).toarray()
X


# In[ ]:




