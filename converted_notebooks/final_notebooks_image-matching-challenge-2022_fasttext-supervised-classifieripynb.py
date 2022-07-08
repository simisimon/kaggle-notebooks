#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import numpy as np
import pandas as pd
import os
import re
import fasttext
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import csv


# In[ ]:


src = '../input/feedback-prize-effectiveness/'


# In[ ]:


train = pd.read_csv(src + 'train.csv')
train.head(10)


# In[ ]:


test = pd.read_csv(src + 'test.csv')
test.head()


# # NLP Preprocessing

# In[ ]:


from gensim.utils import simple_preprocess

dataset = pd.read_csv(src + 'train.csv')[['discourse_text', 'discourse_effectiveness']].rename(columns = {'discourse_text': 'text', 'discourse_effectiveness': 'eff'})
test = pd.read_csv(src + 'test.csv')[['discourse_text']].rename(columns = {'discourse_text': 'text'})

# NLP Preprocess
dataset.iloc[:, 0] = dataset.iloc[:, 0].apply(lambda x: ' '.join(simple_preprocess(x)))
test.iloc[:, 0] = test.iloc[:, 0].apply(lambda x: ' '.join(simple_preprocess(x)))

# Prefixing each row of the category column with '__label__'
dataset.iloc[:, 1] = dataset.iloc[:, 1].apply(lambda x: '__label__' + x)


# In[ ]:


dataset[['eff', 'text']].to_csv('train.txt',
                                 index = False,
                                 sep = ' ',
                                 header = None,
                                 quoting = csv.QUOTE_NONE,
                                 quotechar = "",
                                 escapechar = " ")


# # Training the fastText classifier

# In[ ]:


model = fasttext.train_supervised('train.txt', wordNgrams = 2)


# In[ ]:


test_df = pd.read_csv(src + 'test.csv')
#test_df['discourse_effectiveness'] = None


# # Submission

# In[ ]:


result = pd.DataFrame(columns = ['discourse_id','Ineffective','Adequate','Effective'])

for i in range(len(test_df)):
    discourse_id = test_df.discourse_id[i]
    sentence = test_df.discourse_text[i]
    try:
        label, predictions = model.predict(sentence, k=3)
        result = result.append({'discourse_id' : discourse_id, 
                                'Ineffective': predictions[label.index('__label__Ineffective')],
                                'Adequate' : predictions[label.index('__label__Adequate')],
                                'Effective' : predictions[label.index('__label__Effective')]
                               }, ignore_index=True)
    except:
        result = result.append({'discourse_id' : discourse_id, 
                                'Ineffective': 0,
                                'Adequate' : 0,
                                'Effective' : 0
                               }, ignore_index=True)


# In[ ]:


result.to_csv('submission.csv', index = False)


# In[ ]:


get_ipython().system('rm train.txt')


# In[ ]:


#!cat 'submission.csv'


# In[ ]:




