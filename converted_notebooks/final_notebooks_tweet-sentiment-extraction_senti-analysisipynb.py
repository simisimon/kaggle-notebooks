#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import string


# In[ ]:


train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')


# In[ ]:


train[train['text'].isna()]


# In[ ]:


train.drop(314, inplace = True)


# In[ ]:


train['text'] = train['text'].apply(lambda x: x.lower())
test['text'] = test['text'].apply(lambda x: x.lower())
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(
    train, train_size = 0.80, random_state = 0)


# In[ ]:


pos_train = X_train[X_train['sentiment'] == 'positive']
neutral_train = X_train[X_train['sentiment'] == 'neutral']
neg_train = X_train[X_train['sentiment'] == 'negative']


# In[ ]:


cv = CountVectorizer(max_df=0.95, min_df=2,max_features=10000,stop_words='english')
X_train_cv = cv.fit_transform(X_train['text'])
X_pos = cv.transform(pos_train['text'])
X_neutral = cv.transform(neutral_train['text'])
X_neg = cv.transform(neg_train['text'])
pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())
pos_words = {}
neutral_words = {}
neg_words = {}
for k in cv.get_feature_names():
    pos = pos_count_df[k].sum()
    neutral = neutral_count_df[k].sum()
    neg = neg_count_df[k].sum()
    pos_words[k] = pos/pos_train.shape[0]
    neutral_words[k] = neutral/neutral_train.shape[0]
    neg_words[k] = neg/neg_train.shape[0]
neg_words_adj = {}
pos_words_adj = {}
neutral_words_adj = {}
for key, value in neg_words.items():
    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])
for key, value in pos_words.items():
    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])
for key, value in neutral_words.items():
    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])


# In[ ]:


def calculate_selected_text(df_row, tol = 0):
    tweet = df_row['text']
    sentiment = df_row['sentiment']
    if(sentiment == 'neutral'):
        return tweet
    elif(sentiment == 'positive'):
        dict_to_use = pos_words_adj
    elif(sentiment == 'negative'):
        dict_to_use = neg_words_adj
    words = tweet.split()
    words_len = len(words)
    subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]
    
    score = 0
    selection_str = ''
    lst = sorted(subsets, key = len)
    
    
    for i in range(len(subsets)):
        new_sum = 0
        for p in range(len(lst[i])):
            if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in dict_to_use.keys()):
                new_sum += dict_to_use[lst[i][p].translate(str.maketrans('','',string.punctuation))]
        if(new_sum > score + tol):
            score = new_sum
            selection_str = lst[i]
    if(len(selection_str) == 0):
        selection_str = words
    return ' '.join(selection_str)


# In[ ]:


pd.options.mode.chained_assignment = None


# In[ ]:


tol = 0.001
X_val['predicted_selection'] = ''
for index, row in X_val.iterrows():
    selected_text = calculate_selected_text(row, tol)
    X_val.loc[X_val['textID'] == row['textID'], ['predicted_selection']] = selected_text


# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[ ]:


X_val['jaccard'] = X_val.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)
print('The jaccard score for the validation set is:', np.mean(X_val['jaccard']))


# In[ ]:


pos_tr = train[train['sentiment'] == 'positive']
neutral_tr = train[train['sentiment'] == 'neutral']
neg_tr = train[train['sentiment'] == 'negative']


# In[ ]:


cv = CountVectorizer(max_df=0.95, min_df=2,max_features=10000,stop_words='english')
final_cv = cv.fit_transform(train['text'])
X_pos = cv.transform(pos_tr['text'])
X_neutral = cv.transform(neutral_tr['text'])
X_neg = cv.transform(neg_tr['text'])
pos_final_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
neutral_final_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
neg_final_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())


# In[ ]:


pos_words = {}
neutral_words = {}
neg_words = {}
for k in cv.get_feature_names():
    pos = pos_final_count_df[k].sum()
    neutral = neutral_final_count_df[k].sum()
    neg = neg_final_count_df[k].sum()
    pos_words[k] = pos/(pos_tr.shape[0])
    neutral_words[k] = neutral/(neutral_tr.shape[0])
    neg_words[k] = neg/(neg_tr.shape[0])


# In[ ]:


neg_words_adj = {}
pos_words_adj = {}
neutral_words_adj = {}
for key, value in neg_words.items():
    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])
for key, value in pos_words.items():
    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])
for key, value in neutral_words.items():
    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])


# In[ ]:


tol = 0.001
for index, row in test.iterrows():
    selected_text = calculate_selected_text(row, tol)
    sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text


# In[ ]:


sample.to_csv('submission.csv', index = False)

