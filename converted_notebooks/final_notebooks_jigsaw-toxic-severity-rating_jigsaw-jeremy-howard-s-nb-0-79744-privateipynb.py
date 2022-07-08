#!/usr/bin/env python
# coding: utf-8

# # ☣️ Jigsaw - Jeremy Howard's NB [0.79744 Private]
# 
# ## This is version 7 of my notebook [☣️ Jigsaw - Incredibly Simple Naive Bayes [0.768]](https://www.kaggle.com/julian3833/jigsaw-incredibly-simple-naive-bayes-0-768?scriptVersionId=79432344), a NB-SVM adapted from [NB-SVM strong linear baseline](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline/) by the great [Jeremy Howard](https://www.kaggle.com/jhoward)!!
# 
# ## It would have landed in position 164, obtaining a bronze medal.
# 
# # Please, _DO_ upvote!

# # Imports

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# # Create train data
# 
# The competition was multioutput
# 
# We turn it into a binary toxic/ no-toxic classification

# In[ ]:


df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) > 0 ).astype(int)
df = df.rename(columns={'comment_text': 'text'})
df.sample(5)


# # Undersample
# 
# The dataset is very unbalanced. Here we undersample the majority class. Other strategies might work better.

# In[ ]:


df['y'].value_counts(normalize=True)


# In[ ]:


min_len = (df['y'] == 1).sum()


# In[ ]:


df_y0_undersample = df[df['y'] == 0].sample(n=min_len, random_state=201)


# In[ ]:


df = pd.concat([df[df['y'] == 1], df_y0_undersample])


# In[ ]:


df['y'].value_counts()


# # TF-IDF

# In[ ]:


# vec = TfidfVectorizer()
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1 )


# In[ ]:


X = vec.fit_transform(df['text'])
X


# # Fit NB-SVM
# 
# ### Adapted from [NB-SVM strong linear baseline](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline/) by the great [Jeremy Howard](https://www.kaggle.com/jhoward)!!
# 

# In[ ]:


#model = MultinomialNB()
#model.fit(X, df['y'])

def pr(X, y, y_i):
    p = X[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def train_model(X, y):
    y = y.values
    r = np.log(pr(X, y, 1) / pr(X, y, 0))
    m = LogisticRegression(C=4, dual=True, solver='liblinear')
    x_nb = X.multiply(r)
    return m.fit(x_nb, y), r


# In[ ]:


model, r = train_model(X, df['y'])


# # Per-label models

# In[ ]:


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
models_and_rs = {l: train_model(X, df[l]) for l in labels}


# # Validate

# In[ ]:


df_val = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data.csv")


# In[ ]:


X_less_toxic = vec.transform(df_val['less_toxic'])
X_more_toxic = vec.transform(df_val['more_toxic'])


# In[ ]:


p1 = model.predict_proba(X_less_toxic.multiply(r))
p2 = model.predict_proba(X_more_toxic.multiply(r))


# In[ ]:


(p1[:, 1] < p2[:, 1]).mean()


# In[ ]:


# Summation of prediction of each model
r1 = np.array([model_label.predict_proba(X_less_toxic.multiply(r_label))[:, 1] for model_label, r_label in models_and_rs.values()]).sum(axis=0)
r2 = np.array([model_label.predict_proba(X_more_toxic.multiply(r_label))[:, 1] for model_label, r_label in models_and_rs.values()]).sum(axis=0)


# In[ ]:


# Validation of the summation approach
(r1 < r2).mean()


# In[ ]:


# Validation of both approaches combined
(p1[:, 1] + r1 < p2[:, 1] + r2).mean()


# # Submission

# In[ ]:


df_sub = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")
X_test = vec.transform(df_sub['text'])
p3 = model.predict_proba(X_test.multiply(r))
r3 = np.array([model_label.predict_proba(X_test.multiply(r_label))[:, 1] for model_label, r_label in models_and_rs.values()]).sum(axis=0)


# In[ ]:


df_sub


# In[ ]:


df_sub['score'] = p3[:, 1] + r3


# In[ ]:


df_sub['score'].count()


# In[ ]:


# 9 comments will fail if compared one with the other
df_sub['score'].nunique()


# In[ ]:


df_sub[['comment_id', 'score']].to_csv("submission.csv", index=False)


# # Please, _DO_ upvote!
