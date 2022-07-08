#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action="ignore")

import time 

#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_union
from sklearn.linear_model import LogisticRegression


#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import matplotlib_venn as venn


#settings
start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

get_ipython().run_line_magic('matplotlib', 'inline')



# In[ ]:


train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


train.nunique()


# In[ ]:


train['comment_text']


# In[ ]:


train['comment_text'][2]


# In[ ]:


test.head()


# In[ ]:


test['comment_text'][0]


# In[ ]:


test.shape


# In[ ]:


train.describe().T


# In[ ]:


train.isnull().sum()


# In[ ]:


import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import regexp_tokenize   # module for tokenizing strings
from nltk.tokenize import TreebankWordTokenizer


# In[ ]:


def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[ ]:


train['comment_text'] = train['comment_text'].apply(lambda x: clean_text(x))


# In[ ]:


train['comment_text']


# In[ ]:


test['comment_text']=test['comment_text'].apply(lambda x: clean_text(x))


# In[ ]:


test['comment_text']


# In[ ]:


def remove_stopwords(text):
    
    words = [word for word in text if word not in stopwords.words('english')]
    return words


# In[ ]:





# In[ ]:


# Checking the count of the various types of words

cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = train[cols].values

train_df = train['comment_text']
test_df = test['comment_text']


# In[ ]:


val_counts = train[cols].sum()

plt.figure(figsize=(8,5))
ax = sns.barplot(val_counts.index, val_counts.values, alpha=0.8)

plt.title("Comments per Classes")
plt.xlabel("Various Comments Type")
plt.ylabel("Counts of the Comments")

rects = ax.patches
labels = val_counts.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+5, label, ha="center", va="bottom")


plt.show()


# In[ ]:


from wordcloud import WordCloud
words = ' '.join([text for text in train['comment_text'] ])


word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       #colormap='PuRd', 
                       margin=0,
                       max_words=500, # Maximum numbers of words we want to see 
                       min_word_length=3, # Minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white").generate(words)

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('Comments and their Nature', fontsize = 40)
plt.axis("off")
plt.show()


# In[ ]:


words = ' '.join([text for text in test['comment_text'] ])


word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       #colormap='PuRd', 
                       margin=0,
                       max_words=500, # Maximum numbers of words we want to see 
                       min_word_length=3, # Minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white").generate(words)

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="bilinear")
plt.title('Comments and their Nature', fontsize = 40)
plt.axis("off")
plt.show()


# In[ ]:


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=30000)
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=30000)
vectorizer = make_union(word_vectorizer, char_vectorizer, n_jobs=2)


# In[ ]:


all_text = pd.concat([train,test])


# In[ ]:


vectorizer.fit(all_text)
train_features = vectorizer.transform(train_df)
test_features = vectorizer.transform(test_df)


# In[ ]:


scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in cols:
    train_target = train[class_name]
    classifier = LogisticRegression(solver='sag')

    cv_score = np.mean(cross_val_score(
        classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]


# In[ ]:


print('Total CV score is {}'.format(np.mean(scores)))


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




