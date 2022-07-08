#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
# Tools for preprocessing input data
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
# Tools for creating ngrams and vectorizing input data
from gensim.models import Word2Vec, Phrases
from nltk.tokenize import word_tokenize
import nltk
import gensim
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import spacy  # For preprocessing


# ## Loading The Dataset

# In[ ]:


df = pd.read_csv('../input/google-war-news/war-news.csv',encoding= 'unicode_escape')


# In[ ]:


df.head()


# ## Removing NULL Rows

# In[ ]:


df.isnull().sum()


# python -m spacy download en_core_web_sm

# In[ ]:


df.dropna(subset=['Summary'], inplace=True)


# In[ ]:


df.isnull().sum()


# ## Text Preprocessing - Punctuations,Stopwords,Stemming,Lemmatize,Tokenize

# In[ ]:


from nltk.stem.snowball import SnowballStemmer

def stemm_text(text):
    stemmer = SnowballStemmer("english")
    return ' '.join([stemmer.stem(w) for w in text.split(' ')])


# In[ ]:


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split(' ')])


# In[ ]:


T = df['Summary'].str.split(' \n\n---\n\n').str[0]
T = T.str.replace('-',' ').str.replace('[^\w\s]','').str.replace('\n',' ').str.lower()
stop = stopwords.words('english')
T = T.apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
T = T.apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))
T = T.apply(stemm_text)
T = T.apply(lemmatize_text)
T = T.apply(word_tokenize)


# In[ ]:


T


# In[ ]:


type(T)


# In[ ]:


T1 = T.values.tolist()


# ## Word2Vec Model

# In[ ]:


# Creating the model and setting values for the various parameters
num_features = 100  # Word vector dimensionality
min_word_count = 2 # Minimum word count
num_workers = 4     # Number of parallel threads
context = 5        # Context window size
downsampling = 1e-3 # (0.001) Downsample setting for frequent words

# Initializing the train model
from gensim.models import word2vec
print("Training model....")
model = word2vec.Word2Vec(workers=num_workers,
                          vector_size=num_features,
                          min_count=min_word_count,
                          window=context,
                          sample=downsampling)

model.build_vocab(T1, progress_per=1000)

model.train(T1, total_examples=model.corpus_count, epochs=model.epochs)

# Saving the model for later use. Can be loaded using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)


# In[ ]:


len(T1[0])


# ## Embedding of a Word

# In[ ]:


vector = model.wv['biden']
print(len(vector))
print(vector)


# ## Similarity Score

# In[ ]:


model.wv.similarity('putin', 'trump')


# In[ ]:


model.wv.similarity('biden', 'trump')


# ## Top 10 Similar Words

# In[ ]:


model.wv.most_similar("afghanistan")


# In[ ]:


model.wv.most_similar("trump")


# ## Plot of Vocabulary using PCA in 2D Space

# In[ ]:


X = model.wv[model.wv.index_to_key] # gives embedding of vocab


# In[ ]:


pca = PCA(n_components=2)

result = pca.fit_transform(X)

plt.figure(figsize=(12,8))
# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.index_to_key)

for i, word in enumerate(words):
   plt.annotate(word, xy=(result[i, 0], result[i, 1]))


plt.show()

