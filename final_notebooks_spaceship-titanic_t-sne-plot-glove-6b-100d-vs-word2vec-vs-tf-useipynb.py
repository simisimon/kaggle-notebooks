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
from gensim.models import Word2Vec, Phrases
from nltk.tokenize import word_tokenize
import nltk
import seaborn as sns
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import gensim
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import spacy  # For preprocessing


# In[ ]:


df = pd.read_csv('../input/google-war-news/war-news.csv',encoding='latin1')


# In[ ]:


df.head()


# ## Data Preprocessing

# In[ ]:


df.isnull().sum()


# In[ ]:


df.dropna(subset=['Summary'], inplace=True)


# In[ ]:


from nltk.stem.snowball import SnowballStemmer

# Use English stemmer.


def stemm_text(text):
    stemmer = SnowballStemmer("english")
    return ' '.join([stemmer.stem(w) for w in text.split(' ')])


# In[ ]:


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split(' ')])


# In[ ]:


T = df['Summary'].str.split(' \n\n---\n\n').str[0]
T = T.str.replace('-',' ').str.replace('[^\w\s]','').str.replace('\n',' ').str.lower()
stop = stopwords.words('english')
T = T.apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
T = T.apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))
#T = T.apply(stemm_text)
#T = T.apply(lemmatize_text)


# In[ ]:


T


# In[ ]:


type(T)


# In[ ]:


T1 = T.values.tolist()


# ## Glove - 100d

# In[ ]:


def Sentence2Vec(T,embedding_dim = 100,max_length = 300):
    glove_path = '../input/glove6b100dtxt/glove.6B.100d.txt'
    path = glove_path
    tokenizer = Tokenizer()
    text=T
    tokenizer.fit_on_texts(text)
    word_index=tokenizer.word_index
    print("number of word in vocabulary",len(word_index))
    vocab_size = 5000
    trunc_type = 'post'
    oov_tok = '<OOV>'
    padding_type = 'post'
    #print("words in vocab",word_index)
    text_sequence=tokenizer.texts_to_sequences(text)
    text_sequence = pad_sequences(text_sequence, maxlen=max_length, truncating=trunc_type,padding=padding_type)
    print("word in sentences are replaced with word ID",text_sequence)
    size_of_vocabulary=len(tokenizer.word_index) + 1
    print("The size of vocabulary ",size_of_vocabulary)
    embeddings_index = dict()

    f = open(path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((size_of_vocabulary, embedding_dim))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    text_shape = text_sequence.shape
    X_train = np.empty((text_shape[0],text_shape[1],embedding_matrix.shape[1]))
    for i in range(text_sequence.shape[0]):
        for j in range(text_sequence.shape[1]):
            X_train[i,j,:] = embedding_matrix[text_sequence[i][j]]
    print(X_train.shape)

    return X_train,embeddings_index,word_index


# In[ ]:


X_train,embeddings_index,word_index = Sentence2Vec(T)


# In[ ]:


L = list(word_index.keys())


# In[ ]:


emb_matrix = np.zeros((len(L)+1, 100))

for word, index in word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    emb_matrix[index, :] = embedding_vector


# In[ ]:


emb_matrix.shape


# In[ ]:


from sklearn.manifold import TSNE


# In[ ]:


tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(emb_matrix)


# In[ ]:


df3 = pd.DataFrame()
df3["comp-1"] = z[:,0]
df3["comp-2"] = z[:,1]

plt.figure(figsize=(12,6))
sns.scatterplot(x="comp-1", y="comp-2",data=df3).set(title="Glove T-SNE projection") 


# ## Word2Vec - 100d

# In[ ]:


T1 = T.apply(word_tokenize)


# In[ ]:


# Creating the model and setting values for the various parameters
num_features = 100  # Word vector dimensionality
min_word_count = 2 # Minimum word count
num_workers = 4     # Number of parallel threads
context = 10        # Context window size
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


# In[ ]:


X = model.wv[model.wv.index_to_key]


# In[ ]:


X.shape


# In[ ]:


tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(X)


# In[ ]:


df3 = pd.DataFrame()
df3["comp-1"] = z[:,0]
df3["comp-2"] = z[:,1]

plt.figure(figsize=(12,6))
sns.scatterplot(x="comp-1", y="comp-2",data=df3).set(title="Glove T-SNE projection") 


# ## Generating Vectors using the TensorFlow2: Universal Sentence Encoder 4

# In[ ]:


import tensorflow_hub as hub
#download the model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


# In[ ]:


embeddings = embed(T)


# In[ ]:


#create list from np arrays
use= np.array(embeddings).tolist()


# In[ ]:


tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(use)


# In[ ]:


df3 = pd.DataFrame()
df3["comp-1"] = z[:,0]
df3["comp-2"] = z[:,1]

plt.figure(figsize=(12,6))
sns.scatterplot(x="comp-1", y="comp-2",data=df3).set(title="TensorFlow2: Universal Sentence Encoder:  T-SNE projection") 

