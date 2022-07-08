#!/usr/bin/env python
# coding: utf-8

# <a id='top'></a>
# # NLP - Document Retrieval for Question Answering
# 
# 
# Question answering (QA) is a task that answers user's questions using a large collection of documents; it consists of two steps: (1) sort possible documents that contain the answer of a given question; and (2) extract the content from these documents and elaborate an answer to the user. In this notebook, we are going to explore techniques for identify the most similar document for a question, problem called _Document Retrieval_.
# 
# **Statement**: Given a question, the document retriever have to return the most likely $k$ documents that contain the answer to the question.
# 
# We are going to experiment Document Retrieval in [Stanford Question Answering Dataset](https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset). In the end, I hope you are going to be able to apply some algorithms to handle with this problem.
# 
# > **Summary** - Document Retrieval for Question Answering.   
# > Content for intermediate level in Machine Learning and Data Science!   
# 
# ## Table of Contents
# - [Data Exploration](#data)
# - [Document Retrieval](#document)
#     - TF-IDF
#     - Word2Vec
# - [Conclusion](#discussion)
# 
# ![image](https://qa.fastforwardlabs.com/images/copied_from_nb/my_icons/QAworkflow.png)
# _Image from [Intro to Automated Question Answering](https://qa.fastforwardlabs.com/methods/background/2020/04/28/Intro-to-QA.html)._

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install wikipedia==1.4.0\n!pip install scikit-learn==1.0.2\n!pip install gensim==4.0.1\n')


# ## What do we want to do?
# 
# We want to create a Document Retrieval, like a search tool, such as Wikipedia. Let's explore the `wikipedia` library, to see the retriever in action.

# In[ ]:


import wikipedia as wiki

k = 5
question = "What are the tourist hotspots in Portugal?"

results = wiki.search(question, results=k)
print('Question:', question)
print('Pages:  ', results)


# ### Discussion
# 
# For this question, Wikipedia's Document Retrieval returned the 5 most likely pages that contain the answer to the question.

# <a id="data"></a>
# 
# ---
# # Data Exploration
# 
# In this section, we are going to load a `json` file into a `pandas.DataFrame`. At last, elaborate our list of documents.
# 
# [Back to Top](#top)

# In[ ]:


import json
import numpy as np
import pandas as pd


# In[ ]:


import os

# list the available data
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# based on: https://www.kaggle.com/code/sanjay11100/squad-stanford-q-a-json-to-pandas-dataframe
def squad_json_to_dataframe(file_path, record_path=['data','paragraphs','qas','answers']):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    """
    file = json.loads(open(file_path).read())
    # parsing different level's in the json file
    js = pd.json_normalize(file, record_path)
    m = pd.json_normalize(file, record_path[:-1])
    r = pd.json_normalize(file,record_path[:-2])
    # combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    m['context'] = idx
    data = m[['id','question','context','answers']].set_index('id').reset_index()
    data['c_id'] = data['context'].factorize()[0]
    return data


# In[ ]:


# loading the data
file_path = '/kaggle/input/stanford-question-answering-dataset/train-v1.1.json'
data = squad_json_to_dataframe(file_path)
data


# In[ ]:


# how many documents do we have?
data['c_id'].unique().size


# ## Get the Unique Documents
# 
# Let's select the unique documents in our `data`. This will be the list of documents to search for the answers.

# In[ ]:


documents = data[['context', 'c_id']].drop_duplicates().reset_index(drop=True)
documents


# <a href='#top'><span class="label label-info" style="font-size: 125%">Back to Top</span></a>

# <a id="document"></a>
# 
# ---
# # Document Retrieval
# 
# In this section, we are going to explore the techniques to retrieve documents. First, we are going to create our document `vectorizer`. We use this `vectorizer` to encode the documents and the questions into vectors. After, we can search for a question comparing with the document vectors. In the end, the algorithm will return the $k$ most similar document vectors to a question vector.
# 
# [Back to Top](#top)

# ## TF-IDF
# 
# "In information retrieval, TF-IDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling." [Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# defining the TF-IDF
tfidf_configs = {
    'lowercase': True,
    'analyzer': 'word',
    'stop_words': 'english',
    'binary': True,
    'max_df': 0.9,
    'max_features': 10_000
}
# defining the number of documents to retrieve
retriever_configs = {
    'n_neighbors': 3,
    'metric': 'cosine'
}

# defining our pipeline
embedding = TfidfVectorizer(**tfidf_configs)
retriever = NearestNeighbors(**retriever_configs)


# In[ ]:


# let's train the model to retrieve the document id 'c_id'
X = embedding.fit_transform(documents['context'])
retriever.fit(X, documents['c_id'])


# Let's test the vectorizer, what information our model is using to extract the vector?

# In[ ]:


def transform_text(vectorizer, text):
    '''
    Print the text and the vector[TF-IDF]
    vectorizer: sklearn.vectorizer
    text: str
    '''
    print('Text:', text)
    vector = vectorizer.transform([text])
    vector = vectorizer.inverse_transform(vector)
    print('Vect:', vector)


# In[ ]:


# vectorize the question
transform_text(embedding, question)


# What is the most similar document to this question?

# In[ ]:


# predict the most similar document
X = embedding.transform([question])
c_id = retriever.kneighbors(X, return_distance=False)[0][0]
selected = documents.iloc[c_id]['context']

# vectorize the document
transform_text(embedding, selected)


# ### Evaluation

# In[ ]:


get_ipython().run_cell_magic('time', '', "# predict one document for each question\nX = embedding.transform(data['question'])\ny_test = data['c_id']\ny_pred = retriever.kneighbors(X, return_distance=False)\n")


# In[ ]:


# top documents predicted for each question
y_pred


# In[ ]:


def top_accuracy(y_true, y_pred) -> float:
    right, count = 0, 0
    for y_t in y_true:
        count += 1
        if y_t in y_pred:
            right += 1
    return right / count if count > 0 else 0


# In[ ]:


acc = top_accuracy(y_test, y_pred)
print('Accuracy:', f'{acc:.4f}')
print('Quantity:', int(acc*len(y_pred)), 'from', len(y_pred))


# ### Discussion
# 
# 1. This is a difficult problem, because we have multiples documents (in this notebook, ~19k documents) and the answer can be in one or more documents. Thus, the retriever usually returns $k$ documents, because it is not complete/fair return only one document.
# 2. We reach a high accuracy with top-3 (98.92%); in top-1 a low accuray (43.22%) becase we have a lot of documents, and some are pretty similar. Actually, this top-1 and top-3 are very good accuracy for this problem.
# 3. TF-IDF has some problems: (1) this algorithm is only able to compute similarity between questions and documents that present the same words, so it can not capture synonyms; and (2) cannot understand the question context or the meaning of the words.

# ## Word2Vec / Embedding
# 
# "Word2vec is a technique for natural language processing published in 2013. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence." [Wikipedia](https://en.wikipedia.org/wiki/Word2vec)

# In[ ]:


from gensim.parsing.preprocessing import preprocess_string

# create a corpus of tokens
corpus = documents['context'].tolist()
corpus = [preprocess_string(t) for t in corpus]


# In[ ]:


from gensim.models import Word2Vec
import gensim.downloader

# you can download a pretrained Word2Vec
# - or you can train your own model

# download a model
# 'glove-wiki-gigaword-300' (376.1 MB)
# 'word2vec-ruscorpora-300' (198.8 MB)
# 'word2vec-google-news-300' (1.6 GB)
# vectorizer = gensim.downloader.load('word2vec-ruscorpora-300')

# train your own model
vectorizer = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, workers=4).wv


# In[ ]:


# similar words to 'tourist'
vectorizer.most_similar('tourist', topn=5)


# In[ ]:


def transform_text2(vectorizer, text, verbose=False):
    '''
    Transform the text in a vector[Word2Vec]
    vectorizer: sklearn.vectorizer
    text: str
    '''
    tokens = preprocess_string(text)
    words = [vectorizer[w] for w in tokens if w in vectorizer]
    if verbose:
        print('Text:', text)
        print('Vector:', [w for w in tokens if w in vectorizer])
    elif len(words):
        return np.mean(words, axis=0)
    else:
        return np.zeros((300), dtype=np.float32)


# In[ ]:


# just testing our Word2Vec
transform_text2(vectorizer, question, verbose=True)


# In[ ]:


# let's train the model to retrieve the document id 'c_id'
retriever = NearestNeighbors(**retriever_configs)

# vectorizer the documents, fit the retriever
X = documents['context'].apply(lambda x: transform_text2(vectorizer, x)).tolist()
retriever.fit(X, documents['c_id'])


# ### Evaluation

# In[ ]:


get_ipython().run_cell_magic('time', '', "# vectorizer the questions\nX = data['question'].apply(lambda x: transform_text2(vectorizer, x)).tolist()\n\n# predict one document for each question\ny_test = data['c_id']\ny_pred = retriever.kneighbors(X, return_distance=False)\n")


# In[ ]:


# top documents predicted for each question
y_pred


# In[ ]:


acc = top_accuracy(y_test, y_pred)
print('Accuracy:', f'{acc:.4f}')
print('Quantity:', int(acc*len(y_pred)), 'from', len(y_pred))


# ### Discussion
# 
# 1. We also reach a good accuracy (97.15%) in top-3; and a really low accuray (3.12%) in top-1. Thus, the TF-IDF was better.
# 2. Maybe, my `vectorizer` didn't receive enough data to be trained. Thus, I suggest use pretrained model, like `'word2vec-google-news-300'`.
# 3. Another problem: I simply compute the average of the words to compose the document/question embedding; we do have other pooling strategies to work with sentences. Or, we can try more robust embedding techniques, such as BERT, MT5, DPR, etc.

# <a href='#top'><span class="label label-info" style="font-size: 125%">Back to Top</span></a>

# <a id="discussion"></a>
# 
# ---
# # Conclusion
# 
# 1. As mentioned, this problem is really complex, due to the number of documents.
# 2. TF-IDF reached a great top-3 accuracy (98.92%) for this dataset, and it can increases returning more documents.
# 3. We also have other algorithms to work with Document Retriveal, such as [BM25](https://pypi.org/project/rank-bm25/) and [DPR](https://aclanthology.org/2020.emnlp-main.550/).
# 
# [Back to Top](#top)
# 
# ## Reference
# 
# 1. [Intro to Automated Question Answering](https://qa.fastforwardlabs.com/methods/background/2020/04/28/Intro-to-QA.html)
# 2. [Building a QA System with BERT on Wikipedia](https://qa.fastforwardlabs.com/pytorch/hugging%20face/wikipedia/bert/transformers/2020/05/19/Getting_Started_with_QA.html)
# 3. [Evaluating QA: Metrics, Predictions, and the Null Response](https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html)
# 4. [Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2020.emnlp-main.550/)

# In[ ]:




