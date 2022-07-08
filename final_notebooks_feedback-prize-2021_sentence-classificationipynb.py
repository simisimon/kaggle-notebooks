#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import nltk
import os


# In[ ]:


train_df = pd.read_csv("/kaggle/input/feedback-prize-2021/train.csv")


# In[ ]:


train_df.discourse_type.value_counts().plot.barh()


# In[ ]:


train_texts= list(train_df.discourse_text)


# In[ ]:


train_labels = np.array(train_df.discourse_type)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2) #Unigrams and bigrams

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2


# Create keyword arguments to pass to the 'tf-idf' vectorizer.
kwargs = {
        'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': TOKEN_MODE,  # Split text into word tokens.
        'min_df': MIN_DOCUMENT_FREQUENCY,
}
vectorizer = TfidfVectorizer(**kwargs)

# Learn vocabulary from training texts and vectorize training texts.
x_train = vectorizer.fit_transform(train_texts)


# Select top 'k' of the vectorized features.
selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
selector.fit(x_train, train_labels)
x_train = selector.transform(x_train).astype('float32')


# In[ ]:


def create_test_texts_list():
    total_list = []
    
    test_dir = "../input/feedback-prize-2021/test"
    for filename in os.listdir(test_dir):
        file_path = os.path.join(test_dir, filename)
        # checking if it is a file
        if os.path.isfile(file_path) and os.path.splitext(file_path)[1] == ".txt":
            with open(file_path) as f:
                    total_list.append({
                        'text' : f.read(), 
                        'id' : os.path.splitext(filename)[0]
                    })
    
    return total_list


# In[ ]:


from sklearn.svm import LinearSVC

svm_model = LinearSVC()

print("training")
svm_model.fit(x_train, train_labels)


# In[ ]:


test_texts = create_test_texts_list()
print(test_texts)


# In[ ]:


pred_dicts_list = []

for test_text in test_texts:
    
    total_word_count = 0
    
    tokenized_sentences = nltk.sent_tokenize(test_text["text"])
    
    x_test = vectorizer.transform(tokenized_sentences)
    x_test = selector.transform(x_test).astype('float32')
    preds = svm_model.predict(x_test) #Returns list
    
    
    for i, pred in enumerate(preds):
        
        # Generate prediction strings for each predicted discourse
        tokenized_sentence = tokenized_sentences[i]
        
        if i == 0 or preds[i-1] != pred:
            prediction_string = ""
        
        for x in range(total_word_count, total_word_count + len(tokenized_sentence.split())):
            prediction_string += f"{x} "
        
        total_word_count += len(tokenized_sentence.split())
        
        try:
            if preds[i+1] == pred:
                continue
        except:
            pass
        
        pred_dicts_list.append({
            "id" : test_text["id"],
            "class" : pred, 
            "predictionstring" : prediction_string.strip()
        })


# In[ ]:


submission_df = pd.DataFrame(pred_dicts_list)


# In[ ]:


submission_df.to_csv("submission.csv", index=False)

