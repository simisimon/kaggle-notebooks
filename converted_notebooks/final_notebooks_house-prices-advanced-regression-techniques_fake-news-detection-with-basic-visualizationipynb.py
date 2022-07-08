#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
from string import punctuation
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer


# In[ ]:


fake_news = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
true_news = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
fake_news['target'] = 0
true_news['target'] = 1


# In[ ]:


data0 = pd.concat([fake_news, true_news])
data0 = data0.reset_index(drop=True)


# # Data exploration 
# Check the amount of subjects 

# In[ ]:


print(data0.groupby('subject')['text'].count())

fig = plt.figure(figsize=(8,6))
plt.xticks(rotation = 45)
sns.histplot(data=data0, x="subject")


# Amount of true and false news among dataset

# In[ ]:


print(data0.groupby('target')['text'].count())

fig = plt.figure(figsize=(8,6))
sns.histplot(data=data0, x="target")


# # Data cleaning 

# In[ ]:


data0


# In[ ]:


data1 = data0.copy()
data1 = data1.drop(['title', 'subject', 'date'], axis=1)


# Shuffle the data

# In[ ]:


data2 = data1.copy()
data2 = shuffle(data2, random_state=0)
data2 = data2.reset_index(drop=True)


# In[ ]:


data2


# In[ ]:


def cleaner(text):
    
    text = text.lower()
    
    text = ''.join(c for c in text if not c.isdigit()) #remove digits
    text = ''.join(c for c in text if c not in punctuation) #remove all punctuation
    
    stop_words = stopwords.words('english') # removes words which has less meaning 
    text = ' '.join([w for w in nltk.word_tokenize(text) if not w in stop_words])
    
    wordnet_lemmatizer = WordNetLemmatizer() # with use of morphological analysis of words
    text = [wordnet_lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)]
    
    text = " ".join(w for w in text)
    return text


# In[ ]:


data3 = data2.copy()
data3['text'] = data3['text'].apply(cleaner)


# In[ ]:


data3


# # Visualization of most frequent words using WordCloud 

# Visualization of fake news

# In[ ]:


fake_data = data3[data3['target'] == 0]
fake_data = ''.join([text for text in fake_data.text])
wordcloud = WordCloud(width=800, height=500, max_font_size=100, max_words=100).generate(fake_data)

plt.figure(figsize=(8,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Visualization of true news

# In[ ]:


true_data = data3[data3['target'] == 1]
true_data = ''.join([text for text in true_data.text])
wordcloud = WordCloud(width=800, height=500, max_font_size=100, max_words=100).generate(true_data)

plt.figure(figsize=(8,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


X = data3['text']
y = data3['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


tfidf = TfidfVectorizer()
clf = LinearSVC(C=10)


# In[ ]:


pipeline = Pipeline([
('tfidf', tfidf),
('classifier', clf),  
    ])
    
pipeline.fit(X_train, y_train) 
pred = pipeline.predict(X_test)
    
print(accuracy_score(y_test, pred))


# # Result visualization

# In[ ]:


results = pd.DataFrame(confusion_matrix(y_test, pred),
                       columns=['Fake', 'Real'], index=['Fake', 'Real'])
plt.figure(figsize=(8,6))
sns.heatmap(results, annot=True, cbar=False)


# In[ ]:




