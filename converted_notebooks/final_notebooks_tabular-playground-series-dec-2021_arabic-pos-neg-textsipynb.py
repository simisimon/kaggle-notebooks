#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import tensorflow as tf
#listdir returns a list containing the names of the entries in the directory given as a parameter
labels = os.listdir('../input/arabic-sentiment-twitter-corpus/arabic_tweets') 

# tf.keras.preprocessing.text_dataset_from_directory Generates a 
# tf.data.Dataset from text files in a directory.
raw_data = tf.keras.preprocessing.text_dataset_from_directory(
    '../input/arabic-sentiment-twitter-corpus/arabic_tweets',
    # "inferred" : the labels are generated from the directory structure
    labels = "inferred",
    # "int": the labels are encoded as integers
    label_mode = "int",
    # Maximum size of a text string. Texts longer than this will be shortened 
    # to max_length unless it's None ra7at explanation f kil zit.
    max_length = None,
    # Whether to shuffle the data. If False, sorts the data in alphanumeric order.
    shuffle=True,
    # Finally haja fahmetha mn bkri
    seed=11,
    # Optional float between 0 and 1, fraction of data to reserve for validation
    validation_split=None,
    # Only used if validation_split is set, mahich set alors sotit
    subset=None,
)


# In[ ]:


print("Classes names:\n",raw_data.class_names)


# In[ ]:


x=[]
y=[]
for text_batch, label_batch in raw_data:
    for i in range(len(text_batch)):
        s=text_batch.numpy()[i].decode("utf-8") 
        x.append(s)
        y.append(raw_data.class_names[label_batch.numpy()[i]])
print(len(x))
print(len(y))


# In[ ]:


import pandas as pd 
data =pd.DataFrame({"text":x,"label":y})


# In[ ]:


data.head()


# In[ ]:


print('is null ? :  \n **************** ', data.isnull().sum())
print('data info : \n **************** ', data.info())
print('is duplicated : \n ************* ', data.duplicated().sum())
print('data shape \n :  ************** ', data.shape)


# In[ ]:


data.loc[data.duplicated()]


# In[ ]:


data[data['text']=='والله باينه 😎😎 مين شرنوبي ناو😂\n'] #fu. for this data


# In[ ]:


data.drop_duplicates(inplace=True)
data.duplicated().sum()


# In[ ]:


data.shape


# In[ ]:


import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = list(set(stopwords.words('arabic')))
print(stop_words)


# In[ ]:


from nltk.tokenize import word_tokenize
import re
import string
import sys
import argparse

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations


# In[ ]:


def remove_diacritics(text):
    arabic_diacritics = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(arabic_diacritics, '', str(text))
    return text

def remove_emoji(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    text = remove_emoji(text)
    text = remove_diacritics(text)
    tokens = word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])
    return text


# In[ ]:


data['cleaned_text'] = data['text'].apply(clean_text)
data.head()


# In[ ]:


### convert positive 1 and negative 0
#data['label']=data['label'].apply(lambda X: 0 if X == 'neg' else 1 )
#data.head()


# In[ ]:


#from sklearn.preprocessing import LabelEncoder
#label_encoder = LabelEncoder()
#data['encodedLabel'] = label_encoder.fit_transform(data['label'])
#data.head()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer() #(analyzer='char_wb',ngram_range=(3,5),min_df=0.01,max_df=0.3)
from sklearn.svm import SVC


# In[ ]:


clf=SVC() #(kernel='rbf')


# In[ ]:


from sklearn.pipeline import make_pipeline
pipe=make_pipeline(vectorizer,clf)


# In[ ]:


X_cla = data['cleaned_text'] 
y_cla = data['label']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_cla, y_cla, test_size=0.33, random_state=42)


# In[ ]:


pipe.fit(X_train,y_train)


# In[ ]:


y_pred=pipe.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[ ]:


accuracy_score(y_test,y_pred) #without emoji


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=0.01,max_df=0.3)
clf = RandomForestClassifier()

from sklearn.pipeline import make_pipeline
pipe=make_pipeline(vectorizer,clf)

pipe.fit(X_train, y_train)
pred=pipe.predict(X_test)



# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:


accuracy_score(y_test,pred)


# In[ ]:




