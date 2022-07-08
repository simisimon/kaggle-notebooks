#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import unicode_literals, print_function, division
import re, nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter

import tensorflow as tf
from tensorflow.keras import layers , activations , models , preprocessing , utils

import os
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import string


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The stopwords package.
nltk.download('stopwords')

# Much of the code for the seq2seq model was inspired by https://colab.research.google.com/drive/11os3isH4I4X76dwOAQJ5cSRnfhmUziHm
# As mentioned in the powerpoint presentation.


# In[ ]:


# Read from data, and check the head.
data = pd.read_csv("/kaggle/input/customer-support-on-twitter/twcs/twcs.csv")
data.head(20)


# In[ ]:


# Check data info
data.info()


# In[ ]:


# Data Preprocessing on text.

# Punctuation
punctuations = string.punctuation
print(punctuations)

# Stopwords
stopwords_set = set(stopwords.words('english'))

# To determine the frequency of words
cnt = Counter()


# In[ ]:


# Step 1: Set to lowercase.
data['text'] = data.loc[:, 'text'].str.lower()


# In[ ]:


# Step 2: Replace twitter handles with 'HANDLE'
def replaceHandle(match):
    _sn = match.group(2).lower()
    if not _sn.isnumeric():
        # This is a company screen name
        return match.group(1) + match.group(2)
    return 'HANDLE'

regex = re.compile('(\W@|^@)([a-zA-Z0-9_]+)')

data['text'] = data.loc[:, 'text'].apply(lambda txt: regex.sub(replaceHandle, txt))


# In[ ]:


# Step 3: Remove links!
# This checks for hyperlinks.

def removeUrls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


# In[ ]:


# Step 4: Remove punctuation.
def removePunc(text):
    punc_transform = str.maketrans('','',punctuations)
    return text.translate(punc_transform)
    
# Step 5: Remove emojis non-characters.
def removeOther(text):
    # remove non-letters
    return re.sub(r"[^a-zA-Z0-9]", " ", text)

# Step 6: Remove Stopwords.
def removeStops(text):
    return ' '.join(words for words in str(text).split() if words not in stopwords_set)

def transformData(text):
    return removeStops(removeOther(removePunc(removeUrls(text))))


def findFreq(data):
    for text in data['clean_text'].values:
        for word in text.split():
            cnt[word] +=1
    print("Common words include: ")
    for w in range(30):
        print(cnt.most_common()[w])
    return set([w for (w,wc) in cnt.most_common(10)])

# Step 7: Remove rare words.
def findRare():
    numberInList = 61000
    print("\nNumber of unique words: ")
    print(len(cnt.most_common()))
    rareWords = set([w for (w,c) in cnt.most_common()][:- numberInList: -1])
    return rareWords

def removeRare(text, rareWords):
    return ' '.join(word for word in str(text).split() if word not in rareWords)
   
# Apply functions.
def preProcess(data):
    # Remove punctuation, change to lower case and remove stop words.
    data['clean_text'] = data['text'].apply(lambda text: transformData(text))
    
    # Find frequent words.
    frequentWords = findFreq(data)
    
    # Remove rare words, such as usernames.
    rareWords = findRare()
    data['clean_text'] = data['clean_text'].apply(lambda text:removeRare(text, rareWords))
    return data


data = preProcess(data)


# In[ ]:


# Step 8: Build final table.
# Messages to companies.
firstMessages = data[pd.isnull(data.in_response_to_tweet_id) & data.inbound]

# Get replies on the right, and initial tweets on left.
replyMessages = pd.merge(firstMessages, data, left_on='tweet_id', right_on='in_response_to_tweet_id')

# Remove replies that aren't from companies
replyMessages = replyMessages[replyMessages.inbound_y ^ True]

# Step 9: Create final table, and take only first 10k tweets due to memory problems.
data = replyMessages[['clean_text_x', 'clean_text_y']]
data.columns = ['First', 'Second']
data = data[:10000]





# In[ ]:


# Step 10: Apply stemmer. Use the most commonly used stemmer.

stemmer=PorterStemmer()

def applyStem(text):
    stemmed = " ".join([stemmer.stem(word) for word in text.split()])
    return stemmed
    
data["First"]=data['First'].apply(lambda text : applyStem(text))
data["Second"]=data['Second'].apply(lambda text : applyStem(text))

# Step 9: Take only text with less than 10 words/tokens.
data = data[data['First'].str.split().str.len().lt(10)]
data = data[data['Second'].str.split().str.len().lt(10)]

data.head(20)


# In[ ]:


# Now, we need to translate the text into tokens and check out how many words we'll be using.
# Tokenize the input sentences.
enc_input = list()
for line in data['First']:
    enc_input.append( line )

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts( enc_input )
tokenized_enc_input = tokenizer.texts_to_sequences( enc_input ) 


# Get the maximum input length.
length_list = list()
for token_seq in tokenized_enc_input:
    length_list.append( len( token_seq ))
max_input_length = np.array( length_list ).max()
print( 'Max input length is {}'.format( max_input_length ))
length_list = None

# We need the input to be the same length.
# So pad out the input.
padded_enc_input = preprocessing.sequence.pad_sequences( tokenized_enc_input , maxlen=max_input_length , padding='post' )
encoder_input_data = np.array( padded_enc_input )
print( 'Encoder input data shape -> {}'.format( encoder_input_data.shape ))
padded_enc_input = None

# Find size of vocabulary (number of available words)
enc_input_dict = tokenizer.word_index
num_enc_input_tokens = len( enc_input_dict )+1
print( 'Number of Question tokens = {}'.format( num_enc_input_tokens))


# In[ ]:


# Now, we need to prepare the input for the decoder.
# The input for the decoder is the reply tweets.

# First, add start and end tags.
dec_input = list()
for line in data['Second']:
    dec_input.append( '<START> ' + line + ' <END>' )  

    # Tokenize the tweets.
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts( dec_input ) 
tokenized_dec_input = tokenizer.texts_to_sequences( dec_input ) 

# Get maximum input length.
length_list = list()
for token_seq in tokenized_dec_input:
    length_list.append( len( token_seq ))
    
max_output_length = np.array( length_list ).max()
print( 'Reply tweet max length is {}'.format( max_output_length ))
length_list = None

# Pad the decoder input.
padded_dec_input = preprocessing.sequence.pad_sequences( tokenized_dec_input , maxlen=max_output_length, padding='post' )
decoder_input_data = np.array( padded_dec_input )
print( 'Decoder input data shape -> {}'.format( decoder_input_data.shape ))

dec_input_dict = tokenizer.word_index
num_dec_input_tokens = len( dec_input_dict )+1
print( 'Number of Answer tokens = {}'.format( num_dec_input_tokens))

tokenizer = None


# In[ ]:


# Prepare target data for the decoder (the end result)

decoder_target_data = list()
for token_seq in tokenized_dec_input:
    decoder_target_data.append( token_seq[ 1 : ] ) 
    
    
tokenized_dec_input = None
padded_dec_input = preprocessing.sequence.pad_sequences( decoder_target_data , maxlen=max_output_length, padding='post' )




# In[ ]:


onehot_dec_output = utils.to_categorical( padded_dec_input , num_dec_input_tokens )
padded_dec_input = None
try:
    decoder_target_data = np.array( onehot_dec_output )
except e:
      pass
print( 'Decoder target data shape -> {}'.format( decoder_target_data.shape ))


# In[ ]:


encoder_inputs = tf.keras.layers.Input(shape=( None , ))
encoder_embedding = tf.keras.layers.Embedding( num_enc_input_tokens, 256 , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 128 , return_state=True  )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
decoder_embedding = tf.keras.layers.Embedding( num_dec_input_tokens, 256 , mask_zero=True) (decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM( 128 , return_state=True , return_sequences=True)
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = tf.keras.layers.Dense( num_dec_input_tokens , activation=tf.keras.activations.softmax ) 
output = decoder_dense ( decoder_outputs )

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

model.summary()


# In[ ]:


model.fit([encoder_input_data , decoder_input_data], decoder_target_data, epochs=80 ) 


# In[ ]:


def make_inference_models():
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 128 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 128 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model


# In[ ]:


def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( enc_input_dict[ word ] ) 
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=max_input_length , padding='post')


# In[ ]:


# To see the vocabulary. 
print(enc_input_dict.keys())


# In[ ]:


# Test the model, by typing in example sentences:
enc_model , dec_model = make_inference_models()

for epoch in range( encoder_input_data.shape[0] ):
    inp =  applyStem(input( 'How can I help? ' ))
    if inp == "end":
        break
    final_input = ""
    for word in inp.split(' '):
        if word in enc_input_dict.keys():
            final_input += " "
            final_input += word
            
    states_values = enc_model.predict( str_to_tokens( final_input) )
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = dec_input_dict['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word , index in dec_input_dict.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format( word )
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length:
            stop_condition = True
            
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ] 

    print( decoded_translation )
    
    


# In[ ]:


# Test the model by looking at the same data.
enc_model , dec_model = make_inference_models()

passes = 0
total = 0
for epoch in range( encoder_input_data.shape[0] ):
    sampled_arr = [1]
    words_expected = ""
    expected = (decoder_input_data[ epoch ])
    e = np.zeros( ( 1 , 1 ) )
    states_values = enc_model.predict(e+encoder_input_data[ epoch ] )
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = dec_input_dict['start']
    stop_condition = False
    decoded_translation = ' start'
    
    for e in expected:
        for word , index in dec_input_dict.items() :
            if e == index :
                words_expected += ' {}'.format( word )
            
                
    while not stop_condition :
        to = [ empty_target_seq ] + states_values
        dec_outputs , h , c = dec_model.predict(to )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        sampled_arr.append(sampled_word_index)
        
        for word , index in dec_input_dict.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format( word )
                sampled_word = word
                
        if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length:
            stop_condition = True
            
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ] 

    total += 1
    
    # Uncomment the below code to see exactly what is being tested.
    # print("Expected:")
    # print()
    # print(expected)
    # print("( " + words_expected+ ")")
    # print("and Predicted: ")
    # print( sampled_arr)
    # print("( " + decoded_translation + ")")
    if (words_expected== decoded_translation):
        passes += 1
        # print("PASS \n")
    #else:
        # print("FAIL \n")
        
print(str(passes) + " passed!")
print("out of " + str(total) + " tests!")

percent  = str(passes/total)
print(percent+"%")

