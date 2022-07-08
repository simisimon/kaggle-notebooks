#!/usr/bin/env python
# coding: utf-8

# --**edits by [@Federico CABRERA PAEZ](https://www.kaggle.com/federicocabrerapaez): This noteook was originally published by [@anasofiauzsoy](https://www.kaggle.com/anasofiauzsoy) under this [link](https://www.kaggle.com/anasofiauzsoy/tutorial-notebook) as a tutorial for the ["Contradictory, My Dear Watson" competition](https://www.kaggle.com/c/contradictory-my-dear-watson/overview). 
# I will mark all my edits in the text with bold and with # FC: in the code.**--
# 
# 
# Natural Language Inferencing (NLI) is a classic NLP (Natural Language Processing) problem that involves taking two sentences (the _premise_ and the _hypothesis_ ), and deciding how they are related- if the premise entails the hypothesis, contradicts it, or neither.
# 
# In this tutorial we'll look at the _Contradictory, My Dear Watson_ competition dataset, build a preliminary model using Tensorflow 2, Keras, and BERT, and prepare a submission file.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


os.environ["WANDB_API_KEY"] = "0" ## 0 to silence warning of Weights & Biases


# In[ ]:


import matplotlib.pyplot as plt
import tensorflow as tf


# ## Downloading Data

# The training set contains a premise, a hypothesis, a label (0 = entailment, 1 = neutral, 2 = contradiction), and the language of the text. For more information about what these mean and how the data is structured, check out the data page: https://www.kaggle.com/c/contradictory-my-dear-watson/data

# In[ ]:


#FC to download the data outside of kaggle
#!kaggle competitions download -c contradictory-my-dear-watson 


# In[ ]:


train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")
test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")


# We can use the pandas head() function to take a quick look at the training set.

# In[ ]:


train.head()


# In[ ]:


#FC : only French texts
train[train.lang_abv=='fr'].head()


# Let's look at one of the pairs of sentences.

# In[ ]:


print("Entailment example","\n")
print(train.premise.values[2])
print(train.hypothesis.values[2])
print(train.label.values[2])


# In[ ]:


print("Neutral example","\n")
print(train.premise.values[69])
print(train.hypothesis.values[69])
print(train.label.values[69])


# In[ ]:


print("Contradictory example","\n")
print(train.premise.values[41])
print(train.hypothesis.values[41])
print(train.label.values[41])


# Let's look at the size of the dataset and the distribution of languages in the training set.

# Dataset from https://www.kaggle.com/hugoarmandopazvivas/contradictory-my-dear-watson-hapv

# In[ ]:


#!pip3 install datasets version==1
#!pip3 install sentencepiece


# We are going to import more data to train our model 

# In[ ]:


get_ipython().system('pip install nlp')
from nlp import load_dataset


# In[ ]:


def load_mnli(use_validation=True):
    result = []
    dataset = load_dataset('multi_nli')
    print(dataset['train'])
    keys = ['train', 'validation_matched','validation_mismatched'] if use_validation else ['train']
    for k in keys:
        for record in dataset[k]:
            c1, c2, c3 = record['premise'], record['hypothesis'], record['label']
            if c1 and c2 and c3 in {0,1,2}:
                result.append((c1,c2,c3,'en'))
    result = pd.DataFrame(result, columns=['premise','hypothesis', 'label','lang_abv'])
    return result


# In[ ]:


mnli = load_mnli()


# In[ ]:


total_train = train[['id', 'premise', 'hypothesis','lang_abv', 'language', 'label']]
total_train


# In[ ]:


mnli = mnli[['premise', 'hypothesis', 'lang_abv', 'label']]
mnli.insert(0, 'language', 'English')
mnli = mnli[['premise', 'hypothesis', 'lang_abv', 'language', 'label']]
mnli.insert(0, 'id', 'xxx')
mnli


# In[ ]:


total_train = pd.concat([total_train, mnli], axis = 0)
total_train


# In[ ]:


train.describe(include='all')


# In[ ]:


test.describe(include='all')


# In[ ]:


labels, frequencies = np.unique(train.language.values, return_counts = True)

plt.figure(figsize = (10,10))
plt.pie(frequencies,labels = labels, autopct = '%1.1f%%')
plt.show()


# In[ ]:


import seaborn as sns
#explore the distribution of classes and languages
fig, ax = plt.subplots(figsize = (12,5))

#for maximum aesthetics
palette = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True)

graph1 = sns.countplot(train['language'], hue = train['label'])#, palette = palette)

#set title
graph1.set_title('Distribution of Languages and Labels')

plt.tight_layout()
plt.show()


# ## Preparing Data for Input

# --**The tutorial use a pre-trained BERT model combined with a classifier to make the predictions. In order to have better results than the Tutorial we will use a bigger model with the same architecture call RoBERTa (Robust optimization BERT), this model use basically 10 times more data and train more time than the original model** 
# ![image.png](attachment:bb914455-66cd-428c-9e22-28c4d1ea2f61.png)!
# 
# **First, the input text data will be fed into the model (after they have been converted into tokens), which in turn outputs embbeddings of the words. The advantage to other types of embeddings is that the BERT embeddings (or RoBERTa embeddings because is the same architecture) are contextualized. Predictions with contextualized embeddings are more accurate than with non-contextualized embeddings.**
# 
# **After we receive the embeddings for the words in our text from RoBERTa, we can input them into the classifier which will then in turn return the prediction labels 0,1, or 2.**--
# 
# To start out, we can use a pretrained model. Here, we'll use a multilingual BERT model from huggingface. For more information about BERT, see: https://github.com/google-research/bert/blob/master/multilingual.md
# 
# First, we download the tokenizer.
# 
# --** We will first break down our text into tokens by using RoBERTa own tokenizer using AutoTokenizer and the model name in order for the tokenizer to know how to tokenize. This will download all the necessary files. This model includes more than 100 languages which is useful since our data also contains multiple languages.**--

# In[ ]:


from transformers import BertTokenizer, TFBertModel, TFAutoModel,AutoTokenizer

#model_name = "bert-base-multilingual-cased"
#tokenizer = BertTokenizer.from_pretrained(model_name) # FC: this is the tokenizer we will use on our text data to tokenize it

model_name = "joeddav/xlm-roberta-large-xnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Tokenizers turn sequences of words into arrays of numbers. Let's look at an example:

# In[ ]:


list(tokenizer.tokenize("I love machine learning")) # FC: tokenize only create a list of words


# In[ ]:


# FC we make a function in order to have a list of the id for each word and the separator 
def encode_sentence(s):
   tokens = list(tokenizer.tokenize(s)) # FC: split the sentence into tokens that are either words or sub-words
   tokens.append('[SEP]') # FC: a token called [SEP] (=separator) is added to mark end of each sentence
   return tokenizer.convert_tokens_to_ids(tokens) # FC: instead of returning the list of tokens, a list of each token ID is returned


# In[ ]:


encode_sentence("I love machine learning") # FC: the output is a number for each word plus the ID for the [SEP] token


# BERT uses three kind of input data- input word IDs, input masks, and input type IDs.
# 
# These allow the model to know that the premise and hypothesis are distinct sentences, and also to ignore any padding from the tokenizer.
# 
# We add a [CLS] token to denote the beginning of the inputs, and a [SEP] token to denote the separation between the premise and the hypothesis. We also need to pad all of the inputs to be the same size. For more information about BERT inputs, see: https://huggingface.co/transformers/model_doc/bert.html#tfbertmodel
# 
# **To prepare our token IDs which we received through our function encode_sentence() to become input into BERT, we first concatenate the token ID list of each sentence in the hypothesis and premise column (remember that each sentence is already separated by the [SEP] token at the end which we added above) into one list and add another token ID at the very beginning (the ID for the token '[CLS]') which denotes the beginning. However, the output list of encode_sentence() will have a different length for each sentence from our data set which means we can't just construct a nice table out of all those concatenated lists to feed them to BERT. Instead, we will have to add zeros at the end of each ID list until it has the length of the longest list in the data set (corresponding to the longest hypothesis/premise pair). This process is called padding. This way, every ID list will have the same length and BERT will be able to accept them as input. This will be the variable "input word IDs" mentioned above and our first input variable for the BERT model.**
# 
# **However, we also need to tell BERT which of the IDs in the "input word IDs" actually belong to tokens that it should embed and which of them it should ignore because they are just padding. This is where input masks come into play: The input mask variable has the same size as the input word IDs variable but contains a 1 for each entry that is an actual token ID (which BERT should consider) and a 0 for eacht entry that is just padding and which BERT should ignore. This will be our second input argument "input masks" for the BERT model.**
# 
# **Lastly, BERT (but not RoBERTa because it wasn't train in predicting the likelihood that sentence B belongs after sentence A) also needs to know which of the input word IDs belong to which sentence (i.e., hypthesis or premise). We can explain this to BERT by using the variable 'input type IDs' mentioned above. Again, this variable has the same size as as the input word IDs variable but this time it contains a 1 for each entry that belongs to sentence B (i.e., the premise) and a 0 for each entry that belongs to sentence A (i.e., the hypothesis), including our start token ID for '[CLS]'. "Input type IDs" will be our third and last input argument for BERT.**--
# 
# Now, we're going to encode all of our premise/hypothesis pairs for input into BERT.

# In[ ]:


def bert_encode(hypotheses, premises, tokenizer): # FC: for RoBERTa we remove the input_type_ids from the inputs of the model
    
  num_examples = len(hypotheses)
  
  sentence1 = tf.ragged.constant([   # FC: constructs a constant ragged tensor. every entry has a different length
      encode_sentence(s) for s in np.array(hypotheses)])
  
  sentence2 = tf.ragged.constant([
      encode_sentence(s) for s in np.array(premises)])
  
  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0] # FC: list of IDs for the token '[CLS]' to denote each beginning
  
  input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1) # FC: put everything together. every row still has a different length.
  
  #input_word_ids2 = tf.concat([cls, sentence2, sentence1], axis=-1)
  
  #input_word_ids = tf.concat([input_word_ids1, input_word_ids2], axis=0) # we duplicate the dataset inverting sentence 1 and 2
    
  input_mask = tf.ones_like(input_word_ids).to_tensor() # FC: first, a tensor with just ones in it is constructed in the same size as input_word_ids. Then, by applying to_tensor the ends of each row are padded with zeros to give every row the same length

  # type is not need for the RoBERTa model it will not be include in the output of this function
  type_cls = tf.zeros_like(cls) # FC: creates a tensor same shape as cls with only zeros in it
  
  type_s1 = tf.zeros_like(sentence1)
  
  type_s2 = tf.ones_like(sentence2) # FC: creates a tensor same shape as sentence2 with only ones in it to mark the 2nd sentence
  
  input_type_ids = tf.concat(
      [type_cls, type_s1, type_s2], axis=-1).to_tensor() # FC: concatenates everything and again adds padding 
  
  inputs = {
      'input_word_ids': input_word_ids.to_tensor(), # FC: input_word_ids hasn't been padded yet - do it here now
      'input_mask': input_mask
      
      #,'input_type_ids': input_type_ids
  }

  return inputs


# In[ ]:


train_input = bert_encode(train.premise.values, train.hypothesis.values, tokenizer)


# In[ ]:


train_input


# In[ ]:


total_train_input = bert_encode(total_train.premise.values, total_train.hypothesis.values, tokenizer)


# In[ ]:


train.label.values.shape


# In[ ]:


total_train.label.values.shape


# In[ ]:


print(np.count_nonzero(train_input['input_word_ids'], axis=1))


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


x = np.count_nonzero(train_input['input_word_ids'], axis=1)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=True, facecolor='b', alpha=0.75)


plt.xlabel('input word lenght')
plt.ylabel('Probability')
plt.title('Distribution of word length on the train set')
plt.text(60, .021, r'max_length=245')
plt.xlim(0, 250)
#plt.ylim(0, 0.03)
plt.grid(True)
plt.show()


# In[ ]:


test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


x = np.count_nonzero(test_input['input_word_ids'], axis=1)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=True, facecolor='b', alpha=0.75)


plt.xlabel('input word lenght')
plt.ylabel('Probability')
plt.title('Distribution of word length on the test set')
plt.text(60, .021, r'max_length=236')
plt.xlim(0, 250)
#plt.ylim(0, 0.03)
plt.grid(True)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


x = np.count_nonzero(total_train_input['input_word_ids'], axis=1)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=True, facecolor='b', alpha=0.75)


plt.xlabel('input word lenght')
plt.ylabel('Probability')
plt.title('Distribution of word length on the test set')
plt.text(60, .021, r'max_length=236')
plt.xlim(0, 250)
#plt.ylim(0, 0.03)
plt.grid(True)
plt.show()


# ## Creating & Training Model

# Now, we can incorporate the BERT transformer into a Keras Functional Model. For more information about the Keras Functional API, see: https://www.tensorflow.org/guide/keras/functional.
# 
# This model was inspired by the model in this notebook: https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert#BERT-and-Its-Implementation-on-this-Competition, which is a wonderful introduction to NLP!
# 
# --**Now, we are ready to build the actual model. As mentioned above, the final model will consist of a RoBERTa Large model that performs contextual embedding of the input token IDs which are then passed to a classifier that will return probabilites for each of the possible three labels "entailment" (0), "neutral" (1), or "contradiction" (2). The classifier consists of a regular densely-connected neural network.**--
# 
# ![image.png](attachment:d8ff3ce5-2ade-4c35-a477-7bcee08bff69.png)

# In[ ]:


max_len = 236 #: FC 50 in the initial tutorial

def build_model():
    #encoder = TFBertModel.from_pretrained(model_name) 
    # FC: constructs a RoBERTa model pre-trained on the above described language model 'xlm-roberta-large-xnli'
    encoder = TFAutoModel.from_pretrained('joeddav/xlm-roberta-large-xnli')
    # FC: now we adjust the model so that it can accept our input by telling the model what the input looks like:
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids") # FC: tf.keras.Input constructs a symbolic tensor object whith certain attributes: "shape" tells it that the expected input will be in batches of max_len-dimensional vectors; "dtype" tells it that the data type will be int32; "name" will be the name string for the input layer
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask") # FC: repeat the same for the other two input variables
    # FC: the input type is only needed for the BERT model
    #input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")
    
    # FC: now follows, what we want to happen with our input:
    # FC: first, our input goes into the BERT model bert_encoder. It will return a tuple and the contextualized embeddings that we need are stored in the first element of that tuple
    embedding = encoder([input_word_ids, input_mask])[0] # FC: add_input_type_ids for the BERT model
    # FC: we only need the output corresponding to the first token [CLS], which is a 2D-tensor with size (#sentence pairs, 768) and is accessd with embedding[:,0,:]. This will be input for our classifier, which is a regular densely-connected neural network constructed through tf.keras.layers.Dense. The inputs mean: "3" is the dimensionality of the output space, which means that the output has shape (#sentence pairs,3). More practically speaking, for each sentence pair that we input, the output will have 3 probability values for each of the 3 possible labels (entailment, neutral, contradiction). They will be in range(0,1) and add up to 1; "activation" denotes the activation function, in this case 'softmax', which connects a real vector to a vector of categorical possibilities.
    
    # I tried to put another layer put it doesn't help in performance
    #output = tf.keras.layers.Dense(10, activation='softmax')(embedding[:,0,:]) #FC: no need of a GlobalAveragePooling for BERT
    
    output = tf.keras.layers.Dense(3, activation='softmax')(embedding[:,0,:])
    # FC: we also have the posibility of making a globalAveragepooling of all the embeddings, but the resuls are not better 
    #output = tf.keras.layers.GlobalAveragePooling1D()(embedding) 
    #output = tf.keras.layers.Dense(3, activation='softmax')(output) 
    
       
    model = tf.keras.Model(inputs=[input_word_ids, input_mask], outputs=output) # FC: based on the code in the lines above, a model is now constructed and passed into the variable model
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy']) # FC: we tell the model how we want it to train and evaluate: "tf.keras.optimizers.Adam": use an optimizer that implements the Adam algorithm. "lr" denotes the learning rate; "loss" denotes the loss function to use; "metrics" specifies which kind of metrics to use for training and testing
    
    return model 


# Why do we only need embedding[0][:,0,:] from the BERT's output?**
# 
# **[Here](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) is a wonderful blog article (including a notebook) which explains this in great detail (see in particular the section "Unpacking the BERT output tensor" for an illustrative visualization). I will try to give a short summary here: embedding[0] consists of a 3D tensor with a number (embedding) for each token in the sentence pair (columns) for each sentence pair (rows) for each hidden unit in BERT (768). Since he output corresponding to the first token [CLS] (a.k.a. embedding[0][:,0,:], this is a 2D tensor with size (#sentence pairs, 768)) can be thought of as an embedding for each individual sentence pair, it is enough to just use that as the input for our classification model.**--

# Let's set up our TPU.
# 
# More info about setting up the TPU can be found on this [Kaggle documentation page](https://www.kaggle.com/docs/tpu#tpu1).**--

# In[ ]:


try:
    # detect and init the TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # FC: detect and init the TPU: TPUClusterResolver() locates the TPUs on the network
    # instantiate a distribution strategy
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    
    strategy = tf.distribute.experimental.TPUStrategy(tpu) # FC: "strategy" contains the necessary distributed training code that will work on the TPUs
except ValueError: # FC: in case Accelerator is not set to TPU in the Notebook Settings
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync) # FC: returns the number of cores


# In[ ]:


try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
  tpu = None
  gpus = tf.config.experimental.list_logical_devices("GPU")

if tpu:
  tf.tpu.experimental.initialize_tpu_system(tpu)
  strategy = tf.distribute.experimental.TPUStrategy(tpu,) # Going back and forth between TPU and host is expensive. Better to run 128 batches on the TPU before reporting back.
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
elif len(gpus) > 1:
  strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
  print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
  strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
  print('Running on single GPU ', gpus[0].name)
else:
  strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
  print('Running on CPU')
print("Number of accelerators: ", strategy.num_replicas_in_sync)


# In[ ]:


# instantiating the model in the strategy scope creates the model on the TPU

with strategy.scope(): # FC: defines the compute distribution policy for building the model. or in other words: makes sure that the model is created on the TPU/GPU/CPU, depending on to what the Accelerator is set in the Notebook Settings
    model = build_model() # FC: our model is being built
    model.summary()       # FC: let's look at some of its properties

tf.keras.utils.plot_model(model, "my_model.png", show_shapes=True) # FC: I added this line because it gives a nice visualization showing the individual components of our model


# --**The graph above illustrates in a very detailed way what our model and its inputs look like: input_word_ids, input_mask, and input_type_ids are the 3 input variables for the BERT model, which in turn returns a tuple. The word embeddings that are stored in the first entry of the tuple are then given to the classifier which then returns 3 categorical probabilities. The question marks stand for the number of rows in the input data which are of course unknown.**--

# In[ ]:


# We can freeze the RoBERTa weights in order to save some time
print(model.layers[2])
model.layers[2].trainable=True


# In[ ]:


# We need to put the train set with the same size of the model
for key in train_input.keys():
    train_input[key] = train_input[key][:,:max_len]


# In[ ]:


# We need to put the train set with the same size of the model
for key in total_train_input.keys():
    total_train_input[key] = total_train_input[key][:,:max_len]


# In[ ]:


early_stop = tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
# FC: make sure that TPU in Accelerator under Notebook Settings is turned on so that model trains on the TPU. Otherwise this line will crash
model.fit(total_train_input, total_train.label.values, epochs = 10, verbose = 1, validation_split = 0.01,
         batch_size=16*strategy.num_replicas_in_sync
          ,callbacks=[early_stop]
         ) # FC: now we fit the model to our training data that we prepared before. The number of training epochs is 2, verbose = 1 shows progress bar, # of rows in each batch is 64, and 20% of the data is used for validation


# In[ ]:


test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")
test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer) # FC: finally we prepare our competition data for the model


# In[ ]:


# same for the test set we need to put it in the same size of the model
for key in test_input.keys():
    test_input[key] = test_input[key][:,:max_len]


# In[ ]:


test.head()


# ## Generating & Submitting Predictions

# In[ ]:


predictions = [np.argmax(i) for i in model.predict(test_input)] # FC; ve the model predict three categorical probabilities, choose the highest probability, and save the respective label ID (0,1, or 2)


# The submission file will consist of the ID column and a prediction column. We can just copy the ID column from the test file, make it a dataframe, and then add our prediction column.

# In[ ]:


submission = test.id.copy().to_frame()
submission['prediction'] = predictions


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv", index = False)


# And now we've created our submission file, which can be submitted to the competition. Good luck!
