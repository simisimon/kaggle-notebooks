#!/usr/bin/env python
# coding: utf-8

#  #  <div style="text-align: center">Gendered Pronoun Explainability  </div> 
# <img src='https://storage.googleapis.com/kaggle-media/competitions/GoogleAI-GenderedPronoun/PronounResolution.png' width=600 height=600>
# <div style="text-align:center"> last update: <b>10/02/2019</b></div>
# 
# 
# 
# You can Fork code  and  Follow me on:
# 
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# > ###### [Kaggle](https://www.kaggle.com/mjbahmani/)
# -------------------------------------------------------------------------------------------------------------
#  <b>I hope you find this kernel helpful and some <font color='red'>UPVOTES</font> would be very much appreciated.</b>
#     
#  -----------

#  <a id="top"></a> <br>
# ## Notebook  Content
# 1. [Introduction](#1)
# 1. [Load packages](#2)
#     1. [import](21)
#     1. [Setup](22)
#     1. [Version](23)
# 1. [Problem Definition](#3)
#     1. [Problem Feature](#31)
#     1. [Aim](#32)
#     1. [Variables](#33)
#     1. [Evaluation](#34)
# 1. [Exploratory Data Analysis(EDA)](#4)
#     1. [Data Collection](#41)
#     1. [Visualization](#42)
#     1. [Data Preprocessing](#43)
#         1. [Some new features](#431)
# 1. [Conclusion](#5)

#  <a id="1"></a> <br>
# ## 1- Introduction
# Pronoun resolution is part of coreference resolution, the task of pairing an expression to its referring entity. This is an important task for natural language understanding, and the resolution of ambiguous pronouns is a longstanding challenge.

#  <a id="2"></a> <br>
#  ## 2- Load packages
#   <a id="21"></a> <br>
# ## 2-1 Import

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.patches as patch
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
from scipy.stats import norm
from scipy import linalg
from sklearn import svm
import tensorflow as tf
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import glob
import sys
import os


#  <a id="22"></a> <br>
# ##  2-2 Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('precision', '4')
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
np.set_printoptions(suppress=True)
pd.set_option("display.precision", 15)


#  <a id="23"></a> <br>
# ## 2-3 Version
# 

# In[ ]:


print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))


# <a id="3"></a> 
# <br>
# ## 3- Problem Definition
# I think one of the important things when you start a new machine learning project is Defining your problem. that means you should understand business problem.( **Problem Formalization**)
# 
# Problem Definition has four steps that have illustrated in the picture below:
# <img src="http://s8.picofile.com/file/8338227734/ProblemDefination.png">
# 
# **Kagglers** are challenged to build pronoun resolution systems that perform equally well regardless of pronoun gender. Stage two's final evaluation will use a new dataset following the same format.
# 

# <a id="31"></a> 
# ### 3-1 Problem Feature
# In this competition, you must identify the target of a pronoun within a text passage. The source text is taken from Wikipedia articles. You are provided with the pronoun and two candidate names to which the pronoun could refer. You must create an algorithm capable of deciding whether the pronoun refers to name A, name B, or neither.
# 

# <a id="32"></a> 
# ### 3-2 Aim
# Unlike many **Kaggle challenges**, this competition does not provide an explicit labeled training set. Files are also available on the GAP Dataset Github Repo. Note that the labels for the test set are available on this page. However, your final score and ranking will be determined in stage 2, against a withheld private test set.
# 
# 1. test_stage_1.tsv - the test set data for stage 1
# 1. sample_submission_stage_1.csv - a file showing the correct submission format for stage 1

# <a id="33"></a> 
# ### 3-3 Variables
# 
# 1. ID - Unique identifier for an example (Matches to Id in output file format)
# 1. Text - Text containing the ambiguous pronoun and two candidate names (about a paragraph in length)
# 1. Pronoun - The target pronoun (text)
# 1. Pronoun-offset The character offset of Pronoun in Text
# 1. A - The first name candidate (text)
# 1. A-offset - The character offset of name A in Text
# 1. B - The second name candidate
# 1. B-offset - The character offset of name B in Text
# 1. URL - The URL of the source Wikipedia page for the example
# 

# <a id="34"></a> 
# ## 3-4 evaluation
# Submissions are evaluated using the multi-class logarithmic loss. Each pronoun has been labeled with whether it refers to A, B, or NEITHER. For each pronoun, you must submit a set of predicted probabilities (one for each class). The formula is :
# <img src='http://s8.picofile.com/file/8351608076/1.png'>

# <a id="4"></a> 
# ## 4- Exploratory Data Analysis(EDA)
#  In this section, we'll analysis how to use graphical and numerical techniques to begin uncovering the structure of your data. 
#  
# * Which variables suggest interesting relationships?
# * Which observations are unusual?
# * Analysis of the features!
# 
# By the end of the section, you'll be able to answer these questions and more, while generating graphics that are both insightful and beautiful.  then We will review analytical and statistical operations:
# 
# *  Data Collection
# *  Visualization
# *  Data Preprocessing
# *  Data Cleaning

#  <a id="41"></a> <br>
# ## 4-1 Data Collection

# In[ ]:


print(os.listdir("../input/"))


# In[ ]:


# import Dataset to play with it
test = pd.read_csv('../input/test_stage_1.tsv', delimiter='\t')


# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission_stage_1.csv')
sample_submission.head()


# In[ ]:


test.head()


# In[ ]:


print(test.columns)


# In[ ]:


test.describe()


# In[ ]:


test.shape


# In[ ]:


test.isna().sum()


# In[ ]:


type(test)


#  <a id="42"></a> <br>
# ## 4-2 Visualization

# <a id="421"></a> 
# ### 4-2-1 hist

# In[ ]:


#acoustic_data means signal
test.hist();


# <a id="422"></a> 
# ### 4-2-2 scatter_matrix

# In[ ]:


pd.plotting.scatter_matrix(test,figsize=(10,10))
plt.figure();


# <a id="423"></a> 
# ### 4-2-3 jointplot

# In[ ]:


sns.jointplot(x='Pronoun-offset',y='A-offset' ,data=test, kind='reg')


# <a id="424"></a> 
# ### 4-2-4 Scatter_matrix

# In[ ]:


sns.swarmplot(x='Pronoun-offset',y='B-offset',data=test);


# <a id="425"></a> 
# ### 4-2-5 WordCloud

# In[ ]:


from wordcloud import WordCloud as wc
from nltk.corpus import stopwords
def generate_wordcloud(text): 
    wordcloud = wc(relative_scaling = 1.0,stopwords = eng_stopwords).generate(text)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.margins(x=0, y=0)
    plt.show()


# In[ ]:


from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))


# In[ ]:


text =" ".join(test.Text)
generate_wordcloud(text)


# <a id="426"></a> 
# ### 4-2-6 Distplot

# In[ ]:


sns.distplot(test["Pronoun-offset"])


# <a id="427"></a> 
# ### 4-2-7 kdeplot

# In[ ]:


sns.kdeplot(test["Pronoun-offset"] )


#  <a id="43"></a> <br>
# ## 4-3 Data Preprocessing

# In[ ]:


test.head()


# In[ ]:


test.describe()


# In[ ]:


test.Pronoun.tail()


# In[ ]:


test.shape


# In[ ]:


test.isna().sum()


# <a id="431"></a> <br>
# ## 4-3-1 Some new features
# In this section, I will extract a few new statistical features from the text field

# ### Number of words in the text

# In[ ]:


test["num_words"] = test["Text"].apply(lambda x: len(str(x).split()))


# In[ ]:


#MJ Bahmani
print('maximum of num_words in test',test["num_words"].max())
print('min of num_words in test',test["num_words"].min())


# ### Number of unique words in the text

# In[ ]:


test["num_unique_words"] = test["Text"].apply(lambda x: len(set(str(x).split())))
print('maximum of num_unique_words in test',test["num_unique_words"].max())
print('mean of num_unique_words in test',test["num_unique_words"].mean())


# ### Number of characters in the text

# In[ ]:


test["num_chars"] = test["Text"].apply(lambda x: len(str(x)))
print('maximum of num_chars in data_df',test["num_chars"].max())


# ### Number of stopwords in the text

# In[ ]:


from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))


# In[ ]:


test["num_stopwords"] = test["Text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

print('maximum of num_stopwords in test',test["num_stopwords"].max())


# ### Number of punctuations in the text
# 

# In[ ]:


import string
test["num_punctuations"] =test['Text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
print('maximum of num_punctuations in data_df',test["num_punctuations"].max())


# ### Number of title case words in the text

# In[ ]:


test["num_words_upper"] = test["Text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
print('maximum of num_words_upper in test',test["num_words_upper"].max())


# In[ ]:


print(test.columns)
test.head(1)


# In[ ]:


pronoun=test["Pronoun"]


# In[ ]:


np.unique(pronoun)


# In[ ]:


test["Pronoun_binary"] = test["Pronoun"]


# In[ ]:


test["Pronoun_binary"]=test["Pronoun_binary"].str.replace('He','0')
test["Pronoun_binary"]=test["Pronoun_binary"].str.replace('he','0')
test["Pronoun_binary"]=test["Pronoun_binary"].str.replace('she','1')
test["Pronoun_binary"]=test["Pronoun_binary"].str.replace('She','1')
test["Pronoun_binary"]=test["Pronoun_binary"].str.replace('His','2')
test["Pronoun_binary"]=test["Pronoun_binary"].str.replace('his','2')
test["Pronoun_binary"]=test["Pronoun_binary"].str.replace('him','3')
test["Pronoun_binary"]=test["Pronoun_binary"].str.replace('her','4')
test["Pronoun_binary"]=test["Pronoun_binary"].str.replace('Her','4')


# In[ ]:


sns.violinplot(data=test,x="Pronoun_binary", y="num_words")


# you can follow me on:
# > ###### [ GitHub](https://github.com/mjbahmani/)
# > ###### [Kaggle](https://www.kaggle.com/mjbahmani/)
# 
#  <b>I hope you find this kernel helpful and some <font color='red'>UPVOTES</font> would be very much appreciated.<b/>
#  

# Go to first step: [Course Home Page](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# Go to next step : [Titanic](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
# 

# # Not Completed yet!!!
