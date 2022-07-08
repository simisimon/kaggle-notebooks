#!/usr/bin/env python
# coding: utf-8

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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #All script by Ong Jia Ying https://www.kaggle.com/code/jyingong/data-cleaning-sentiment-analysis-reddit/notebook 

# #Loading Packages

# In[ ]:


#Code by Ong Jia Ying https://www.kaggle.com/code/jyingong/data-cleaning-sentiment-analysis-reddit

# import praw
# from praw.models import MoreComments
import datetime
import pandas as pd
import numpy as np
import re
import pprint
pd.options.mode.chained_assignment = None

from textblob import TextBlob

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import tokenize
import spacy
#conda install -c conda-forge spacy-model-en_core_web_sm
import en_core_web_sm
nlp = en_core_web_sm.load()
nltk.download('averaged_perceptron_tagger')

import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import seaborn as sns

import emoji


# In[ ]:


df_comment = pd.read_csv("../input/end-of-usafghan-war-tweet-data/US-Afghan_war_tweets.csv")
df_comment.head()


# In[ ]:


#check number of rows.  
len(df_comment)


# In[ ]:


#check for for any empty cells 
df_comment.isna().sum()


# In[ ]:


#I made that to avoid the next Error ('float' object has no attribute 'split'). Anyway it didn't work.

#AttributeError: 'float' object has no attribute 'split'

df_comment['text'].apply(str)


# In[ ]:


#split sentences by delimited "\n\n"
 
df_comment["text"] = [item.split("\n\n") for item in df_comment.text]

#explode nested list into individual rows 
df_comment = df_comment.explode("text").rename_axis("index_name").reset_index()

#replace double space with empty string
df_comment["text"] = df_comment.text.str.replace("&#x200B;", "")


# #Dealing with Missing values

# In[ ]:


#Code by Parul Pandey  https://www.kaggle.com/parulpandey/a-guide-to-handling-missing-values-in-python


from sklearn.impute import SimpleImputer
df_most_frequent = df_comment.copy()
#setting strategy to 'mean' to impute by the mean
mean_imputer = SimpleImputer(strategy='most_frequent')# strategy can also be mean or median 
df_most_frequent.iloc[:,:] = mean_imputer.fit_transform(df_most_frequent)


# In[ ]:


df_most_frequent.isnull().sum()


# In[ ]:


#for replies with parent comments within, remove parent comment and retain replies  
#those are fields with string that start with ">"
#remove bullet points 
df_most_frequent.loc[df_most_frequent.text.str.startswith(">")] = ""
df_most_frequent["comment"] = [i.strip() for i in df_most_frequent.text]
df_most_frequent["comment"] = [re.sub(r"^[0-9]", " ", i) for i in df_most_frequent.text]


# In[ ]:


#see table of items with https links and markup links 
df_most_frequent.loc[df_most_frequent.comment.str.contains("https")]


# In[ ]:


#I don't have a 2nd csv file (details csv). Save for next time.

#do a temporary table to see the usernames for these comments 

df_temp_https = df_most_frequent[df_most_frequent.comment.str.contains("https")]
#df_temp_details = df_details.reset_index() #Original has 2 csv files
#df_temp = pd.merge(df_temp_https, df_temp_details, how = "inner", left_on = "index_name", right_on = "index")
#df_temp
df_temp_https


# In[ ]:


# define function to remove both links and markup links 
# also remove \' from dont\'t
def remove_https(item):
    #remove https links
    item_1 = re.sub(r"[(+*)]\S*https?:\S*[(+*)]", "", item)
    #remove https links with no brackets
    item_2 = re.sub('http://\S+|https://\S+', " ", item_1)
    #remove link markups []
    #note that this will also remove comment fields with ["Delete"] 
    item_3 = re.sub(r"[\(\[].*?[\)\]]", " ", item_2)
#     #remove \ in don\'t
#     item_4 = re.sub("[\"\']", "'", item_3)
    return item_3

df_most_frequent["text"] = [remove_https(x) for x in df_most_frequent.text]


# #It was suppose to be Username. User_location there are many and takes a long time. Save it for the next time.

# In[ ]:


#check the temporary table to see if links/ markuplinks/ \' 
#all links has been removed
#unecessary comments (highlighted in yellow) can be removed later by filtering out unecessary usernames
df_temp_https["text"] = [remove_https(x) for x in df_temp_https.text]
df_temp_https.style.apply(lambda x: ['background: lightyellow' if x.user_location == "lahore" \
                               or x.user_location =="lahore" else '' for i in x], axis=1)


# In[ ]:


#We don't have index name. Save for next time

#implode and remove column index_name
#df_comment_1 = df_comment.groupby("index_name")["comment"].apply(lambda x: " ".join(x)).reset_index().drop("index_name", axis = 1)


# #Text Clean-Up

# In[ ]:


df= df_most_frequent


# In[ ]:


#ensure that comment attribute is of correct data type
df["comment"] = df.comment.astype("str")
df["comment"] = [item.lower() for item in df.comment]

#remove apostrophe at the beginning and end of each word (e.g. 'like, 'this, or', this')
df["comment"] = [re.sub(r"(\B'\b)|(\b'\B)", ' ', item) for item in df.comment]
df["comment"] = [re.sub(r'…', ' ', item) for item in df.comment]
df["comment"] = [item.replace('\\',' ') for item in df.comment]
df["comment"] = [item.replace('/',' ') for item in df.comment]


# In[ ]:


#TEST
#have a overview/ general sensing of types of contractions we have 
#create a temp list of tokenized sentences 
df_token_temp = [item.split() for item in df["comment"]]
df_token_temp = [i for word in df_token_temp for i in word]
df_contraction_temp = [re.findall("(?=\S*['-])([a-zA-Z'-]+)", i) for i in df_token_temp]
df_contraction_temp_1 = [i for item in df_contraction_temp if item != [] for i in item]
df_contraction_temp_2 = [i for n, i in enumerate(df_contraction_temp_1) if i not in df_contraction_temp_1[:n]]
print(df_contraction_temp_2)


# #Why my test above is empty? I've no clue about what went wrong.

# In[ ]:


#define a function to clean up these contractions 
#for \'s such as he's, she's we will just replace with he and she as is is a stop word and will be removed 
def decontract(phrase):  
    phrase = re.sub(r"can\'t", "cannot", phrase)
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"let\'s", "let us", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'s", "", phrase)
    #"kpkb'ing"
    phrase = re.sub(r"\'ing", "", phrase)
    
    phrase = re.sub(r"can’t", "cannot", phrase)
    phrase = re.sub(r"won’t", "will not", phrase)
    phrase = re.sub(r"let’s", "let us", phrase)
    phrase = re.sub(r"n’t", " not", phrase)
    phrase = re.sub(r"’m", " am", phrase)
    phrase = re.sub(r"’ll", " will", phrase)
    phrase = re.sub(r"’re", " are", phrase)
    phrase = re.sub(r"’d", " would", phrase)
    phrase = re.sub(r"’ve", " have", phrase)
    phrase = re.sub(r"’s", "", phrase)
    #"kpkb'ing"
    phrase = re.sub(r"’ing", "", phrase)
    return phrase


# In[ ]:


#TEST
#test it on df_contraction_temp_2
df_contraction_temp_3 = [decontract(i) for i in df_contraction_temp_2]
print(df_contraction_temp_3)


# #Really?? Another empty test above?

# In[ ]:


#decontract words in dataframe
df["comment"] = [decontract(i) for i in df.comment]


# In[ ]:


#define a function to find all emojis
def extract_emojis(s):
    return ''.join(c for c in s if c in emoji.UNICODE_EMOJI['en'])


# In[ ]:


#Extract the list of emojis to convert
emoji_lst = [extract_emojis(i) for i in df.comment.tolist()]
emoji_lst = list(filter(None, emoji_lst))
emoji_lst 


# #More empty brackets above (again)?

# In[ ]:


#define a function that converts emojis to words/ phrase
def convert_emoji(phrase):
    phrase = re.sub(r"😢", " sad ", phrase)
    phrase = re.sub(r"🤨", " not confident ", phrase)
    phrase = re.sub(r"🙄", " annoying ",  phrase)
    phrase = re.sub(r"😂", " laugh ", phrase)
    return phrase

df["comment"] = [convert_emoji(i) for i in df.comment]


# In[ ]:


#define a function that converts all typos
def clean_typo(phrase):
    phrase = re.sub(r"-ish", "", phrase)
    phrase = re.sub(r"rrcent", "recent", phrase)
    phrase = re.sub(r"govenment", " government ", phrase)
    phrase = re.sub(r"diffit", "definitely", phrase)
    phrase = re.sub(r"overexxagearting", "overexaggerate", phrase)
    phrase = re.sub(r"en-bloc", "enbloc", phrase)
    phrase = re.sub(r" dnt ", "do not", phrase)
    phrase = re.sub(r" underdeclared ", " under declare ", phrase)
    phrase = re.sub(r" lgbt ", " lgbtq ", phrase)
    phrase = re.sub(r" 9wn ", " own ", phrase)
    phrase = re.sub(r" rocher ", " rochor ", phrase)
    phrase = re.sub(r" pinnacle ", " duxton ", phrase)
    phrase = re.sub(r" cdb ", " cbd ", phrase)
    phrase = re.sub(r" hivemind ", " hive mind ", phrase)
    phrase = re.sub(r" claw back ", " clawback ", phrase)
    phrase = re.sub(r" discludes ", " excludes ", phrase)
    phrase = re.sub(r" hugeee ", " huge ", phrase)
    phrase = re.sub(r" birthrate ", " birth rate ", phrase)
    phrase = re.sub(r" oligations ", " obligations ", phrase)
    phrase = re.sub(r" wayyy ", " way ", phrase)
    phrase = re.sub(r" plhs ", " plh ", phrase)
    phrase = re.sub(r" noobhdbbuyer ", " noob hdb buyer ", phrase)
    return phrase
    
df["comment"] = [clean_typo(i) for i in df.comment]


# In[ ]:


#define a function to convert all short-forms/ short terms 
def clean_short(phrase):
    phrase = re.sub(r"fyi", "for your information", phrase)
    phrase = re.sub(r"tbh", "to be honest", phrase)
    phrase = re.sub(r" esp ", " especially ", phrase)
    phrase = re.sub(r" info ", "information", phrase)
    phrase = re.sub(r"gonna", "going to", phrase)
    phrase = re.sub(r"stats", "statistics", phrase)
    phrase = re.sub(r"rm ", " room ", phrase)
    phrase = phrase.replace("i.e.", " ")
    phrase = re.sub(r"idk", "i do not know", phrase)
    phrase = re.sub(r"haha", "laugh", phrase)
    phrase = re.sub(r"yr", " year", phrase)
    phrase = re.sub(r" sg ", " singapore ", phrase)
    phrase = re.sub(r" mil ", " million ", phrase)
    phrase = re.sub(r" =", " same ", phrase)
    phrase = re.sub(r" msr. ", " mortage serving ratio ", phrase)
    phrase = re.sub(r" eip ", " ethnic integration policy ", phrase)
    phrase = re.sub(r" g ", " government ", phrase)
    phrase = re.sub(r"^imo ", " in my opinion ", phrase)
    phrase = re.sub(r" pp ", " private property ", phrase)
    phrase = re.sub(r" grad ", " graduate ", phrase)
    phrase = re.sub(r" ns ", " national service ", phrase)
    phrase = re.sub(r" bc ", " because ", phrase)
    phrase = re.sub(r" u ", " you ", phrase)
    phrase = re.sub(r" ur ", " your ", phrase)
    phrase = re.sub(r"^yo ", " year ", phrase)
    phrase = re.sub(r" vs ", " versus ", phrase)
    phrase = re.sub(r" irl ", " in reality ", phrase)
    phrase = re.sub(r" tfr ", " total fertility rate ", phrase)
    phrase = re.sub(r" fk ", " fuck ", phrase)
    phrase = re.sub(r" fked ", " fuck ", phrase)
    phrase = re.sub(r" fucked ", " fuck ", phrase)
    phrase = re.sub(r".  um.", " cynical. ", phrase)
    phrase = re.sub(r" pre-", " before ", phrase)
    phrase = re.sub(r" ed ", " education ", phrase)
    return phrase

df["comment"] = [clean_short(i) for i in df.comment]


# In[ ]:


#define a function that converts singlish
def singlish_clean(phrase):
    phrase = re.sub(r"yup", " yes", phrase)
    phrase = re.sub(r" yah ", " yes", phrase)
    phrase = re.sub(r"yeah", "yes", phrase)
    phrase = re.sub(r" ya ", "  yes", phrase)
    phrase = re.sub(r"song ah", "good", phrase)
    phrase = re.sub(r" lah", " ", phrase)
    phrase = re.sub(r"hurray", "congratulation", phrase)
    phrase = re.sub(r"^um", "unsure", phrase)
    phrase = re.sub(r" sian ", " tired of ", phrase)
    phrase = re.sub(r" eh", " ", phrase)
    phrase = re.sub(r" hentak kaki ", " stagnant ", phrase)
    phrase = re.sub(r" ulu ", " remote ", phrase)
    phrase = re.sub(r" kpkb ", " complain ", phrase)
    phrase = re.sub(r" leh.", " .", phrase)
    phrase = re.sub(r"sinkies", " rude ", phrase)
    phrase = re.sub(r"sinkie", " rude ", phrase)
    phrase = re.sub(r"shitty", "shit", phrase)
    return phrase

df["comment"] = [singlish_clean(i) for i in df.comment]


# In[ ]:


def others_clean(phrase):
    phrase = re.sub(r" govt ", " government ", phrase)
    phrase = re.sub(r"14 000", "14k", phrase)
    phrase = re.sub(r"14000", "14k", phrase)
    phrase = re.sub(r"14,000", "14k", phrase)
    phrase = re.sub(r"flipper", "flip ", phrase)
    phrase = re.sub(r"flip s", "flip", phrase)
    phrase = re.sub(r"flipping", "flip ", phrase)
    phrase = re.sub(r"hdbs", "hdb", phrase)
    phrase = re.sub(r"dont", "do not", phrase)
    phrase = re.sub(r"cant", "cannot", phrase)
    phrase = re.sub(r"shouldnt", "should not", phrase)
    phrase = re.sub(r"condominiums", "condo ", phrase)
    phrase = re.sub(r"condominium", "condo ", phrase)
    phrase = re.sub(r"btos", "bto", phrase)
    phrase = re.sub(r"non-", "not ", phrase)
    phrase = re.sub(r" x+ ", " ", phrase)
    phrase = re.sub(r" ccr or ", " ", phrase)
    phrase = re.sub(r" its ", " it ", phrase)
    return phrase

df["comment"] = [others_clean(i) for i in df.comment]


# In[ ]:


df.head()


# #Extracting Sentiments using Text Blob
# 
# Negative: polarity < 0
# 
# Positive: polarity > 0
# 
# Neutral: polarity = 0

# In[ ]:


df_sentiment = df
df_sentiment.head()


# In[ ]:


#separate each comment into invididual sentences
df_sentiment["comment"] = [tokenize.sent_tokenize(item) for item in df_sentiment.comment]


# In[ ]:


#split each sentence into individual rows
df_sentiment_1 = df_sentiment.explode("comment").reset_index(drop = True)


# In[ ]:


df_sentiment_1.head()


# In[ ]:


#define a function to obtain get polariy and subjectivity

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity 
    sentiment_subjectivity = blob.sentiment.subjectivity 
    if sentiment_polarity > 0:
        sentiment_label = "positive"
    elif sentiment_polarity < 0:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    #store result in a dictionary
    result = {"polarity": sentiment_polarity, 
             "subjectivity": sentiment_subjectivity,
             "sentiment": sentiment_label}
    return result  


# In[ ]:


#apply function and create new column to store result
#df_sentiment_1["sentiment_result"] = df_sentiment_1.comment.apply(get_sentiment)


# #Error: The `text` argument passed to `__init__(text)` must be a string, not <class 'float'>

# In[ ]:


#split result (stored as dictionary) into individual key columns 
#sentiment = pd.json_normalize(df_sentiment_1["sentiment_result"])


# #Since the snippet above showed an error, I couldn't have "sentiment_result"

# In[ ]:


#No Sentiment above, so No concatenation too.

#concatenate both dataframe together horizontally
#df_1 = pd.concat([df_sentiment_1,sentiment], axis = 1)
#df_1.head()


# #Sentiments, Polarity and Subjectivity Analysis

# In[ ]:


plt.style.use("ggplot")

positive = len(df[df.sentiment == "positive"])
negative = len(df[df.sentiment == "negative"])
neutral = len(df[df.sentiment == "neutral"])

sentiment = [positive, neutral, negative]
sentiment_cat = ["positive", "neutral", "negative"]

sentiment.reverse()
sentiment_cat.reverse()

fig, ax = plt.subplots(figsize=(10,5))

palette = ["maroon", "darkslategrey", "seagreen"]

hbars = plt.barh(sentiment_cat, sentiment, color = palette, alpha = 0.5)

ax.bar_label(hbars, fmt='%.0f', color = "grey", padding = 5)

plt.xticks(np.arange(0,560,50).tolist())

plt.xlabel("Number of Comments")
plt.title("Overall Sentiment Distribution, 898 sentences", size = 13)
plt.show()


# #Acknowledgement: 
#     
# Ong Jia Ying https://www.kaggle.com/code/jyingong/data-cleaning-sentiment-analysis-reddit/notebook    
