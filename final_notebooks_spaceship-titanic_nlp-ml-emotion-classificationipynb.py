#!/usr/bin/env python
# coding: utf-8

# In machine learning, a model represents the process of creating a save file after running a machine learning algorithm on training data. Fueled by data, the model is an offshoot of artificial intelligence that uses algorithms such as logistic regression and naive bayes classifier to identify text emotions in a dataset. By doing so, it is able to find patterns or make predictions without human intervention. Therefore, it is also known as the resulting output that enables systems to identify patterns, learn from data, and make decisions with a certain level of precision and confidence. In here, we will train two models over a set of data, providing it with two algorithms that can be used to learn from those data. This is to enable us to compare model performance and ultimately determine the best model which is able to be used to analyze text emotions. 
# 
# However, before training or building any model, the data scientist will first import a new dataset which labels a few emotions in each review so that the model can based on them to classify the emotions. Along this line, the dataset is obtained from Kaggle published by Anjaneya Tripathi in 2021. 

# In[ ]:


# install pipline
get_ipython().system(' pip install simple-colors')
get_ipython().system(' pip install neattext')
get_ipython().system(' pip install emoji')


# In[ ]:


# import libraries
import matplotlib.pyplot as plt
import neattext.functions as nfx
import numpy as np
import pandas as pd
import seaborn as sns

import nltk
import string
import re


# In[ ]:


# import emotion-label-train.csv
mldf = pd.read_csv("../input/emotion-classification-nlp/emotion-labels-train.csv")
mldf.head(3)


# Based on the figure above, the dataset that is used by us contains 2 columns, text and label. Each column will represent a different meaning of data. For example, a text column is used to store the reviews made by reviewers and the label column is used to store the emotions from the reviewers. 
# 
# ## 1.0 Exploratory Data Analysis
# 
# Here, data preprocessing and exploratory data analysis are the initial steps in the data science workflow. It is a process where a data scientist uses certain python built-in functions to explore the fetal health dataset in order to gain a deeper understanding of the important properties of the data. By doing so, it enables us to prepare the data according to the insights gained from this step and determine hypotheses that will serve as the basis for further analysis and modeling.
# 
# ### 1.1 Data Understanding
# Before building powerful machine learning models, first getting familiar with the training dataset by understanding all the features found in the dataset is needed.

# In[ ]:


# general info about the dataset
print("Rows     : " ,mldf.shape[0])
print("Columns  : " ,mldf.shape[1])
print("Features : " ,mldf.columns.tolist())


# The dataset contains 3142 records and 2 attributes. Each data in the text column is unique. However, the data in the label column can be used to conduct more specific observations by using a statistical graph as shown below.

# In[ ]:


from simple_colors import *

# number and types of different emotion in dataset 
print(blue("Total types of emotions: "), len(mldf['label'].unique()))
print(blue("Types of emotions      :"))
mldf['label'].value_counts()


# ### 1.2 Exploratory Data Analysis
# Exploratory Data Analysis (EDA) can be defined as a process of exploring a fetal health dataset through visualizations to observe if there is any interesting or valuable information. Generally speaking, it is an initial investigation method that aims to use both visual and quantitative methods to understand what the data is telling.
# 
# Below is an example of visualizing the target variable emotions

# In[ ]:


# create an emotion dataframe for the graph used
emotion_mldf = mldf['label'].value_counts().to_frame()

# draw graph
ax = emotion_mldf.plot(kind='barh', color='#4863A0', width=0.6)

# count percentage
total = len(mldf['label'])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_width()/total)
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height()/2
    ax.annotate(percentage, (x, y))

# remove frame 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# set labels
plt.title('Types of Emotions', fontdict={'color':'dimgray', 'size': 16}, x=0, y=1.05) 
ax.xaxis.set_label_coords(0.1,-0.15)
ax.set_xlabel('Emotional count')
ax.yaxis.set_label_coords(-0.17, 1.0)
ax.set_ylabel('Emotion')
plt.show()


# The bar chart created in horizontal form provides us with the summary of the four categorical features in the dataset. We can observe that the distribution of sadness, anger, and joy are having a quite similar occupation with each other. Whereas, the fear emotion is accounted for the most in the dataset. 
# 
# ## 2.0 Data Pre-processing
# 
# Data preprocessing is a data mining technique that involves transforming raw data into an understandable format. Real-world data is often incomplete, inconsistent, and/or lacking in certain behaviors or trends, and is likely to contain many errors. Data preprocessing is a proven method of resolving such issues. Data preprocessing prepares raw data for further processing.
# 
# In this section, the process will be divided into 2 parts :
# 
# Data Quality Assessment which is used to check the quality status in the dataset through detecting incomplete, inconsistent and redundant data
# 
# Data Cleaning which is the process of fixing or removing incorrect, corrupted, incorrectly formatted, duplicate, or incomplete data within a dataset
# 
# ### 2.1 Data Quality Assessment
# In this section, the process will be used to assess the data quality by observing:
# 
# Redundant data - duplicate rows exist
# 
# Incomplete data - lacking attribute values, lacking certain attributes of interest, or containing only aggregate data
# 
# **(1) Redundant Data**
# 
# Checking data redundancy for each column is the process of examining the quality of the dataset. This is because it allows us to understand if the dataset is at risk of being misinterpreted. Therefore, the following code snippet is used to find the total number of duplicate rows in the dataset to observe if the data has the potential to confuse the data consumers.

# In[ ]:


# check redundant data 
mldf.duplicated().sum()

# output - there is no duplicate row in the dataset


# According to the result, there is no redundant data in the dataset, which indicates that the same piece of data does not exist in two or more separate rows.
# 
# **(2) Incomplete Data**
# 
# In addition, figuring out if the training dataset consists of missing data is also crucial. If there is any missing data, it may lead to our research running into serious problems. So, examining each column for any incomplete data is a way for us to know if the dataset is ‘healthy’ and if we will encounter difficulties in later analysis. 

# In[ ]:


# check null data 
mldf.isnull().sum(axis = 0)

# output - there is no incomplete data 


# According to the result, there is no incomplete data in any of the columns, which indicates that all data is fully filled into every row and column in the dataset. 
# 
# ### 2.2 Data Cleaning (NLP)
# 
# Data cleansing or text cleaning, in general, is a process of lowering mistakes and improving data quality. Correcting violations of data quality and removing insignificant textual data is an important process that cannot be ignored. This is due to the fact that if the dataset contains bad data, it will definitely degrade the model performance. So to avoid wrong solutions in the case of classification models, the cleaning process includes the following action: -
# 
# * User handles - removing user handles to reduce less useful data.
# * Lower case - lowering case to maintain a consistent flow during text mining.
# * Emojis - converting emojis to text so machines can understand what they mean.
# * Punctuations - removing punctuations to minimize ambiguity. 
# * Stop words - removing stop words to minimize noise.	
# * Number - removing numbers to reduce less useful data.
# * Single letter - removing single letters to reduce less meaningful words. 
# * Lemmatization - lemmatizing words to return words into their root form.
# * Unwanted words - removing unwanted words to reduce noise.      
# 
# The following code snippet assembles all the text cleaning methods which have been discussed above. 

# In[ ]:


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# In[ ]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from emoji import demojize

# remove userhandles
mldf['text_userhandles'] = mldf['text'].apply(nfx.remove_userhandles)

# lower casing
mldf["text_lower"] = mldf["text_userhandles"].str.lower()

# handle emojis
mldf["text_emoji"] = mldf["text_lower"].apply(lambda text: demojize(text))

# remove puntuation
punc_to_remove = string.punctuation

def remove_punctuation(text):
    return text.translate(str.maketrans('','', punc_to_remove))
mldf["text_punc"] = mldf["text_emoji"].apply(lambda text: remove_punctuation(text))

# remove stopwords
STOPWORDS = set(stopwords.words("english"))

def remove_stopword(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
mldf["text_stop"] = mldf["text_punc"].apply(lambda text: remove_stopword(text))

# lemmatization
lemmatizer = WordNetLemmatizer()
wordnet_map={"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

def lemmatized_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word , wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
    
mldf["text_lemma"] = mldf["text_stop"].apply(lambda text: lemmatized_words(text))

# removal of numbers
mldf['text_num'] = mldf['text_lemma'].str.replace(r'\d+','')

# remove single letter in text
mldf['text_letter'] = mldf['text_num'].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')

# remove most common words but not carrying much meaning 
unwanted_words = ["amp","gonna","cant'","im","lol","nd","youre","would","want","thats", "there","still","wanna","ive","also","could","didnt","he","youre","do", "nt","let","yall", "ur", "bb", "bc", "doesnt", "id", "want", "wanna", "wasnt", "dont", "get"]

text = lambda x: ' '.join(w for w in x.split() if not w in unwanted_words)
mldf["cleantext"] = mldf['text_letter'].apply(text)


# In[ ]:


mldf.head(2)


# As we can see, multiple columns have been created in the data frame during the cleaning process. However, some of the columns that are no longer needed will be removed as shown below. 

# In[ ]:


# check total columns of the dataset before cleaning
print("Dataset Features (Before): " ,mldf.columns.tolist())

# remove unnecessary columns
mldf.drop(['text_lower', 'text_num', 'text_letter', 'text_userhandles', 'text_lemma', 'text_stop', 'text_punc', 'text_emoji'], axis=1, inplace=True)

# check total columns of the dataset to ensure unnecessary columns are deleted - successful
print("Dataset Features (After) : " ,mldf.columns.tolist())


# The result shows that the columns in the training dataset have been removed from 11 to 3. Those columns which were previously used to perform text cleaning are no longer required for the following data analysis. So to cut down the space occupied by unnecessary data, columns such as text userhandles, text lower, text emoji, text punc, text stop, text lemma, text num, text letter have been dropped. 
# 
# 
# ## 3.0 Extract Keywords
# 
# As the name implies, keywords extraction refers to a text analysis technique for extracting the most common words or phrases from a list of text. To be concrete, the algorithm involves using NLP and ML concepts to decompose human language so that data scientists are able to identify in less time and faster which parts of the available data cover the topics they are searching for. Through this, they can save a lot of time manually processing high quality keywords. 
# 
# The following code snippet is about using a function to extract most mentioned attributes in the textual data so that it can simplify the task of finding relevant words in each emotion.  

# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator
from collections import Counter
from PIL import Image

def extract_keywords(text, num = 100):
    tokens = [tok for tok in text.split()]
    most_common_tokens = Counter(tokens).most_common(num)
    return dict(most_common_tokens)

# extract joy keywords
joyList = mldf[mldf['label'] == 'joy']['cleantext'].tolist()    # create a joy list 
joyDoc = ' '.join(joyList)                                      # create a joy document which store all the joy words in list
joyKeywords = extract_keywords(joyDoc)                          # extract the keywords

# extract anger keywords
angerList = mldf[mldf['label'] == 'anger']['cleantext'].tolist() 
angerDoc = ' '.join(angerList)                                   
angerKeywords = extract_keywords(angerDoc)   

# extract fear keywords
fearList = mldf[mldf['label'] == 'fear']['cleantext'].tolist() 
fearDoc = ' '.join(fearList)                                   
fearKeywords = extract_keywords(fearDoc)   

# extract sadness keywords
sadnessList = mldf[mldf['label'] == 'sadness']['cleantext'].tolist() 
sadnessDoc = ' '.join(sadnessList)                                   
sadnessKeywords = extract_keywords(sadnessDoc)   


# Once a collection of keywords for each emotion has been generated, the next step is to visualize the word frequency. This is to give prominence to words that appear frequently in the source text. Below are some examples of creating word clouds to quickly show which terms are included in emotion documents, such as joyDoc, sadnessDoc, angerDoc, and fearDoc. Through this, it can inspire us by highlighting the frequency of keywords in the text-based data. 
# 
# **Emotion - Joy**
# 
# The following code snippet is to display a word cloud of 1000 joyful keywords. 
# 
# image link: https://www.alamy.com/the-word-joy-concept-written-in-colorful-retro-shapes-and-colors-illustration-image387713480.html
# 

# In[ ]:


def plot_wordcloud(docx):
    mywordcloud = WordCloud(background_color="white").generate(docx)
    plt.figure(figsize = (13,10))
    plt.imshow(mywordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.show()


# In[ ]:


plot_wordcloud(joyDoc)


# According to the result, the words ‘happy’ and ‘smile’ are the most frequently mentioned characteristics, and ‘love’ is the most popular of them all. 
# 
# **Emotion - Anger**
# 
# The following code snippet is to display a word cloud of 1000 angry keywords. 

# In[ ]:


plot_wordcloud(angerDoc)


# Based on the result, the words ‘like’ and ‘go’ are the most frequently mentioned words, and ‘people’ are the most popular of them all. Although the first three words are seen did not match up much with any angry words, the words ‘angry’, ‘offen’, and ‘fuck’ are relevant to what we wanted on our angry list.
# 
# **Emotion - Sadness**
# 
# The following code snippet is to display a word cloud of 1000 sadness keywords.

# In[ ]:


plot_wordcloud(sadnessDoc)


# According to the result, the words ‘sad’ and ‘depression’ are the most frequently mentioned characteristics, and ‘sadness’ is the most popular among them. 
# 
# **Emotion - Fear**
# 
# The following code snippet is to display a word cloud of 1000 fear keywords.

# In[ ]:


plot_wordcloud(fearDoc)


# According to the result, the words ‘go’ and ‘make’ are the most frequently mentioned words, and ‘fear’ is the most popular among them. 
# 
# ## 4.0 Build model
# In this section, the process of building ML models involves using two learning algorithms, Naive Bayes classifier and logistic regression, to train the models in learning certain events. Yet, before starting the development of any ML model, the first step is to set up the dependent variable (cleantext) and the independent variable (label) so that algorithms are able to find patterns in the training data that map the independent variable to the dependent variable. Also, determining the percentage of resources to allocate for the training and testing parts is needed. In fact, there are several theories behind how many percent of data should be split between training and testing. But, in this paper, the data scientist uses 80% of the data for training and 20% of the data for testing. This is to avoid the model using the same data for both training and testing purposes. 

# In[ ]:


Xfeature = mldf['cleantext']
Ylabel = mldf['label']


# In[ ]:


# victorize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cv = CountVectorizer()
X = cv.fit_transform(Xfeature)


# In[ ]:


# split dataset into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Ylabel, test_size=0.2, random_state=42)
print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)


# Notice that there are 2890 rows used for training the model and 723 rows used for testing the model.
# 
# ### 4.1 Naive Bayes classifier model
# 
# After splitting the dataset into two parts, the data scientist uses Naive Bayes techniques to train a model called nvModel and evaluates its performance by using the model accuracy score.

# In[ ]:


# train / build model
from sklearn.naive_bayes import MultinomialNB

nvModel = MultinomialNB()
nvModel.fit(X_train, y_train)


# In[ ]:


print("Length of the model classes: ", len(nvModel.classes_))
print("Type of the model classes  : ", nvModel.classes_)
print("Model accuracy             : ", nvModel.score(X_test, y_test))


# Result shows that the Naive Bayes classifier model accuracy is 0.8188 or about 82% on the holdout dataset.
# 
# ### 4.2 Logistic regression model
# 
# Apart from Naive Bayes technique, logistic regression is another technique used by the data scientist to train a model and also evaluates its performance by using the model accuracy score.

# In[ ]:


# train / build model
from sklearn.linear_model import LogisticRegression

lrModel = LogisticRegression()
lrModel.fit(X_train, y_train)


# In[ ]:


print("Length of the model classes: ", len(lrModel.classes_))
print("Type of the model classes  : ", lrModel.classes_)
print("Model accuracy             : ", lrModel.score(X_test, y_test)) 


# According to the results, the accuracy of the logistic regression model on the holdout dataset is 0.8450 or about 85%, which is **better than the naive Bayes model.**
# 
# ### 4.3 Model evaluation 
# 
# In machine learning, model evaluation is an integral part of using different evaluation metrics to determine the performance of a model. Essentially, it is a performance measure of how well the algorithm is before putting a model to use. To this end, there are several techniques that can be applied to monitor the model quality. Classification, confusion matrix and chi square are some of the example metrics that can evaluate model performance during training and also ensure that the model will work well in the production environment. Here, we will develop two confusion matrices based on the Naive Bayes and logistic regression algorithms that includes various statistics needed to judge the two models. Following that, we will also develop two classification reports to give a better picture of the model accuracies.
# 
# Without only focusing on the accuracy score, the following code snippet is a creation of two confusion matrices that is used to measure machine learning classification problems based on the testing dataset. By definition, a confusion matrix is an N X N table, where N is the number of target classes. For the problem at hand, the number of model classes we obtain includes anger, joy, sadness and fear. So there are 4 types of classes in the model that can lead us to a 4 X 4 matrix.
# 

# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
from simple_colors import *

# naive bayes 
plot_confusion_matrix(nvModel, X_test, y_test, cmap="Blues")
plt.title("Confusion Matrix for Naive Bayes Classification")
plt.show()

# logistic regression
plot_confusion_matrix(lrModel, X_test, y_test, cmap="Blues")
plt.title("Confusion Matrix for Logistic Regression")
plt.show()


# After comparing the two confusion matrices, logistic regression shows that it outperforms Naive Bayes in every way. This indicates that logistic regression not only has a better accuracy score than Naive Bayes but its overall correct and incorrect predictions made by a classifier are also more accurate than Naive Bayes.
# 
# In addition to evaluating the predictions, the confusion matrix can also measure the performance of a classification model by calculating precision, precision, recall, and F1-score. Basically, this is an expanded version or elements of the confusion matrix which can help data scientists to gain a better insight and understanding into the performance of the model. 
# 
# * Recall - It should be as high as possible because it is about calculating the model’s ability to predict positive values. 
# * Precious - It should be as high as possible because it is what determines whether the model is reliable.
# * F1-score - The closer it is to 1, the better, because a low f1-score is an indication of both poor precision and poor recall.
# 
# Below is a code snippet that can be used to generate a classification report with recall, precious, f1-score and accuracy score. 
# 

# In[ ]:


# naive bayes 
preds = nvModel.predict(X_test)
print(blue("Accuracy on train data by naive bayes: "), accuracy_score(y_train, nvModel.predict(X_train))*100)
print(blue("Accuracy on test data by naive bayes : "), accuracy_score(y_test, preds)*100, '\n')
# comprehensive report on the classification
print(classification_report(y_test, preds))

# logistic regression
preds = lrModel.predict(X_test)
print(blue("Accuracy on train data by logistic regression: "), accuracy_score(y_train, lrModel.predict(X_train))*100)
print(blue("Accuracy on test data by logistic regression : "), accuracy_score(y_test, preds)*100, '\n')
# comprehensive report on the classification
print(classification_report(y_test, preds))


# As per analysis, all the precision, recall, and f1-scores in the logistic regression algorithm are performed better than Naive Bayes, so the data scientist will decide to use a logistic regression model to classify emotions in the review dataset. 
