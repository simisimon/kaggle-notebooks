#!/usr/bin/env python
# coding: utf-8

# 
# # Introduction 😃😃😃
# 
# ## Smoking Status & Prediction Dataset
# 
# - A dataset for Body signal of smoking
# 
# 
# ### About Dataset
# 
# ##### This dataset is a collection of basic health biological signal data.
# 
# - The goal is to determine the presence or absence of smoking through bio-signals.
# 
# - The dataset is divided into type.
# 
# #### columns
# ##### data shape : (55692, 27)
# 
# - ID : index
# - gender
# - age : 5-years gap
# - height(cm)
# - weight(kg)
# - waist(cm) : Waist circumference length
# - eyesight(left)
# - eyesight(right)
# - hearing(left)
# - hearing(right)
# - systolic : Blood pressure
# - relaxation : Blood pressure
# - fasting blood sugar
# - Cholesterol : total
# - triglyceride
# - HDL : cholesterol type
# - LDL : cholesterol type
# - hemoglobin
# - Urine protein
# - serum creatinine
# - AST : glutamic oxaloacetic transaminase type
# - ALT : glutamic oxaloacetic transaminase type
# - Gtp : γ-GTP
# - oral : Oral Examination status
# - dental caries
# - tartar : tartar status
# - smoking : 0 or 1
# 
# 
# # Work plan 🤝🤝🤝🤝🤝
# 
# - 1- Data Exploration & Analysis 🤝🤝🤝
# - 2- Building a Machine Learning Model / classification score Volume
# 
# 
# # Data Exploration & Analysis 🤝🤝🤝
# 

# In[ ]:


#Importing the basic librarires fot analysis

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")  #using style ggplot

get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objects as go
import plotly.express as px


# In[ ]:


#Importing the dataset
df =pd.read_csv("../input/body-signal-of-smoking/smoking.csv")


# look the data set
df.head()


# In[ ]:


# looking the shape DataSet
df.shape


# In[ ]:


#Checking the dtypes of all the columns

df.info()


# In[ ]:


#Checking the dtypes of all the columns

df.info()


# - No any missing value

# In[ ]:


# look  describe data set
df.describe().round(2)


# In[ ]:


# check unique value
df.nunique().sort_values()


# # Some of Visualisations

# In[ ]:


# how much percentage Gender in the dataset

df['gender'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True)


# #### The percentage gender in the dataset:
# - Female = 36.4%
# - Male =63.6 %

# In[ ]:


# how much percentage smoking  in the dataset

df['smoking'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True)


# #### The percentage gender in the dataset:
# - Non-smoking = 63.3%
# - Smoking =36.7 %

# In[ ]:


# boxplot for show describe age 

plt.boxplot(df["age"])
plt.show()


# #### Describe age :
# - mean =       44
# - min   =      20
# - max    =     85

# In[ ]:


# boxplot for show describe height 

plt.boxplot(df["height(cm)"])
plt.show()


# #### Describe height- cm :
# - mean=        164
# - min  =       130
# - max   =     190

# In[ ]:


# boxplot for show describe weight 

plt.boxplot(df["weight(kg)"])
plt.show()


# #### Describe weight(kg) :
# - mean= 65
# - min = 30
# - max = 135

# In[ ]:


#make groupby to show the average age smoking

ag=df.groupby("smoking")["age"].mean()
ag


# In[ ]:


# graph average age smoking
ag.plot(kind="pie",explode=[0,0.1],autopct='%1.1f%%',shadow=True)


# #### The average age smoking
# - age 45 = non- somkign
# - age 41 = somking

# In[ ]:


# group by for show the average age , weight and height by the gender

summary=df.groupby(["gender","smoking"])["age","weight(kg)","height(cm)"].mean().round(0)
summary


# In[ ]:


# graph the group by

summary.plot(kind="bar",figsize=(15,7))


# # Analysis Results 🙉🙈🙊
# 
# #### From my point view , I see the important columns :
# - Gender ,  smoking ,  age , weight(kg) , height(cm) 
# 
# #### After make some analysis , visual graph and explore the data set , I see some results .
# 
# 
# #### The shape DataSet
# - Rows= 55692 ,  Columns =  27
# 
# 
# #### The percentage gender in the dataset:
# - Female = 36.4%
# - Male =63.6 %
# 
# 
# #### The percentage gender in the dataset:
# - Non-smoking = 63.3%
# - Smoking =36.7 %
# 
# 
# #### Describe age :
# - mean = 44
# - min = 20
# - max = 85
# 
# 
# #### Describe height- cm :
# - mean= 164
# - min = 130
# - max = 190
# 
# 
# #### Describe weight(kg) :
# - mean= 65
# - min = 30
# - max = 135
# 
# 
# #### The average age smoking
# - age 45 = non- somkign
# - age 41 = somking
# 
# 
# ### The average data
# 
# #### Female
# - somking avg /> age= 46 ,weght =56 kg ,height 157 cm
# - non-somking ave / >age= 49 ,weght =56 kg ,height 165 cm
# 
# 
# #### Male
# - somking avg /> age= 41 ,weght =72 kg ,height 170 cm
# - non-somking ave / >age= 42 ,weght =71 kg ,height 170 cm

# # Building a Machine Learning Model / classification score Volume

# In[ ]:


#Importing the basic librarires for building model - classification

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,r2_score


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import  MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance


# In[ ]:


# show the data set
df.head()


# In[ ]:


#show the data type
df.info()


# - we need change the data type from object to number

# In[ ]:


# change datatype column gender 

le = LabelEncoder()
le.fit(df["gender"])
df["gender"]=le.transform(df["gender"])  

# change datatype column oral 

l = LabelEncoder()
l.fit(df["oral"])
df["oral"]=l.transform(df["oral"])

# change datatype column tartar 


a = LabelEncoder()
a.fit(df["tartar"])
df["tartar"]=a.transform(df["tartar"])


# In[ ]:


df.info()


# In[ ]:


# drop column ID , I don`t need this column
df.drop(columns="ID",inplace=True)


# In[ ]:


df.head()


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


#Defined X value and y value , and split the data train
X = df.drop(columns="smoking")           
y = df["smoking"]    # y = quality

# split the data train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("X Train : ", X_train.shape)
print("X Test  : ", X_test.shape)
print("Y Train : ", y_train.shape)
print("Y Test  : ", y_test.shape)


# In[ ]:


#Defined object from library classification 

LR = LogisticRegression()
DTR = DecisionTreeClassifier()
RFR = RandomForestClassifier(n_estimators=150)
KNR = KNeighborsClassifier()
MLP = MLPClassifier()
XGB = XGBClassifier()
SVR=SVC()


# In[ ]:


# make for loop for classification 

li = [LR,DTR,RFR,KNR,MLP,KNR,XGB,SVR]
d = {}
for i in li:
    i.fit(X_train,y_train)
    ypred = i.predict(X_test)
    print(i,":",accuracy_score(y_test,ypred)*100)
    d.update({str(i):i.score(X_test,y_test)*100})


# In[ ]:


# make graph about Accuracy

plt.figure(figsize=(30, 6))
plt.title("Algorithm vs Accuracy")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.plot(d.keys(),d.values(),marker='o',color='red')
plt.show()


# # RandomForestClassifier
# 
# - This the best model , let's see the important columns

# In[ ]:


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

ypred = rf.predict(X_test)

print("Score the X-train with Y-train is : ", rf.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", rf.score(X_test,y_test))
print("Accuracy Score :",accuracy_score(y_test,ypred)*100)


# In[ ]:


#feature_importances

sort = rf.feature_importances_.argsort()
plt.barh(df.columns[sort], rf.feature_importances_[sort])
plt.xlabel("Feature Importance")


# # Model Selection Results 😃😃😃
# 
# - Logistic Regression = 70.6 %
# - Decision Tree Classifier = 77.7 %
# - Random Forest Classifier = 82.6 %
# - K Neighbors Classifier = 70 %
# - MLP Classifier = 75 %
# - K Neighbors Classifier = 70 %
# - XGB Classifier = 78 %
# - SVC = 72.5 %
# 
# 
# 
# ### So , the best model Random Forest Classifier
# 
# ### You can change parameter in the library , maybe get better accuracy
# 
# 
# # Notes 😃😃😃😃
# 
# - Thank for reading my analysis and my classification. 😃😃😃😃
# 
# - If you any questions or advice me please write in the comment . ❤️❤️❤️❤️
# 
# - If anyone has a model with a higher percentage, please tell me 🤝🤝🤝, it`s will support me .
# 
# 
# # Vote ❤️😃
# - If you liked my work upvote me ,
# 
# 
# # The End 🤝🎉🤝🎉
