#!/usr/bin/env python
# coding: utf-8

# <h2 style="text-align:center">Wine Quality Classifier using Machine Learning</h2>

# **Data Link:** *https://www.kaggle.com/datasets/rajyellow46/wine-quality*

# In[ ]:


# Importing packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

import tensorflow as tf
from tensorflow.keras.regularizers import l2

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, ConfusionMatrixDisplay


# In[ ]:


# Reading data from csv file

dataPath = "../input/wine-quality/winequalityN.csv"


# In[ ]:


data = pd.read_csv(dataPath)


# In[ ]:


df = data.copy()


# In[ ]:


# Viewing the data

df.head()


# In[ ]:


df.shape

# The dataset contains 6497 rows and 13 columns - with 1 dependent target feature


# In[ ]:


df.isna().sum()


# In[ ]:


# The dataset has null values, they have to be cleaned

df.dropna(inplace=True)


# In[ ]:


df.isna().sum()

# The dataset is clean and free from null values


# In[ ]:


df.dtypes


# In[ ]:


df['type'] = df['type'].astype("category").cat.codes

# Now the preprocessing works are done and the dataset is ready to be analysed


# In[ ]:


df.head()


# In[ ]:


# Descriptive Statistics of the Features

df.describe()


# In[ ]:


# Splitting the features and target as X and y respectively

X = df.drop("type",axis=1)
y = df['type']


# In[ ]:


# Checking for outliers in the data and removing if any

def remove_outlier(df, col_name):
    plt.figure(figsize=(20,20))
    f, axes = plt.subplots(1, 2,figsize=(12,4))
    sns.boxplot(data = df,x = col_name, ax=axes[0], color='skyblue').set_title("Before Outlier Removal: "+col_name)
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3-Q1
    df[col_name] = df[col_name].apply(lambda x : Q1-1.5*IQR if x < (Q1-1.5*IQR) else (Q3+1.5*IQR if x>(Q3+1.5*IQR) else x))
    sns.boxplot(data = df, x = col_name, ax=axes[1], color='pink').set_title("After Outlier Removal: "+col_name)
    plt.show()
    return df


# In[ ]:


for col in X.columns:
    df = remove_outlier(df,col)
plt.show()


# In[ ]:


# Checking the distribution of values in the feature variables

fig, axes = plt.subplots(4, 3, figsize=(20,13))
fig.suptitle('Independent Features'.upper(), fontsize=20)
feat = X.columns
f = 0
for i in range(4):
    for j in range(3):
        sns.histplot(data = df,x= feat[f],ax=axes[i,j],color="skyblue")
        axes[i,j].set_title(feat[f].upper(),fontsize=16)
        f += 1
fig.tight_layout()


# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(data = df,x="quality",y="alcohol",hue="type")
plt.legend(loc="best")
plt.show()


# In[ ]:


# Checking the correlation of the independent features and the dependent target

plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[ ]:


for a in range(len(df.corr().columns)):
    for b in range(a):
        if abs(df.corr().iloc[a,b]) >0.7:
            name = df.corr().columns[a]
            print(df.corr().columns[a],df.corr().columns[b])


# In[ ]:


# Total sulfur dioxide can be dropped as it correlates with free sulfur dioxide


# In[ ]:


X = X.drop("total sulfur dioxide",axis=1)
X.head()


# In[ ]:


X['quality'].unique()


# In[ ]:


X['quality'].mean()


# In[ ]:


# A feature best quality is created with quality feature and the condition of it to be best if quality is greater than 5

X['best quality'] = X['quality'].apply(lambda x: 1 if x>=5. else 0)
X.head()


# In[ ]:


X.drop("quality",axis=1)
X.head()


# In[ ]:


# Splitting the data for training and testing

X_train,X_test,y_train,y_test = train_test_split(X.values,y.values,test_size=0.25,random_state=123)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[ ]:


# Scaling the values

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


# Defining the base learners

models = {}
# Logistic Regression
lr = LogisticRegression()
models["Logistic Regression"] = lr
# KNN
knn = KNeighborsClassifier()
models["K Neighbors Classifier"] = knn
# SVC
svc = SVC(kernel="linear")
models["Support Vector Classifier"] = svc
# Decision Tree Classifier
dtc = DecisionTreeClassifier()
models["Decision Tree Classifier"] = dtc
rf = RandomForestClassifier(n_estimators=10, criterion="entropy",random_state=0)
models["Random Forest"] = rf


# In[ ]:


models


# In[ ]:


for model in models:
    models[model].fit(X_train,y_train)


# In[ ]:


def model_performance(modelName,model,X_test,y_test):
    print("_______________________________________________")    
    print("Model:",modelName)
    y_pred = model.predict(X_test)
    print("Accuracy Score:",accuracy_score(y_test,y_pred))
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix:\n",cm)
    cmd = ConfusionMatrixDisplay(cm,display_labels=["white","red"])
    cmd.plot()
    print("Classification Report:\n",classification_report(y_test,y_pred))
    plt.show()
    print("_______________________________________________")


# In[ ]:


# Performance of the base learners

for model in models:
    model_performance(model,models[model],X_test,y_test)


# In[ ]:


models_list = []
for model_ in models:
    models_list.append((model_,models[model_]))
models_list


# In[ ]:


# Defining the ensemble model

ensemble = VotingClassifier(estimators = models_list, voting="hard")


# In[ ]:


ensemble.fit(X_train,y_train)


# In[ ]:


# Performance of the ensemble model

model_performance("Ensemble of all Base Learners",ensemble,X_test,y_test)


# In[ ]:


X.shape


# In[ ]:


# Defining the Artifical Neural Network

factor=0.0001
rate=0.4

# Model Structure
model=tf.keras.models.Sequential([
                                  tf.keras.layers.Dense(160,input_shape=(12,),activation="relu",kernel_regularizer=l2(factor)),
                                  tf.keras.layers.Dropout(rate),
                                  tf.keras.layers.Dense(120,activation="relu",kernel_regularizer=l2(factor)),
                                  tf.keras.layers.Dropout(rate),
                                  tf.keras.layers.Dense(80,activation='relu',kernel_regularizer=l2(factor)),
                                  tf.keras.layers.Dropout(rate),
                                  tf.keras.layers.Dense(40,activation='relu',kernel_regularizer=l2(factor)),
                                  tf.keras.layers.Dropout(rate),
                                  tf.keras.layers.Dense(20,activation='relu',kernel_regularizer=l2(factor)),
                                  tf.keras.layers.Dropout(rate),
                                  tf.keras.layers.Dense(10,activation='relu',kernel_regularizer=l2(factor)),
                                  tf.keras.layers.Dropout(rate),
                                  tf.keras.layers.Dense(units=1, activation='sigmoid')])


# In[ ]:


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs={}):
    if(logs.get("val_accuracy")>0.95):
      print("Reached the accuracy required (ie) 90%", logs)
      self.model.stop_training=True
callback=myCallback()


# In[ ]:


def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()


# In[ ]:


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


history=model.fit(X_train, y_train, batch_size = 256,verbose=2, 
                  epochs = 100,callbacks=[callback],validation_split=0.25)


# In[ ]:


# Performance and Metrics of Artifical Neural Network

plot_metric(history, 'accuracy')


# In[ ]:


y_pred=model.predict(X_test)>0.5
y_pred = y_pred.astype("int")


# In[ ]:


print("_______________________________________________")    
print("Model: ANN Model")
print("Accuracy Score:",accuracy_score(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",cm)
cmd = ConfusionMatrixDisplay(cm,display_labels=["white","red"])
cmd.plot()
print("Classification Report:\n",classification_report(y_test,y_pred))
plt.show()
print("_______________________________________________")

