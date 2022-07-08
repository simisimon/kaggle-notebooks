#!/usr/bin/env python
# coding: utf-8

# ## Context
# 
# Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.
# 
# People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.
# 
# ## Attribute Information
# 
# 1) Age: age of the patient [years]
# 
# 2) Sex: sex of the patient [M: Male, F: Female]
# 
# 3) ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
# 
# 4) RestingBP: resting blood pressure [mm Hg]
# 
# 5) Cholesterol: serum cholesterol [mm/dl]
# 
# 6) FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
# 
# 7) RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
# 
# 8) MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
# 
# 9) ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
# 
# 10) Oldpeak: oldpeak = ST [Numeric value measured in depression]
# 
# 11) ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
# 
# 12) HeartDisease: output class [1: heart disease, 0: Normal]
# 
# ## Source
# 
# This dataset was created by combining different datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes. The five datasets used for its curation are:
# 
# Cleveland: 303 observations
# Hungarian: 294 observations
# Switzerland: 123 observations
# Long Beach VA: 200 observations
# Stalog (Heart) Data Set: 270 observations
# Total: 1190 observations
# Duplicated: 272 observations
# 
# Final dataset: 918 observations

# In[ ]:


import os, types
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np

df = pd.read_csv('../input/heart-failure-prediction/heart.csv')
df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# In[ ]:


df.shape


# In[ ]:


df.describe()


# ### Redifining DataFrame

# In[ ]:


numerical = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
categorical = ['Sex','ChestPainType','FastingBS','RestingECG','ST_Slope','ExerciseAngina','HeartDisease']
df = df[numerical + categorical]
df.shape


# ### Analyzing Numerical Variables 

# In[ ]:


sns.set(style='whitegrid', palette='deep', font_scale=1.1, rc={'figure.figsize':[8,5]})


# In[ ]:


df[numerical].hist(bins=15, figsize=(15,6), layout=(2,4));


# ### Analyzing Categorical Variables

# In[ ]:


fig, ax = plt.subplots(2, 3, figsize=(20,10))

for variable, subplot in zip(categorical, ax.flatten()):
    sns.countplot(x=df[variable], hue='HeartDisease', data=df, ax=subplot, palette='Set2')
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
        


# ### Analyzing Relationships Between Numerical Variables

# In[ ]:


sns.pairplot(df[numerical + ['HeartDisease']], hue='HeartDisease')


# ### Analyzing Relationships Between Numerical nd Categorical Variables

# In[ ]:


bins = np.linspace(df.Age.min(), df.Age.max(), 10)
g = sns.FacetGrid(df, col="Sex", hue="HeartDisease", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


bins = np.linspace(df.Cholesterol.min(), df.Cholesterol.max(), 10)
g = sns.FacetGrid(df, col="Sex", hue="HeartDisease", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Cholesterol', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


bins = np.linspace(df.RestingBP.min(), df.RestingBP.max(), 10)
g = sns.FacetGrid(df, col="Sex", hue="HeartDisease", palette="Set1", col_wrap=2)
g.map(plt.hist, 'RestingBP', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


bins = np.linspace(df.MaxHR.min(), df.MaxHR.max(), 10)
g = sns.FacetGrid(df, col="Sex", hue="HeartDisease", palette="Set1", col_wrap=2)
g.map(plt.hist, 'MaxHR', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


bins = np.linspace(df.Oldpeak.min(), df.Oldpeak.max(), 10)
g = sns.FacetGrid(df, col="Sex", hue="HeartDisease", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Oldpeak', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


sns.set(style="white") 
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(5,5))
cmap = sns.diverging_palette(255, 10, as_cmap=True)
sns.heatmap(df.corr(), mask=mask, annot=True, square=True, cmap=cmap,vmin=-1, vmax=1, ax=ax)  

bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)


# ## Convert Categorical features to numerical values

# ### Sex:

# In[ ]:


df['Sex'].replace(to_replace=['M','F'], value=[0,1],inplace=True)
df.head()


# ### ChestPainType:

# In[ ]:


ChestPainType_dummy = pd.get_dummies(df['ChestPainType'])
ChestPainType_dummy.rename(columns={'TA':'ChestPainType-TA','ATA':'ChestPainType-ATA','NAP':'ChestPainType-NAP','ASY':'ChestPainType-ASY'}, inplace=True)
df = pd.concat([df,ChestPainType_dummy],axis=1)
df.drop('ChestPainType',axis=1,inplace=True)
df.head()


# ### RestingECG:

# In[ ]:


RestingECG_dummy = pd.get_dummies(df['RestingECG'])
RestingECG_dummy.rename(columns={'Normal':'RestingECG-Normal','ST':'RestingECG-ST','LVH':'RestingECG-LVH'}, inplace=True)
df = pd.concat([df,RestingECG_dummy],axis=1)
df.drop('RestingECG',axis=1,inplace=True)
df.head()


# ### ExerciseAngina:

# In[ ]:


df['ExerciseAngina'].replace(to_replace=['N','Y'], value=[0,1],inplace=True)
df.head()


# ### ST_Slope:

# In[ ]:


ST_Slope_dummy = pd.get_dummies(df['ST_Slope'])
ST_Slope_dummy.rename(columns={'Up':'ST_Slope-Up','Flat':'ST_Slope-Flat','Down':'ST_Slope-Down'}, inplace=True)
df = pd.concat([df,ST_Slope_dummy],axis=1)
df.drop('ST_Slope',axis=1,inplace=True)
df.head()


# In[ ]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Confusion Matrix Function.

# In[ ]:


def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); 
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['no heart disease', 'heart disease']); ax.yaxis.set_ticklabels(['no heart disease', 'heart disease'])


# ## Feature Selection

# Let's define feature sets, X:

# In[ ]:


X = df[['Age','Sex','RestingBP','Cholesterol','FastingBS','MaxHR','ExerciseAngina','Oldpeak','ChestPainType-ASY','ChestPainType-ATA','ChestPainType-NAP','ChestPainType-TA','RestingECG-LVH','RestingECG-Normal','RestingECG-ST','ST_Slope-Down','ST_Slope-Flat','ST_Slope-Up']]
X.head()


# What are our lables?

# In[ ]:


Y = df['HeartDisease'].values
Y[0:5]


# ### Normalize Data

# In[ ]:


transform = preprocessing.StandardScaler()
X = transform.fit(X).transform(X)


# ## Model Development

# Train/Test split:

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 2)


# ### Logistic Regression

# In[ ]:


parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}

lr=LogisticRegression()


# In[ ]:


logreg_cv = GridSearchCV(lr,parameters,cv=10)
logreg_cv.fit(X_train, Y_train)


# In[ ]:


print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# In[ ]:


print('Accuracy is ', logreg_cv.score(X_test,Y_test))


# Confusion Matrix:

# In[ ]:


yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# ### Support Vector Machine (SVM)

# In[ ]:


parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()


# In[ ]:


svm_cv = GridSearchCV(svm, parameters, cv=10)
svm_cv.fit(X_train, Y_train)


# In[ ]:


print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)


# In[ ]:


print('Accuracy is', svm_cv.score(X_test, Y_test))


# Confusion Matrix:

# In[ ]:


yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# ### Decision Tree

# In[ ]:


parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()


# In[ ]:


tree_cv = GridSearchCV(tree, parameters, cv=10)
tree_cv.fit(X_train,Y_train)


# In[ ]:


print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)


# In[ ]:


print('Accuracy is', tree_cv.score(X_test,Y_test))


# Confusion Matrix:

# In[ ]:


yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# ### K-Nearest Neighbors (KNN)

# In[ ]:


parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()


# In[ ]:


knn_cv = GridSearchCV(KNN, parameters, cv=10)
knn_cv.fit(X_train, Y_train)


# In[ ]:


print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)


# In[ ]:


print('Accuracy is', knn_cv.score(X_test,Y_test))


# Confusion Matrix:

# In[ ]:


yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# ## Finding Best Model and Accuracy

# In[ ]:


models = {'kneighbors': knn_cv.best_score_,
         'DecisionTree': tree_cv.best_score_,
         'SVM': svm_cv.best_score_,
         'LogisticRegression': logreg_cv.best_score_ }

best_model = max(models, key = models.get)
print('The best model is',best_model, 'with a score of',models[best_model])

