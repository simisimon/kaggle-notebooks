#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Data Manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import missingno

# Data Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, label_binarize

# Machine Learning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

# Lets ignore warnings, cuz why not?
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## 2. Load Dataset

# In[ ]:


# read data from csv files
train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
gender_data = pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


# view training set
train_data.head()


# In[ ]:


# view test set
test_data.head()


# In[ ]:


# view sample submission
gender_data.head()


# ## 3. Data Overview

# In[ ]:


print('Training Examples = {}'.format(train_data.shape[0]))
print('Test Examples = {}\n'.format(test_data.shape[0]))
print('Training Shape = {}'.format(train_data.shape))
print('Test Shape = {}\n'.format(test_data.shape))
print('Training Columns = {}'.format(train_data.columns))
print('Test Columns = {}\n'.format(test_data.columns))


# In[ ]:


print(train_data.info())
train_data.isnull().sum()


# In[ ]:


print(test_data.info())
test_data.isnull().sum()


# In[ ]:


# visualize missing values of training set
missingno.matrix(train_data, figsize =(28, 14))


# > In the above graph of training set, the white lines represent missing values. Its clear that there are a lot of missing values in "Cabin" column followed by "Age" column and 2 missing values in "Embarked" column.

# In[ ]:


# visualize missing values of test set
missingno.matrix(test_data, figsize =(28, 14))


# > In the above graph of test set, its clear that there are a lot of missing values in "Cabin" column (same as training set) followed by "Age" column and 1 missing value in "Fare" column.

# In[ ]:


train_data.describe()


# In[ ]:


test_data.describe()


# Now we will see the correlation between numerical features of the data by plotting a heat map.

# In[ ]:


df_num = train_data[['Age', 'SibSp', 'Parch', 'Fare']]
fig = plt.figure(figsize=(10, 6))
ax = sns.heatmap(df_num.corr(), annot=True)


# ## 4. Exploratory Data Analysis (EDA)

# **Let's exclude column which are not relevent. "PassengerId" and "Name" can be excluded, as well as "Cabin" column because only few of its rows have values, we would be better off dropping it altogther.**
# > **Note**: In this solution I'm dropping "Ticket" also but if we group it according to frequencies then we can include it in our features, you can try it out once.

# In[ ]:


excluded_cols = ['PassengerId', 'Name', 'Cabin', 'Ticket']
df_train = train_data.drop(excluded_cols, axis=1)
df_test = test_data.drop(excluded_cols, axis=1)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# We have 9 columns in training set, lets analyse each column one by one.

# In[ ]:


# Plot function for categorical features
def cat_plot(feature):
    fig = plt.figure(figsize = (14, 8))
    sns.countplot(x=feature, hue='Survived', data=df_train)
    
    plt.xlabel(feature, size=15, labelpad=5)
    plt.ylabel('Passengers', size=15, labelpad=15)    
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 14})
    plt.title('Survival Count in {}'.format(feature), size=15)
    plt.show()
    
    print(df_train[feature].value_counts())


# In[ ]:


# Plot function for continuous features
def con_plot(feature):
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(12, 16))
    fig.tight_layout(h_pad=6)
    
    survived = df_train['Survived'] == 1

    # Distribution of survival in feature
    sns.distplot(df_train[~survived][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0])
    sns.distplot(df_train[survived][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0])

    # Distribution of feature in dataset
    sns.distplot(df_train[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1])
    sns.distplot(df_test[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1])
    
    axs[0].set_xlabel(feature, size=15, labelpad=5)
    axs[1].set_xlabel(feature, size=15, labelpad=5)
    axs[0].set_ylabel('')
    axs[1].set_ylabel('')
    
    axs[0].tick_params(axis='x', labelsize=15)
    axs[1].tick_params(axis='x', labelsize=15)
    axs[0].tick_params(axis='y', labelsize=15)
    axs[1].tick_params(axis='y', labelsize=15)
    
    axs[0].legend(loc='upper right', prop={'size': 14})
    axs[1].legend(loc='upper right', prop={'size': 14})
    
    axs[0].set_title('{} Survival Distribution'.format(feature), size=15)
    axs[1].set_title('{} Distribution'.format(feature), size=15)
    
    plt.show()


# ### 4.1 Survived -> Target

# This is our target column, meaning this is what our model will predict (**0** or **1**)
# * **1** = Survived
# * **0** = Not Survived

# In[ ]:


sns.countplot(df_train['Survived'])
plt.xlabel('Survival')
plt.ylabel('Passenger Count')
plt.title('Training Set Survival Distribution')
plt.show()
print(df_train['Survived'].value_counts())


# ### 4.2 Pclass -> Feature

# Pclass means Passenger Class and it is a categorical feature with 3 values:
# * **1** = Upper Class
# * **2** = Middle Class
# * **3** = Lower Class

# In[ ]:


cat_plot('Pclass')


# ### 4.3 Sex -> Feature

# Sex is self explainatory, it is a categorical feature with two values:
# * **male**
# * **female**

# In[ ]:


cat_plot('Sex')


# ### 4.4 Age -> Feature

# In[ ]:


con_plot('Age')


# Age has many missing values so lets find out a way to replace those missing values. As Pclass and Sex are highly correlated to Age, it would be wise to use the mean of these for missing values of Age.

# In[ ]:


for df in [df_train, df_test]:
    df['Age'].fillna(df.groupby(['Pclass','Sex'])['Age'].transform('mean'), inplace=True)

print('No of missing ages in training set: {}'.format(df_train['Age'].isnull().sum()))
print('No of missing ages in test set: {}'.format(df_test['Age'].isnull().sum()))


# ### 4.5 SibSp -> Feature

# In[ ]:


cat_plot('SibSp')


# ### 4.6 Parch -> Feature

# In[ ]:


cat_plot('Parch')


# ### 4.7 Fare -> Feature

# In[ ]:


con_plot('Fare')


# Fare column has only 1 missing value in the test set, so lets replace it using the mean value.

# In[ ]:


df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean())
df_test['Fare'].isnull().sum()


# ### 4.8 Embarked -> Feature

# In[ ]:


cat_plot('Embarked')


# Embarked column has 2 missing values in the training set, so lets remove those rows from the training set.

# In[ ]:


df_train.dropna(subset=['Embarked'], inplace=True)
df_train['Embarked'].isnull().sum()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# ## 5. Feature Engineering

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# ### 5.1 Label Encoding

# Now we will convert non-numeric features (Sex and Embarked) to numeric features using Label Encoder.

# In[ ]:


le = LabelEncoder()

for df in [df_train, df_test]:
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# ### 5.2 Splitting Training and Test Data

# In[ ]:


Y_train = df_train.loc[:, ['Survived']]
X_train = df_train.drop(axis=1, columns=['Survived'])


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split( X_train, Y_train, test_size = 0.2, random_state = 42 )


# ### 5.3 Scale and Transform Data

# In[ ]:


ss = StandardScaler()
ss.fit(X_train)
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)


# ## 6. Training Multiple Models

# In[ ]:


def model_prediction(model):
    cv = cross_val_score(model, x_train, y_train, cv=5)
    print('Cross Validation Score: {}'.format(cv))
    print('Mean Cross Validation Score: {}'.format(cv.mean()))
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = round(accuracy_score(y_pred, y_test) * 100, 2)
    print('Model Accuracy: {} %'.format(accuracy))
    model_pred = model.predict(ss.transform(df_test))
    return model_pred


# In[ ]:


def model_submission(y_pred):
    prediction = {'PassengerId': test_data.PassengerId, 'Survived': y_pred}
    submission = pd.DataFrame(data=prediction)
    print(submission['Survived'].value_counts())
    return submission


# ### 6.1 Logistic Regression

# In[ ]:


lr = LogisticRegression(max_iter = 2000, C=1.5, penalty='l1', solver='liblinear')
lr_pred = model_prediction(lr)
lr_submission = model_submission(lr_pred)
lr_submission.to_csv('lr_submission.csv', index=False)


# ### 6.2 Stochastic Gradient Descent Classifier

# In[ ]:


sgd = SGDClassifier()
sgd_pred = model_prediction(sgd)
sgd_submission = model_submission(sgd_pred)
sgd_submission.to_csv('sgd_submission.csv', index=False)


# ### 6.3 Gaussian Naive Bayes

# In[ ]:


gnb = GaussianNB()
gnb_pred = model_prediction(gnb)
gnb_submission = model_submission(gnb_pred)
gnb_submission.to_csv('gnb_submission.csv', index=False)


# ### 6.4 Support Vector Machine

# In[ ]:


svc = SVC(gamma=1.0, probability=True)
svc_pred = model_prediction(svc)
svc_submission = model_submission(svc_pred)
svc_submission.to_csv('svc_submission.csv', index=False)


# ### 6.5 Linear Support Vector Machine

# In[ ]:


linear_svc = LinearSVC()
linear_svc_pred = model_prediction(linear_svc)
linear_svc_submission = model_submission(linear_svc_pred)
linear_svc_submission.to_csv('linear_svc_submission.csv', index=False)


# ### 6.6 K Nearest Neighbour Classifier

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=13)
knn_pred = model_prediction(knn)
knn_submission = model_submission(knn_pred)
knn_submission.to_csv('knn_submission.csv', index=False)


# ### 6.7 Decision Tree Classifier

# In[ ]:


dt = DecisionTreeClassifier(random_state=1)
dt_pred = model_prediction(dt)
dt_submission = model_submission(dt_pred)
dt_submission.to_csv('dt_submission.csv', index=False)


# ### 6.8 Random Forest Classifier

# In[ ]:


rf = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=7, n_estimators=550, random_state=42)
rf_pred = model_prediction(rf)
rf_submission = model_submission(rf_pred)
rf_submission.to_csv('rf_submission.csv', index=False)


# ### 6.9 AdaBoost Classifier

# In[ ]:


adc = AdaBoostClassifier(random_state=1)
adc_pred = model_prediction(adc)
adc_submission = model_submission(adc_pred)
adc_submission.to_csv('adc_submission.csv', index=False)


# ### 6.10 eXtreme Gradient Boosting Classifier

# In[ ]:


xgb = XGBClassifier(colsample_bytree=0.8,
                    gamma=0.5,
                    learning_rate=0.5,
                    max_depth=None,
                    min_child_weight=0.01,
                    n_estimators=550,
                    reg_alpha=1,
                    reg_lambda=10,
                    sampling_method='uniform',
                    subsample=0.65)
xgb_pred = model_prediction(xgb)
xgb_submission = model_submission(xgb_pred)
xgb_submission.to_csv('xgb_submission.csv', index=False)


# In[ ]:




