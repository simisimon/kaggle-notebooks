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


# In[ ]:


data_test = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
data_test.head()


# In[ ]:


data_test.info()


# In[ ]:


data_train = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
data_train.tail()


# # **EDA for Data Train**

# In[ ]:


data_train['HomePlanet'].count()


# In[ ]:


data_train.info()


# In[ ]:


data_train.isna().sum()


# In[ ]:


from sklearn.preprocessing import LabelEncoder 
label_encoder= LabelEncoder()

data_train['Transported'] = label_encoder.fit_transform(data_train['Transported'])
data_train['VIP'] = label_encoder.fit_transform(data_train['VIP'])
data_train['CryoSleep'] = label_encoder.fit_transform(data_train['CryoSleep'])



# In[ ]:


data_train.dropna(inplace=True) #let's eliminate nan values


# In[ ]:


data_train.isna().sum() #we're checking the missing values once again


# In[ ]:


data_train


# In[ ]:


#let's transform the categorical data to be numerical data so we can apply the machine learning algorithm into the dataset
data_train['HomePlanet'] = label_encoder.fit_transform(data_train['HomePlanet']) 


# In[ ]:


data_train.describe()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.subplot(2,1,1)
plt.hist(data_train['Age'])
plt.title("histogram of Age Distribution")

plt.subplot(2,1,2)
plt.hist(data_train['HomePlanet'])
plt.title("histogram of HomePlanet Distribution")


# In[ ]:


data_train[data_train["HomePlanet"]==0] #Europe


# In[ ]:


data_train[data_train["HomePlanet"]==1] #Earth


# In[ ]:


data_train[data_train["HomePlanet"]==2] #Mars


# In[ ]:


data_train["Destination"]


# In[ ]:


import seaborn as sns

#let's plot the destination vs age 

sns.barplot(x= data_train["Destination"], y= data_train['Age']) 


# In[ ]:


#let's plot homeplanet vs age 
sns.barplot(x= data_train["HomePlanet"], y= data_train['Age']) 


# # ****EDA for Data Test****

# In[ ]:


data_test.head()


# In[ ]:


data_test.isna().sum() #Checking NaN values


# In[ ]:


data_test.info()


# In[ ]:


data_test['Age'] #There are some NaN Values in "Age" column


# In[ ]:


#fill NaN value of Age with Mean
mean = data_test['Age'].mean()
data_test['Age'] = data_test['Age'].replace(np.nan, mean)
data_test


# In[ ]:


data_test.isna().sum()


# In[ ]:


from sklearn.preprocessing import LabelEncoder 
label_encoder= LabelEncoder()

data_test['HomePlanet'] = label_encoder.fit_transform(data_test['HomePlanet'])
data_test['CryoSleep'] = label_encoder.fit_transform(data_test['CryoSleep'])
data_test['VIP'] = label_encoder.fit_transform(data_test['VIP'])


# In[ ]:


data_test.head()


# In[ ]:


plt.subplot(2,1,1)
plt.hist(data_test['Age'])
plt.title("histogram of Age Distribution")

plt.subplot(2,1,2)
plt.hist(data_test['HomePlanet'])
plt.title("histogram of HomePlanet Distribution")


# In[ ]:


import seaborn as sns

#let's plot the destination vs age 

sns.barplot(x= data_test["Destination"], y= data_test['Age']) 


# In[ ]:


#let's plot homeplanet vs age 
sns.barplot(x= data_test["HomePlanet"], y= data_test['Age']) 


# # ****Let's Process the data train and data test

# In[ ]:


no= np.arange(0,400,1)
no= pd.DataFrame(no)


# In[ ]:


data_train


# In[ ]:


X_Train= data_train.iloc[:, [2,5,6,7,8,9,10,11]].values


# In[ ]:


#dealing with missing value on integer object
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_Train)
X_Train = imputer.transform(X_Train)


# In[ ]:


y_train= data_train['Transported'].values


# > * **After doing the EDA, let's predict wheter a passengger will be transported or not by executing the dataset using classification algorthms**
# > * **I use 2 classification algorithms to predict the result**
# > * **I use classification algorithm because the result of prediction will be classification (Transported(1) or not(0)******

# # **first, we will try to predict the transported passengers using decision Tree**

# 

# Decision Tree is one of Machine Learning Algorithm classification that we can use to predict something based on dataset that we have

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_Train, y_train)


# In[ ]:


X_test= data_test.iloc[:, [2,5,6,7,8,9,10,11]].values


# In[ ]:


#dealing with missing value on integer object
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_test)
X_test = imputer.transform(X_test)


# In[ ]:


prediction = classifier.predict(X_test)


# In[ ]:


print("prediction result =", prediction)
print("training data = ", y_train)



# # **second, we will try to predict the transported passengers with random forest classifier**

# Random Forest one of Machine Learning algorithm classification that we can use to make prediction.
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier_2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_2.fit(X_Train, y_train)


# In[ ]:


prediction_2= classifier_2.predict(X_test)


# In[ ]:


print("prediction result =", prediction_2)
print("training data = ", y_train)


# In[ ]:


plt.hist(prediction_2)


# In[ ]:


plt.hist(y_train)


# In[ ]:


my_submission_titanic= pd.DataFrame({'PassengerId':data_test.PassengerId, 'Transported': prediction_2 })

my_submission_titanic.to_csv('submission.csv', index=False)


# In[ ]:


my_submission_titanic_2= pd.DataFrame({'PassengerId':data_test.PassengerId, 'Transported': prediction})

my_submission_titanic_2.to_csv('submission.csv', index=False)

