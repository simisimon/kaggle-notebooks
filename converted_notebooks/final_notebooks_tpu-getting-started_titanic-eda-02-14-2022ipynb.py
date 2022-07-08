#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # **1. Define the Problem**
# 
# **What sorts of people were more likely to survive?**

# # **2. Observe data** 
# **What we got?**
# 
# There are 10 variables in the train dataset and test dataset:
# * Passengerid
# * Name
# * survival: Survival, 0 = No, 1 = Yes
# * pclass: Ticket class for socio-economic status, 1 = 1st(Upper), 2 = 2nd(Middle), 3 = 3rd(Lower)
# * sex: Sex
# * Age: Age in years
# * sibsp: # of siblings / spouses aboard the Titanic
# * parch: # of parents / children aboard the Titanic
# * ticket: Ticket number
# * fare: Passenger fare
# * cabin: Cabin number, reference: https://www.encyclopedia-titanica.org/cabins.html
# * embarked: Port of Embarkation, C = Cherbourg, Q = Queenstown, S = Southampton
# 

# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
print(train.shape)
print(train.head(5))
print('_________________________________________\n','Train Data Unique:\n',train.nunique())
print('_________________________________________\n','Train Data Description:\n',train.describe())
print(train.info())


# **Observations of Train Data :**
# * The shape of train dataset is (891, 12).
# * The unique value of “PassengerId” and “Name” both are 891, means there are 891 individuals aboard. Nominal
# * Age from 0.42 years old to 80 years old. However, there are missing data in the “Age” column: 177(891-714) out of 891. Continuous
# * SibSp from 0 to 8 (0,1,2,3,4,5,8). Discrete
# * Parch from 0 to 6 (0,1,2,3,4,5,6). Discrete
# * Ticket : There are 681 distinct ticket numbers out of 891 tickets, and the data type is object, which means repeated data. Look deep into the data would find same ticket number occurs with same Fare, and these people may from a family since they have same last name. Discrete
# * Fare from 0 to 512.3292. Continuous
# * Cabin has plenty missing data, we only got 204 counts out of 891 individuals.
# * Embarked has only 2 missing data. Discrete
# 

# In[ ]:


test = pd.read_csv("/kaggle/input/titanic/test.csv")
print(test.shape)
print(test.head(5))
print('_________________________________________\n','Test Data Unique:\n',test.nunique())
print('_________________________________________\n','Test Data Description:\n',test.describe())
print(test.info())


# **Observations of Test Data :**
# * The shape of train dataset is (418, 11).
# * Aged from 0.17 to 76, there are 86(418-332) missing value.
# * SibSp from 0 to 8 (0,1,2,3,4,5,8). Discrete
# * Parch from 0 to 9 (0,1,2,3,4,5,6,9). Discrete
# * Fare from 0 to 512.3292

# # **3. Pre-processing**
# 
# (1) "Survived" column is dependent variable. We separate it from train dataset to train_survived, a new dataset.

# In[ ]:


train_survired = train.Survived
train_x = train.drop('Survived', 1, inplace=False)
print('train_x shape: ', train_x.shape)
print('train_survived shape: ', train_survired.shape)


# (2) Combine train dataset and test dataset, which can process same method for making up missing data.

# In[ ]:


train_test_x = train_x.append(test)
print('train_test_x shape: ', train_test_x.shape)


# (3) Observe over-all dataset

# In[ ]:


print('train_test_x element: \n', train_test_x.columns.values)
print('_________________________________________\n','train_test_x description: \n', train_test_x.describe())
print(train_test_x.info())
print('_________________________________________\n','train_test_x unique: \n', train_test_x.nunique())


# # **4.EDA vs. Single independent variable**
# 
# **(1) Survived**

# In[ ]:


# descrption
train_eda = train.drop('PassengerId', 1, inplace=False)
print('How many people aboard: ', train_eda['Name'].count())
print('How many people survived: ', train_eda.Survived.sum())
print('Survived Rate: ', train_eda['Survived'].sum()/ train_eda['Survived'].count())
print(train_eda.groupby(['Survived'], as_index = False).mean())

# plot
sns.countplot(x='Survived', data=train)


# **(2) Class & Survived - Survived rate: Pclass 1 > Pclass 2 > Pclass 3**

# In[ ]:


# description
print(train_eda[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).agg(['count','sum']))
print(train_eda[['Pclass', 'Survived', 'Age', 'SibSp', 'Parch', 'Fare']].groupby(['Pclass'], as_index = False).mean())

# plot
pclass_survived = train_eda[train_eda['Survived']==1]['Pclass'].value_counts()
pclass_dead = train_eda[train_eda['Survived']==0]['Pclass'].value_counts()
pclass = pd.DataFrame([pclass_survived, pclass_dead], index=['survived', 'dead'])
print(pclass.head(5))
pclass.plot(kind = 'bar')


# **(3) Sex & Survived - Survived rate: Female > Male**

# In[ ]:


# description
print(train_eda[['Sex', 'Survived']].groupby(['Sex'], as_index = False).agg(['count','sum']))
print(train_eda[['Sex', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].groupby(['Sex'], as_index = False).mean())

# plot
sex_survived = train_eda[train_eda['Survived']==1]['Sex'].value_counts()
sex_dead = train_eda[train_eda['Survived']==0]['Sex'].value_counts()
sex = pd.DataFrame([sex_survived, sex_dead], index=['survived', 'dead'])
sex.plot(kind = 'bar')


# **(4) Age & Survived - Survived rate: toddler & aged**

# In[ ]:


# description
# Age is continuous variable
# plot
age_all = train_eda['Age'].value_counts()
age_survived = train_eda[train_eda['Survived']==1]['Age'].value_counts()
age_dead = train_eda[train_eda['Survived']==0]['Age'].value_counts()
age = pd.DataFrame([age_all, age_survived, age_dead], index = ['ALL','Survived', 'Dead']).T #T:Transpose
age.plot(kind = 'line', figsize=(30,8))

# → Too mess to find anything valued. Maybe group Age variable to 8 bins would help.


# In[ ]:


# Age & Survived (cont.)

# 將age分組_分8組
Age_range = ['A','B','C','D','E','F','G','H','I','J']
train_eda['Age_range'] = pd.cut(train_eda['Age'], bins = len(Age_range))
print(train_eda[['Age_range','Sex', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare']].groupby(['Age_range'], as_index = False).mean())

age_survived = train_eda[train_eda['Survived']==1]['Age_range'].value_counts()
age_dead = train_eda[train_eda['Survived']==0]['Age_range'].value_counts()
age = pd.DataFrame([age_survived, age_dead], index = ['Survived', 'Dead']).T #T:Transpose
age.plot(kind = 'line', figsize=(30,8))


# **(5) Fare & Survived**

# In[ ]:


# description
fare_survived = train_eda[train_eda['Survived']==1]['Fare']
fare_dead = train_eda[train_eda['Survived']==0]['Fare']
fare = pd.DataFrame([fare_survived, fare_dead],  index = ['Survived', 'Dead']).T
plt.hist(fare_survived, fare_dead, bins = 30,label = ['Survived', 'Dead'])


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

y = train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was sussfully saved")


# References:
# * https://chtseng.wordpress.com/2017/12/24/kaggle-titanic%E5%80%96%E5%AD%98%E9%A0%90%E6%B8%AC-1/
