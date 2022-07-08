#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# First we will put our .csv files to pandas data frames

# In[ ]:


train_url='/kaggle/input/titanic/train.csv'
test_url='/kaggle/input/titanic/test.csv'
gender_submission_url='/kaggle/input/titanic/gender_submission.csv'
df_train=pd.read_csv(train_url)
df_test=pd.read_csv(test_url)


# In[ ]:


y_test=pd.read_csv(gender_submission_url)


# In[ ]:


y_test.set_index('PassengerId')


# In[ ]:


y_train=df_train.drop(columns=['Name', 'Ticket', 'Cabin', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'PassengerId'])


# In[ ]:


x_train=df_train.drop(columns=['Name', 'Ticket', 'Cabin'])


# In[ ]:


x_test = df_test.drop(columns=['Name', 'Ticket', 'Cabin'])


# We calculate the mean Age to replace missing values of age of passengers.

# In[ ]:


age_mean_train = x_train['Age'].mean()
#x['Age'].fillna(value=age_mean, inplace=True)
age_mean_test = x_test['Age'].mean()


# We calculate the mean Fare to put for missing fares.

# In[ ]:


fare_mean_test = x_test['Fare'].mean()


# In[ ]:


x_train['Age'].fillna(value=age_mean_train, inplace=True)
x_test['Age'].fillna(value=age_mean_test, inplace=True)
x_test['Fare'].fillna(value=fare_mean_test, inplace=True)


# In[ ]:


x_train['Embarked'].fillna(value='S', inplace=True)
x_test['Embarked'].fillna(value='S', inplace=True)


# In[ ]:


x_train.set_index('PassengerId')


# We should look to see what is the most frequent value for Embarked and replace the empty cells with that

# In[ ]:


x_train['Embarked'].value_counts()


# Now we should do 'one hot encoding' to map the categorical variables to numerical so our models are not thrown off

# In[ ]:


x_train['Embarked']=x_train['Embarked'].replace(to_replace="S",value="1")
x_train['Embarked']=x_train['Embarked'].replace(to_replace="C",value="2")
x_train['Embarked']=x_train['Embarked'].replace(to_replace="Q",value="3")
x_train['Sex']=x_train['Sex'].replace(to_replace="male",value="1")
x_train['Sex']=x_train['Sex'].replace(to_replace="female",value="2")


# In[ ]:


x_test['Embarked']=x_test['Embarked'].replace(to_replace="S",value="1")
x_test['Embarked']=x_test['Embarked'].replace(to_replace="C",value="2")
x_test['Embarked']=x_test['Embarked'].replace(to_replace="Q",value="3")
x_test['Sex']=x_test['Sex'].replace(to_replace="male",value="1")
x_test['Sex']=x_test['Sex'].replace(to_replace="female",value="2")


# In[ ]:


sns.set_theme(style="ticks", palette="pastel")
# Draw a nested boxplot to show age by passenger class and sex
sns.boxplot(x="Pclass", y="Age",
            hue="Survived", palette=["m", "g"],
            data=x_train)
sns.despine(offset=10, trim=True)


# In[ ]:


sns.regplot(x="Pclass", y="Survived", data=x_train)


# In[ ]:


sns.regplot(x="Age", y="Survived", data=x_train)


# In[ ]:


sns.regplot(x="Fare", y="Survived", data=x_train)


# In[ ]:


sns.catplot(x="Pclass", y="Age", hue="Survived",
            kind="violin", split=True, data=x_train)


# In[ ]:


sns.catplot(x="Embarked", y="Age", hue="Survived",
            kind="violin", split=True, data=x_train)


# In[ ]:


x_train.head()


# In[ ]:


x_train=x_train.drop(columns=['Survived', 'SibSp', 'Parch', 'Embarked'])


# In[ ]:


x_test=x_test.drop(columns=['SibSp', 'Parch', 'Embarked'])


# In[ ]:


x_test.set_index('PassengerId')


# In[ ]:


x_test.head()


# In[ ]:


x_train.head()


# We need to import sklearn and standardize our data so that our models can properly process our data

# In[ ]:


from sklearn import preprocessing


# In[ ]:


x_train= preprocessing.StandardScaler().fit(x_train).transform(x_train)
x_train[0:5]


# In[ ]:


x_test= preprocessing.StandardScaler().fit(x_test).transform(x_test)
x_test[0:5]


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# In[ ]:


parameters = {'kernel':('linear','rbf','sigmoid'),
              'C': (1,2,3),
              'gamma':(1,2,3)}
svm = SVC()


# In[ ]:


y = y_train.values.ravel()
y_train = np.array(y).astype(int)


# In[ ]:


svm_cv = GridSearchCV(svm, parameters, cv=10)
svm_cv.fit(x_train, y_train)


# In[ ]:


submission = svm_cv.predict(x_test)


# In[ ]:


submission = pd.DataFrame(submission)


# In[ ]:


#to add another column with index
#submission.reset_index().set_index('index', drop=False)


# In[ ]:


submission.rename(columns={0:'Survived'}, inplace=True)


# In[ ]:


submission.head()


# In[ ]:


y_test['PassengerId']=df_test['PassengerId']


# In[ ]:


y_test.set_index('PassengerId')


# In[ ]:


print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)


# In[ ]:


submission = pd.DataFrame(submission)


# In[ ]:


#submission["PassengerId"]=passengerid


# In[ ]:


submission.rename(columns={0: "Survived"}, inplace=True)


# In[ ]:


submission.head()


# In[ ]:


submission['PassengerId']=df_test['PassengerId']


# In[ ]:


submission = submission[['PassengerId', 'Survived']]


# In[ ]:


submission.head()


# In[ ]:


submission.set_index('PassengerId', inplace=True)


# In[ ]:


#submission.columns=["PassengerId", "Survived"]
submission.shape


# In[ ]:


submission.tail()


# In[ ]:


submission.to_csv('submission.csv')

