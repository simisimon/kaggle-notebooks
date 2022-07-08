#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# In[ ]:


df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')


# # Data Processing

# ## General data overview

# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# Preliminary conclusions:
# - Data is not complete, there are missing values in several columns (Age, Embarked, Cabin, Fare). We will need to fill in the missing points in useful columns
# - It seems that we won't need some of the columns. Specifically, columns Name, Cabin, and Ticket don't seem to carry useful information
# - We will probably need to encode Sex and Embarked, depending on the algorithm of choice

# ## Feature removal

# In[ ]:


def remove_features(df):
    '''Return a dataframe without unuseful columns.'''
    if 'Survived' in df.columns.values:
        return df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']].copy()
    else:
        return df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()


# In[ ]:


train_set = remove_features(df_train)


# In[ ]:


test_set = remove_features(df_test)


# In[ ]:


train_set.head()


# In[ ]:


train_set.info()


# ## Missing data handling

# We need to handle following missing data:
# 1. Age - will be filled by median in a grouped manner
# 2. Fare - only one point is missing in the test set, thus will be filled with a median value of a test set
# 3. Cabin - dropped due to too many missing values and nature of the property
# 4. Embarked - only two points are missing, thus will be filled with mode value
# 
# Since Age property is of importance and has a significiant portion of missing values, we will first evaluate the distribution of those missing values against other properties. This is done in order to verify that there are no specific patterns in missing data.

# ### Age

# In[ ]:


for column in train_set.columns.values:
    if column not in ['Age', 'Fare'] :
        display(pd.crosstab(train_set[column], train_set['Age'].isnull(), margins=True))
        print("_" * 40)


# There is some notable disbalance in missing points in Embarked property, but we will disregard this fact because disbalanced class is quite small.
# 
# We will fill in the missing points after train/dev split in order to avoid data leak.

# ### Embarked

# In[ ]:


train_set['Embarked'].mode()[0]


# In[ ]:


train_set.loc[train_set['Embarked'].isnull(), 'Embarked'] = train_set['Embarked'].mode()[0]
train_set['Embarked'].value_counts()


# ## Feature altering

# In[ ]:


train_set.head()


# In[ ]:


le = LabelEncoder()


# In[ ]:


train_set['Sex'] = le.fit_transform(train_set['Sex'])
test_set['Sex'] = le.transform(test_set['Sex'])


# In[ ]:


train_set['Embarked'] = le.fit_transform(train_set['Embarked'])
test_set['Embarked'] = le.transform(test_set['Embarked'])


# In[ ]:


train_set.head()


# # EDA

# In[ ]:


sns.catplot(x='Pclass', col='Survived', hue='Sex', data=train_set, kind='count');


# In[ ]:


plt.figure(figsize=(16,10))
sns.histplot(x='Fare', hue='Survived', data=train_set, element='step', stat="density", common_norm=False, alpha=0.1, kde=True);


# In[ ]:


plt.figure(figsize=(16, 8))
sns.boxplot(y='Survived', x='Fare', hue='Pclass', data=train_set, orient='h');


# In[ ]:


plt.figure(figsize=(16, 8))
sns.boxplot(y='Sex', x='Age', hue='Survived', data=train_set, orient='h');


# # ML Model

# In[ ]:


RS = 8  # Random State


# In[ ]:


X = train_set.iloc[:,:-1].copy()
y = train_set['Survived'].copy()


# In[ ]:


X.head()


# In[ ]:


X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, shuffle=True, random_state = RS)


# ## Filling in missing Age points

# In[ ]:


mean_ages_df = X_train.groupby(['Pclass', 'Sex']).mean()['Age']
mean_ages_df


# In[ ]:


for Pclass in range(1, 4):
    for Sex in range(0, 2):
        X_train.loc[(X_train['Age'].isnull()) & 
                    (X_train['Pclass'] == Pclass) &
                    (X_train['Sex'] == Sex), 'Age'] = mean_ages_df[Pclass, Sex]


# In[ ]:


mean_ages_df_dev = X_dev.groupby(['Pclass', 'Sex']).mean()['Age']
mean_ages_df_dev


# In[ ]:


for Pclass in range(1, 4):
    for Sex in range(0, 2):
        X_dev.loc[(X_dev['Age'].isnull()) & 
                  (X_dev['Pclass'] == Pclass) &
                  (X_dev['Sex'] == Sex), 'Age'] = mean_ages_df_dev[Pclass, Sex]


# ## Random Forest

# In[ ]:


rfc = RandomForestClassifier(
    random_state=RS,
    verbose=1
)


# In[ ]:


rfc.fit(X_train, y_train)


# In[ ]:


rfc.score(X_dev, y_dev)


# In[ ]:


y_pred = rfc.predict(X_dev)


# In[ ]:


print(classification_report(y_dev, y_pred))


# Good enough for my first independent submission :D

# ## Test set prediction

# In[ ]:


test_set.head()


# In[ ]:


test_set.info()


# In[ ]:


test_set.loc[test_set['Fare'].isnull(), 'Fare'] = test_set['Fare'].median()


# In[ ]:


mean_ages_df_test = test_set.groupby(['Pclass', 'Sex']).mean()['Age']
mean_ages_df_test


# In[ ]:


for Pclass in range(1, 4):
    for Sex in range(0, 2):
        test_set.loc[(test_set['Age'].isnull()) & 
                     (test_set['Pclass'] == Pclass) &
                     (test_set['Sex'] == Sex), 'Age'] = mean_ages_df_test[Pclass, Sex]


# In[ ]:


test_set.info()


# In[ ]:


predictions = rfc.predict(test_set)


# In[ ]:


predictions = pd.DataFrame(data = predictions,
                          columns=['Survived'])
predictions


# In[ ]:


result = pd.concat([df_test['PassengerId'], predictions], axis=1)
result


# In[ ]:


result = result.set_index('PassengerId')
result


# In[ ]:


result.to_csv('submission.csv')


# Submitted result scored for 0.74401
