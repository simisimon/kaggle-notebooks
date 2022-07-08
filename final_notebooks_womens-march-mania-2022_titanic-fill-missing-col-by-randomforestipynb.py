#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from pandas.core.frame import DataFrame

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.model_selection import cross_val_score

import seaborn as sns
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


df = pd.read_csv("/kaggle/input/titanic/train.csv")
df.head()


# # I - Explore Data Analysis

# In[ ]:


df.info()


# In[ ]:


df.Survived.value_counts(normalize=True).plot(kind="pie")
df['Survived'].value_counts(normalize=True) * 100


# # II - Preprocessing

# ## 1. Fill Embarked

# In[ ]:


def fill_embarked(df: DataFrame):
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df

df = fill_embarked(df)


# ## 2. Encode features

# ### Age

# In[ ]:


def encode_age(df: DataFrame):
    for i in range(len(df['Name'])):
        if not df['Age'][i] == None:
            if df['Age'][i] <= 5: 
                df['Age'][i] = 0
            if 5 < df['Age'][i] <= 15: 
                df['Age'][i] = 1
            if 15 < df['Age'][i] <= 63: 
                df['Age'][i] = 2
            if df['Age'][i] > 63: 
                df['Age'][i] = 3
    return df

df = encode_age(df)


# In[ ]:


fig = plt.figure(figsize=(15,6))
i=1
for title in range(4):
    fig.add_subplot(1, 4, i)
    plt.title('Title : {}'.format(title))
    df.Survived[df['Age'] == title].value_counts().plot(kind='pie')
    i += 1


# ### Sex

# In[ ]:


def encode_sex(df: DataFrame): 
    replacement = {
        'male': 0, 'female': 1
    }
    df['Sex'] = df['Sex'].apply(lambda x: replacement.get(x))
    return df

df = encode_sex(df)


# ### Embarked

# In[ ]:


def encode_embarked(df: DataFrame): 
    replacement = {
        'C': 0, 'Q': 1, 'S': 2
    }
    df['Embarked'] = df['Embarked'].apply(lambda x: replacement.get(x))
    return df
df = encode_embarked(df)


# ## 3. Create Ticket Frequency Column

# In[ ]:


def create_ticket_freq_col(df: DataFrame):
    df['Ticket_Freq'] = df.groupby('Ticket')['Ticket'].transform('count')
    return df

df = create_ticket_freq_col(df)


# In[ ]:


fig, axs = plt.subplots(figsize=(10, 5))
sns.countplot(x='Ticket_Freq', hue='Survived', data=df)

plt.xlabel('Ticket Frequency', size=20, labelpad=20)
plt.ylabel('Passenger Count', size=20, labelpad=20)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Count of Survival in {} Feature'.format('Ticket Frequency'), size=24, y=1.05)

plt.show()


# ## 4. Transform

# ### Name

# In[ ]:


def transform_name(df: DataFrame): 
    df['Name'] = df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
    return df

df = transform_name(df)


# In[ ]:


fig = plt.figure(figsize=(15,6))

i=1
for title in df['Name'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Title : {}'.format(title))
    df.Survived[df['Name'] == title].value_counts().plot(kind='pie')
    i += 1


# In[ ]:


def encode_name(df: DataFrame):
    replacement = {
        'Don': 0, 'Rev': 0, 'Jonkheer': 0, 'Capt': 0,
        'Mr': 1,
        'Dr': 2,
        'Col': 3, 'Major': 3,
        'Master': 4,
        'Miss': 5,
        'Mrs': 6,
        'Dona': 7, 'Mme': 7, 'Ms': 7, 'Mlle': 7, 'Sir': 7, 'Lady': 7, 'the Countess': 7
    }
    df['Name'] = df['Name'].apply(lambda x: replacement.get(x))
    return df

df = encode_name(df)


# In[ ]:


fig = plt.figure(figsize=(12,7))

i=1
for title in df['Name'].unique():
    fig.add_subplot(3, 4, i)
    plt.title('Title : {}'.format(title))
    df.Survived[df['Name'] == title].value_counts().plot(kind='pie')
    i += 1


# ### Cabin

# In[ ]:


def transform_cabin(df: DataFrame): 
    df['Cabin'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else None)
    return df

df = transform_cabin(df)


# In[ ]:


fig = plt.figure(figsize=(15,6))

i=1
for title in df['Cabin'].unique():
    if title == None: 
        continue
    fig.add_subplot(3, 4, i)
    plt.title('Title : {}'.format(title))
    df.Survived[df['Cabin'] == title].value_counts().plot(kind='pie')
    i += 1


# In[ ]:


def encode_cabin(df: DataFrame):
    replacement = {
        'A': 0, 'B': 1, 'C': 3, 'D': 4, 'E': 5, 'F': 6,  'G': 7, 'T': 8
    }
    df['Cabin'] = df['Cabin'].apply(lambda x: replacement.get(x))
    return df

df = encode_cabin(df)


# ## 4. Drop columns

# In[ ]:


df = df.drop(columns=["PassengerId", "Ticket"])


# In[ ]:


df.head()


# ## 5. Fill columns

# ### Fare

# In[ ]:


def fill_fare(df: DataFrame):
    median_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
    df['Fare'] = df['Fare'].fillna(median_fare)
    df['Fare'] = StandardScaler().fit_transform(df['Fare'].values.reshape(-1, 1))
    return df

df = fill_fare(df)


# ### Age

# In[ ]:


def discard_missing_row(nameCol: str, df: DataFrame):
    data = []
    num_missing_row = 0
    for row in range(len(df)):
        val = df[nameCol][row]
        if val != val:  # If element is the empty element
            num_missing_row += 1
            continue
        data.append(df.iloc[row, :])

    return pd.DataFrame(data, columns=[col for col in df])


# In[ ]:


def fill_empty_element(arr, val):
    for i in range(len(arr)):
        if arr[i] != arr[i]:  # If element is empty
            arr[i] = val
    return arr


# In[ ]:


def decision_tree(df: DataFrame, features, target, min_samples_leaf):
    le = preprocessing.LabelEncoder()
    df = df.apply(le.fit_transform)

    x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.1, random_state=0)  

    clf = RandomForestClassifier(random_state=47, criterion= 'entropy', min_samples_leaf = min_samples_leaf, bootstrap=True)
    clf = clf.fit(x_train.values, y_train.values)
    
    y_pred = clf.predict(x_test.values)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    
    scores = cross_val_score(clf, df[features], df[target], cv=5)
    print("Cross-validation:", scores)
    return clf


# In[ ]:


def fill_col(nameCol, model, df, features):
    modified_df = df     

    #Find the index of row having missing value for nameCol attribute
    list_index = modified_df[modified_df[nameCol].isnull()].index.tolist()

    #Use Decision Tree model to predict the nameCol 
    list_label = model.predict(df.loc[list_index, features].values)
    
    #Fill the missing value
    for index in range (len(list_index)):
            modified_df.at[list_index[index],nameCol] = str(list_label[index])

    return modified_df


# In[ ]:


# Reallocating the Age col
df = df[[c for c in df if not c == "Age"] + ["Age"]]

model_df = discard_missing_row("Age", df)

model_age = decision_tree(model_df, ["Name", "Pclass", "Sex", "SibSp", "Parch", "Ticket_Freq"], "Age", 5)

df = fill_col("Age", model_age, df, ["Name", "Pclass", "Sex", "SibSp", "Parch", "Ticket_Freq"])


# In[ ]:


df.info()


# ### Cabin

# In[ ]:


model_df = discard_missing_row("Cabin", df)

model_cabin = decision_tree(model_df, ["Pclass", "Fare", "Embarked"], "Cabin", 1)

df = fill_col("Cabin", model_cabin, df, ["Pclass", "Fare", "Embarked"])


# In[ ]:


df = df.drop(columns=["Fare", "Embarked"])


# In[ ]:


df.head()


# # III - Model

# In[ ]:


df = df[[c for c in df if not c == "Survived"] + ["Survived"]]

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.2, random_state=34)  

clf = RandomForestClassifier(n_estimators=50, random_state=0, criterion='entropy', min_samples_leaf=5)

clf = clf.fit(x_train.values, y_train.values)
y_pred = clf.predict(x_test.values)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

scores = cross_val_score(clf, df.iloc[:,:-1], df.iloc[:,-1], cv=5)
print("Cross-validation:", scores)


# # IV - Test

# In[ ]:


test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


test.info()


# In[ ]:


test = fill_embarked(test)

test = encode_age(test)
test = encode_sex(test)
test = encode_embarked(test)

test = create_ticket_freq_col(test)

test = transform_name(test)
test = encode_name(test)

test = transform_cabin(test)
test = encode_cabin(test)


# In[ ]:


result = pd.DataFrame()
result["PassengerId"] = test.PassengerId

test = test.drop(columns=["PassengerId", "Ticket"])


# In[ ]:


test.info()


# In[ ]:


# Fill Fare col
test = fill_fare(test)

# Reallocating the Age col
test = test[[c for c in test if not c == "Age"] + ["Age"]]

# Fill Age col
test = fill_col("Age", model_age, test, ["Pclass", "Name", "Sex", "SibSp", "Parch", "Ticket_Freq"])
# Fill Cabin col
test = fill_col("Cabin", model_cabin, test, ["Pclass", "Fare", "Embarked"])


# In[ ]:


test = test.drop(columns=["Fare","Embarked"])


# In[ ]:


test.head()


# In[ ]:


y_pred = clf.predict(test.values)
result['Survived'] = y_pred
result.head()


# In[ ]:


result.to_csv("submission.csv", index=False)

