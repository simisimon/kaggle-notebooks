#!/usr/bin/env python
# coding: utf-8

# # Do stuff with data

# Import modules:

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier


# In[ ]:


# Hello everyone. Nice to meet you, guys!


# Read data from files:

# In[ ]:


train_data = pd.read_csv("../input/spaceship-titanic/train.csv")
test_data = pd.read_csv("../input/spaceship-titanic/test.csv")
# Result:
train_data


# Add family name to each person:

# In[ ]:


def get_surname(name):
    return name[name.find(' ')+1:] if isinstance(name, str) else "UNKNOWN"


def add_surname_column(dataframe):
    dataframe['Surname'] = dataframe['Name'].map(get_surname)


add_surname_column(train_data)
add_surname_column(test_data)

train_data[["Name", "Surname"]]


# Turning cabin number into Deck.
# Yes, here I used random to fill out people that do not have a cabin (sorry).

# In[ ]:


cabin_set = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
cabins_per_deck = dict()


def get_cabin_number(cabin: str) -> int:
    return int(cabin[cabin.find('/')+1:cabin.rfind('/')])


def get_deck(cabin):
    if isinstance(cabin, str) and cabin[0] in cabin_set:
        # Count, how many cabins in each deck by storing the max id value:
        deck_number = ord(cabin[0]) - ord('A') if cabin[0] != 'T' else 7
        previous_value = (0 if deck_number not in cabins_per_deck
                          else cabins_per_deck[deck_number])
        cabin_id = get_cabin_number(cabin)
        cabins_per_deck[deck_number] = max(previous_value, cabin_id)
        # Return the value for column:
        return deck_number
    return random.randint(0, 7)


def add_deck_column(dataframe):
    dataframe['Deck'] = dataframe['Cabin'].map(get_deck)


add_deck_column(train_data)
add_deck_column(test_data)

print_data = train_data.groupby(['Deck'],
                                sort=False).size().reset_index(name='Count')
print_data['Cabins number'] = print_data['Deck'].map(cabins_per_deck.get)
print_data


# Turning `cabin` string into number:

# In[ ]:


def random_cabin_id(row):
    return random.randint(0, cabins_per_deck[row['Deck']])


def add_cabin_num_column(dataframe):
    dataframe['CabinNum'] = dataframe.apply((lambda c:
                                            get_cabin_number(c['Cabin']) if
                                            isinstance(c['Cabin'], str) else
                                            random_cabin_id(c)),
                                            axis=1)


add_cabin_num_column(train_data)
add_cabin_num_column(test_data)

train_data[["Cabin", "CabinNum"]]


# Find max number of people in 1 group:

# In[ ]:


train_data['PassengerId'].map(lambda id: int(id[-2:])).max()


# Ok, as we see, there may occur some big families. 
# 
# Then lets count the number of family for each guy:

# In[ ]:


def get_family_group(PassengerId):
    return int(PassengerId[:4])


def add_family_size_column(dataframe):
    dataframe['GroupId'] = dataframe['PassengerId'].map(get_family_group)
    groups_counts = dataframe['GroupId'].value_counts()
    dataframe['FamilySize'] = dataframe.apply(lambda row:
                                              groups_counts[row['GroupId']],
                                              axis=1)


add_family_size_column(train_data)
add_family_size_column(test_data)

print(f"Max number of family members is : {train_data['FamilySize'].max()}")
train_data[["GroupId", "Name", "FamilySize"]]


# Add `money` and `money per person` columns:

# In[ ]:


def return_zero_if_nan(value) -> float:
    return 0 if np.isnan(value) else value


def add_more_columns(dataframe):
    dataframe['Fare'] = (dataframe['RoomService'].map(return_zero_if_nan) +
                         dataframe['FoodCourt'].map(return_zero_if_nan) +
                         dataframe['ShoppingMall'].map(return_zero_if_nan) +
                         dataframe['Spa'].map(return_zero_if_nan) +
                         dataframe['VRDeck'].map(return_zero_if_nan))
    dataframe['FarePerPerson'] = dataframe['Fare']/dataframe['FamilySize']


add_more_columns(train_data)
add_more_columns(test_data)

train_data[["Name", "Fare", "FarePerPerson"]]


# <b>GENDER</b>
# I tried to create `sex` field by using the `gender_guesser` python package.
# But there were many strange names that guesser could not attempt to recongize that is why I deleted the code that adds a `Sex` field.

# The last thing to do is to convert an age field to class field. To do this we need to see the whole picture of all ages.
# 
# Firstly, lets see a graph of the ages on the ship. 

# In[ ]:


count_column = 'Count'
ages = train_data.groupby(['Age'],
                          sort=False).size().reset_index(name=count_column)
transported_ages = train_data.loc[train_data["Transported"]]
transported_ages = transported_ages["Age"].value_counts()
total_ages = train_data["Age"].value_counts()

x = ages['Age'].to_numpy()
y = ages[count_column].to_numpy()


def interpolate_color(a: tuple, b: tuple, alpha: float) -> tuple:
    new_color = list(a)
    for i, color_component in enumerate(new_color):
        new_color[i] = color_component + alpha * (b[i] - color_component)
    return tuple(new_color)


def color_age(age: float) -> tuple:
    they_alive_color = (1.0, 0.0, 0.0)
    they_transported_color = (0.0, 0.0, 0.0)
    if age not in transported_ages:
        transported_ages[age] = 0
    return interpolate_color(they_alive_color, they_transported_color,
                             transported_ages[age]/total_ages[age])


fig, ax = plt.subplots()

ax.bar(x, y, color=[color_age(age) for age in x])

plt.xticks(np.arange(0, max(x), 2.5))
plt.setp(ax.get_xticklabels(), rotation=80, ha='right')
fig.set_figwidth(12)
fig.set_figheight(6)

plt.show()


# On this diagram bad cases are darker but cases when people was not transported are more red.
# Ok as we see on the diagram there are 4 main classes:
# 1. From 0 to 5 years old. (~75)
# 2. From 5 to 12.5. (~50) + from 45 to 65
# 2. From 12.5 to 17.5. (~150) + from 30 to 45.
# 3. From 17.5 to 24. (~300)
# 4. From 24 to 30. (~250)
# 5. From 30 to 65 -- see previously.
# 6. From 65 and older. (~20)

# In[ ]:


def fix_nan_age(dataframe):
    dataframe['Age'] = dataframe['Age'].replace(np.nan,
                                                dataframe['Age'].median())


fix_nan_age(train_data)
fix_nan_age(test_data)


# And now we replace age with the class:

# In[ ]:


def get_age_class(age) -> int:
    if isinstance(age, float):
        if 0 <= age < 5:
            return 1
        if 5 <= age < 12.5 or 45 <= age < 65:
            return 2
        if 12.5 <= age < 17.5 or 30 <= age < 45:
            return 3
        if 17.5 <= age < 24:
            return 4
        if 24 <= age < 30:
            return 5
        return 6
    raise ValueError("Bad age passed!")


def add_age_class_column(dataframe):
    dataframe['AgeClass'] = dataframe['Age'].map(get_age_class)


add_age_class_column(train_data)
add_age_class_column(test_data)

train_data[["PassengerId", "Age", "AgeClass"]]


# That is all we gonna do with the data. All things left is to fix types a little:

# In[ ]:


train_data.dtypes


# In[ ]:


featuries = ['CryoSleep', 'VIP', 'Deck', 'CabinNum', 'GroupId', 'FamilySize',
             'Fare', 'FarePerPerson', 'AgeClass', 'Surname']


def column_to_category(dataframe, column):
    dataframe[column] = dataframe[column].astype("category").cat.codes


for dataframe in [train_data, test_data]:
    for feature in featuries:
        if dataframe.dtypes[feature] == np.dtype(object):
            column_to_category(dataframe, feature)


# # Now we start the ML itself:

# In[ ]:


x_train = train_data[featuries]
y_train = train_data['Transported']
x_test = test_data[featuries]

# get parameters via GridCV
clf = ExtraTreesClassifier(class_weight='balanced',
                           criterion='gini',
                           max_features='log2',
                           n_estimators=200)
ada = AdaBoostClassifier(base_estimator=clf, n_estimators=200,
                         algorithm='SAMME')
ada.fit(x_train, y_train)

print(ada.score(x_train, y_train))

predictions = ada.predict(x_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Transported': predictions})
output.to_csv('submission.csv', index=False)
print("Submission was saved :)")

