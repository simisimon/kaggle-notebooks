#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('../input/dapprojekt22/train.csv')


# # 1. Motivation

# The NBA dataset we are working with is a **time series dataset**. This means that data is recorded over consistent intervals of time. In our case this interval is equivalent to the time between consecutive games and seasons.
# 
# While making predictions about the outcome of a game in real life, we have information only about games played up to that point in time. Rows in our original dataset, however, don't contain information about all previous games, only the last played game. Furthermore, the last played game doesn't contain information about the teams that are playing the next game. Therefore, we need to transform our data to:
# 
# 1. contain **information corresponding to the home and away team we are making predictions about**
# 2. contain **information about games played only previous to the current game**, and not any future games (otherwise this would be a considered a data leak)
# 
# One way to perform this transformation is to consider the average of all statistics from games the home and away team previously played in. This approach can have multiple variations. For example, we can consider the following variations:
# 
# 1. **SAME POSITION** - only games that the home and away team played in the same position (i.e. games in which the home team was also the home team, and vice versa)
# 2. **WINDOW AVERAGE** - taking only a window of previous games (i.e. taking information only from the last k games)
# 3. **WEIGHTED AVERAGE** - multiplying previous games with different weights (e.g. more recent games get more weight)

# # 2. Idea

# While after transforming the data once, we don't need to transform it again, we can see from the previous list it is probably good to try different transformations before settling on the best one. It is therefore desirable for this transformtion to take as little time as possible.
# 
# The following section presents two approaches to performing this data transformation.

# ## 1. Naive approach
# The naive approach to perform this data transformation is to iterate over all games in the dataset and then to iterate over all previous games to collect information.
# 
# #### Complexity:
# 
# Time complexity - $O(n^2)$
# 
# Space complexity - $O(nc)$ - which corresponds to the copy of the original dataset, since it is needed for iterating over all previous games.
# 
# where $n$ is the number of games in the dataset and $c$ is the number of columns that are being averaged.

# ## 2. O(n) time approach
# 
# The idea of this approach is to avoid iterating over the previous games for every game in the dataset, and to instead keep a summary of the previously played games for each team in an additional data structure.
# 
# First we initialize two additional data structures that help us store two pieces of information:
# 1. the number of games a team has participated in so far
# 2. a running summation of every column we want to average for every team
# 
# The algorithm goes as follows:
# 
# For each game:
# 1. Collect information about the current game and add it to the above mentioned data structures.
# 2. Replace the information about the current game with the average information from the previous games for NEXT_WINNER and NEXT_AWAY teams. We do this by dividing the running summation with the number of games a team has participated in.
# 
# **A concrete implementation of this approach is shown in the section Data Transformation in O(n) time.**
# 
# #### Complexity:
# 
# Time complexity - $O(n)$
# 
# Space complexity - $O(tc)$
# 
# where $n$ is the number of games in the dataset, $t$ is the number of teams, and $c$ is the number of columns that are being averaged and $n>>t$.
# 
# Since $n>>t$, we can see that this approach actually saves space as well, as we can use the original dataset to store the transformed values, instead of storing them in a copy of the original dataset.

# # 3. Data Preprocessing
# In this section we preprocess the data before performing the transformation.

# In[ ]:


#Dropping columns with only one unqiue value
one_unique = train.loc[:, train.nunique()==1]
train.drop(one_unique.columns, axis=1, inplace=True)

#Dropping additional columns deemed unncessary for this demonstration
train.drop(["TEAM_NAME_AWAY", "TEAM_CITY_AWAY", "TEAM_NAME_HOME", "TEAM_CITY_HOME", "TEAM_ID_HOME", "TEAM_ID_AWAY"], axis=1, inplace=True)
train.drop(["NEXT_WEIGHT"], axis=1, inplace=True)
train.drop(["MIN_HOME", "MIN_AWAY"], axis=1, inplace=True)


# In[ ]:


from sklearn import preprocessing

#Encoding team abbreviations to numerical values
datatypes = train.dtypes
categorical = datatypes[datatypes == 'object'].index.tolist()
print(categorical)

label_encoder = preprocessing.LabelEncoder()
train[categorical] = train[categorical].apply(label_encoder.fit_transform)


# In[ ]:


#Ensuring there is no null values
null_values = train.isnull().sum().sum()
print("Broj NaN vrijednosti:", null_values)


# In[ ]:


#Renaming the winner column, as it will correspond to the current row after the transformation.
train.rename(columns={"NEXT_WINNER": "WINNER"}, inplace=True)


# In[ ]:


from sklearn.ensemble import IsolationForest

#Dropping outliers
iso = IsolationForest()
yhat = iso.fit_predict(train)
mask = yhat != -1


train = train[mask == 1]


# In[ ]:


#Since we removed some rows, we reset the index.
train.reset_index(drop=True, inplace=True)


# # 4. Data Transformation in O(n) time

# #### Preparation steps
# The following cells set up the dataset and additional structures for the algorithm.

# In[ ]:


#The following list contains columns that we don't need to average.
columns_skip_list = ["NEXT_HOME", "NEXT_AWAY", "WINNER", "TEAM_ABBREVIATION_HOME", "TEAM_ABBREVIATION_AWAY"]

#A minor implementation detail - to make the transformation more convenient we want all of these columns to be at the end.
#The only columns that aren't already at the end are the team abbreviations.
abb_home = train.pop('TEAM_ABBREVIATION_HOME') 
abb_away = train.pop('TEAM_ABBREVIATION_AWAY')
train['TEAM_ABBREVIATION_HOME'] = abb_home
train['TEAM_ABBREVIATION_AWAY'] = abb_away


# In[ ]:


train.columns


# In[ ]:


#Initializing additional data structures

#Number of columns = (169-5)/2 = 82
team_dict = dict() #size=t*(c/2), where t is the number of teams and c is the number of columns we want to average
for team in range(0,30):
    team_dict[team] = [0]*82
    
team_counts = [0]*30 #size=t, where t is the number of teams


# #### The algorithm
# The following cells perform the data transformation.

# In[ ]:


import math
import time

start_time = time.time()
for index in range(len(train)):
    home_team = train.loc[index, "TEAM_ABBREVIATION_HOME"]
    away_team = train.loc[index, "TEAM_ABBREVIATION_AWAY"]
    next_home = train.loc[index, "NEXT_HOME"]
    next_away = train.loc[index, "NEXT_AWAY"]
    
    #We note every occurrence of a team in the dataset thus far
    team_counts[home_team] += 1
    team_counts[away_team] += 1
    
    #We iterate over all columns in a row, except the last 5 which are the ones we don't need to average
    for col_index, col in enumerate(train.columns.tolist()[:-5]):
        
        #We calculate the index in the dictionary
        col_index_for_dict = int(math.floor(col_index/2))
        
        if "AWAY" in col:
            team_dict[away_team][col_index_for_dict] += train.loc[index, col]
            
            if(team_counts[next_away] == 0):
                #If there is no info for the team we are predicting the outcome to we mark it with -1
                #Later we can remove all games with this value, or treat them in some other way
                res_away = -1
            else:
                res_away = team_dict[next_away][col_index_for_dict]/team_counts[next_away]

            train.loc[index, col] = res_away
            
        else:
            team_dict[home_team][col_index_for_dict] += train.loc[index, col]
        
            if(team_counts[next_home] == 0):
                res_home = -1
            else:
                res_home = team_dict[next_home][col_index_for_dict]/team_counts[next_home]
            train.loc[index, col] = res_home

    if(index == 99):
        print("--- Time for 100 rows: %s seconds ---" % (time.time() - start_time))
print("--- Time for the whole datset: %s seconds ---" % (time.time() - start_time))


# In[ ]:


#We delete the current teams' abbreviations since they are not needed anymore.
train.drop(["TEAM_ABBREVIATION_HOME", "TEAM_ABBREVIATION_AWAY"], axis=1, inplace=True)

#The row now contains relevant information for the game we are trying to predict.
train.rename(columns={"NEXT_HOME": "HOME", "NEXT_AWAY": "AWAY"}, inplace=True)


# Just for reference, the following code prints the size of different structures in bytes:

# In[ ]:


import sys

print("Team_dict size in bytes", sys.getsizeof(team_dict))
print("Team_counts size in bytes", sys.getsizeof(team_counts))
print("Dataset size in bytes", sys.getsizeof(train))


# We can see that storing a dataset copy is much more memory expensive than storing two additional smaller data structures with summaries of previously played games.

# # 5. Conclusion

# This notebook presents a more efficient way of transforming data. It is worth noting that the presented implementation can be easily tweaked to calculate the average values in other ways (such as the ones presented in the beginning of the notebook). While most of them will require some extra space to store less general summaries than the one implemented in this notebook, they will still be more memory efficient since a copy of the original dataset is not needed.
