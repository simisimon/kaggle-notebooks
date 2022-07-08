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


import random
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


# In[ ]:


inp_dict = {'WTeamID':'str', 'LTeamID':'str'}
inp = '../input/ncaam-march-mania-2021/'
ss = pd.read_csv(inp+'MSampleSubmissionStage1.csv')
sd = pd.read_csv(inp+'MRegularSeasonCompactResults.csv',dtype=inp_dict)
td = pd.read_csv(inp+'MNCAATourneyCompactResults.csv',dtype=inp_dict)
ts = pd.read_csv(inp+'MNCAATourneySeeds.csv',dtype={'TeamID':'str'})


# In[ ]:


ts['Seed'] = ts['Seed'].map(lambda s: s[1:].strip('ab'))
sd['DScore'] = sd['WScore'] - sd['LScore']


# In[ ]:


for i in range(len(td.index)):
    if random.choices([0,1]) == [1]:
        td.at[i, 'Team1'] = td.at[i, 'WTeamID']
        td.at[i, 'Team2'] = td.at[i, 'LTeamID']
        td.at[i, 'target'] = 1.0
    else:
        td.at[i, 'Team1'] = td.at[i, 'LTeamID']
        td.at[i, 'Team2'] = td.at[i, 'WTeamID']
        td.at[i, 'target'] = 0.0


# In[ ]:


ss['Team1'] = ss['ID'].map(lambda s: s[5:9])
ss['Team2'] = ss['ID'].map(lambda s: s[10:])
ss['Season'] = ss['ID'].map(lambda s: s[:4])


# In[ ]:


statistics = {}
def calculate_stat(season, team):
    if (season, team) in statistics.keys():
        return
    t_w = sd.loc[(sd['Season']==season)&(sd['WTeamID']==team),'DScore']
    t_l = sd.loc[(sd['Season']==season)&(sd['LTeamID']==team),'DScore']
    t_wc = len(t_w.index)
    t_lc = len(t_l.index)
    t_ws = t_w.sum()
    t_ls = t_l.sum()
    statistics[(season, team)] = {}
    statistics[(season, team)]['WinRate'] = t_wc / (t_wc+t_lc)
    statistics[(season, team)]['ScoreDiff'] = t_ws - t_ls
    statistics[(season, team)]['Seed'] = int(ts.loc[(ts['Season']==season)&(ts['TeamID']==team),'Seed'].any())


# In[ ]:


def features(df):
    for i in df.index:
        season = int(df.at[i, 'Season'])
        team1 = df.at[i, 'Team1']
        team2 = df.at[i, 'Team2']
        calculate_stat(season, team1)
        calculate_stat(season, team2)
        df.at[i, 'T1WinRate'] = statistics[(season, team1)]['WinRate']
        df.at[i, 'T2WinRate'] = statistics[(season, team2)]['WinRate']
        df.at[i, 'T1ScoreDiff'] = statistics[(season, team1)]['ScoreDiff']
        df.at[i, 'T2ScoreDiff'] = statistics[(season, team2)]['ScoreDiff']
        df.at[i, 'T1Seed'] = statistics[(season, team1)]['Seed']
        df.at[i, 'T2Seed'] = statistics[(season, team2)]['Seed']
    return df


# In[ ]:


td = features(td)
ss = features(ss)


# In[ ]:


ss.head()


# In[ ]:


td.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
# model = LogisticRegression()
model = RandomForestClassifier(n_estimators=100, random_state=0)
cols = ['T1ScoreDiff','T2ScoreDiff','T1WinRate','T2WinRate','T1Seed','T2Seed']


# In[ ]:


def get_train_test(df, test_season):
    train_df = df.loc[df['Season']!=test_season, cols+['target']]
    test_df = df.loc[df['Season']==test_season, cols+['target']]
    return train_df, test_df


# In[ ]:


gloss = 0
seasons = [2015, 2016, 2017, 2018, 2019]
final_imputer = SimpleImputer(strategy='median')
for season in seasons:
    train, test = get_train_test(td, season)
    X_train = train.drop('target', axis=1)
    y_train = train['target']
    X_test = test.drop('target', axis=1)
    final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
    final_X_test = pd.DataFrame(final_imputer.transform(X_test))
    model.fit(final_X_train, y_train)
    pred = model.predict_proba(final_X_test)[:,1]
    loss = log_loss(test['target'], pred)
    print(season, loss)
    gloss += loss


# In[ ]:


print('average', gloss/len(seasons))


# In[ ]:


model.fit(td[cols], td['target'])
pred = model.predict_proba(ss[cols])[:,1]
ss['Pred'] = pred.clip(0, 1)
ss.to_csv('submission.csv', columns=['ID','Pred'], index=None)


# Please give an upvote if you like it or feel free to ask any questions
