#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Part 1 : Understanding the data
# 
# ![](https://cdn-images-1.medium.com/max/1200/1*wdGUB-2bxIMCzbUJOGw65w.jpeg)
# 
# In the first stage of the competition, Kagglers will rely on results of past tournaments to build and test models. External data can be used as well for improving the results.
# 
# In the second stage, competitors will forecast outcomes of all possible matchups in the 2020 NCAA Division I Men’s and Women’s Basketball Championships. You don't need to participate in the first stage to enter the second. The first stage exists to incentivize model building and provide a means to score predictions. The real competition is forecasting the 2020 results.
# 
# As the official public cloud provider of the **NCAA**, **Google Cloud** is proud to provide a competition to help participants strengthen their knowledge of basketball, statistics, data modeling, and cloud technology. As part of its journey to the cloud, the NCAA has migrated 80+ years of historical and play-by-play data, from **90 championships and 24 sports, to Google Cloud Platform (GCP)**. 
# 
# 
# 
# # **NCAA Division I Basketball Tournament**
# 
#  
# ![](https://cdn-images-1.medium.com/max/800/1*2fU29HRgk4ySJ_l1N5NaJw.jpeg)
# 
# This tournament is a knockout tournament where the loser is immediately eliminated from the tournament. Since it is mostly played in march, hence it has been accorded the title of **March Madness**. The first edition took place in 1939 and has been regularly held since then. the Women’s Championship was inaugurated in the 1981–82 season.
# 
# # Format
# 
# The male edition tournament comprises of **68** teams that compete in **7** rounds for the National Championship Title. However, the number of Teams in the Women’s edition is **64**.
# 
# ![](https://cdn-images-1.medium.com/max/800/1*TaaEJ3zTwhuU67QPqqrkaA.png)
# 
# ---
# 
# # Selection
# 
# The selection procedure takes place by two methods:
# 
# ![](https://cdn-images-1.medium.com/max/800/1*s7gpAnvzL-mQ0lKlzc8xXQ.png)
# 
# ## 1. Automatic
# 
# 32 Teams get selected in this way.
# 
# -   Men’s Division 1 Team comprises of **353** Teams.
# 
# ![](https://cdn-images-1.medium.com/max/800/1*DBT72cUKGLIvXmjO7mBgyQ.png)
# 
# -   Each one of those teams belongs to **32** [conferences](https://en.wikipedia.org/wiki/List_of_NCAA_conferences).
# 
# ![](https://cdn-images-1.medium.com/max/800/1*rq4HBtMnQeGsiI7hOmBfIA.png)
# 
# -   Each of those conferences conducts a tournament and if a time wins the tournament, they get selected for the NCAA.
# 
#   
# 
# ## 2. At Large
# 
# The second selection process is called ‘At Large’ where The NCAA selection committee convenes at the final days of the regular season and decides which 36 teams which are not the Automatic qualifiers can be sent to the playoffs. This selection is based on multiple stats and rankings.
# 
# ---
# 
# ## Selection Sunday
# 
# These “at-large” teams are announced in a nationally televised event on the Sunday preceding the [“First Four” play-in games](https://en.wikipedia.org/wiki/NCAA_Men%27s_Division_I_Basketball_Opening_Round_game "NCAA Men's Division I Basketball Opening Round game"). This Sunday is called ‘Selection Sunday and is on March 15.
# 
# ## Seeding
# 
# After all the 68(64 in case of Women), have been decided, the selection committee ranks them in a process called seeding where each team gets a ranking from 1 to 68. Then **First Four** play-in games are contested between teams holding the four lowest-seeded automatic bids and the four lowest-seeded at-large bids.
# 
# The Teams are then split into 4 regions of 16 Teams each. Each team is now ranked from 1 to 16 in each region. After the [First Four](https://en.wikipedia.org/wiki/First_Four "First Four"), the tournament occurs during the course of three weekends, at pre-selected neutral sites across the United States. Here, the first round matches are determined by pitting the top team in the region with the lowest-seeded team in that region and so on. This ranking is the team’s seed.
# 
# # March Madness Begins
# ![](https://www.ncaa.com/sites/default/files/public/styles/original/public-s3/images/2020/02/12/kelly-campbell-depaul-2020-ncaa.jpg?itok=suRzKxqw)
# 
# ## First Round
# 
# The First round consisting of 64 teams playing in 32 games over the course of a week. From here 32 teams emerge as winners and go on to the second round.
# 
# ## Sweet Sixteen
# 
# Next, the sweet sixteen round takes place, which sees the elimination of 16 teams. Rest of the 16 teams move forward.
# 
# ## Elite Eight
# 
# The next fight is for the Elite Eight as only 8 teams remain in the competition.
# 
# ## Final Four
# ![](https://media.giphy.com/media/2fMOp0fPmvwwCgLXUK/giphy.gif)
# The penultimate round of the tournament where the 4 teams contest to reserve a place in the finals. Four teams, one from each region (East, South, Midwest, and West), compete in a preselected location for the national championship.
#  
#  ---
# 
# # Submission Logic
# 
# Submission File
# The file you submit will depend on whether the competition is in stage 1 (historical model building) or stage 2 (the 2020 tournament). Sample submission files will be provided for both stages. The format is a list of every possible matchup between the tournament teams. Since team1 vs. team2 is the same as team2 vs. team1, we only include the game pairs where team1 has the lower team id. For example, in a tournament of 64 teams, you will predict (64*63)/2  = 2,016 matchups. 
# 
# Each game has a unique id created by concatenating the season in which the game was played, the team1 id, and the team2 id. For example, "2015_3106_3107" indicates team 3106 played team 3107 in the year 2015. You must predict the probability that the team with the lower id beats the team with the higher id.
# 
# The resulting submission format looks like the following, where "pred" represents the predicted probability that the first team will win:

# ### Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
import gc
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix


# In[ ]:


Tourney_Compact_Results = pd.read_csv('../input/ncaam-march-mania-2021/MNCAATourneyCompactResults.csv')
Tourney_Seeds = pd.read_csv('../input/ncaam-march-mania-2021/MNCAATourneySeeds.csv')


# In[ ]:


RegularSeason_Compact_Results = pd.read_csv('../input/ncaam-march-mania-2021/MRegularSeasonCompactResults.csv')
MSeasons = pd.read_csv('../input/ncaam-march-mania-2021/MSeasons.csv')
MTeams=pd.read_csv('../input/ncaam-march-mania-2021/MTeams.csv')


# In[ ]:


RegularSeason_Compact_Results.shape


# In[ ]:


Tourney_Compact_Results.head()


# In[ ]:


Tourney_Compact_Results.shape


# In[ ]:


Tourney_Compact_Results.groupby('Season').mean().head()


# In[ ]:


Tourney_Compact_Results.head()


# In[ ]:


Tourney_Results_Compact=pd.merge(Tourney_Compact_Results, Tourney_Seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
Tourney_Results_Compact.rename(columns={'Seed':'WinningSeed'},inplace=True)
Tourney_Results_Compact=Tourney_Results_Compact.drop(['TeamID'],axis=1)

Tourney_Results_Compact = pd.merge(Tourney_Results_Compact, Tourney_Seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
Tourney_Results_Compact.rename(columns={'Seed':'LoosingSeed'}, inplace=True)
Tourney_Results_Compact=Tourney_Results_Compact.drop(['TeamID','NumOT','WLoc'],axis=1)


Tourney_Results_Compact


# In[ ]:


Tourney_Results_Compact=Tourney_Results_Compact.drop(['WScore','LScore'],axis=1)
Tourney_Results_Compact.head()


# In[ ]:


Tourney_Results_Compact.shape


# In[ ]:


Tourney_Results_Compact['WinningSeed'] = Tourney_Results_Compact['WinningSeed'].str.extract('(\d+)', expand=True)
Tourney_Results_Compact['LoosingSeed'] = Tourney_Results_Compact['LoosingSeed'].str.extract('(\d+)', expand=True)
Tourney_Results_Compact.WinningSeed = pd.to_numeric(Tourney_Results_Compact.WinningSeed, errors='coerce')
Tourney_Results_Compact.LoosingSeed = pd.to_numeric(Tourney_Results_Compact.LoosingSeed, errors='coerce')


# In[ ]:


RegularSeason_Compact_Results.head()


# In[ ]:


num_win = RegularSeason_Compact_Results.groupby(['Season', 'WTeamID']).count()
num_win = num_win.reset_index()[['Season', 'WTeamID', 'DayNum']].rename(columns={"DayNum": "NumWins", "WTeamID": "TeamID"})

num_loss = RegularSeason_Compact_Results.groupby(['Season', 'LTeamID']).count()
num_loss = num_loss.reset_index()[['Season', 'LTeamID', 'DayNum']].rename(columns={"DayNum": "NumLosses", "LTeamID": "TeamID"})

RegularSeason_Compact_Results['ScoreGap'] = RegularSeason_Compact_Results['WScore'] - RegularSeason_Compact_Results['LScore']

gap_win = RegularSeason_Compact_Results.groupby(['Season', 'WTeamID']).mean().reset_index()
gap_win = gap_win[['Season', 'WTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"})

gap_loss = RegularSeason_Compact_Results.groupby(['Season', 'LTeamID']).mean().reset_index()
gap_loss = gap_loss[['Season', 'LTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"})


# In[ ]:


RegularSeason_Compact_Results.shape


# In[ ]:


season_winning_team = RegularSeason_Compact_Results.groupby(['Season', 'WTeamID']).count().reset_index()[['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"})
season_losing_team = RegularSeason_Compact_Results.groupby(['Season', 'LTeamID']).count().reset_index()[['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})


# In[ ]:


RegularSeason_Compact_Results = pd.concat((season_winning_team, season_losing_team)).drop_duplicates().sort_values(['Season', 'TeamID']).reset_index(drop=True)
RegularSeason_Compact_Results.head()


# In[ ]:


"""
season_winning_team = RegularSeason_Compact_Results[['Season', 'WTeamID', 'WScore']]
season_losing_team = RegularSeason_Compact_Results[['Season', 'LTeamID', 'LScore']]
season_winning_team.rename(columns={'WTeamID':'TeamID','WScore':'Score'}, inplace=True)
season_losing_team.rename(columns={'LTeamID':'TeamID','LScore':'Score'}, inplace=True)
RegularSeason_Compact_Results = pd.concat((season_winning_team, season_losing_team)).reset_index(drop=True)
RegularSeason_Compact_Results
"""


# In[ ]:


RegularSeason_Compact_Results = RegularSeason_Compact_Results.merge(num_win, on=['Season', 'TeamID'], how='left')
RegularSeason_Compact_Results = RegularSeason_Compact_Results.merge(num_loss, on=['Season', 'TeamID'], how='left')
RegularSeason_Compact_Results = RegularSeason_Compact_Results.merge(gap_win, on=['Season', 'TeamID'], how='left')
RegularSeason_Compact_Results = RegularSeason_Compact_Results.merge(gap_loss, on=['Season', 'TeamID'], how='left')
RegularSeason_Compact_Results.head()


# In[ ]:


RegularSeason_Compact_Results.fillna(0, inplace=True) 


# In[ ]:


RegularSeason_Compact_Results['WinRatio'] = RegularSeason_Compact_Results['NumWins'] / (RegularSeason_Compact_Results['NumWins'] + RegularSeason_Compact_Results['NumLosses'])
RegularSeason_Compact_Results['GapAvg'] = (
    (RegularSeason_Compact_Results['NumWins'] * RegularSeason_Compact_Results['GapWins'] - 
    RegularSeason_Compact_Results['NumLosses'] * RegularSeason_Compact_Results['GapLosses'])
    / (RegularSeason_Compact_Results['NumWins'] + RegularSeason_Compact_Results['NumLosses'])
)
RegularSeason_Compact_Results.head()


# In[ ]:


RegularSeason_Compact_Results.drop(['NumWins', 'NumLosses', 'GapWins', 'GapLosses'], axis=1, inplace=True)
RegularSeason_Compact_Results.head(1)


# In[ ]:


#RegularSeason_Compact_Results_Final = RegularSeason_Compact_Results.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()
#RegularSeason_Compact_Results_Final


# In[ ]:


Tourney_Results_Compact.shape


# In[ ]:


RegularSeason_Compact_Results.shape


# In[ ]:


Tourney_Results_Compact = pd.merge(Tourney_Results_Compact, RegularSeason_Compact_Results, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left').rename(columns={
    'WinRatio': 'WinRatioW',
    'GapAvg': 'GapAvgW'})
Tourney_Results_Compact.rename(columns={'Score':'WScoreTotal'}, inplace=True)
Tourney_Results_Compact


# In[ ]:


Tourney_Results_Compact = Tourney_Results_Compact.drop('TeamID', axis=1)
Tourney_Results_Compact = pd.merge(Tourney_Results_Compact, RegularSeason_Compact_Results, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left').rename(columns={
    'WinRatio': 'WinRatioL',
    'GapAvg': 'GapAvgL'})
Tourney_Results_Compact.rename(columns={'Score':'LScoreTotal'}, inplace=True)
Tourney_Results_Compact = Tourney_Results_Compact.drop('TeamID', axis=1)
Tourney_Results_Compact.to_csv('Tourney_Win_Results_Train.csv', index=False)
Tourney_Results_Compact=Tourney_Results_Compact[Tourney_Results_Compact['Season'] < 2015] 
Tourney_Results_Compact


# In[ ]:


Tourney_Win_Results=Tourney_Results_Compact.drop(['Season','WTeamID','LTeamID','DayNum'],axis=1)
Tourney_Win_Results


# In[ ]:


#Tourney_Win_Results.rename(columns={'WinningSeed':'Seed1', 'LoosingSeed':'Seed2', 'WScoreTotal':'ScoreT1', 'LScoreTotal':'ScoreT2'}, inplace=True)
Tourney_Win_Results.rename(columns={'WinningSeed':'Seed1', 'LoosingSeed':'Seed2'}, inplace=True)


# In[ ]:


Tourney_Win_Results.head()


# In[ ]:


tourney_lose_result = Tourney_Win_Results.copy()
tourney_lose_result['Seed1'] = Tourney_Win_Results['Seed2']
tourney_lose_result['Seed2'] = Tourney_Win_Results['Seed1']
#tourney_lose_result['ScoreT1'] = Tourney_Win_Results['ScoreT2']
#tourney_lose_result['ScoreT2'] = Tourney_Win_Results['ScoreT1']
tourney_lose_result


# **Training Data**

# In[ ]:


Tourney_Win_Results['Seed_diff'] = Tourney_Win_Results['Seed1'] - Tourney_Win_Results['Seed2']
#Tourney_Win_Results['ScoreT_diff'] = Tourney_Win_Results['ScoreT1'] - Tourney_Win_Results['ScoreT2']
tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']
#tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']


# In[ ]:


Tourney_Win_Results.head()


# In[ ]:


tourney_lose_result.head()


# In[ ]:


Tourney_Win_Results['result'] = 1
tourney_lose_result['result'] = 0
tourney_result_Final = pd.concat((Tourney_Win_Results, tourney_lose_result)).reset_index(drop=True)



# In[ ]:


tourney_result_Final.head()


# In[ ]:


tourney_result_Final1 = tourney_result_Final[[
    'Seed1', 'Seed2','GapAvgL','WinRatioL','GapAvgW','WinRatioW', 'result','Seed_diff']]


# In[ ]:


tourney_result_Final1.shape


# **Test Data**

# In[ ]:


test_df = pd.read_csv('../input/ncaam-march-mania-2021/MSampleSubmissionStage1.csv')


# In[ ]:


test_df.head()


# In[ ]:


test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))
test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))
test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))
test_df


# In[ ]:


test_df = pd.merge(test_df, Tourney_Seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, Tourney_Seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)


# In[ ]:


RegularSeason_Compact_Results.head(1)


# In[ ]:


test_df.head(1)


# In[ ]:


test_df = pd.merge(test_df, RegularSeason_Compact_Results, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left').rename(columns={
    'WinRatio': 'WinRatioW',
    'GapAvg': 'GapAvgW'})
test_df.rename(columns={'Score':'ScoreT1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, RegularSeason_Compact_Results, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left').rename(columns={
    'WinRatio': 'WinRatioL',
    'GapAvg': 'GapAvgL'})
test_df.rename(columns={'Score':'ScoreT2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df


# In[ ]:


test_df['Seed1'] = test_df['Seed1'].str.extract('(\d+)', expand=True)
test_df['Seed2'] = test_df['Seed2'].str.extract('(\d+)', expand=True)
test_df.Seed1 = pd.to_numeric(test_df.Seed1, errors='coerce')
test_df.Seed2 = pd.to_numeric(test_df.Seed2, errors='coerce')


# In[ ]:


test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']
#test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']
test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)
test_df


# **Logistic regression**

# In[ ]:


tourney_result_Final1


# In[ ]:


X = tourney_result_Final1.drop('result', axis=1)
y = tourney_result_Final1.result


# In[ ]:


X.head(1)


# In[ ]:


test_df.head(1)


# In[ ]:


# LGB
lgb_num_leaves_max = 255
lgb_in_leaf = 50
lgb_lr = 0.0001
lgb_bagging = 7


# In[ ]:


params_lgb = {'num_leaves': lgb_num_leaves_max,
              'min_data_in_leaf': lgb_in_leaf,
              'objective': 'binary',
              'max_depth': -1,
              'learning_rate': lgb_lr,
              "boosting_type": "gbdt",
              "bagging_seed": lgb_bagging,
              "metric": 'logloss',
              "verbosity": -1,
              'random_state': 42,
             }


# In[ ]:


lgb_params = {'objective': 'binary',
              'metric': 'binary_logloss',
              'boosting': 'gbdt',
              'num_leaves': 32,
              'feature_fraction': 0.6,
              'bagging_fraction': 0.6,
              'bagging_freq': 5,
              'learning_rate': 0.05
}


# In[ ]:


NFOLDS = 10
folds = KFold(n_splits=NFOLDS)

columns = X.columns
splits = folds.split(X, y)
y_preds_lgb = np.zeros(test_df.shape[0])
y_train_lgb = np.zeros(X.shape[0])
y_oof = np.zeros(X.shape[0])

feature_importances = pd.DataFrame()
feature_importances['feature'] = columns
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params_lgb, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    
    y_train_lgb += clf.predict(X) / NFOLDS
    y_preds_lgb += clf.predict(test_df) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()


# In[ ]:


submission_df = pd.read_csv('../input/ncaam-march-mania-2021/MSampleSubmissionStage1.csv')
y_preds = y_preds_lgb
submission_df['Pred'] = y_preds
submission_df.to_csv('submission.csv', index=False)


# In[ ]:




