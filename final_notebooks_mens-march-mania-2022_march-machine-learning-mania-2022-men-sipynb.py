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


import re # Importing the Regular Expression Module...


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Notebook Configuration...

# Amount of data we want to load into the Model...
DATA_ROWS = None
# Dataframe, the amount of rows and cols to visualize...
NROWS = 50
NCOLS = 15
# Main data location path...
BASE_PATH = '...'


# In[ ]:


# Configure notebook display settings to only use 2 decimal places, tables look nicer.
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', NCOLS) 
pd.set_option('display.max_rows', NROWS)


# # Reading the Datasets

# In[ ]:


# Read the requiered datasets.
season_data = pd.read_csv('/kaggle/input/mens-march-mania-2022/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
seeds = pd.read_csv('/kaggle/input/mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneySeeds.csv')

public_rating = pd.read_csv('/kaggle/input/mens-march-mania-2022/MDataFiles_Stage1/MMasseyOrdinals.csv')


# In[ ]:


season_data.info()


# In[ ]:


season_data.head()


# In[ ]:


season_data.describe()


# # Creating Features for the Model.

# In[ ]:


def score_gap(df):
    """
    """
    df['ScoreGap'] = df['WScore'] - df['LScore']
    return df


# In[ ]:


season_data = score_gap(season_data)


# In[ ]:


season_data.head()


# In[ ]:


# Creates an empty list of all the teams, Winners + Lossers to merge data back....
def create_team_list(df, group_list = ['Season', 'WTeamID'], team_id = 'WTeamID'):
    group = df.groupby(group_list).count().reset_index()
    group = group[group_list].rename(columns={team_id: "TeamID"})
    return group


# In[ ]:


winners = create_team_list(season_data, group_list = ['Season', 'WTeamID'], team_id = 'WTeamID')
lossers = create_team_list(season_data, group_list = ['Season', 'LTeamID'], team_id = 'LTeamID')

# Create an empty train dataset.
team_agg_features = pd.concat([winners, lossers], axis = 0).drop_duplicates().sort_values(['Season', 'TeamID']).reset_index(drop = True)


# In[ ]:


team_agg_features.head()


# In[ ]:


# Creating aggregated features...
def winner_aggregated_features(df, group_list = ['Season', 'WTeamID']):
    tmp = df.groupby(group_list).agg(NumWins       = ('WTeamID', 'count'), 
                                     AvgWinsGap    = ('ScoreGap', 'mean'),
                                     W_TotalPoints = ('WScore', 'sum'),
                                     W_MaxPoints   = ('WScore', 'max'),
                                     W_MinPoints   = ('WScore', 'min'),
                                    )
    tmp = tmp.reset_index()
    tmp = tmp.rename(columns={"WTeamID": "TeamID"})
    
    return tmp


# In[ ]:


# Creating aggregated features...
def losser_aggregated_features(df, group_list = ['Season', 'LTeamID']):
    tmp = df.groupby(group_list).agg(NumLosses       = ('LTeamID', 'count'), 
                                     AvgLossesGap    = ('ScoreGap', 'mean'),
                                     L_TotalPoints = ('LScore', 'sum'),
                                     L_MaxPoints   = ('LScore', 'max'),
                                     L_MinPoints   = ('LScore', 'min'),
                                    )
    tmp = tmp.reset_index()
    tmp = tmp.rename(columns={"LTeamID": "TeamID"})
    return tmp


# In[ ]:


winner_team_aggregation = winner_aggregated_features(season_data)
losser_team_aggregation = losser_aggregated_features(season_data)


# In[ ]:


def merge_back(df):
    df = df.merge(winner_team_aggregation, on = ['Season', 'TeamID'], how = 'left')
    df = df.merge(losser_team_aggregation, on = ['Season', 'TeamID'], how = 'left')
    df.fillna(0, inplace = True) 
    return df


# In[ ]:


team_agg_features = merge_back(team_agg_features)


# In[ ]:


def calculate_features(df):
    """
    
    """
    df['WinRatio'] = df['NumWins'] / (df['NumWins'] + df['NumLosses'])
    df['AvgScoreGap'] = ((df['NumWins'] * df['AvgWinsGap'] - df['NumLosses'] * df['AvgLossesGap']) / (df['NumWins'] + df['NumLosses']))
    df['PointsRatio'] = df['W_TotalPoints'] / (df['L_TotalPoints'] + df['W_TotalPoints'])
    return df

team_agg_features = calculate_features(team_agg_features)


# In[ ]:


team_agg_features = team_agg_features[['Season','TeamID','WinRatio', 'AvgScoreGap','PointsRatio']]


# In[ ]:


team_agg_features.head()


# # Creating the Training Dataset...

# In[ ]:


tournament_data = pd.read_csv('/kaggle/input/mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
tournament_data.head()


# In[ ]:


tournament_data.describe()


# In[ ]:


tournament_data = tournament_data.rename(columns = {'WTeamID' : 'W_TeamID', 'LTeamID' : 'L_TeamID', 'WScore' : 'W_Score', 'LScore' : 'L_Score'})  


# In[ ]:


# Drop non importante features from the dataset...
tournament_data.drop(['NumOT', 'WLoc'], axis = 1, inplace = True)


# In[ ]:


MIN_SEASON = 2015
# Remove data before 2003, no all the data is available...
tournament_data = tournament_data[tournament_data['Season'] >= MIN_SEASON].reset_index(drop = True)


# In[ ]:


tournament_data.head()


# In[ ]:


def merge_seed(df, seed_df, left_on = ['Season', 'W_TeamID'], field_name = 'SeedW'):
    df = pd.merge(df,seed_df, how = 'left', left_on = left_on, right_on = ['Season', 'TeamID'])
    df = df.drop('TeamID', axis = 1).rename(columns = {'Seed': field_name})
    return df


# In[ ]:


tournament_data = merge_seed(tournament_data, seeds, left_on = ['Season', 'W_TeamID'], field_name = 'W_Seed')
tournament_data = merge_seed(tournament_data, seeds, left_on = ['Season', 'L_TeamID'], field_name = 'L_Seed')


# In[ ]:


def seed_number(row):
    return int(re.sub("[^0-9]", "", row))

tournament_data['W_Seed'] = tournament_data['W_Seed'].apply(seed_number)
tournament_data['L_Seed'] = tournament_data['L_Seed'].apply(seed_number)


# In[ ]:


def merge_agg_features(df, agg_features):
    for result in ['W', 'L']:
        df = pd.merge(df, agg_features, how = 'left', left_on = ['Season', result +'_'+ 'TeamID'], right_on = ['Season', 'TeamID'])
        avoid = ['Season', 'TeamID']
        new_names = {col: result +'_'+ col for col in agg_features.columns if col not in avoid}
        df = df.rename(columns = new_names)        
        df = df.drop(columns = 'TeamID', axis = 1)
    return df

tournament_data = merge_agg_features(tournament_data, team_agg_features)


# In[ ]:


tournament_data.head()


# In[ ]:


def replace_win_loser(df):
    team_a = df.copy()
    team_b = df.copy()
    
    team_a_dict, team_b_dict = {}, {}
    
    for col in team_a.columns:
        if col.find('W_') == 0:
            new_col_name = str(col).replace('W_', 'A_')
            team_a_dict[col] = new_col_name
        if col.find('L_') == 0:
            new_col_name = col.replace('L_', 'B_')    
            team_a_dict[col] = new_col_name
            
    for col in team_b.columns:
        if col.find('W_') == 0:
            new_col_name = str(col).replace('W_', 'B_')
            team_b_dict[col] = new_col_name
        if col.find('L_') == 0:
            new_col_name = col.replace('L_', 'A_')
            team_b_dict[col] = new_col_name

    team_a = team_a.rename(columns = team_a_dict)
    team_b = team_b.rename(columns = team_b_dict)
    
    merged_df = pd.concat([team_a, team_b], axis = 0, sort = False)
    return merged_df


# In[ ]:


tournament_data = replace_win_loser(tournament_data)


# In[ ]:


def calculate_differences(df):
    """
    
    """
    df['SeedDiff'] = df['A_Seed'] - df['B_Seed']
    df['WinRatioDiff'] = df['A_WinRatio'] - df['B_WinRatio']
    df['GapAvgDiff'] = df['A_AvgScoreGap'] - df['B_AvgScoreGap']    
    df['PointsRatioDiff'] = df['A_PointsRatio'] - df['A_PointsRatio']
    return df

tournament_data = calculate_differences(tournament_data)


# ---

# # Creating the Target Variable

# In[ ]:


tournament_data['ScoreDiff'] = tournament_data['A_Score'] - tournament_data['B_Score']
tournament_data['A_Win'] = (tournament_data['ScoreDiff'] > 0).astype(int)
tournament_data = tournament_data.drop(columns=['A_Score', 'B_Score'])


# In[ ]:


tournament_data.head()


# In[ ]:


tournament_data.info()


# ---

# # Creating the Test Dataset

# In[ ]:


sub_stage_one = pd.read_csv('/kaggle/input/mens-march-mania-2022/MDataFiles_Stage1/MSampleSubmissionStage1.csv')
tst_data = sub_stage_one.copy()


# In[ ]:


tst_data.shape


# In[ ]:


def separate_id(df):
    """
    
    """
    df['Season']  = df['ID'].apply(lambda x: int(x.split('_')[0]))
    df['TeamIdA'] = df['ID'].apply(lambda x: int(x.split('_')[1]))
    df['TeamIdB'] = df['ID'].apply(lambda x: int(x.split('_')[2]))
    return df

tst_data = separate_id(tst_data)


# In[ ]:


tst_data = merge_seed(tst_data, seeds, left_on = ['Season', 'TeamIdA'], field_name = 'A_Seed')
tst_data = merge_seed(tst_data, seeds, left_on = ['Season', 'TeamIdB'], field_name = 'B_Seed')


# In[ ]:


tst_data['A_Seed'] = tst_data['A_Seed'].apply(seed_number)
tst_data['B_Seed'] = tst_data['B_Seed'].apply(seed_number)


# In[ ]:


tst_data = tst_data.rename(columns = {'TeamIdA': 'A_TeamID', 'TeamIdB': 'B_TeamID'})


# In[ ]:


tst_data


# In[ ]:


team_agg_features


# In[ ]:


def merge_agg_features(df, agg_features):
    for result in ['A', 'B']:
        df = pd.merge(df, agg_features, how = 'left', left_on = ['Season', result +'_'+ 'TeamID'], right_on = ['Season', 'TeamID'])
        avoid = ['Season', 'TeamID']
        new_names = {col: result +'_'+ col for col in agg_features.columns if col not in avoid}
        df = df.rename(columns = new_names)        
        df = df.drop(columns = 'TeamID', axis = 1)
    return df

tst_data = merge_agg_features(tst_data, team_agg_features)


# In[ ]:


tst_data = calculate_differences(tst_data)


# In[ ]:


tst_data


# In[ ]:


tst_data.shape


# # Building the Model...
# Ok, so up to this point we have been aggregating and merging data from the Seasons Datasets...
# 
# * We used the season data to create aggregated features by team and season. and merge this back to the tournament data
# * Using the merged data we calculate the outcomes of each of the games, Win or Lost.
# * This new calculated variable will be our target.

# In[ ]:


from sklearn import tree
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# In[ ]:


target_feature = 'A_Win'
avoid = ['ScoreDiff', 'Season', 'DayNum', 'A_Win']
features = [col for col in tournament_data.columns if col not in avoid]


# In[ ]:


features


# In[ ]:


features = ['A_TeamID',
            'B_TeamID',
            'A_Seed',
            'B_Seed',
            'A_WinRatio',
            'A_AvgScoreGap',
            'A_PointsRatio',
            'B_WinRatio',
            'B_AvgScoreGap',
            'B_PointsRatio',
            'SeedDiff',
            'WinRatioDiff',
            'GapAvgDiff',
            'PointsRatioDiff',
           ]


# In[ ]:


# Develop a CV loop to avoid leaking data from future tournaments...
def kfold_model(train_df, tst_df):
    cvs = []
    preds_test = []
    seasons = train_df['Season'].unique()
    
    for season in seasons[1:]:
        print(f'\nValidating on season {season}')
        X_train = train_df[train_df['Season'] < season][features].reset_index(drop = True).copy()
        X_val = train_df[train_df['Season'] == season][features].reset_index(drop = True).copy()
        
        y_train = train_df[train_df['Season'] < season][target_feature].reset_index(drop = True).copy()
        y_val = train_df[train_df['Season'] == season][target_feature].reset_index(drop = True).copy()
        
        tst_dataset = tst_data[features].copy()
        
        
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)        
        X_val = scaler.transform(X_val)
        tst_dataset = scaler.transform(tst_dataset)
        
        model = XGBClassifier(n_estimators = 2048, random_state = 3)
        model.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = 0, early_stopping_rounds = 128)
        pred = model.predict_proba(X_val)[:, 1]
        
        pred_test = model.predict_proba(tst_dataset)[:, 1]
        preds_test.append(pred_test)
        
        loss = log_loss(y_val, pred)
        cvs.append(loss)
        
        print(f'\t -> Scored {loss:.4f}')
    print(f'\nLocal Cross Validation Score Is: {np.mean(cvs):.3f}', '\n')
    return preds_test


# In[ ]:


predictions = kfold_model(tournament_data, tournament_data)


# In[ ]:


mean_predictions = np.mean(predictions, 0)

sub = tst_data[['ID', 'Pred']].copy()
sub['Pred'] = mean_predictions
sub.to_csv('submission_02242022.csv', index = False)


# ---

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




