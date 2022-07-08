#!/usr/bin/env python
# coding: utf-8

# V.2
# 
# -deleted prediction year from train
# 
# V.3 
# 
# -add visualization
# 
# -add 538 rating
# 
# -add new FE

# In[ ]:


import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import lightgbm
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


woman538 = pd.read_csv('../input/ncaa-women-538-team-ratings/538ratingsWomen.csv').drop(['TeamName'],axis = 1)
woman538.head(3)


# # **X prepare**

# In[ ]:


WNCAATourneyDetailedResults = pd.read_csv('../input/ncaaw-march-mania-2021/WDataFiles_Stage2/WNCAATourneyDetailedResults.csv')
#WNCAATourneyCompactResults = pd.read_csv('../input/ncaaw-march-mania-2021/WDataFiles_Stage2/WNCAATourneyCompactResults.csv')
#TourneyResults = WNCAATourneyDetailedResults.merge(WNCAATourneyCompactResults, on= WNCAATourneyCompactResults.columns.to_list(), how='right')
TourneyResults = WNCAATourneyDetailedResults
TourneyResults['TypeCompetition'] = 'Tourney'

WRegularSeasonDetailedResults = pd.read_csv('../input/ncaaw-march-mania-2021/WDataFiles_Stage2/WRegularSeasonDetailedResults.csv')
#WRegularSeasonCompactResults = pd.read_csv('../input/ncaaw-march-mania-2021/WDataFiles_Stage2/WRegularSeasonCompactResults.csv')
#SeasonResults = WRegularSeasonDetailedResults.merge(WRegularSeasonCompactResults, on= WRegularSeasonCompactResults.columns.to_list(), how='right')
SeasonResults = WRegularSeasonDetailedResults
SeasonResults['TypeCompetition'] = 'Season'

X = TourneyResults.merge(SeasonResults, on= SeasonResults.columns.to_list(), how='outer')
X.head()


# In[ ]:


A = X[X['WLoc']=='A'].rename(columns={
    'WTeamID': 'BTeamID',
    'WScore': 'BScore',
    'LTeamID': 'ATeamID',
    'LScore': 'AScore',
}).drop('WLoc', axis=1)

H = X[X['WLoc']=='H'].rename(columns={
    'WTeamID': 'ATeamID',
    'WScore': 'AScore',
    'LTeamID': 'BTeamID',
    'LScore': 'BScore',
}).drop('WLoc', axis=1)

N = X[X['WLoc']=='N'].rename(columns={
    'WTeamID': 'ATeamID',
    'WScore': 'AScore',
    'LTeamID': 'BTeamID',
    'LScore': 'BScore',
})

import random
for index in N.index:
    if random.randint(0, 1) == 1:
        N.at[index, 'WLoc'] = N.at[index, 'ATeamID']
        N.at[index, 'ATeamID'] = N.at[index, 'BTeamID']
        N.at[index, 'BTeamID'] = N.at[index, 'WLoc']
        
        N.at[index, 'WLoc'] = N.at[index, 'AScore']
        N.at[index, 'AScore'] = N.at[index, 'BScore']
        N.at[index, 'BScore'] = N.at[index, 'WLoc']

N = N.drop('WLoc', axis=1)  

X = H.append(A, ignore_index=False,sort=False)
X = X.append(N, ignore_index=False,sort=False)
X.head()


# In[ ]:


X['FGM'] = X['WFGM'] - X['LFGM']
X['FGA'] = X['WFGA'] - X['LFGA']
X['FGM3'] = X['WFGM3'] - X['LFGM3']
X['FGA3'] = X['WFGA3'] - X['LFGA3']
X['FTM'] = X['WFTM'] - X['LFTM']
X['FTA'] = X['WFTA'] - X['LFTA']
X['OR'] = X['WOR'] - X['LOR']
X['DR'] = X['WDR'] - X['LDR']
X['Ast'] = X['WAst'] - X['LAst']
X['TO'] = X['WTO'] - X['LTO']
X['Stl'] = X['WStl'] - X['LStl']
X['Blk'] = X['WBlk'] - X['LBlk']
X['PF'] = X['WPF'] - X['LPF']

X = X.drop(['WFGM','WFGA','WFGM3','WFGA3','WFTM','WFTA','WOR','WDR','WAst','WTO','WStl','WBlk','WPF',
 'LFGM','LFGA','LFGM3','LFGA3','LFTM','LFTA','LOR','LDR','LAst','LTO','LStl','LBlk','LPF'],axis = 1)
X.head()


# In[ ]:


MeanStatA = X.groupby(['Season', 'ATeamID']).mean().reset_index().drop(['DayNum','BTeamID'],axis = 1).rename(columns={
    'AScore': 'MeanAScore_Home',
    'BScore': 'MeanBScore_Home',
    'NumOT':'NumOT_Home',
    'FGM':'FGM_Home',
    'FGA':'FGA_Home',
    'FGM3':'FGM3_Home',
    'FGA3':'FGA3_Home',
    'FTM':'FTM_Home',
    'FTA':'FTA_Home',
    'OR':'OR_Home',
    'DR':'DR_Home',
    'Ast':'Ast_Home',
    'TO':'TO_Home',
    'Stl':'Stl_Home',
    'Blk':'Blk_Home',
    'PF':'PF_Home'})
MeanStatA.head()


# In[ ]:


MeanStatB = X.groupby(['Season', 'BTeamID']).mean().reset_index().drop(['DayNum','ATeamID'],axis = 1).rename(columns={
    'AScore': 'MeanAScore_Away',
    'BScore': 'MeanBScore_Away',
    'NumOT':'NumOT_Away',
    'FGM':'FGM_Away',
    'FGA':'FGA_Away',
    'FGM3':'FGM3_Away',
    'FGA3':'FGA3_Away',
    'FTM':'FTM_Away',
    'FTA':'FTA_Away',
    'OR':'OR_Away',
    'DR':'DR_Away',
    'Ast':'Ast_Away',
    'TO':'TO_Away',
    'Stl':'Stl_Away',
    'Blk':'Blk_Away',
    'PF':'PF_Away'})
MeanStatB.head()


# In[ ]:


X = X.drop(['NumOT','FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF'],axis = 1)
X.head()


# In[ ]:


X = X.merge(MeanStatA, how='left', left_on=['Season','ATeamID'], right_on=['Season','ATeamID'])
X = X.merge(MeanStatB, how='left', left_on=['Season','BTeamID'], right_on=['Season','BTeamID'])
X.head()


# In[ ]:


X = X.drop(['DayNum'],axis = 1)
X.head()


# In[ ]:


X = X.merge(woman538, how='left', left_on=['Season','ATeamID'], right_on=['Season','TeamID']).drop(['TeamID'], axis=1)
X = X.merge(woman538, how='left', left_on=['Season','BTeamID'], right_on=['Season','TeamID']).drop(['TeamID'], axis=1)
X.head()


# In[ ]:


X['538rating'] = X['538rating_x'] - X['538rating_y']
X = X.drop(['538rating_x', '538rating_y'],axis = 1)
X.head()


# In[ ]:


#ADD SEED
Seeds = pd.read_csv('../input/ncaaw-march-mania-2021/WDataFiles_Stage2/WNCAATourneySeeds.csv')
X = X.merge(Seeds, how='left', left_on=['Season', 'ATeamID'], right_on=['Season', 'TeamID']).drop('TeamID', axis=1).rename(columns={'Seed': 'ASeed'})
X = X.merge(Seeds, how='left', left_on=['Season', 'BTeamID'], right_on=['Season', 'TeamID']).drop('TeamID', axis=1).rename(columns={'Seed': 'BSeed'})
#SEED TO FLOAT
X['ASeed'] = X['ASeed'].str.replace(r'[^0-9]', '').astype('float')
X['BSeed'] = X['BSeed'].str.replace(r'[^0-9]', '').astype('float')

X['Seed'] = X['ASeed'] - X['BSeed']
X = X.drop(['ASeed', 'BSeed'],axis = 1)
X.head()


# In[ ]:


X = X.dropna(subset=['Seed'])
#X = X.dropna(subset=['538rating'])


# In[ ]:


X['HomeWin'] = (X['AScore']-X['BScore'] > 0).astype(int)
X.head()


# # **test prepare**

# In[ ]:


test = pd.read_csv('../input/ncaaw-march-mania-2021/WDataFiles_Stage2/WSampleSubmissionStage2.csv')
submission = pd.read_csv('../input/ncaaw-march-mania-2021/WDataFiles_Stage2/WSampleSubmissionStage2.csv')

test['Season'] = test['ID'].apply(lambda x: int(x.split('_')[0]))
test['ATeamID'] = test['ID'].apply(lambda x: int(x.split('_')[1]))
test['BTeamID'] = test['ID'].apply(lambda x: int(x.split('_')[2]))
test = test.drop(['Pred','ID'], axis=1)
test['TypeCompetition'] = 'Tourney'

test.head()


# In[ ]:


test = test.merge(MeanStatA, how='left', left_on=['Season','ATeamID'], right_on=['Season','ATeamID'])
test = test.merge(MeanStatB, how='left', left_on=['Season','BTeamID'], right_on=['Season','BTeamID'])
test.head()


# In[ ]:


test = test.merge(woman538, how='left', left_on=['Season','ATeamID'], right_on=['Season','TeamID']).drop(['TeamID'], axis=1)
test = test.merge(woman538, how='left', left_on=['Season','BTeamID'], right_on=['Season','TeamID']).drop(['TeamID'], axis=1)

test['538rating'] = test['538rating_x'] - test['538rating_y']
test = test.drop(['538rating_x', '538rating_y'],axis = 1)
test.head()


# In[ ]:


#ADD SEED
test = test.merge(Seeds, how='left', left_on=['Season', 'ATeamID'], right_on=['Season', 'TeamID']).drop('TeamID', axis=1).rename(columns={'Seed': 'ASeed'})
test = test.merge(Seeds, how='left', left_on=['Season', 'BTeamID'], right_on=['Season', 'TeamID']).drop('TeamID', axis=1).rename(columns={'Seed': 'BSeed'})
#SEED TO FLOAT
test['ASeed'] = test['ASeed'].str.replace(r'[^0-9]', '').astype('float')
test['BSeed'] = test['BSeed'].str.replace(r'[^0-9]', '').astype('float')
test['Seed'] = test['ASeed'] - test['BSeed']
test = test.drop(['ASeed', 'BSeed'],axis = 1)
test.head()


# In[ ]:


temp = X.append(test, ignore_index=False,sort=False)
temp = pd.get_dummies(temp,dtype=bool)
X = temp[:len(X)]
test = temp[len(X):]
temp = pd.DataFrame
test = test.drop(['AScore','BScore','HomeWin'],axis = 1)
test.head()


# # Train

# In[ ]:


lgbm_parameters= {
    'cat_feature': [0,1,2],
    'n_estimators': 20000,
    'objective': 'binary',
    'metric': 'binary_logloss',
}


# In[ ]:


test_pred = np.zeros(len(test))
test_pred = []

kf = KFold(n_splits=10, shuffle=True)

for year in test['Season'].unique():
    
    #X_year = X[(X['Season'] >= year-3)&(X['Season'] <= year+3)]
    X_year = X[((X['TypeCompetition_Season'] == True)&(X['Season'] == year))|(X['Season'] != year)]
    
    y_year = X_year['HomeWin']
    X_year = X_year.drop(['AScore','BScore','HomeWin'], axis=1)
    test_year = test[test['Season'] == year]
 
    lgbm_val_pred = np.zeros(len(y_year))
    lgbm_test_pred = np.zeros(len(test_year))
    logloss = []
    
    for trn_idx, val_idx in kf.split(X_year,y_year):
        x_train_idx = X_year.iloc[trn_idx]
        y_train_idx = y_year.iloc[trn_idx]
        x_valid_idx = X_year.iloc[val_idx]
        y_valid_idx = y_year.iloc[val_idx]

        lgbm_model = LGBMClassifier(**lgbm_parameters)
        lgbm_model.fit(x_train_idx, y_train_idx, eval_set = ((x_valid_idx,y_valid_idx)),verbose = False, early_stopping_rounds = 100,categorical_feature=[0,1,2])
        lgbm_test_pred += lgbm_model.predict_proba(test_year)[:,1]/10
        logloss.append(log_loss(y_valid_idx, lgbm_model.predict_proba(x_valid_idx)[:,1])) 
        
    test_pred += lgbm_test_pred.tolist()
    
    print('Year_Predict:',year,'Log_Loss:',np.mean(logloss))


# In[ ]:


submission.Pred = test_pred   
submission.to_csv('submission.csv', index=False)


# In[ ]:


plt.rcParams["figure.figsize"] = (4, 8)
lightgbm.plot_importance(lgbm_model,height=.9)


# # Show Result

# In[ ]:


Teams = pd.read_csv('../input/ncaaw-march-mania-2021/WDataFiles_Stage2/WTeams.csv')
Seeds = Seeds.merge(Teams,how = 'left',left_on = ['TeamID'],right_on = ['TeamID'])
Slots = pd.read_csv('../input/ncaaw-march-mania-2021/WDataFiles_Stage2/WNCAATourneySlots.csv')
Teams = pd.read_csv('../input/ncaaw-march-mania-2021/WDataFiles_Stage2/WTeams.csv')
test['Pred'] = test_pred 
test = test[['ATeamID','BTeamID','Pred']]
test = test.merge(Seeds[Seeds['Season']==2021],how= 'left',left_on = 'ATeamID',right_on = 'TeamID').drop(['TeamID','Season'], axis=1)
test = test.merge(Seeds[Seeds['Season']==2021],how= 'left',left_on = 'BTeamID',right_on = 'TeamID').drop(['TeamID'], axis=1).rename(columns={'Seed_x': 'ASeed','Seed_y': 'BSeed','TeamName_x':'ATeamName','TeamName_y':'BTeamName'})
test = test.drop(['Season','ATeamID','BTeamID'],axis = 1)

FirstRound = Slots[0:32]
SecondRound = Slots[32:48]
Sweet16 = Slots[48:56]
EliteEight = Slots[56:60]
FinalFour = Slots[60:62]
Final = Slots[62:]
FirstRound['ASeed'] = FirstRound['StrongSeed']
FirstRound['BSeed'] = FirstRound['WeakSeed']

def merger(Round,test=test):  
    Round = Round.merge(test,how = 'left',left_on = ['ASeed','BSeed'],right_on = ['ASeed','BSeed'])
    for index in Round[Round['Pred'].isna() == True].index:
            Round.at[index, 'Temp'] = Round.at[index, 'ASeed']
            Round.at[index, 'ASeed'] = Round.at[index, 'BSeed']
            Round.at[index, 'BSeed'] = Round.at[index, 'Temp']
            Round = Round.drop(['Temp'],axis = 1)
            
    Round = Round.merge(test,how = 'left',left_on = ['ASeed','BSeed'],right_on = ['ASeed','BSeed'])
    Round = Round.drop(['Pred_x','ATeamName_x','BTeamName_x'],axis = 1).rename(columns={'Pred_y': 'Pred','ATeamName_y':'ATeamName','BTeamName_y':'BTeamName'})
    Round['Win'] = Round['Pred'] > 0.5
    Round[['Win_Seed']] = Round[Round['Win'] == True][['ASeed']]
    Round[['Win_Name']] = Round[Round['Win'] == True][['ATeamName']]
    Round['Win_Seed'].fillna(value=Round[Round['Win'] == False]['BSeed'], inplace=True)
    Round['Win_Name'].fillna(value=Round[Round['Win'] == False]['BTeamName'], inplace=True)
    return Round 


# # First Round Predict

# In[ ]:


FirstRound = merger(FirstRound)
FirstRound_Win = FirstRound[['Slot','Win_Seed']]
for game in FirstRound.values:
    print('{:4}({:6.2%}){:^16}{:2}{:^16}({:6.2%}){:>4}'.format(game[3],game[5], game[6],'vs', game[7],1-game[5],game[4]))
    print('{:>25} {:<} {:<}'.format('WINNER:',game[10],game[9]))
    print('-'*58)


# # Second Round Predict

# In[ ]:


SecondRound = SecondRound.merge(FirstRound_Win,how= 'left', left_on = 'StrongSeed',right_on = 'Slot')
SecondRound = SecondRound.merge(FirstRound_Win,how= 'left', left_on = 'WeakSeed',right_on = 'Slot').drop(['Slot','Slot_y'],axis=1).rename(columns={'Win_Seed_x': 'ASeed','Win_Seed_y': 'BSeed','Slot_x':'Slot'})
SecondRound = merger(SecondRound)
SecondRound_Win = SecondRound[['Slot','Win_Seed']]
for game in SecondRound.values:
    print('{:4}({:6.2%}){:^16}{:2}{:^16}({:6.2%}){:>4}'.format(game[3],game[5], game[6],'vs', game[7],1-game[5],game[4]))
    print('{:>25} {:<} {:<}'.format('WINNER:',game[10],game[9]))
    print('-'*58)


# # Sweet 16 Predict

# In[ ]:


Sweet16 = Sweet16.merge(SecondRound_Win,how= 'left', left_on = 'StrongSeed',right_on = 'Slot')
Sweet16 = Sweet16.merge(SecondRound_Win,how= 'left', left_on = 'WeakSeed',right_on = 'Slot').drop(['Slot','Slot_y'],axis=1).rename(columns={'Win_Seed_x': 'ASeed','Win_Seed_y': 'BSeed','Slot_x':'Slot'})
Sweet16 = merger(Sweet16)
Sweet16_Win = Sweet16[['Slot','Win_Seed']]
for game in Sweet16.values:
    print('{:4}({:6.2%}){:^16}{:2}{:^16}({:6.2%}){:>4}'.format(game[3],game[5], game[6],'vs', game[7],1-game[5],game[4]))
    print('{:>25} {:<} {:<}'.format('WINNER:',game[10],game[9]))
    print('-'*58)


# # Elite Eight Predict

# In[ ]:


EliteEight = EliteEight.merge(Sweet16_Win,how= 'left', left_on = 'StrongSeed',right_on = 'Slot')
EliteEight = EliteEight.merge(Sweet16_Win,how= 'left', left_on = 'WeakSeed',right_on = 'Slot').drop(['Slot','Slot_y'],axis=1).rename(columns={'Win_Seed_x': 'ASeed','Win_Seed_y': 'BSeed','Slot_x':'Slot'})
EliteEight = merger(EliteEight)
EliteEight_Win = EliteEight[['Slot','Win_Seed']]
for game in EliteEight.values:
    print('{:4}({:6.2%}){:^16}{:2}{:^16}({:6.2%}){:>4}'.format(game[3],game[5], game[6],'vs', game[7],1-game[5],game[4]))
    print('{:>25} {:<} {:<}'.format('WINNER:',game[10],game[9]))
    print('-'*58)


# # Final Four Predict

# In[ ]:


FinalFour = FinalFour.merge(EliteEight_Win,how= 'left', left_on = 'StrongSeed',right_on = 'Slot')
FinalFour = FinalFour.merge(EliteEight_Win,how= 'left', left_on = 'WeakSeed',right_on = 'Slot').drop(['Slot','Slot_y'],axis=1).rename(columns={'Win_Seed_x': 'ASeed','Win_Seed_y': 'BSeed','Slot_x':'Slot'})
FinalFour = merger(FinalFour)
FinalFour_Win = FinalFour[['Slot','Win_Seed']]
for game in FinalFour.values:
    print('{:4}({:6.2%}){:^16}{:2}{:^16}({:6.2%}){:>4}'.format(game[3],game[5], game[6],'vs', game[7],1-game[5],game[4]))
    print('{:>25} {:<} {:<}'.format('WINNER:',game[10],game[9]))
    print('-'*58)


# # Final Predict

# In[ ]:


Final = Final.merge(FinalFour_Win,how= 'left', left_on = 'StrongSeed',right_on = 'Slot')
Final = Final.merge(FinalFour_Win,how= 'left', left_on = 'WeakSeed',right_on = 'Slot').drop(['Slot','Slot_y'],axis=1).rename(columns={'Win_Seed_x': 'ASeed','Win_Seed_y': 'BSeed','Slot_x':'Slot'})
Final = merger(Final)
Final_Win = Final[['Slot','Win_Seed']]
for game in Final.values:
    print('{:4}({:6.2%}){:^16}{:2}{:^16}({:6.2%}){:>4}'.format(game[3],game[5], game[6],'vs', game[7],1-game[5],game[4]))
    print('{:>25} {:<} {:<}'.format('WINNER:',game[10],game[9]))
    print('-'*58)

