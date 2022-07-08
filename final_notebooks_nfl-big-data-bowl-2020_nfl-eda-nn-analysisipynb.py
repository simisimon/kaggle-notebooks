#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import os
import numpy as np
import pandas as pd
import sklearn.metrics as mtr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization,LeakyReLU,PReLU,ELU,ThresholdedReLU,Concatenate
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from  keras.callbacks import EarlyStopping,ModelCheckpoint
import datetime
import warnings
warnings.filterwarnings('ignore')

TRAIN_OFFLINE = False

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)


# In[ ]:


if TRAIN_OFFLINE:
    train = pd.read_csv('../input/train.csv', dtype={'WindSpeed': 'object'})
else:
    train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})


# In[ ]:


outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()


# In[ ]:


def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans

def strtofloat(x):
    try:
        return float(x)
    except:
        return -1

def map_weather(txt):
    ans = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        ans*=0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans*3
    if 'sunny' in txt or 'sun' in txt:
        return ans*2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2*ans
    if 'snow' in txt:
        return -3*ans
    return 0

def OffensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0, 'QB' : 0, 'RB' : 0, 'TE' : 0, 'WR' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic

def DefensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic

def orientation_to_cat(x):
    x = np.clip(x, 0, 360 - 1)
    try:
        return str(int(x/15))
    except:
        return "nan"
def preprocess(train):
    ## GameClock
    train['GameClock_sec'] = train['GameClock'].apply(strtoseconds)
    train["GameClock_minute"] = train["GameClock"].apply(lambda x : x.split(":")[0]).astype("object")

    ## Height
    train['PlayerHeight_dense'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    ## Time
    train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

    ## Age
    seconds_in_year = 60*60*24*365.25
    train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
    train["PlayerAge_ob"] = train['PlayerAge'].astype(np.int).astype("object")

    ## WindSpeed
    train['WindSpeed_ob'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    train['WindSpeed_dense'] = train['WindSpeed_ob'].apply(strtofloat)

    ## Weather
    train['GameWeather_process'] = train['GameWeather'].str.lower()
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    train['GameWeather_dense'] = train['GameWeather_process'].apply(map_weather)

    ## Rusher
    train['IsRusher'] = (train['NflId'] == train['NflIdRusher'])
    train['IsRusher_ob'] = (train['NflId'] == train['NflIdRusher']).astype("object")
    temp = train[train["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team":"RusherTeam"})
    train = train.merge(temp, on = "PlayId")
    train["IsRusherTeam"] = train["Team"] == train["RusherTeam"]

    ## dense -> categorical
    train["Quarter_ob"] = train["Quarter"].astype("object")
    train["Down_ob"] = train["Down"].astype("object")
    train["JerseyNumber_ob"] = train["JerseyNumber"].astype("object")
    train["YardLine_ob"] = train["YardLine"].astype("object")
    # train["DefendersInTheBox_ob"] = train["DefendersInTheBox"].astype("object")
    # train["Week_ob"] = train["Week"].astype("object")
    # train["TimeDelta_ob"] = train["TimeDelta"].astype("object")


    ## Orientation and Dir
    train["Orientation_ob"] = train["Orientation"].apply(lambda x : orientation_to_cat(x)).astype("object")
    train["Dir_ob"] = train["Dir"].apply(lambda x : orientation_to_cat(x)).astype("object")

    train["Orientation_sin"] = train["Orientation"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
    train["Orientation_cos"] = train["Orientation"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
    train["Dir_sin"] = train["Dir"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
    train["Dir_cos"] = train["Dir"].apply(lambda x : np.cos(x/360 * 2 * np.pi))

    ## diff Score
    train["diffScoreBeforePlay"] = train["HomeScoreBeforePlay"] - train["VisitorScoreBeforePlay"]
    train["diffScoreBeforePlay_binary_ob"] = (train["HomeScoreBeforePlay"] > train["VisitorScoreBeforePlay"]).astype("object")

    ## Turf
    Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 
    train['Turf'] = train['Turf'].map(Turf)

    ## OffensePersonnel
    temp = train["OffensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(OffensePersonnelSplit(x)))
    temp.columns = ["Offense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]
    train = train.merge(temp, on = "PlayId")

    ## DefensePersonnel
    temp = train["DefensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(DefensePersonnelSplit(x)))
    temp.columns = ["Defense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]
    train = train.merge(temp, on = "PlayId")

    ## sort
    #train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index(drop = True)
    train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'IsRusherTeam', 'IsRusher']).reset_index(drop = True)
    return train


# ## *Anchoring offense moving left from {0,0}*

# In[ ]:


def create_features(df, deploy=False):
    def new_X(x_coordinate, play_direction):
        if play_direction == 'left':
            return 120.0 - x_coordinate
        else:
            return x_coordinate

    def new_line(rush_team, field_position, yardline):
        if rush_team == field_position:
            # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
            return 10.0 + yardline
        else:
            # half the field plus the yards between midfield and the line of scrimmage
            return 60.0 + (50 - yardline)

    def new_orientation(angle, play_direction):
        if play_direction == 'left':
            new_angle = 360.0 - angle
            if new_angle == 360.0:
                new_angle = 0.0
            return new_angle
        else:
            return angle

    def euclidean_distance(x1,y1,x2,y2):
        x_diff = (x1-x2)**2
        y_diff = (y1-y2)**2

        return np.sqrt(x_diff + y_diff)

    def back_direction(orientation):
        if orientation > 180.0:
            return 1
        else:
            return 0

    def update_yardline(df):
        new_yardline = df[df['NflId'] == df['NflIdRusher']]
        new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)
        new_yardline = new_yardline[['GameId','PlayId','YardLine']]

        return new_yardline

    def update_orientation(df, yardline):
        df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)
        df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
        df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)

        df = df.drop('YardLine', axis=1)
        df = pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

        return df

    def back_features(df):
        carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine']]
        carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
        carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
        carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
        carriers = carriers.rename(columns={'X':'back_X',
                                            'Y':'back_Y'})
        carriers = carriers[['GameId','PlayId','NflIdRusher','back_X','back_Y','back_from_scrimmage','back_oriented_down_field','back_moving_down_field']]

        return carriers

    def features_relative_to_back(df, carriers):
        player_distance = df[['GameId','PlayId','NflId','X','Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[['X','Y','back_X','back_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field'])\
                                         .agg({'dist_to_back':['min','max','mean','std']})\
                                         .reset_index()
        player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
                                   'min_dist','max_dist','mean_dist','std_dist']

        return player_distance

    def defense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

        defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
        defense['def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        defense = defense.groupby(['GameId','PlayId'])\
                         .agg({'def_dist_to_back':['min','max','mean','std']})\
                         .reset_index()
        defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']

        return defense
    
    def rusher_features(df):       
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Dir', 'S', 'A', 'X', 'Y']]
        rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS', 'RusherA', 'RusherX', 'RusherY']
        
        radian_angle = (90 - rusher['RusherDir']) * np.pi / 180.0
        v_horizontal = np.abs(rusher['RusherS'] * np.cos(radian_angle))
        v_vertical = np.abs(rusher['RusherS'] * np.sin(radian_angle)) 
        
        rusher['v_horizontal'] = v_horizontal
        rusher['v_vertical'] = v_vertical
        
        rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS','RusherA','RusherX', 'RusherY','v_horizontal', 'v_vertical']
        
        return rusher
    
    def static_features(df):           
        add_new_feas = []

        ## Height
        df['PlayerHeight_dense'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
        
        add_new_feas.append('PlayerHeight_dense')

        ## Time
        df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
        df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

        df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
        df['PlayerBirthDate'] =df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

        ## Age
        seconds_in_year = 60*60*24*365.25
        df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
        add_new_feas.append('PlayerAge')

        ## WindSpeed
        df['WindSpeed_ob'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
        df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
        df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
        df['WindSpeed_dense'] = df['WindSpeed_ob'].apply(strtofloat)
        add_new_feas.append('WindSpeed_dense')

        ## Weather
        df['GameWeather_process'] = df['GameWeather'].str.lower()
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
        df['GameWeather_dense'] = df['GameWeather_process'].apply(map_weather)
        add_new_feas.append('GameWeather_dense')

        ## Orientation and Dir
        df["Orientation_ob"] = df["Orientation"].apply(lambda x : orientation_to_cat(x)).astype("object")
        df["Dir_ob"] = df["Dir"].apply(lambda x : orientation_to_cat(x)).astype("object")

        df["Orientation_sin"] = df["Orientation"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
        df["Orientation_cos"] = df["Orientation"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
        df["Dir_sin"] = df["Dir"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
        df["Dir_cos"] = df["Dir"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
        add_new_feas.append("Dir_sin")
        add_new_feas.append("Dir_cos")

        ## diff Score
        df["diffScoreBeforePlay"] = df["HomeScoreBeforePlay"] - df["VisitorScoreBeforePlay"]
        add_new_feas.append("diffScoreBeforePlay")
        
        
        static_features = df[df['NflId'] == df['NflIdRusher']][add_new_feas+['GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir',
                                                            'YardLine','Quarter','Down','Distance','DefendersInTheBox']].drop_duplicates()
        static_features.fillna(-999,inplace=True)

        return static_features


    def combine_features(relative_to_back, defense, rushing, static, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,rushing,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

        return df
    
    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    rush_feats = rusher_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats, rush_feats, static_feats, deploy=deploy)
    
    return basetable


# In[ ]:


get_ipython().run_line_magic('time', 'train_basetable = create_features(train, False)')


# # *Splitting data into Training/Val*

# In[ ]:


X = train_basetable.copy()
yards = X.Yards

y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1

X.drop(['GameId','PlayId','Yards'], axis=1, inplace=True)
feature_columns = X.columns.values


# In[ ]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X.shape)


# # *Building Neural Network and RandomForest Regressor*

# In[ ]:


from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import re
from keras.losses import binary_crossentropy
from  keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import codecs
from keras.utils import to_categorical
from sklearn.metrics import f1_score

class CRPSCallback(Callback):
    
    def __init__(self,validation, predict_batch_size=20, include_on_batch=False):
        super(CRPSCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch
        
        print('validation shape',len(self.validation))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('CRPS_score_val' in self.params['metrics']):
            self.params['metrics'].append('CRPS_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['CRPS_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):
        logs['CRPS_score_val'] = float('-inf')
            
        if (self.validation):
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_pred = self.model.predict(X_valid)
            y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
            y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
            val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
            val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s


# In[ ]:


# Calculate CRPS score
def crps_score(y_prediction, y_valid, shape=X.shape[0]):
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_prediction, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * shape)
    crps = np.round(val_s, 6)
    
    return crps


# # *Neural Network Optimizer*

# In[ ]:


def get_nn(x_tr, y_tr, x_val, y_val, shape):
    K.clear_session()
    inp = Input(shape = (x_tr.shape[1],))
    x = Dense(1277, input_dim=X.shape[1], activation='relu')(inp)
    x = Dropout(0.33709)(x)
    x = BatchNormalization()(x)
    x = Dense(427, activation='relu')(x)
    x = Dropout(0.85564)(x)
    x = BatchNormalization()(x)
    x = Dense(426, activation='relu')(x)
    x = Dropout(0.65879)(x)
    x = BatchNormalization()(x)
    
    out = Dense(199, activation='softmax')(x)
    model = Model(inp,out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
    
    es = EarlyStopping(monitor='CRPS_score_val', 
                       mode='min',
                       restore_best_weights=True, 
                       verbose=1, 
                       patience=10)

    mc = ModelCheckpoint('best_model.h5',monitor='CRPS_score_val',mode='min',
                                   save_best_only=True, verbose=1, save_weights_only=True)
    
    bsz = 1024
    steps = x_tr.shape[0]/bsz

    model.fit(x_tr, y_tr,callbacks=[CRPSCallback(validation = (x_val,y_val)),es,mc], epochs=100, batch_size=bsz,verbose=1)
    model.load_weights("best_model.h5")
    
    y_pred = model.predict(x_val)
    y_valid = y_val
    crps = crps_score(y_pred, y_valid, shape=shape)

    return model,crps


# # *RandomForest Optimizer*

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

def get_rf(x_tr, y_tr, x_val, y_val, shape):
    model = RandomForestRegressor(bootstrap=False, max_features=0.43709, min_samples_leaf=19, 
                                  min_samples_split=11, n_estimators=63, n_jobs=-1, random_state=42)
    model.fit(x_tr, y_tr)
    
    y_pred = model.predict(x_val)
    y_valid = y_val
    crps = crps_score(y_pred, y_valid, shape=shape)
    
    return model, crps


# # *Training the Model*

# In[ ]:


from sklearn.model_selection import train_test_split, KFold
import time

loop = 2
fold = 5

oof_nn = np.zeros([loop, y.shape[0], y.shape[1]])
oof_rf = np.zeros([loop, y.shape[0], y.shape[1]])

models_nn = []
crps_csv_nn = []
models_rf = []
crps_csv_rf = []

feature_importance = np.zeros([loop, fold, X.shape[1]])

s_time = time.time()

for k in range(loop):
    kfold = KFold(fold, random_state = 42 + k, shuffle = True)
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
        print("-----------")
        print(f'Loop {k+1}/{loop}' + f' Fold {k_fold+1}/{fold}')
        print("-----------")
        tr_x, tr_y = X[tr_inds], y[tr_inds]
        val_x, val_y = X[val_inds], y[val_inds]
        
        # Train NN
        nn, crps_nn = get_nn(tr_x, tr_y, val_x, val_y, shape=val_x.shape[0])
        models_nn.append(nn)
        print("the %d fold crps (NN) is %f"%((k_fold+1), crps_nn))
        crps_csv_nn.append(crps_nn)
        
        # Train RF
        rf, crps_rf = get_rf(tr_x, tr_y, val_x, val_y, shape=val_x.shape[0])
        models_rf.append(rf)
        print("the %d fold crps (RF) is %f"%((k_fold+1), crps_rf))
        crps_csv_rf.append(crps_rf)
        
        # Feature Importance
        feature_importance[k, k_fold, :] = rf.feature_importances_
        
        #Predict OOF
        oof_nn[k, val_inds, :] = nn.predict(val_x)
        oof_rf[k, val_inds, :] = rf.predict(val_x)


# In[ ]:


crps_oof_nn = []
crps_oof_rf = []

for k in range(loop):
    crps_oof_nn.append(crps_score(oof_nn[k,...], y))
    crps_oof_rf.append(crps_score(oof_rf[k,...], y))


# In[ ]:


print("mean crps (NeuralNetwork) is %f"%np.mean(crps_csv_nn))
print("mean crps (RandomForest) is %f"%np.mean(crps_csv_rf))


# In[ ]:


print("mean OOF crps (NeuralNetwork) is %f"%np.mean(crps_oof_nn))
print("mean OOF crps (RandomForest) is %f"%np.mean(crps_oof_rf))


# # *Feature Importance*

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

feature_importances = pd.DataFrame(np.mean(feature_importance, axis=0).T, columns=[[f'fold_{fold_n}' for fold_n in range(fold)]])
feature_importances['feature'] = feature_columns
feature_importances['average'] = feature_importances[[f'fold_{fold_n}' for fold_n in range(fold)]].mean(axis=1)
feature_importances.sort_values(by=('average',), ascending=False).head(10)


# In[ ]:


feature_importance_flatten = pd.DataFrame()
for i in range(len(feature_importances.columns)-2):
    col = ['feature', feature_importances.columns.values[i][0]]
    feature_importance_flatten = pd.concat([feature_importance_flatten, feature_importances[col].rename(columns={f'fold_{i}': 'importance'})], axis=0)

plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importance_flatten.sort_values(by=('importance',), ascending=False), x=('importance',), y=('feature',))
plt.title(f'Feature Importances over {loop} loops and {fold} folds')
plt.show();


# # *Weight Optimization Blending*

# In[ ]:


def weight_opt(oof_nn, oof_rf, y_true):
    weight_nn = np.inf
    best_crps = np.inf
    
    for i in np.arange(0, 1.01, 0.05):
        crps_blend = np.zeros(oof_nn.shape[0])
        for k in range(oof_nn.shape[0]):
            crps_blend[k] = crps_score(i * oof_nn[k,...] + ((i-1) * oof_rf[k,...], y_true))
        if np.mean(crps_blend) < best_crps:
            best_crps = np.mean(crps_blend)
            weight_nn = round(i, 2)
            
        print(str(round(i, 2)) + ' : mean crps (Blend) is ', round(np.mean(crps_blend), 6))
        
    print('-'*36)
    print('Best weight for NN: ', weight_nn)
    print('Best weight for RF: ', round(1-weight_nn, 2))
    print('Best mean crps (Blend): ', round(best_crps, 6))
    
    return weight_nn, round(1-weight_nn, 2)


# In[ ]:


weight_nn, weight_rf = weight_opt(oof_nn, oof_rf, y)


# 

# In[ ]:


def predict(x_te, models_nn, models_rf, weight_nn, weight_rf):
    model_num_nn = len(models_nn)
    model_num_rf = len(models_rf)
    for k,m in enumerate(models_nn):
        if k==0:
            y_pred_nn = m.predict(x_te, batch_size=1024)
            y_pred_rf = models_rf[k].predict(x_te)
        else:
            y_pred_nn += m.predict(x_te, batch_size=1024)
            y_pred_rf += models_rf[k].predict(x_te)
            
    y_pred_nn = y_pred_nn / model_num_nn
    y_pred_rf = y_pred_rf / model_num_rf
    
    return weight_nn * y_pred_nn + weight_rf * y_pred_rf


# In[ ]:


get_ipython().run_cell_magic('time', '', "if  TRAIN_OFFLINE==False:\n    from kaggle.competitions import nflrush\n    env = nflrush.make_env()\n    iter_test = env.iter_test()\n\n    for (test_df, sample_prediction_df) in iter_test:\n        basetable = create_features(test_df, deploy=True)\n        basetable.drop(['GameId','PlayId'], axis=1, inplace=True)\n        scaled_basetable = scaler.transform(basetable)\n\n        y_pred = predict(scaled_basetable)\n        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]\n\n        preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)\n        env.predict(preds_df)\n\n    env.write_submission_file()\n")


# # ***Deep Dive with an extensive EDA***

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.patches as patches
import seaborn as sns 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import warnings
warnings.filterwarnings('ignore')

#import sparklines
import colorcet as cc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from IPython.display import HTML
from IPython.display import Image
from IPython.display import display
from IPython.core.display import display
from IPython.core.display import HTML
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from PIL import Image

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import scipy 
from scipy import constants
import math

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import colorcet as cc
plt.style.use('seaborn') 
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
##%config InlineBackend.figure_format = 'retina'   < - keep in case 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
#USE THIS in some form:
# th_props = [('font-size', '13px'), ('background-color', 'white'), ('color', '#666666')]
# td_props = [('font-size', '15px'), ('background-color', 'white')]
#styles = [dict(selector="td", props=td_props), dict(selector="th", props=th_props)]
# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###?sns.set_context('paper')  #Everything is smaller, use ? 
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
##This helps set size of all fontssns.set(font_scale=1.5)

#~~~~~~~~~~~~~~~~~~~~~~~~~ B O K E H ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.io import show
from bokeh.io import push_notebook
from bokeh.io import output_notebook
from bokeh.io import output_file
from bokeh.io import curdoc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.plotting import show                  
from bokeh.plotting import figure                  
from bokeh.plotting import output_notebook 
from bokeh.plotting import output_file

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.models import ColumnDataSource
from bokeh.models import Circle
from bokeh.models import Grid 
from bokeh.models import LinearAxis
from bokeh.models import Plot
from bokeh.models import Slider
from bokeh.models import CategoricalColorMapper
from bokeh.models import FactorRange
from bokeh.models.tools import HoverTool
from bokeh.models import FixedTicker
from bokeh.models import PrintfTickFormatter
from bokeh.models.glyphs import HBar

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.core.properties import value

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.palettes import Blues4
from bokeh.palettes import Spectral5
from bokeh.palettes import Blues8

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.layouts import row
from bokeh.layouts import column
from bokeh.layouts import gridplot

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.sampledata.perceptions import probly

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.transform import factor_cmap

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ M L  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
import gc, pickle, tqdm, os, datetime



# In[ ]:


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Import raw data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
gold = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
dontbreak = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
from kaggle.competitions import nflrush

killed_columns=['xyz','etc']
def drop_these_columns(your_df,your_list):
    #KILL KOLUMNS
    your_df.drop(your_list,axis=1,inplace=True)
    return(your_df)
YRS = dontbreak[dontbreak.NflId==dontbreak.NflIdRusher].copy()
YR1 = YRS[YRS.Season==2017]
YR2 = YRS[YRS.Season==2018]


# In[ ]:


from bokeh.transform import factor_cmap
from bokeh.palettes import Blues8
from bokeh.palettes import Blues, Spectral6, Viridis, Viridis256, GnBu, Viridis256
from bokeh.palettes import Category20b,Category20c,Plasma,Inferno,Category20
from bokeh.palettes import cividis, inferno, grey


# In[ ]:


print(f"The total number of games in the training data is {df['GameId'].nunique()}")
print(f"The total number of plays in the training data is {df['PlayId'].nunique()}")
print(f"The NFL seasons in the training data are {df['Season'].unique().tolist()}")


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


nullvalues = df.loc[:, df.isnull().any()].isnull().sum().sort_values(ascending=False)

print(nullvalues)


# In[ ]:


f,ax = plt.subplots(figsize=(12,10))
sns.heatmap(df.iloc[:,2:].corr(),annot=True, linewidths=.1, fmt='.1f', ax=ax)

plt.show();


# In[ ]:


from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px

temp_df = df.query("NflIdRusher == NflId")
fig = px.histogram(temp_df, x="Yards")
plt.style.use("classic")
layout = go.Layout(title=go.layout.Title(text="Distribution of Yards (Target)", x=0.5), font=dict(size=14), width=800, height=600)
fig.update_layout(layout)
fig.show();


# In[ ]:


my_data = df[['PlayerCollegeName','NflId', 'DisplayName']].drop_duplicates().copy()

college_attended = my_data["PlayerCollegeName"].value_counts()

df_cc = pd.DataFrame({'CollegeName':college_attended.index, 'Count':college_attended.values}).sort_values("Count", ascending = False)

#df_cc.Count.astype('int', inplace=True)

df_cc = df_cc[df_cc.CollegeName != 'Louisiana State']

df_cc.at[42,'Count']=51


# In[ ]:


# LSU AND Louisiana State is taken as different colleges!!! That’s not the case!
# LSU has a massive number of players currently in the NFL, and so let's consolidate the values...

df_cc.sort_values('Count',ascending=False, inplace=True)

#pd.set_option('display.max_rows', 500)
df_cc.index = df_cc.index + 1


# In[ ]:


mysource = ColumnDataSource(df_cc)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
p = figure(y_range=df_cc.CollegeName[:50],    
           
  title = '\nNFL Player Count by College Attended\n',
  x_axis_label ='# of NFL players that attended the college prior\n',
  plot_width=800,
  plot_height=600,
  tools="hover",
  toolbar_location=None,   
 
)

p.hbar(y='CollegeName',  # center of your y coordinate launcher, 40 points as def above ... 
    left=0, # or left=20, etc
    right='Count',    # right is 40 points... 
    height=1,
    alpha=0.4,
    #color='orange',    #color=Spectral3  #color=Blues8,   
    #background_fill_color="#efe8e2", 
    #     fill_color=Blues8,
    #     fill_alpha=0.4, 
    
    fill_color=factor_cmap(
        'CollegeName',
        palette=inferno(50),  #cividis(50),  #d3['Category20b'][4],  #Category20b(2),  #[2],   #Category20b,   #Viridis256,    #GnBu[8], #,#Spectral6,             #viridis(50),  #[3], #Spectral6,  #|Blues[2],
        factors=df_cc.CollegeName[:50].tolist()    
    ),

    source = mysource,
    fill_alpha=1.0,
    #line_color='blue'  
) 

p.title.text_font_size = '12pt'

# Y TICKS:
p.yaxis.major_tick_line_color = None
p.axis.minor_tick_line_color = None

p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
p.yaxis.minor_tick_line_color = None

# GRID:
# p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None   

# HOVER:
hover = HoverTool()
hover.tooltips = [
    ("College Name:", "@CollegeName"),
    ("Ranking by Count", "$index"),
    ("Number of gradutes that entered the NFL:", "@Count"),
]

p.add_tools(hover)

output_notebook(hide_banner=True)
show(p);


# Below is a quick way of referring to the pertinent column without having to actually type out the full name.  Just use `df.iloc[:,<the number index of column below>]`. i.e. listing out df.Team you could use `df.iloc[:,2]`.  This helps when trying to list out numerous columns...where you could then enter `df.iloc[:,[2,4,6,7,10]]`.  I find it faster...

# In[ ]:


refer = pd.DataFrame(df.columns)
refer.columns=['Mapper']
refer.index.name='Ref:'
refer.style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])])


# In[ ]:


df.head(1).T. \
style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])])


# Every game has a unique id, every play has a unique play id, and all single play related data is included such as game venue information, player information, and even the offense and defense formations at the time of the snap. This is really detailed information and its amazing they have this much data available on a play by play basis, the sky is the limit in terms of what you could do from an analysis perspective. I especially think that the score over the course of time may impact the manner and style of run plays.
# 
# 

# <div class="h3"><i>Summary of our dataset:</i></div>
# <p style="margin-top: 50px">It is always important to look at our entire dataset and examine the descriptive statistics:</p>
# 
# &ensp; **Number of football teams in the NFL:** &ensp; 32  
# &ensp; **Number of unique NFL players in our dataset:** &ensp; 2,231  
# &ensp; **Number of 2017 Season players:** &ensp; 1,788  
# &ensp; **Number of 2018 Season players:** &ensp; 1,783   
# &ensp; **Number of players playing both yrs:** &ensp; 1,340    
# &ensp; **Number of players allowed per team:** &ensp; 53    
# &ensp; **Number of games a team plays in a NFL season:** &ensp; 16      
# &ensp; **Number of weeks in a NFL season:** &ensp; 17   
# &ensp; **Total unique NFL games played per season:** &ensp; 256  
# &ensp; **Number of NFL seasons in the dataset:** &ensp; 2  
# &ensp; **Dataset NFL season years:** &ensp; 2017 and 2018 Seasons    
# &ensp; **Dataset total number of unique NFL games:** &ensp; 512  
# &ensp; **Number of unique run plays in our dataset:** &ensp; 23,171  
# &ensp; **Number of 2017 Season run plays:** &ensp; 11,900  
# &ensp; **Number of 2018 Season run plays:** &ensp; 11,271  
# &ensp; **Number of unique NFL jersey numbers:** &ensp; 99  
# &ensp; **Number of players on roster that never played:** &ensp; 11  
# &ensp; **Size of a typical NFL field (in acres):** &ensp; 1.32

# In[ ]:


def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='green', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='yellow')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='red')
        plt.text(62, 50, '<- Player Yardline at Snap', color='red')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
    return fig, ax

create_football_field()
plt.show();


# In[ ]:


import math
def get_dx_dy(angle, dist):
    cartesianAngleRadians = (450-angle)*math.pi/180.0
    dx = dist * math.cos(cartesianAngleRadians)
    dy = dist * math.sin(cartesianAngleRadians)
    return dx, dy

play_id = 20181007011551
fig, ax = create_football_field()
df.query("PlayId == @play_id and Team == 'away'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=50, legend='Away')
df.query("PlayId == @play_id and Team == 'home'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=50, legend='Home')
df.query("PlayId == @play_id and NflIdRusher == NflId") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='red', s=100, legend='Rusher')
rusher_row = df.query("PlayId == @play_id and NflIdRusher == NflId")
yards_covered = rusher_row["Yards"].values[0]

x = rusher_row["X"].values[0]
y = rusher_row["Y"].values[0]
rusher_dir = rusher_row["Dir"].values[0]
rusher_speed = rusher_row["S"].values[0]
dx, dy = get_dx_dy(rusher_dir, rusher_speed)

ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3)
plt.title(f'Play # {play_id} and yard distance is {yards_covered}', fontsize=15)
plt.legend()
plt.show();


# In[ ]:


fig, ax = create_football_field()
train.query("PlayId == 20170907000118 and Team == 'away'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='yellow', s=30, legend='Away')
train.query("PlayId == 20170907000118 and Team == 'home'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='red', s=30, legend='Home')
plt.title('Play # 20170907000118')
plt.legend()
plt.show();


# In[ ]:


rusher_dir


# In[ ]:


def get_plot(play_id):
    fig, ax = create_football_field()
    df.query("PlayId == @play_id and Team == 'away'") \
        .plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=50, legend='Away')
    df.query("PlayId == @play_id and Team == 'home'") \
        .plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=50, legend='Home')
    df.query("PlayId == @play_id and NflIdRusher == NflId") \
        .plot(x='X', y='Y', kind='scatter', ax=ax, color='red', s=100, legend='Rusher')
    rusher_row = df.query("PlayId == @play_id and NflIdRusher == NflId")
    yards_covered = rusher_row["Yards"].values[0]

    x = rusher_row["X"].values[0]
    y = rusher_row["Y"].values[0]
    rusher_dir = rusher_row["Dir"].values[0]
    rusher_speed = rusher_row["S"].values[0]
    dx, dy = get_dx_dy(rusher_dir, rusher_speed)

    ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3)
    plt.title(f'Play # {play_id} and yard distance is {yards_covered}', fontsize=15)
    plt.legend()
    return plt

temp_df = df.groupby("PlayId").first()
temp_df = temp_df.sort_values(by="Yards").reset_index().head()

for play_id in temp_df["PlayId"].values:
    plt = get_plot(play_id)
    plt.show();


# In[ ]:


playid = 20181230154157
train.query("PlayId == @playid").head()


# In[ ]:


temp_df = df.groupby("PlayId").first()
temp_df = temp_df[temp_df["Yards"]==0].reset_index().head()

for play_id in temp_df["PlayId"].values:
    plt = get_plot(play_id)
    plt.show();


# In[ ]:


yl = train.query("PlayId == @playid")['YardLine'].tolist()[0]
fig, ax = create_football_field(highlight_line=True,
                                highlight_line_number=yl+54)
train.query("PlayId == @playid and Team == 'away'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='yellow', s=30, legend='Away')
train.query("PlayId == @playid and Team == 'home'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='red', s=30, legend='Home')
plt.title(f'Play # {playid}')
plt.legend()
plt.show();


# In[ ]:


temp_df = df.groupby("PlayId").first()
temp_df = temp_df[temp_df["Yards"]==0].reset_index().head()

for play_id in temp_df["PlayId"].values:
    plt = get_plot(play_id)
    plt.show();


# In[ ]:


temp_df = df.groupby("PlayId").first()
temp_df = temp_df[temp_df["Yards"]>10].reset_index().head()

for play_id in temp_df["PlayId"].values:
    plt = get_plot(play_id)
    plt.show();


# In[ ]:


######################################################################### 
#   Creating an example visualization illustrating the core problem     #
#########################################################################

# #Styling
sns.set_style("white", {'grid.linestyle': '--'})

#Creating a synthetic dataset
synthetic_data   = [12,15,19,21,25,29,35,45,65,90,105,190,305,405,420,430,1700,2300,2450,2855,3105]
synthetic_points = ['U','T','S','R','Q','P','O','N','M','L','K','J','I','H','G','F','E','D','C','B','A']
     
#Creating core dataframe
mich24 = pd.DataFrame(synthetic_data,index=synthetic_points)
mich24.columns =['Count']
mich24 = mich24.sort_values(['Count'], ascending=False)
plt.figure(figsize=(15,7))

ax = sns.barplot(mich24.index, 
                 mich24.Count, 
                 color='olive', 
                 alpha=1, 
                 linewidth=.1, 
                 edgecolor="black",
                 saturation=10)

ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.set(xlabel="\n\n\n", ylabel='Count\n')
ax.set_xticklabels(mich24.index, color = 'black', alpha=.8)

for item in ax.get_xticklabels(): 
    item.set_rotation(0)
    
for i, v in enumerate(mich24["Count"].iteritems()):        
    ax.text(i ,v[1], "{:,}".format(v[1]), color='gray', va ='bottom', rotation=0, ha='center')
    

ax.tick_params(axis='x', which='major', pad=9)    
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)  
#################################################plt.tight_layout()

plt.axvline(4.5, 0,0.95, linewidth=1.4, color="#00274C", label="= 'Charbonnet Cut' proposed", linestyle="--")

plt.legend(loc='center', fontsize=13)

#  plt.text(3+0.2, 4.5, "An annotation", horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.text(0, -425, "\nThis is a synthetic dataset I created to illustrate a core problem seen when plotting histograms/boxplots with highly variable data", fontsize=12)

#Remove unnecessary chart junk   
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 
# #sns.despine()

plt.title('\n\n\n\nCreating a splitting point for a better visualization, if we also plot the 2nd tier/level data...''\n\n',fontsize=14, loc="left")    

plt.text(4.5,600,"|--- This region contains a lack of VISUAL INSIGHT so, we should split data based on Charbonnet Cut ---|", fontsize=12.5)

plt.show();



# In[ ]:


sns.set_style("white", {'grid.linestyle': '--'})

#Creating a synthetic dataset
synthetic_data   = [12,15,19,21,25,29,35,45,65,90,105,190,305,405,420,430]
synthetic_points = ['U','T','S','R','Q','P','O','N','M','L','K','J','I','H','G','F']
     
#Creating core dataframe
mich24 = pd.DataFrame(synthetic_data,index=synthetic_points)
mich24.columns =['Count']
mich24 = mich24.sort_values(['Count'], ascending=False)
plt.figure(figsize=(12,6))

ax = sns.barplot(mich24.index, 
                 mich24.Count, 
                 color='olivedrab', 
                 alpha=1, 
                 linewidth=.1, 
                 edgecolor="black",
                 saturation=10)

ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.set(xlabel="\n\n\n", ylabel='Count\n')
ax.set_xticklabels(mich24.index, color = 'black', alpha=.8)

for item in ax.get_xticklabels(): 
    item.set_rotation(0)
    
for i, v in enumerate(mich24["Count"].iteritems()):        
    ax.text(i ,v[1], "{:,}".format(v[1]), color='black', va ='bottom', rotation=0, ha='center')
    

ax.tick_params(axis='x', which='major', pad=9)    
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)  
plt.tight_layout()

plt.legend(loc='center', fontsize=13)

plt.text(0, -65, "\n Now we can see the relationship in the heights of the '2nd tier' (East of the Charbonnet Cut) data...", fontsize=15, color="#00274C")

#Remove unnecessary chart junk   
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 
# #sns.despine()

plt.title('2nd Tier data has been plotted, and we can see the relationships without data being drowned out...\n\n',fontsize=15, loc="left")    
plt.show();


# Let's now begin: Initial Examination of overall running (rushing) yards per play:
# * Let's take a look at the most important feature, the yards, which we will need to be able to predict going forward after our machine learning model is created

# In[ ]:


tf = df.query("NflIdRusher == NflId")

sns.set_style("white", {'grid.linestyle': '--'})

fig, ax = plt.subplots(figsize=(10,5))
ax.set_xlim(-10,26)

c = [ 'r' if i < 0 else 'b' for i in tf.Yards]

sns.distplot(tf.Yards, kde=False, color='turquoise', bins=100, 
            hist_kws={"linewidth": 0.5, 'edgecolor':'blue'})

## Remove the x-tick labels:  plt.xticks([])
plt.yticks([])
## This method also hides the tick marks

plt.title('\n Overall distribution of yards gained during an individual running play\n',fontsize=14)
plt.xlabel('\n Yards (yd) Gained -->\n',fontsize=14)
sns.despine(top=True, right=True, left=True, bottom=True)
plt.tight_layout()

plt.tight_layout()
plt.show();


# In[ ]:


plt.style.use('default')
plt.figure(figsize=(12,10))
temp_df = df.query("NflIdRusher == NflId")
sns.scatterplot(temp_df["Dis"], temp_df["Yards"])
plt.xlabel('Distance covered', fontsize=10)
plt.ylabel('Yards (Target)', fontsize=10)
plt.title("Distance covered by Rusher Vs Yards (target)", fontsize=15)
plt.show();


# In[ ]:


plt.figure(figsize=(12,10))
temp_df = df.query("NflIdRusher == NflId")
sns.scatterplot(temp_df["S"], temp_df["Yards"])
plt.xlabel('Rusher Speed', fontsize=10)
plt.ylabel('Yards (Target)', fontsize=10)
plt.title("Rusher Speed Vs Yards (target)", fontsize=15)
plt.show();


# In[ ]:


plt.figure(figsize=(12,10))
temp_df = df.query("NflIdRusher == NflId")
sns.scatterplot(temp_df["A"], temp_df["Yards"])
plt.xlabel('Rusher Acceleration', fontsize=10)
plt.ylabel('Yards (Target)', fontsize=10)
plt.title("Rusher Acceleration Vs Yards (target)", fontsize=15)
plt.show();


# In[ ]:


plt.style.use('dark_background')
#aaa is our temp df
aaa = gold
aaa['IsRunner'] = aaa.NflId == aaa.NflIdRusher

#bbb is now the unique run play runners in the year 2018 only
bbb = aaa[aaa.IsRunner & (aaa.Season == 2018)]

#ccc is now a specific actual game
ccc=bbb[bbb.GameId==2018121601] # we grab random game #1 
ccc = ccc[['Yards']][:]
ccc['colors'] = ['gold' if x <= 0 else 'limegreen' for x in ccc['Yards']]
##ccc.sort_values('Yards', inplace=True)
ccc.reset_index(inplace=True)

plt.figure(figsize=(8,10))
plt.hlines(y=ccc.index, xmin=0, xmax=ccc.Yards, color=ccc.colors, alpha=0.8, linewidth=8)
plt.gca().set(ylabel='$Play$\n', xlabel='\n$Yards$')
plt.yticks(fontsize=8)
plt.title('\n Positive & Negative Yards for random NFL game #1 - (2018 Season)\n', fontdict={'size':14})
plt.grid(linestyle='--', alpha=0.5)
sns.despine(top=True, right=True, left=True, bottom=True)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show();


# In[ ]:


plt.style.use('dark_background')

ccc=bbb[bbb.GameId==2018121500]
ccc = ccc[['Yards']][:]
ccc['colors'] = ['gold' if x <= 0 else 'limegreen' for x in ccc['Yards']]
##ccc.sort_values('Yards', inplace=True)
ccc.reset_index(inplace=True)
plt.figure(figsize=(8,10), dpi= 300)
plt.hlines(y=ccc.index, xmin=0, xmax=ccc.Yards, color=ccc.colors, alpha=0.8, linewidth=9)
plt.gca().set(ylabel='$Play$\n', xlabel='\n$Yards$')
plt.yticks(fontsize=6)
plt.title('\nPositive and Negative Yards for random NFL game #2 - (2018 Season)\n', fontdict={'size':14})
plt.grid(linestyle='--', alpha=0.5)
sns.despine(top=True, right=True, left=True, bottom=True)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show();


# In[ ]:


plt.style.use('dark_background')

ccc=bbb[bbb.GameId==2018121501]
ccc = ccc[['Yards']][:]
ccc['colors'] = ['gold' if x <= 0 else 'limegreen' for x in ccc['Yards']]
##ccc.sort_values('Yards', inplace=True)
ccc.reset_index(inplace=True)
plt.figure(figsize=(8,10), dpi= 300)
plt.hlines(y=ccc.index, xmin=0, xmax=ccc.Yards, color=ccc.colors, alpha=0.8, linewidth=9)
plt.gca().set(ylabel='$Play$\n', xlabel='\n$Yards$')
plt.yticks(fontsize=6)
plt.title('\nPositive and Negative Yards for random NFL game #3 - (2018 Season)\n', fontdict={'size':14})
plt.grid(linestyle='--', alpha=0.5)
sns.despine(top=True, right=True, left=True, bottom=True)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show();


# ## *Insights from the above plots*
# * **Game 1:**
#   * A **high number** of run plays (49 total plays from the two teams), with one run for over 25 yards (a big gain), and a fair amount of long runs.  
#   * Only two runs produced negative yards (one hurt, it was for a loss of 5 yards), so looks like the Offense on both teams is doing pretty good, with only two of the run play for zero-yards gained.  
#   * Good game to watch to see trends in running, we have a lot of sample points here.  
#   * We can also analyze WHY there were so many long run plays, i.e. what defensive formation was being run in this game ?  This game was most likely a battle between two teams with strong runningbacks. 
# * **Game 2:**
#   * Longest run was for 14 yards, interestingly:  
#   * **10** runs were zero-yard gain, 
#   * **6** were negative gain, and in general we see very little yardage gained even when it was positive.  
#   * There is some **excellent**  defense going on in this game, it must have been a hardcore defensive battle...
# * **Game 3:**
#   * One very long run for **40 yards!**  
#   * Let's think about this for a second:  If the offense runs lets say 20 run plays in a game, and most of them were for 2-4 yards, you can quickly see how devasting one run of 40 yards can be.  
#   * A good NFL rusher can gain 100 total yards in a game, but if one of those alone was for 40 yards, that is a BIG deal.  
#   * And you can also see how a running back is worth his weight in gold, as he can change the outcome of a game with yardage gains.  
# * A good **rule of thumb** is that a pretty good running back is averaging over 4 yards per carry, and below that is 'ok'

# ## *2018 - Top 10 Longest Rushes:*
# ### *Let's take a look at the top 10 most spectacular rushes:*

# In[ ]:


cm = sns.light_palette("green", as_cmap=True)
tom = df.query("NflIdRusher == NflId")
tom = tom[tom.Season==2018] 

fixup = {"ARZ":"ARI","BLT":"BAL","CLV":"CLE","HST":"HOU"}
tom.PossessionTeam.replace(fixup, inplace=True)


tom.groupby(['PossessionTeam'], as_index=False)['Yards'].agg(max).set_index('PossessionTeam').sort_values('Yards', ascending=False)[:15].style.set_caption('TOP TEN LONGEST RUNS:').background_gradient(cmap=cm)


# <u>The first was NOT a kickoff return, it was a **handoff**</u>.  &nbsp; Now look at HOW far back he is in the endzone !   
# The guy ran probably 107 yards on that play (but gets credit for 99 yards from line of scrimmage at their own 1 yard line; that day he ran for a total of a staggering 238 yards).  This tied the NFL record for longest rush, set 35 years earlier.   
# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/unbelievable.png" width="700px">
# 
# Derrick Henry from the Tennessee Titans is a [*beast*](https://twitter.com/NFL/status/1070863698791550976?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1070863698791550976&ref_url=https%3A%2F%2Fwww.theguardian.com%2Fsport%2F2018%2Fdec%2F06%2Fderrick-henry-touchdown-titans-jaguars-nfl-99-yards).  
# The best part:  He celebrated by striking the Heisman pose, which is perfectly fine, since he won it in 2015.  
# 
# 1. Derrick Henry - Tennessee Titans
# 1. Lamar Miller	- Houston Texans 
# 1. Nick Chubb	- Cleveland Browns
# 1. Adrian Peterson	- Washington Redskins	
# 

# <div class="h4"><i>Yards vs Down:</i></div>
# * Plotting the distribution of yards by series 'Down'.  Note that we have configured many of the plots to show granular ultra-precise 1-yard increments ! 
# * In this case, removed a few outliers so as to see the general trend of the data 
#     

# * >Note: &nbsp; Creating a new term called 'unit grid'. When creating plots where the y or x axis, depending on the plot, is in a relatively short magnitiude range (lets say approximately 20 units or below), I find it helps to actually use the grid lines to expand to the plot on a somewhat granular basis.  When that is needed or helpful, spacing the grid at 'unit' levels shall be known as 'unit grid', i.e. the grids are every 1 unit on the scale.  I think it helps the viewer quickly quantify actual values, to the point where it approaches the information transfer of a barplot...

# In[ ]:


tf = df.query("NflIdRusher == NflId")
sns.set_style("ticks", {'grid.linestyle': '--'})

flierprops = dict(markerfacecolor='0.75', 
                  markersize=1,
                  linestyle='none')

fig, ax = plt.subplots(figsize=(8,6))

ax.set_ylim(-7, 14)
ax.set_title('Yards Gained by Down\n', fontsize=14)

sns.boxplot(x='Down',
            y='Yards',
            data=tf,
            ax=ax,
            showfliers=False, 
            #color='blue'
            )

ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 

ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))

ax.set(xlabel='')
ax.set_xticklabels(['1st Down', '2nd Down', '3rd Down', '4th Down'])
plt.tight_layout(); plt.show();


# * First and second down we see almost an identical distribution of runs (i.e. In the first two quarters, the teams run about the same distribution of runs in the quarters)
#   * What we find suprising is that in the first two quarters, examining the plots we see that 25% of the plays generated **less** than 1 yard total gained.  
# * In the third down, we see a slight drop in the number of yards gained, and the median yards gained has dropped a solid yard.  In a game of inches, this is a big deal. 
# * Fourth down performance is relatively poor.  (insert number of n)

# <div class="alert alert-block alert-warning">
# <b>Warning:</b> It is important to list the sample size for each of the histograms, because one may draw the erroneous conclusion that the number of times the ball was run was the 'same' for each of the downs, when in fact it wasn't...</div>

# In[ ]:


# VANILLA SWIRL
YDS_by_down = tf.groupby("Down")['Yards'].size()
total_run_plays = YDS_by_down.sum()
df_ydsbydown = pd.DataFrame( {'Down':YDS_by_down.index, 'Count':YDS_by_down.values}).sort_values('Count', ascending=False)
df_ydsbydown.set_index('Down', drop=True, inplace=True)

df_ydsbydown['Percentage']=round(df_ydsbydown.Count/total_run_plays*100, 2)
cm = sns.light_palette("green", as_cmap=True)
df_ydsbydown.style.set_caption('PLAY COUNT PER DOWN:').background_gradient(cmap=cm)


# > **INSIGHT**: &nbsp; For a sample size of all NFL rush plays over the course of <u>two entire years</u>, only **7.83%** of the runs were on 3rd down, and less than **1%** were rushes on 4th down. 

# In[ ]:


plt.figure(figsize=(12,10))
temp_df = df.query("NflIdRusher == NflId")
sns.boxplot(data=temp_df, x="Position", y="Yards", showfliers=False, whis=3.0)
plt.xlabel('Rusher position', fontsize=10)
plt.ylabel('Yards (Target)', fontsize=10)
plt.title("Rusher Position Vs Yards (target)", fontsize=15)
plt.show();


# <div class="h4"><i>Yards vs Quarter of the Game:</i></div>
# * Plotting the distribution of yards by game quarter, where 5Q symbolizes overtime...

# In[ ]:


tf = df.query("NflIdRusher == NflId")
flierprops = dict(markerfacecolor='0.75', 
                  markersize=1,
                  linestyle='none')
fig, ax = plt.subplots(figsize=(9,7))
ax.set_ylim(-7, 17)
ax.set_title('Yards Gained by Game Quarter\n\n', fontsize=12)

sns.boxplot(x='Quarter',
            y='Yards',
            data=tf,
            ax=ax,
            showfliers=False , 
            #color='blue'
            )

ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 

ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
ax.set(xlabel='')
ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])
plt.tight_layout(); plt.show();


# <div class="h4"><i>Yards Gained vs Box Defender Count:</i></div>
# * Plotting the distribution of yards gained vs number of defenders in the box.  We will call this the defensive 'density' count...
# * A helpful reference image I created is shown below. 
# 
# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/thebox.png" width="400px">

# In[ ]:


dff = tf[tf.DefendersInTheBox>2]
dff.DefendersInTheBox = dff.DefendersInTheBox.astype('int')

flierprops = dict(markerfacecolor='0.75', 
                  markersize=1,
                  linestyle='none')

fig, ax = plt.subplots(figsize=(9,7))
ax.set_ylim(-7, 23)
ax.set_title('Yards Gained vs number of Defenders in the box\n\n', fontsize=12)
sns.boxplot(x='DefendersInTheBox',
            y='Yards',
            data=dff,
            ax=ax,
            showfliers=False , 
            #color='blue'
            )
            #flierprops=flierprops)

ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 

ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
# ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
ax.set(xlabel="\nNumber of defensive players in the 'Box'\n\n")
# ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])

plt.tight_layout()
plt.show();


# * When there are nine defensive players in the box, 25% of the runs gained LESS than 0 yards, and half the runs were for LESS than 2 yards. 
# * It is very rare for defenses to line up with four or less players, but when they do, the Offense seems to gain a fair amount of yards.

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

sns.pairplot(train.query("NflIdRusher == NflId").sample(1000)[['S','Dis','A','Yards','DefensePersonnel']],
            hue='DefensePersonnel')
plt.suptitle('Speed, Acceleration, Distance traveled, and Yards Gained for Rushers')
plt.show();


# ### *Yards Gained vs Defensive Personnel 'Groupings':*
# * Plotting the distribution of yards gained vs Defensive Formation...
# * First lets start by looking at the combined 2017/2018 dataset formations by play count (i.e. how many times in the two year season data that particular Defensive Schema were run)
# * We will then look exclusively look at 2018 stats

# In[ ]:


temp101 = pd.DataFrame(tf.DefensePersonnel.value_counts())
temp101.index.name = 'DefensePersonnel'
temp101.columns=['Play Count']

cm = sns.light_palette("green", as_cmap=True)

temp101.style.set_caption('Count of plays by Personnel:').background_gradient(cmap=cm)


# ## *Defensive Personnel's impact on yard gained*
# We can see that there are about 5 common defensive packages that are used. How does the way the defense is aligned correlate with the offensive production (yards gained)?
# 
# What stands out at first glance is that the `4DL - 4LB - 3DB` Defense shows a different distribution in yards gained.
# 
# Per wikipedia: https://en.wikipedia.org/wiki/4%E2%80%934_defense
# 
# *Originally seen as a passing defense against the spread, modern versions of the 4-4 are attacking defenses stocked with multiple blitz packages that can easily be concealed and altered.*

# In[ ]:


top_10_defenses = train.groupby('DefensePersonnel')['GameId'] \
    .count() \
    .sort_values(ascending=False).index[:10] \
    .tolist()
print("Top 10 Defenses:")
top_10_defenses


# In[ ]:


plt.style.use('classic')
train_play = train.groupby('PlayId').first()
train_top10_def = train_play.loc[train_play['DefensePersonnel'].isin(top_10_defenses)]

fig, ax = plt.subplots(figsize=(20, 8))
sns.violinplot(x='DefensePersonnel',
               y='Yards',
               data=train_top10_def,
               ax=ax)
plt.ylim(-10, 20)
plt.title('Yards vs Defensive Personnel', fontsize=14)
plt.show();


# Let's examine the same values, but broken out by percentage, i.e. what percentage of the time did the run play go against a particular DefensePersonnel Schema, and lets grab the top 10, since after that there is an extremely small percentage of plays incorporating that style:

# In[ ]:


temp107 = pd.DataFrame(round(tf.DefensePersonnel.value_counts(normalize=True) * 100,2)).head(10)
temp107.index.name = 'DefensePersonnel'
temp107.columns=['Percentage']
cm = sns.light_palette("green", as_cmap=True)

temp107.style.set_caption('Top 10 Percentage of plays by Defensive Personnel Grouping:').background_gradient(cmap=cm)


# ### ***The Top Five Formations:***
# 
# 1. 4 DL, 2 LB, 5 DB
#   * 4 linemen, 2 linebackers, and 5 defensive backs (6 in the 'box')
# 2. 4 DL, 3 LB, 4 DB	
#   * your conventional *4:3* type defense (7 in the 'box')
# 3. 3 DL, 4 LB, 4 DB
#   * your conventional *3:4* type defense  (7 in the 'box')
# 4. 2 DL, 4 LB, 5 DB	
#   * four linebackers, with only 2 guys on the line  (6 in the 'box')
# 5. 3 DL, 3 LB, 5 DB
#   * a type of formation build to stop the pass (6 in the 'box')
#  

# In[ ]:


sns.set_style("ticks", {'grid.linestyle': '--'})

pers = tf
dff = pers 

flierprops = dict(markerfacecolor='0.2', 
                  markersize=1,
                  linestyle='none')

fig, ax = plt.subplots(figsize=(10,10))
ax.set_ylim(-7, 22)
ax.set_title('\nAverage yards gained by Defensive Personnel Schema\n', fontsize=14)

sns.boxplot(y=dff['DefensePersonnel'].sort_values(ascending=False),
            x=dff['Yards'],
            ax=ax,
            showfliers=False ,
            linewidth=.8
            #color='blue'
            )

ax.yaxis.grid(False)   # Show the horizontal gridlines
ax.xaxis.grid(True)  # Hide x-axis gridlines 

ax.xaxis.set_major_locator(plt.MultipleLocator(1))

# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))

ax.set(xlabel="\n Yards Gained\n\n")

sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_ticks_position('none') 

ax.text(15,17.3, '#1',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=12)

ax.text(15,16.3, '#2',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)

ax.text(15,21.3, '#3',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)

ax.text(15,24.3, '#5',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)

ax.text(15,27.3, '#4',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)

ax.text(9,2, '6 guys on the line',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)

ax.text(0,.2, 'line of scrimmage',
        verticalalignment='bottom', horizontalalignment='right',
        color='blue', fontsize=9)

#-----more control-----#
ax.grid(linestyle='--', 
        linewidth='0.3', 
        color='lightgray', 
        alpha=0.8,
        axis='x'
       )

plt.axvline(0, 0,1, linewidth=.4, color="blue", linestyle="--")

plt.tight_layout()
plt.show();


# ### *Interesting outcomes to look~*
# * This analysis is important to understanding how defense is set up, its critical to understanding how to predict run productivity
# * The `#1` used defensive scheme actually resulted in slightly longer yardage plays, but also slightly higher yards lost for the offense
# * You can see how the first and second scheme is relatively good at containing the run, and as you get lower on the y-axis, you are giving up higher and higher yards
# * The most common is a 4-2-5, which is a good coverage against the pass (you have 5 DB). And since roughly 35-40% of NFL plays are runs, and the other percentage are pass, you can see why this is common. 
# * With a median yardage gain allowed of 4 yards, its pretty good against the run, AND you can see that in some cases you can get losses of up to -6 yards. 75% of the runs against this defense are held to 6 yards or less. 
# * The next most popular is your typical 4-3 defense, where you can see it holds runs to a bit shorter yardage, obviously since you have an extra linebacker involved in the tackling. But what i find interesting is that the third most common defense (3-4) has almost PRECISELY characteristics based on the data, look at the boxplots.  The 3-4 is run less than the 4-3 and the 4-2, but it seems to hold up pretty well against the run. The 3-4 in a flexible defense, and provides some great advantages when it comes to rushing the quarterback and defending against the pass. The 3-4 can be confusing for opposing quarterbacks, who find that wherever they look downfield there is a linebacker.  IF one could argue that the 3-4 is a better defense than the 4-3 in terms of rushing the QB, AND it holds up relatively well against the run (as it appears it does), then it would appear more teams SHOULD be running the 3-4 !  
# * The `#2` and `#3` most occurring run defense resulted in almost precisely the same running yards allowed distribution
# * Having 6 men on the line may appear to be a great idea against the run (and it does seems to 'squash' the run), you see that although it lowers the potential yards a runner could get, it offers no real ability to gain you negative yards on run plays, and its penetration ability are limited.  It does work well against runs, BUT if the play is a pass, you are devasted as you have very few DB to stop the pass. 
# * When there are nine defensive players in the box, 25% of the runs gained LESS than 0 yards, and half the runs were for LESS than 2 yards. 
# * It is very rare for defenses to line up with four or less players, but when they do, the Offense seems to gain a fair amount of yards.

# In[ ]:


plt.figure(figsize=(10,10))
temp_df = df.query("NflIdRusher == NflId")
sns.boxplot(data=temp_df, y="PossessionTeam", x="Yards", showfliers=False, whis=3.0)
plt.ylabel('PossessionTeam', fontsize=10)
plt.xlabel('Yards (Target)', fontsize=10)
plt.title("Possession team Vs Yards (target)", fontsize=15)
plt.show();


# In[ ]:


plt.style.use("default")
plt.figure(figsize=(15,10))
temp_df = df.query("NflIdRusher == NflId")
#sns.catplot(data=temp_df, x="Quarter", y="Yards", kind="violin")
sns.catplot(data=temp_df, x="Quarter", y="Yards", kind="boxen")
plt.xlabel('Quarter', fontsize=10)
plt.ylabel('Yards (Target)', fontsize=10)
plt.title("Quarter Vs Yards (target)", fontsize=15)
plt.show();


# ## *Running strategies change as the game goes on...*
# How are the yards gained impacted by the time in the game? Many times teams run the ball at the end of the game when they are ahead, in order to run out the gameclock and win. In these situations the run is expected more and defenses can scheme against it.
# 
# It doesn't look like the quarter has a huge impact on the running yards gained.

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 5))
ax.set_ylim(-10, 60)
ax.set_title('Yards vs Quarter')
sns.boxenplot(x='Quarter',
            y='Yards',
            data=train.sample(5000),
            ax=ax)
plt.show();


# # *Defenders In The "Box"*
# 
# The number of defenders in the box is an important part of stopping the running game. Typically defenses will add more players to this area of the field when they really want to stop a run, this comes at a cost leaving wide recievers less covered.
# 
# ![](https://i0.wp.com/www.footballzebras.com/wp-content/uploads/2019/02/Slide1.jpg?resize=596%2C317)

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 5))
ax.set_ylim(-10, 60)
sns.boxenplot(x='DefendersInTheBox',
               y='Yards',
               data=train.query('DefendersInTheBox > 2'),
               ax=ax)
plt.title('Yards vs Defenders in the Box')
plt.show();


# *The above plot shows a big difference in yards gained when looking at the number of defenders in the box. 
# If you've got 8+ defenders in the box you're looking to stop the run big time! And you can see the average rush yardage is lower. 
# Conversely, having 3 men in the box (maybe because they are in prevent defense for a long yard to gain) allows for a average return of about 10 yards!*

# ## *Distribution of Yards gained vs Defenders in the Box*

# In[ ]:


fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(15 , 10))
ax_idx = 0
ax_idx2 = 0
for i in range(4, 10):
    this_ax = axes[ax_idx2][ax_idx]
    #print(ax_idx, ax_idx2)
    sns.distplot(train.query('DefendersInTheBox == @i')['Yards'],
                ax=this_ax,
                color=color_pal[ax_idx2])
    this_ax.set_title(f'{i} Defenders in the box')
    this_ax.set_xlim(-10, 20)
    ax_idx += 1
    if ax_idx == 2:
        ax_idx = 0
        ax_idx2 += 1
plt.show();


# # *The Ball Carriers*
# Let's look at ball carriers (the players who typically are handed off the ball) and see if any individual players stand out. We will only look at players with more than 100 plays. Then we can plot the top and bottom 10 players.

# In[ ]:


train.query("NflIdRusher == NflId") \
    .groupby('DisplayName')['Yards'] \
    .agg(['count','mean']) \
    .query('count > 100') \
    .sort_values('mean', ascending=True) \
    .tail(10)['mean'] \
    .plot(kind='barh',
          figsize=(15, 5),
          color=color_pal[5],
          xlim=(0,6),
          title='Top 10 Players by Average Yards')
plt.show()
train.query("NflIdRusher == NflId") \
    .groupby('DisplayName')['Yards'] \
    .agg(['count','mean']) \
    .query('count > 100') \
    .sort_values('mean', ascending=True) \
    .head(10)['mean'] \
    .plot(kind='barh',
          figsize=(15, 5),
          color=color_pal[0],
          xlim=(0,6),
          title='Bottom 10 Players by Average Yards')
plt.show();


# # *3-4 vs 4-3 vs 4-2 (nickel) vs 2-4 Defense - Impact on Yards gained?*
# 
# Next we will use the defensive scheme data to see how the difference in down linemen impacts the yards gained.
# While there are many defensive packages, commonly defenses will run a 4-3 or 3-4 defense.
# 
# You can read more about them here: https://bleacherreport.com/articles/1289011-showcasing-the-biggest-differences-in-the-4-3-and-3-4-pass-rush
# 
# Basically you have "Down linemen" and "Linebackers" the number of players you put in each of these positions determines the defensive scheme.
# 
# _ | _
# - | - 
# ![](https://usercontent2.hubstatic.com/14167861_f496.jpg) | ![](https://img.bleacherreport.net/img/images/photos/002/822/262/07d6727c91db0f152422689a0c095c4c_crop_north.jpg?h=533&w=800&q=70&crop_x=center&crop_y=top)
# ![](https://usercontent2.hubstatic.com/14148753_f496.jpg) | ![](https://img.bleacherreport.net/img/images/photos/002/811/505/1f3630fb49f2e47dd3d62f01f6956940_crop_north.png?1395418910&w=630&h=420)

# In[ ]:


def label_bars(ax, bars, text_format, **kwargs):
    """
    Attaches a label on every bar of a regular or horizontal bar chart
    """
    ys = [bar.get_y() for bar in bars]
    y_is_constant = all(y == ys[0] for y in ys)  # -> regular bar chart, since all all bars start on the same y level (0)

    if y_is_constant:
        _label_bar(ax, bars, text_format, **kwargs)
    else:
        _label_barh(ax, bars, text_format, **kwargs)


def _label_bar(ax, bars, text_format, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    """
    max_y_value = ax.get_ylim()[1]
    inside_distance = max_y_value * 0.05
    outside_distance = max_y_value * 0.01

    for bar in bars:
        text = text_format.format(bar.get_height())
        text_x = bar.get_x() + bar.get_width() / 2

        is_inside = bar.get_height() >= max_y_value * 0.15
        if is_inside:
            color = "white"
            text_y = bar.get_height() - inside_distance
        else:
            color = "black"
            text_y = bar.get_height() + outside_distance

        ax.text(text_x, text_y, text, ha='center', va='bottom', color=color, **kwargs)


def _label_barh(ax, bars, text_format, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    Note: label always outside. otherwise it's too hard to control as numbers can be very long
    """
    max_x_value = ax.get_xlim()[1]
    distance = max_x_value * 0.0025

    for bar in bars:
        text = text_format.format(bar.get_width())

        text_x = bar.get_width() + distance
        text_y = bar.get_y() + bar.get_height() / 2

        ax.text(text_x, text_y, text, va='center', **kwargs)


# In[ ]:


# Create the DL-LB combos
plt.style.use('default')
train['DL_LB'] = train['DefensePersonnel'] \
    .str[:10] \
    .str.replace(' DL, ','-') \
    .str.replace(' LB','') # Clean up and convert to DL-LB combo
top_5_dl_lb_combos = train.groupby('DL_LB').count()['GameId'] \
    .sort_values() \
    .tail(10).index.tolist()
ax = train.loc[train['DL_LB'].isin(top_5_dl_lb_combos)] \
    .groupby('DL_LB').mean()['Yards'] \
    .sort_values(ascending=True) \
    .plot(kind='bar',
          title='Average Yards Top 5 Defensive DL-LB combos',
          figsize=(15, 5),
          color=color_pal[2])

bars = [p for p in ax.patches]
value_format = "{:0.2f}"
label_bars(ax, bars, value_format, fontweight='bold')
plt.show();


# # *Snap to Handoff Time*
# Different types of designed runs develop differently, one way to understand the play design is by looking at the time it takes the quarterback to hand the ball off to the rusher. Lets take a look at the distribution of seconds taken.

# In[ ]:


plt.style.use('default')
train['SnapHandoffSeconds'] = (pd.to_datetime(train['TimeHandoff']) - \
                               pd.to_datetime(train['TimeSnap'])).dt.total_seconds()

(train.groupby('SnapHandoffSeconds').count() / 22 )['GameId'].plot(kind='bar', figsize=(15, 5))

bars = [p for p in ax.patches]
value_format = "{}"
label_bars(ax, bars, value_format, fontweight='bold')
plt.show();


# *Due to lack of percision we don't have much detail about the snap time. Additionally it looks like the sparcity of data for seconds that are not 1 or 2 - cause the average Yards to have large variance.*

# In[ ]:


train.groupby('SnapHandoffSeconds')['Yards'].mean().plot(kind='barh',
                                                         color=color_pal[1],
                                                         figsize=(15, 5),
                                                         title='Average Yards Gained by SnapHandoff Seconds')
plt.show();


# ## *Yards gained average vs season week number:*
# * Plotting average yards gained per play, on a week by week basis as the season transpires

# In[ ]:


sns.set(style="white", palette="muted", color_codes=True)

t = tf[['Week', "Yards"]].groupby('Week').mean().sort_values(by = "Yards")

fig, ax = plt.subplots(figsize=(9,5))

specific_colors=['grey']*17
specific_colors[8]='#ffbf00'
specific_colors[5]='#169016'


sns.set_color_codes('pastel')             
sns.barplot(x=t.index,
            y=t.Yards,
            ax=ax, 
            linewidth=.2, 
            edgecolor="black",
            palette=specific_colors)

ax.set_ylim(0, 5.5)
ax.set_title('\nOverall Average yards gained per play as the season progresses (week by week)\n\n', fontsize=14)
# ax.set(ylabel='Yards Gained\n', rotation='horizontal')
ax.set(xlabel='\nWeek Number (in the season)')
ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 
ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

#-----more control-----#
ax.grid(linestyle='--', 
        linewidth='0.7', 
        color='lightgray', 
        alpha=0.6,
        axis='y'
       )


for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))
ax.spines['top'].set_linewidth(0)  
ax.spines['left'].set_linewidth(.3)  
ax.spines['right'].set_linewidth(0)  
ax.spines['bottom'].set_linewidth(.3) 

plt.ylabel("YDS\n", fontsize=11, rotation=90)

plt.tight_layout()
plt.show();

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

t = tf[['Week', "Yards"]].groupby('Week').mean().sort_values(by = "Yards")
t['WeekInSeason']= t.index
t.reset_index(drop=True, inplace=True)
starter= t.loc[0,'Yards']
t['gain']=t.Yards/starter
t['gainpct']=round(100*(t.gain-1), 3)

fig, ax = plt.subplots(figsize=(9.5,5))

sns.lineplot(x="WeekInSeason", y="gainpct", data=t, 
            color='blue', 
            ax=ax,
            markers=True, marker='o', 
            dashes=True) 

ax.set_title('\nPercent Gain in the average running yards gained per play (week by week)\n\n', fontsize=14)

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.set(ylabel='Gain in average YDS per carry (in %)\n')

ax.set(xlabel='\nWeek Number (in the season)')
ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 

ax.spines['top'].set_linewidth(0)  
ax.spines['left'].set_linewidth(.3)  
ax.spines['right'].set_linewidth(0)  
ax.spines['bottom'].set_linewidth(.3); 

plt.tight_layout()
plt.show(); 


# ### *First Plot:* 
#   * Barplot of average yards gained per game per week    
#   
# ### *Second Plot:*
#   * Our baseline is week 1, and then from there, we compare where the run game is at week by week to that baseline, i.e. if in week 5 the average yards/carry is 10% more compared to week 1 game, then we graph that value in the plot...
#   * Progress:
#     * It appears in the first month of the season, there is a strong climb in runner performance.  By week 6, the runners are peaking in terms of productivity.  Potential fatigue factor kicks in two months into the season, then strong push for the second half of the season as the teams are getting stronger and stronger, making a run towards the playoffs. 

# ### *Number of run plays called per NFL game per team:*
# * Histogram plot of the total number of run plays called per game per season. 
# * This takes into consideration every game played, where each team takes turns calling run plays (and contains both 2017 and 2018 data)

# In[ ]:


sns.set_style("white", {'grid.linestyle': '--'})

t2 = tf.groupby(['GameId','Team'])['PlayId'].count()
t2 = pd.DataFrame(t2)
fig, ax = plt.subplots(figsize=(9,7))

sns.distplot(t2.PlayId, kde=False, color="orange", 
            hist_kws={"linewidth": .9, 'edgecolor':'grey'}, bins=24)

## Remove the x-tick labels:  plt.xticks([])
plt.yticks([])
## This method also hides the tick marks
plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',fontsize=14, loc="left")
plt.xlabel('\nNumber of times the ball was run in the game\n')
sns.despine(top=True, right=True, left=True, bottom=True)
plt.axvline(x=22, color='black', linestyle="--", linewidth=.4)

plt.text(22.8, 114, r'Median: 22 carries', {'color': 'maroon', 'fontsize': 9})
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))

plt.tight_layout()
plt.show();


# * The median number of times there is a run play per team in a game is 22, i.e. if a single running back was used, he would be running roughly 22 plays per game, but there is a fairly wide variation here from 10 up to about 40 plays in a game.  30 is considered a fair number of plays for a running back, beyond 40 is considered *extreme* for a single player...
# * Distribution appears to be bi-modal, where there is a peak at 20 and a peak at about 28 carries.  One could argue this could even be the difference between teams that run the ball a fair amount (as part of their offensive strategy), and those that choose to prefer the pass with a balance of some running plays to keep the defense off guard...
# * This does bring up the fact that to play in the NFL, as a premier running back you will be getting the ball many times, and **durability** becomes a major factor as the season goes on ! 
# * Also, remember that the RB does not run every run play, sometimes there are substitutes made

# ### *Diving Deeper:  *
# 
# The below plot is an exact count visualization of the number of run plays that occurred in a game, specifically in the entire 2018 season 
# * By using the swarmplot, we see the precise distribution - and this gives a better representation of the distribution of values (where 1:1 viz, i.e. one dot is one game that had a specific count of run plays)
# * We also can **quickly** see the second, third, and fourth most run play count in a random game

# In[ ]:


number_plays_2018_perteam = bbb.groupby(['GameId', 'Team'], as_index=False).agg({'PlayId': 'nunique'})

sns.set_style("white", {'grid.linestyle': '--'})
fig, ax = plt.subplots(figsize=(8,8))

#ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(True)  # Hide x-axis gridlines 

ax.xaxis.set_major_locator(plt.MultipleLocator(5))

sns.swarmplot(number_plays_2018_perteam.PlayId, color="b", ax=ax)
sns.despine(top=True, right=True, left=True, bottom=True)

plt.ylabel('The Number of Teams that ran the x-axis play count value\n', fontsize=12)

plt.xlabel('\nTotal Run Plays by a Team in an entire game', fontsize=12)
plt.title('\n          2018 Season: Number of Run Plays Distribution by Team\n',fontsize=14, loc="left")

plt.tight_layout()
plt.show();


# * What we find interesting is the peaks are not **that** pronouced though, i.e. there are many teams that will run the ball 17, 18, 19, 20, up to 22 times in a single game, and also a fair amount of teams that will run the ball 24, 25, 26, up to 27 times in a single game...
# * It should be noted that this is an intriguing factor:
#   * In a single game, there are not a tremendous number of run plays either way, meaning our sample size per team per game of run plays is somewhat limited, so deriving a predictive model will contain many factors with a number of samples that is relatively small, offering a challenge...

# ### *Total Rushing Yards per NFL Team:*
# * The following shows the total run yards per team, over the course of two individual seasons.  
# * We use the total over two years to show the effect the running game can have on a team's performance.  We will plot the yards on a per game average basiss, but the concern here is to show the vast amount of offensive yards that the top teams had over the others.  
# * The New England Patriots won the 2018 season superbowl (against the LA Rams).  The running offense was a major factor in that. 
# * **Note:** I include a new plotting term called **`icicles`** to enhance the visualization of barplots.  Using `icicles`, one can not clutter the plot excessively but still relay x-axis values superimposed onto the chart.  Thus it is not necessary to cover the entire plot with a grid, but rather only the section that specifically needs it and where it is pertinent.  
#   * *This term does not currently exist in mainstream visualization, I'm creating it.*

# In[ ]:


df04 = tf.groupby('PossessionTeam')['Yards'].agg(sum).sort_values(ascending=True)
df04 = pd.DataFrame(df04)
df04['group'] = df04.index

my_range=range(1,33)

fig, ax = plt.subplots(figsize=(9,9))

my_color=np.where( (df04.group == 'NE') | (df04.group == 'NO') | (df04.group == 'LA') , 'darkorange', 'skyblue')

my_size=np.where(df04['group']=='B', 70, 30)
 
plt.hlines(y=my_range, xmin=0, xmax=df04['Yards'], color=my_color, alpha=0.4)
plt.scatter(df04.Yards, my_range, color=my_color, s=my_size, alpha=1)
 
# Add title and exis names
plt.yticks(my_range, df04.group)
plt.title("\nTotal Rushing Yards per Team \n(over the course of two NFL seasons)\n\n", loc='left', fontsize=14)
plt.xlabel('\n Total Rushing Yards')
plt.ylabel('')
##############plt.ylabel('NFL\nTeam\n')

ax.spines['top'].set_linewidth(.3)  
ax.spines['left'].set_linewidth(.3)  
ax.spines['right'].set_linewidth(.3)  
ax.spines['bottom'].set_linewidth(.3)  

plt.text(0, 33.3, r'Top Three:  LA Rams, New England Patriots, and New Orleans Saints absolutely dominating the rushing game...', {'color': 'grey', 'fontsize': 8.5})
sns.despine(top=True, right=True, left=True, bottom=True)

plt.text(4005, 2, '<-- I call these icicles', {'color': 'grey', 'fontsize': 8})

plt.axvline(x=3500, color='lightgrey', ymin = .01, ymax=.82, linestyle="--", linewidth=.4)
plt.axvline(x=4000, color='lightgrey', ymin = .01, ymax=.9, linestyle="--", linewidth=.4)
plt.axvline(x=3000, color='lightgrey', ymin = .01, ymax=.43, linestyle="--", linewidth=.4)
plt.axvline(x=2500, color='lightgrey', ymin = .01, ymax=.07, linestyle="--", linewidth=.4)

plt.tight_layout()
plt.show();


# ### *Correlation between Player's Weight and Jersey Number:*
# *  If you run a correlation between generalized player weight and jersey number and position, you see high correlation, but why ? 
# *  We accidentally have a good feature to use, which is on the surface jersey number shouldn't matter in any of this, **but** if we change JerseyNumber into a categorical bin (1-19), (20-29), we see that it can be quite helpful.  Because only certain positions are actually allowed to wear certain ranges of jersey numbers.  Thus during our modelling we will in fact include jersey number into bins.  

# ### *Analyzing Player Speeds:*
# > The very fastest players on the team are **not** always the ones that run the ball alot, in fact, WR (wide receivers) are generally quite fast, but rushing the ball in the NFL is not a track meet.  It could be if track introduced 300lb guys every 3 yards that were trying to level you. Let's examine the speeds we observed. 

# In[ ]:


sns.set_style("white", {'grid.linestyle': '--'})
speed = bbb.groupby(['DisplayName'])['S'].agg('max').sort_values(ascending=True)
speed = pd.DataFrame(speed)
fig, ax = plt.subplots(figsize=(8,7))
sns.distplot(speed, kde=False, color="indigo", hist_kws={"linewidth": .9, 'edgecolor':'darkgrey'}, bins=38)
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_yticklabels([])
plt.title('\nDistribution of running speed for all players in the 2017/2018 seasons\n', fontsize=12, loc="center")
ax.set(xlabel="\nRunner Speed (yds/sec)\n")

plt.tight_layout()
plt.show();


# * **Rule of thumb:**  double the yards/sec value and you will get mph.  Why ?  Because 3,600 is almost 3,520. (i.e. 3 feet per yard, 1,760 yards per mile, 3,600 seconds per hour, and doubleing 1,760 gets you 3,520, which is very close to 3,600)
# * Sometimes the runner is not moving at the time of the handoff to him, and other times, he is running as fast as he can FROM a standstill.  The question then becomes how quickly can a running back accelerate from zero to full speed. The data we have been given is the speed at the time of the handoff.  
#   * Per competition director: &nbsp; *"in this contest, you're only receiving this tracking information at the moment a handoff is made"*
# * One of the biggest misconceptions that emerges from the NFL Combine each year is the importance of 40-yard dash times, when in reality the 10-yard split is a **very** important indicator of how well the runningback may do in the NFL.  Explosive speed trumps raw speed...

# ### *Offense Formation Strategies:*
# * We will now examine which offense formation strategies appeared to result in the best yardage gained

# In[ ]:


my_x = bbb.groupby('OffenseFormation')['Yards'].mean().sort_values(ascending=False).values
my_y = bbb.groupby('OffenseFormation')['Yards'].mean().index

sns.set(style="white", palette="muted", color_codes=True)
fig, ax = plt.subplots(figsize=(8,4))

sns.barplot(x=my_y, y=my_x, ax=ax, linewidth=.5, edgecolor="black")

ax.set_title('\n2018: Avg YDS gained per play Offense Formations\n', fontsize=14)
ax.set(xlabel='\nOffense Formations')
ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 
ax.yaxis.set_major_locator(plt.MultipleLocator(1))

#-----more control-----#
ax.grid(linestyle='--', 
        linewidth='0.9', 
        color='lightgray', 
        alpha=0.9,
        axis='y'
       )

plt.ylabel("Avg YDS gained per Play\n", rotation=90)
sns.despine(top=True, right=True, left=True, bottom=True)

# Adding transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
plt.tight_layout()
plt.show();


# * There is a relatively large difference in the yards gained based on offensive scheme
# * **Wildcat** - doesn't appear to be that effective.  But it should be noted that it is not run very often in the NFL.  But when it is, it performs pretty poorly.
# * **Shotgun** - performs surprisingly low compared to the other offensive schemes.  One could argue it is a kinda pass play, but more offense* **Empty** - is the clear winner. 'Empty' simply means there is no back in the backfield.  All five eligible receivers are assembled at the line of scrimmage in some fashion.
# * The **I-formation** is one of the more tried and true offensive formations seen in football, and you will likely see it used in short-yardage running siturations.  The I-formation places the runningback 6 - 8 yards behind the line of scrimmage with the quarterback under the center and a fullback splitting them in a three-point stance;  which also means that it is highly likely the defense can see where the runningback is going, but then again, he will probably have a fair amount of speed by the time he hits the line of scrimmage.

# # *Offense Personnel / Defense Personnel*
# Lets have a look at what the top personnel groupings are for the offense and defense

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(20, 6))
train.groupby('PlayId') \
    .first() \
    .groupby('OffensePersonnel') \
    .count()['GameId'] \
    .sort_values(ascending=False) \
    .head(20) \
    .sort_values() \
    .plot(kind='barh',
         title='Offense Personnel # of Plays',
         ax=ax[0])
train.groupby('PlayId') \
    .first() \
    .groupby('DefensePersonnel') \
    .count()['GameId'] \
    .sort_values(ascending=False) \
    .head(20) \
    .sort_values() \
    .plot(kind='barh',
         title='Defense Personnel # of Plays',
         ax=ax[1])
plt.show();


# In[ ]:





# ### *Player Physics*
# *Towards a deeper understanding of the game of football in relation to Newtonian Physics*
# 
# We will focus on the 2018 Season exclusively for this analysis.
# 
# ### *Weight Distribution:*
# * Let's dive into examining player weight information

# In[ ]:


aaa = gold
aaa['IsRunner'] = aaa.NflId == aaa.NflIdRusher
bbb = aaa[aaa.IsRunner & (aaa.Season == 2018)]

fig, ax = plt.subplots(figsize=(10,6))
ax.set_xlim(150,380)
ax.set_title('2018 Season: Player Weight distribution (Runners vs Non-Runners)\n\n', fontsize=14)

sns.kdeplot(bbb.PlayerWeight, shade=True, color="darkorange", ax=ax)
sns.kdeplot(aaa[~aaa.IsRunner & (aaa.Season == 2018)].PlayerWeight, shade=True, color='skyblue', ax=ax)

ax.xaxis.set_major_locator(plt.MultipleLocator(10))
ax.xaxis.set_minor_locator(plt.MultipleLocator(5))

# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))

sns.despine(top=True, right=True, left=True, bottom=True)

# Turn off tick labels
ax.set_yticklabels([])
#ax.set_xticklabels([])

ax.set(xlabel="\nPlayer Weight\n\n")
plt.legend(title="Category:  Ran the ball, or didn't run the ball", loc='upper right', labels=['Runners', 'Non-Runners'])
plt.tight_layout()
plt.show();


# ## *Inferences:*
# * We are gathering up the total number of running plays of all of the combined teams over the entire 2018 seasons, and creating a distribution of the weight of the runner who made the play (in orange), and also during those SAME plays, gathering up the weight distribution of those who did NOT run the ball.  I believe this gives us a very good idea of the weights of the players that were **on** the field during the season (broken out by rushing player versus non-rushing player), and starts to paint a picture of being able to predict the expected yards gained during running plays. 
#   * I care about **who** is on the playing field here, that is the key for future prediction models. 
#   * As long as the weights of the players are updated throughout the season, this also is an extremely granular way of determining kinetic energy on the field as well.
#   * The real point is this - if they aren't on the field, or aren't on the field much, do I really care what their weight is when i figure out my model ? 
# * Thus, of those that ran the ball in the 2018 NFL season, they had an average weight of **217 lbs**, and a median weight of 220 lbs. 
# * Non-Runners had a pretty wide distribution, obviously depends on position they played...
#   * There is a pronounced peak at 310lbs, which is our linemen...

# ### *Player Profile - Mass:*
# * Let's take a look at the weight distribution now for every player in the 2018 NFL season. This time we will just examine all players who played in the season and were on the roster, to get a ballpark on some differences in weight vs position:

# In[ ]:


player_profile=aaa.loc[:,['DisplayName','Position','NflId' 'PlayerBirthDate', 'PlayerWeight', 'PlayerCollegeName']].drop_duplicates()
player_profile_2018=aaa[aaa.Season==2018]
player_profile_2018 = player_profile_2018.loc[: ,['DisplayName','Position','NflId' 'PlayerBirthDate', 'PlayerWeight', 'PlayerCollegeName'] ].drop_duplicates()

player_profile_2018["kg"] = player_profile_2018["PlayerWeight"] * 0.45359237

z = player_profile_2018.groupby('Position')['PlayerWeight'].agg(['min', 'median', 'mean', 'max']).round(1).sort_values(by=['median'], 
                                                                                                                   ascending=False)
z['Avg Mass (kg)'] = (z['mean'] * 0.45359237).round(1)
z


# * We can see that all of the linemen are just over 300lbs. And they make up a LARGE distribution of the players on the field, i.e. there are some BIG BOYS on that field.  
# * It is suprising that FB (fullbacks) are as heavy as they are.  I would imagine one could argue that two things are pretty critical to determining the performance of a running back: 
#   * How big are the offensive linemen ???  (ideally we knew how strong they were as well, but no information contained about that)
#   * How big is the fullback ?  A fullback with some size would really help blocking for the running back and I believe would be directly proportional to the success of the runningback.  
#   * Look at how large the OTs (offensive tackles) are.  One would imagine a run off the OT being a smart play, IF the defensive linebacker at that area was smaller as well...

# ### *Collisions:*
# * Every runningback should PREFER inelastic collisions ! 
# 
# * Why ? **Because it creates the separation that is their advantage** 
# 
# * How ? Ideally with an alpha less than 45 degrees - this way they bounce off and keep going somewhat in the x-axis direction, but ideally it is not an inelastic collision with alpha of 0 or near it, that is going to completely stop runningback momentum 
# 
# * Effectively a runningback wants to maneuver, and when maeuvering is no longer much of an option, to 'bounce' off the tackler
# 
# 
# ### *Why is it so important to analyze the governing dynamics ?*
#   * Check out this *2016 NFL Combine Report* on Dallas Cowboy runningback Ezekiel Elliott
#   * Source: `http://www.nfl.com/player/ezekielelliott/2555224/combine`
# <br>
# <img src="https://github.com/tombresee/Temp/raw/master/ENTER/elliott.png" width="800px">
# * i.e. the key to running effectively I believe is contained within the specific verbage above, these are some key factors for producing actual yards... 
# 

# ### *Stopping a truck:*
# * Converting all the runner's weight values to kilograms
# * THEN we will convert the runner's weight (which is now in kg) to MASS, which is NOT the same thing as weight. 
#   * Technically a body's mass is its weight (in kg) divided by the gravitational constant $g$, which is 9.8 m/s^2, where that $g$ is really the `standard acceleration of gravity`

# In[ ]:


from scipy import constants
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12,12))

#Specifically only using runners like RB, HB, and WR...
ttt = bbb[bbb.Position.isin(['RB','WR','HB'])]

ttt['kg']=ttt["PlayerWeight"] * 0.45359237 / scipy.constants.g

# the momentum is just mass (in kg) X speed in m/s (so convert from yards/sec to mps)
ttt['True Momentum']=ttt['kg'] * ttt['S'] * 0.9144 
tips = ttt[['True Momentum', 'Yards']]

sns.scatterplot(x="True Momentum", y="Yards", data=tips, s=4, ax=ax, color='lime', markers='o', edgecolors='lime')

plt.title('Correlation between Yards Gained and Player Momentum\n',fontsize=14)
plt.suptitle('Kinetic Momentum',fontsize=10, x=0, y=1,ha="left")

ax.set(xlabel="Player Kinetic Momentum $\Rightarrow$\n")
ax.set(ylabel="Yards Gained  $\Rightarrow$\n")

sns.despine(top=True, right=True, left=True, bottom=True)
plt.xticks([])
plt.tight_layout()
plt.show();


# In[ ]:


plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12,12))

#Specifically only using runners like RB, HB, and WR...
ttt = bbb[bbb.Position.isin(['RB','WR','HB'])]

ttt['kg']=ttt["PlayerWeight"] * 0.45359237 / scipy.constants.g

# the momentum is just mass (in kg) X speed in m/s (so convert from yards/sec to mps)
ttt['True Momentum']=ttt['kg'] * ttt['S'] * 0.9144 
tips = ttt[['True Momentum', 'Yards']]

sns.scatterplot(x="True Momentum", y="Yards", data=tips, s=4, ax=ax, color='fuchsia', markers='o', edgecolors='fuchsia')

plt.title('Correlation between Yards Gained beyond 6 and Player Momentum\n',fontsize=14)
plt.suptitle('Kinetic Momentum',fontsize=10, x=0, y=1,ha="left")

ax.set(xlabel="Player Kinetic Momentum $\Rightarrow$\n")
ax.set(ylabel="Yards Gained  $\Rightarrow$\n")

sns.despine(top=True, right=True, left=True, bottom=True)
ax.set_ylim(6,100)

plt.xticks([])

plt.tight_layout()
plt.show();


# > **INSIGHT**: &nbsp; There appears to be a moderate connection between high momentum and yards gained, i.e. beyond a certain momentum threshold, a marked increase in the number of yards beyond 6 is seen...

# In[ ]:


plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12,12))

ttt = bbb[bbb.Position.isin(['RB','WR','HB'])]


# true mass in kg 
ttt['kg']=ttt["PlayerWeight"] * 0.45359237 / scipy.constants.g

# F = ma 
ttt['Force_Newtons']=ttt['kg'] * ttt['A'] * 0.9144
tips = ttt[['Force_Newtons', 'Yards']]

sns.scatterplot(x="Force_Newtons", y="Yards", data=tips, s=4, ax=ax, color='orange', markers='o', edgecolor='orange')
#sns.lmplot(x="Force_Newtons", y="Yards", data=tips)

plt.title('\nCorrelation between Yards Gained and Player Kinetic Force\n',fontsize=14)
plt.suptitle('Kinetic Force',fontsize=10, x=0, y=1,ha="left")
##plt.text(x=4.7, y=14.7, s='Sepal Length vs Width', fontsize=10, weight='bold')

ax.set(xlabel="Player Kinetic Force$\Rightarrow$\n")
ax.set(ylabel="Yards Gained $\Rightarrow$\n")
sns.despine(top=True, right=True, left=True, bottom=True)

plt.xticks([])

plt.tight_layout()
plt.show();


# In[ ]:




