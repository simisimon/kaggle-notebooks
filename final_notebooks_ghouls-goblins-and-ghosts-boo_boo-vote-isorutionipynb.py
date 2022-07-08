#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#ライブラリのインポート欄
import pandas as pd;import numpy as np;import matplotlib.pyplot as plt;import seaborn as sns
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings;import seaborn as sns #可視化
warnings.filterwarnings('ignore');import zipfile#サンプルがzipなので展開する
zipfile.ZipFile('/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip').extractall() 
zipfile.ZipFile('/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip').extractall()
#zipfile.ZipFile('/kaggle/input/ghouls-goblins-and-ghosts-boo/sample_submission.csv.zip').extractall()

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
from sklearn.model_selection import train_test_split # クロスバリデーション用（テストとトレ分ける）
from sklearn.model_selection import cross_val_score
from sklearn import metrics       # 精度検証用
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
# ニューラルネットワーク
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import joblib

from sklearn import svm
from sklearn.linear_model import LogisticRegression


# LightGBM#import lightgbm as lgb
import optuna
import optuna.integration.lightgbm as lgb



from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error




#sklearn モデル　沢山
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, ARDRegression, RidgeCV
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression



# In[ ]:


def N_netlog(train, test, target,tartest):
    # paramate
    hidden_layer_sizes=(100,)
    activation = 'relu'
    solver = 'adam'
    batch_size = 'auto'
    alpha = 0.0001
    random_state = 0
    max_iter = 10000
    early_stopping = True
    # 学習
    clf = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        batch_size=batch_size,
        alpha=alpha,
        random_state=random_state,
        max_iter=max_iter,
    #     early_stopping = early_stopping
        )
    clf.fit(train, target)
    SAVE_TRAINED_DATA_PATH = 'train1.learn'
    # 学習結果を出力
    joblib.dump(clf, SAVE_TRAINED_DATA_PATH)
    # 学習済ファイルのロード
    clf1 = joblib.load(SAVE_TRAINED_DATA_PATH)
    # 学習結果の検証
    #predict_y1 = clf1.predict_proba(test)それぞれの回答確率を出す?
    #predict = clf1.predict(test)
    #accs=accuracy_score(train, target)
    #return  predict,accs

#スコア用
    if len(tartest) >1:
        print(tartest)
        pred = clf1.predict(test)# LightGBM推論]
        pred_r = np.round(np.round(pred, decimals=1)) # 最尤と判断したクラスの値にする
        predict = accuracy_score(tartest, pred_r)  # 最尤と判断したクラスの値にする
    # スコアじゃないとき
    if len(tartest) ==1:
        print(test)
        predict_no = clf1.predict(test)# LightGBM推論
        predict = np.round(np.round(predict_no, decimals=1)) # 最尤と判断したクラスの値にする
    return predict





# In[ ]:


def lightgbm(train, test, target,tartest):# データ用意
    X_train, X_test, Y_train, Y_test = train_test_split(train, target, random_state=0) # random_stateはseed値。
    # LightGBMのパラメータ設定
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',#クラスは0,1,2,...と与えられる(数字は関係ない)#評価指標：正答率
        #'num_iterations': 1000,#1000回学習
        'verbose': -1 #学習情報を非表示
    }
    #'metric': 'multi_logress'かえた
    # LightGBMを利用するのに必要なフォーマットに変換
    lgb_train = lgb.Dataset(X_train, Y_train)
    lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
    best_params, history = {}, []

    # LightGBM学習
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_eval],
                    verbose_eval=100,
                    early_stopping_rounds=100
                   )

    best_params = gbm.params
    print("Best params:", best_params)
    params = best_params
    #ここから推理します
    #スコア用
    if len(tartest) >1:
        print(tartest)
        pred = gbm.predict(test, num_iteration=gbm.best_iteration)# LightGBM推論]
        pred_r = np.round(np.round(pred, decimals=1)) # 最尤と判断したクラスの値にする
        predict = accuracy_score(tartest, pred_r)  # 最尤と判断したクラスの値にする
    # スコアじゃないとき
    if len(tartest) ==1:
        print(test)
        predict_no = gbm.predict(test, num_iteration=gbm.best_iteration)# LightGBM推論
        predict = np.round(np.round(predict_no, decimals=1)) # 最尤と判断したクラスの値にする
    return predict


# In[ ]:


def def_two(ghost,ghoul,goblin, test, target, accuracy,best_cell_gob,reg_dict):#おかしい。うまく行っていない
    #HowDoはどの分析方法にするか
    acc_score_gob=np.zeros((40),dtype = 'float64')#スコアを保存
    vsnp=np.empty((529),dtype="float64")
    vsnpp=np.empty((529),dtype="float64")
    submission=np.empty((529),dtype="int")
    vote=np.zeros((529,2),dtype="int")#投票によるスコア
    ones=np.ones((529),dtype="int")#投票によるスコア
    ghost0=np.zeros(len(ghost));ghost1=np.ones(len(ghost))
    ghoul0=np.zeros(len(ghoul));ghoul1=np.ones(len(ghoul))
    goblin0=np.zeros(len(goblin));goblin1=np.ones(len(goblin))#target作成前段階
    vs = ghost.append(ghoul, ignore_index=True)
    vst = np.append(ghost1,ghoul0)#target作成
    #今回はゴーストが1
    #本番かどうか
    if accuracy == True:#スコア出す
        train_r, test_r, target_r, tartest_r = train_test_split(vs, vst, random_state=0) # random_stateはseed値。
        
        model = LogisticRegression();model.fit(train_r,target_r);vsnp=model.predict(test_r);acc_score_gob[0]=accuracy_score(tartest_r, vsnp)#LogReg
        submission = np.round(np.round(vsnp, decimals=1))
        vote[:len(test_r),0]=vote[:len(test_r),0]+submission[:len(test_r)];vote[:len(test_r),1]=vote[:len(test_r),1]+ones[:len(test_r)]-submission[:len(test_r)]
        #acc_score_gob[1]=lightgbm(train_r, test_r, target_r, tartest_r)
        #acc_score_gob[2]=N_netlog(train_r, test_r, target_r, tartest_r)
        #sklearn沢山
        n=0
        for reg_name, reg in reg_dict.items():
            reg.fit(train_r,target_r);vsnp = reg.predict(test_r);submission = np.round(np.round(vsnp, decimals=1))
            acc_score_gob[n+3]=accuracy_score(tartest_r, submission);n+=1
            vote[:len(test_r),0]=vote[:len(test_r),0]+submission[:len(test_r)];vote[:len(test_r),1]=vote[:len(test_r),1]+ones[:len(test_r)]-submission[:len(test_r)]
        for n in range(len(test_r)):
            submission[n]= (0 if vote[n,1]>vote[n,0] else 1)
        #acc_score_gob[39]=accuracy_score(tartest_r, submission)
        print(acc_score_gob)
        return acc_score_gob
    
    if accuracy == False:#本シミュレーション
        train_r, test_r, target_r, tartest_r = vs, test, vst, [0]
        
        if best_cell_gob==0:#LogReg
            model = LogisticRegression();model.fit(train_r,target_r);vsnp=model.predict(test_r);vsnpp=vsnp
        if best_cell_gob==1:#LogReg
            vsnp=lightgbm(train_r, test_r, target_r, tartest_r);vsnpp=vsnp
        if best_cell_gob==2:#LogReg
            vsnp=N_netlog(train_r, test_r, target_r, tartest_r);vsnpp=vsnp
        if best_cell_gob > 2:#many_sk
            n=0#n初期化
            for reg_name, reg in reg_dict.items():
                #if n == best_cell_gob-3:
                reg.fit(train_r,target_r);vsnp = reg.predict(test_r)
                if n == best_cell_gob-3:
                    vsnpp=vsnp
                n+=1#特定の数のときだけfit
                vote[:len(test_r),0]=vote[:len(test_r),0]+submission[:len(test_r)];vote[:len(test_r),1]=vote[:len(test_r),1]+ones[:len(test_r)]-submission[:len(test_r)]
        if best_cell_gob == 39:
            for n in range(len(test_r)):
                vsnp[n]= (0 if vote[n,1]>vote[n,0] else 1)
            vsnpp=vsnp
        
        submission = np.round(np.round(vsnpp, decimals=1)) # 最尤と判断したクラスの値にする
        
        return submission


# In[ ]:


def def_gob(ghost,ghoul,goblin, test, target, accuracy,best_cell_gob,reg_dict):#ゴブリンかどうか最初に判別するために、ゴブリンを一番区別できる分別機を選ぶ
    acc_score_gob=np.zeros((40),dtype = 'float64')#スコアを保存
    vsnp=np.zeros((529),dtype="float64")
    vsnpp=np.zeros((529),dtype="float64")
    submission=np.empty((529),dtype="bool")
    vote=np.zeros((529,2),dtype="int")#投票によるスコア
    ones=np.ones((529),dtype="bool")#投票によるスコア
    ghost0=np.zeros(len(ghost));ghost1=np.ones(len(ghost))
    ghoul0=np.zeros(len(ghoul));ghoul1=np.ones(len(ghoul))
    goblin0=np.zeros(len(goblin));goblin1=np.ones(len(goblin))#target作成前段階
    vs = goblin.append(ghost, ignore_index=True)#train作成
    vs = vs.append(ghoul, ignore_index=True)#train作成
    vst = np.append(goblin1,ghost0)#target作成
    vst = np.append(vst,ghoul0)
    #本番かどうか
    if accuracy == True:#スコア出す
        train_r, test_r, target_r, tartest_r = train_test_split(vs, vst, random_state=0) # random_stateはseed値。
        #vote[:len(test_r),0]=ones[:len(test_r)]*5
        model = LogisticRegression();model.fit(train_r,target_r);vsnp=model.predict(test_r);acc_score_gob[0]=accuracy_score(tartest_r, vsnp)#LogReg
        submission = np.round(np.round(vsnp, decimals=1))
        vote[:len(test_r),0]=vote[:len(test_r),0]+submission[:len(test_r)];vote[:len(test_r),1]=vote[:len(test_r),1]+ones[:len(test_r)]-submission[:len(test_r)]
        
        #acc_score_gob[1]=lightgbm(train_r, test_r, target_r, tartest_r)
        acc_score_gob[2]=N_netlog(train_r, test_r, target_r, tartest_r)
        
        #sklearn沢山
        n=0
        for reg_name, reg in reg_dict.items():
            reg.fit(train_r,target_r)
            vsnp = reg.predict(test_r);submission = np.round(np.round(vsnp, decimals=1))
        
            acc_score_gob[n+3]=accuracy_score(tartest_r, submission);n+=1
            
            vote[:len(test_r),0]=vote[:len(test_r),0]+submission[:len(test_r)];vote[:len(test_r),1]=vote[:len(test_r),1]+ones[:len(test_r)]-submission[:len(test_r)]
        for n in range(len(test_r)):
            submission[n]= (0 if vote[n,1]>vote[n,0] else 1)
            
        #acc_score_gob[39]=accuracy_score(tartest_r, submission)
        print(acc_score_gob)
        return acc_score_gob
    
    if accuracy == False:#本シミュレーション
        train_r, test_r, target_r, tartest_r = vs, test,vst, [0]
        vsnpp_int=np.zeros((529),dtype="int")
        if best_cell_gob==0:#LogReg
            model = LogisticRegression();model.fit(train_r,target_r);vsnp=model.predict(test_r);vsnpp=vsnp
        if best_cell_gob==1:#LogReg
            vsnp=lightgbm(train_r, test_r, target_r, tartest_r);vsnpp=vsnp
        if best_cell_gob==2:#LogReg
            vsnp=N_netlog(train_r, test_r, target_r, tartest_r);vsnpp=vsnp
        if best_cell_gob > 2:#many_sk
            n=0#n初期化
            
            for reg_name, reg in reg_dict.items():
               
                reg.fit(train_r,target_r);vsnp = reg.predict(test_r)
                vsnpp_int = vsnpp_int +  (np.round(np.round(vsnp, decimals=1))==1)
                
                if n == best_cell_gob-3:
                        vsnpp=vsnp
                        
                n+=1#特定の数のときだけfit
                vote[:len(test_r),0]=vote[:len(test_r),0]+submission[:len(test_r)];vote[:len(test_r),1]=vote[:len(test_r),1]+ones[:len(test_r)]-submission[:len(test_r)]
        if best_cell_gob == 39:
            for n in range(len(test_r)):
                vsnp[n]= (0 if vote[n,1]>vote[n,0] else 1)
            vsnpp=vsnp
        submission = np.round(np.round(vsnpp, decimals=1)) # 最尤と判断したクラスの値にする
        
        submission = (submission==1) | (vsnpp_int>2)
        
        
        return submission
    
    
    
    


# In[ ]:





# In[ ]:


def main_n():
    #sklearn沢山用
    reg_dict = {#"LinearRegression": LinearRegression(),
            #"Ridge": Ridge(),
            #"Lasso": Lasso(),
            #"ElasticNet": ElasticNet(), 
   
            #"KNeighborsRegressor": KNeighborsRegressor(n_neighbors=3),
            #"DecisionTreeRegressor": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            #"SVR": SVR(kernel='rbf', C=1e3, gamma=0.1, epsilon=0.1),
           
            #"SGDRegressor": SGDRegressor(),
            #"MLPRegressor": MLPRegressor(hidden_layer_sizes=(10,10), max_iter=100, early_stopping=True, n_iter_no_change=5),
            "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100), 
            
            #"PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=100, tol=1e-3),
            #"TheilSenRegressor": TheilSenRegressor(random_state=0),
            
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(),
            "AdaBoostRegressor": AdaBoostRegressor(random_state=0, n_estimators=100),
            "BaggingRegressor": BaggingRegressor(base_estimator=SVR(), n_estimators=2),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=0),
            "VotingRegressor": VotingRegressor([('lr', LinearRegression()), ('rf', RandomForestRegressor(n_estimators=2))]),
            #"StackingRegressor": StackingRegressor(estimators=[('lr', RidgeCV()), ('svr', LinearSVR())], final_estimator=RandomForestRegressor(n_estimators=10)),
            #"ARDRegression": ARDRegression(),
            #"HuberRegressor": HuberRegressor(),
                    }

    
    
   # CSVを読み込む
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    
    submission_no=np.empty((529,3),dtype="int")
    submission=[""]*529
    
    #type_pd = train["type"]
    type_array = pd.get_dummies(train['type']);  del train['type']#typeをトレインから分離
    COLOR = pd.get_dummies(train['color']);  del train['color']  ;del train['id']#colorをトレインから分離；idをトレインから分離
    COLOR2 = pd.get_dummies(test['color']);  del test['color'];  ID = test["id"];  del test['id'] #testも同じようにする
    vote = np.zeros((3,529),dtype = 'int')
    target = pd.DataFrame(type_array['Ghost']* 0 + type_array['Ghoul'] * 2 + type_array['Goblin'] * 1)#targetを作成する
    #怪物のデータが別々にいるプログラム用
    ghost=train[type_array['Ghost']==1]
    ghoul=train[type_array['Ghoul']==1]
    goblin=train[type_array['Goblin']==1]
    
    #ここからは色つきでもう一度同じ
    train_c = train.join(COLOR)
    test_c = test.join(COLOR2)
    #怪物のデータが別々にいるプログラム用
    ghost_c=train_c[type_array['Ghost']==1]
    ghoul_c=train_c[type_array['Ghoul']==1]
    goblin_c=train_c[type_array['Goblin']==1]
    
    
    #DAOAのためのスコアで分析
    #ゴブリンかどうか自動判別
    best_cell_gob= 0;accuracy=True;gob_c_or_no=True#色付きがいいならTrue出ないならFalse
    acc_score_gob=def_gob(ghost,ghoul,goblin, test, target, accuracy,best_cell_gob,reg_dict)
    best_cell_gob=np.argmax(acc_score_gob)#色なし最高スコア
    print("serect",best_cell_gob)
    best_cell_gob_c= 0#色あり
    acc_score_gob_c=def_gob(ghost_c,ghoul_c,goblin_c, test_c, target, accuracy,best_cell_gob_c,reg_dict)
    best_cell_gob_c=np.argmax(acc_score_gob_c)#追加して
    print("serect",best_cell_gob)
    gob_c_or_no = (True if best_cell_gob_c > best_cell_gob else False)#色付き色なしどちらがいい？
    
    #ゴブリンか判別本番
    accuracy=False
    if gob_c_or_no:
        submission_no[:,0]=def_gob(ghost_c,ghoul_c,goblin_c, test_c, target, accuracy,best_cell_gob_c,reg_dict)
    if gob_c_or_no==False:
        submission_no[:,0]=def_gob(ghost,ghoul,goblin, test, target, accuracy,best_cell_gob,reg_dict)
    #判別終わり
    
    #ID_goblin=np.array(ID[submission_no==1])#ゴブリン該当するIDを取り出し
    #ID_nogob=np.array(ID[submission_no==0])#ゴブリン該当しないIDを取り出し
    test_nogob=np.array(test[submission_no[:,0]==0])#ゴブリン該当しないテストを取り出し
    test_nogob_c=np.array(test_c[submission_no[:,0]==0])#ゴブリン該当しないテストを取り出し色あり
    #submission[submission_no==1]="Goblin"#ゴブリンを事前に入れておく#いらない
    #magicno=nonono(ID_nogob)
    #submission_no_gob=np.zeros((magicno),dtype="int")
        
    #ここから、ghoulとghostの判別
    best_cell_two= 0;accuracy=True;c_or_no=True#色付きがいいならTrue出ないならFalse
    acc_score_two=def_two(ghost,ghoul,goblin, test_nogob, target, accuracy,best_cell_two,reg_dict)
    best_cell_two=np.argmax(acc_score_two)#色なし最高スコア
    best_cell_two_c= 0#色あり
    acc_score_two_c=def_two(ghost_c,ghoul_c,goblin_c, test_nogob_c, target, accuracy,best_cell_two_c,reg_dict)
    best_cell_two_c=np.argmax(acc_score_two_c)#追加して
    c_or_no = (True if best_cell_two_c > best_cell_two else False)#色付き色なしどちらがいい？
    
     #２つの判別本番
    
    
    accuracy=False
   
    if gob_c_or_no:
        submission_no[:,1]=def_two(ghost_c,ghoul_c,goblin_c, test_c, target, accuracy,best_cell_two_c,reg_dict)
    if gob_c_or_no==False:
        submission_no[:,1]=def_two(ghost,ghoul,goblin, test, target, accuracy,best_cell_two,reg_dict)
    
    #ID_ghost=np.array(ID_nogob[submission_no_gob[:len(ID_nogob)]==1])#ghost該当するIDを取り出し
    #ID_ghoul=np.array(ID_nogob[submission_no_gob[:len(ID_nogob)]==0])#ghoul該当IDを取り出し
    #print(ID_ghost)
    #nghost, nghoul, ngoblin = 0,0,0
    for n in range (len(ID)):
        if submission_no[n,0]==1:
            submission[n]="Goblin"
        if submission_no[n,0]==0:
            submission[n]= ("Ghost" if submission_no[n,1]==1 else "Ghoul")
            

        #if ID[n]==ID_ghost[nghost]:
       #     submission[n]="Ghost";nghost =(nghost+1 if len(ID_ghost)>nghost+1 else 0 )
       # if ID[n]==ID_ghoul[nghoul]:
       #     submission[n]="Ghoul";nghoul = (nghoul+1 if len(ID_ghoul)>nghoul+1 else 0 )
       # if ID[n]==ID_goblin[ngoblin]:
       #     submission[n]="Ghoblin";ngoblin = (ngoblin+1 if len(ID_goblin)>ngoblin+1 else 0 )
    
    s_c= pd.DataFrame({"id": ID, "type": submission})
    
    return s_c


# In[ ]:


#ここでメインを一つずつ実行
submission=main_n()

##ここから推理します
# Kaggle提出用csvファイルの作成
submission.to_csv("submission6.csv", index=False)


# ここから使っていないコード
