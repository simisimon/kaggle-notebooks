#!/usr/bin/env python
# coding: utf-8

# * Binary Classification problem based on real life data
# * We have to predict probabilities and the metric is ROC-AUC
# * This is a sample notebook which gives beginner approach to the data and hyperparameter tuning using Optuna
# 
# 
# I'm a beginner in this field too, Please give suggestions to improove the score!!
# 

# In[ ]:


import pandas as pd 
import numpy as np
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv')


# In[ ]:


train.isnull().sum().values.sum()


# In[ ]:


test.isnull().sum().values.sum()


# No null values!!

# In[ ]:


train.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(19):
    le.fit(list(train['cat'+str(i)])+list(test['cat'+str(i)]))
    train['cat'+str(i)] = le.transform(train['cat'+str(i)])
    test['cat'+str(i)] = le.transform(test['cat'+str(i)])


# In[ ]:


X = train.iloc[:,1:-1].values
y = train.iloc[:,-1].values
X_test = test.iloc[:,1:]


# # Baseline Model

# In[ ]:


X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.15,random_state=42)


# In[ ]:


lg = LGBMClassifier()
lg.fit(X_train,y_train)
y_pred_l = lg.predict_proba(X_dev)[:,1]
roc_auc_score(y_dev,y_pred_l)


# # Hyperparameter tuning using Optuna

# In[ ]:


def fun(trial,data=X,target=y):
    
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2,random_state=42)
    param = {
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_uniform('subsample', 0,1),
        'learning_rate': trial.suggest_uniform('learning_rate', 0, 0.1 ),
        'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'min_child_weight' : trial.suggest_loguniform('min_child_weight' , 1e-5 , 1),
        'cat_smooth' : trial.suggest_int('cat_smooth', 1, 100),
        'cat_l2': trial.suggest_int('cat_l2',1,20),
        'metric': 'auc', 
        'random_state': 13,
        'n_estimators': 10000,
        
    }
    model = LGBMClassifier(**param)  
    
    model.fit(train_x,train_y,eval_set=[(test_x,test_y)],early_stopping_rounds=200,verbose=False)
    
    preds = model.predict_proba(test_x)[:,1]
    
    auc = roc_auc_score(test_y, preds)
    
    return auc


# In[ ]:


study = optuna.create_study(direction='maximize')
study.optimize(fun, n_trials=30)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


# In[ ]:


#plot_optimization_histor: shows the scores from all trials as well as the best score so far at each point.
optuna.visualization.plot_optimization_history(study)


# In[ ]:


#plot_parallel_coordinate: interactively visualizes the hyperparameters and scores
optuna.visualization.plot_parallel_coordinate(study)


# In[ ]:


# plot_slice: shows the evolution of the search. You can see where in the hyperparameter space your search
# went and which parts of the space were explored more.
optuna.visualization.plot_slice(study)


# In[ ]:


#Visualize parameter importances.
optuna.visualization.plot_param_importances(study)


# ## Best Parameters found by Optuna

# In[ ]:


best_params = study.best_params
best_params['n_estimators'] = 10000
best_params['cat_feature'] = [i for i in range(19)]
best_params['random_state'] = 13
best_params['metric'] = 'auc'


# ## Make Predictions

# In[ ]:


columns = [col for col in train.columns if col not in ['id','target'] ]


# In[ ]:


preds = np.zeros(X_test.shape[0])
kf = StratifiedKFold(n_splits = 10 , random_state = 13 , shuffle = True)
auc =[]
n=0

for tr_idx, test_idx in kf.split(train[columns], train['target']):
    
    X_tr, X_val = train[columns].iloc[tr_idx], train[columns].iloc[test_idx]
    y_tr, y_val = train['target'].iloc[tr_idx], train['target'].iloc[test_idx]
    
    model = LGBMClassifier(**best_params)
    
    model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],early_stopping_rounds=200,verbose=False)
    
    preds+=model.predict_proba(X_test)[:,1]/kf.n_splits
    auc.append(roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))
    print(n+1,auc[n])
    n+=1


# In[ ]:


np.mean(auc)


# In[ ]:


submission = pd.DataFrame({'id':test['id'],'target':preds})
submission.to_csv('submit.csv',index=False)


# # Please upvote the notebook if it helped in any way! 
# 
# ## Have a nice day :)

# In[ ]:


submission.head()

