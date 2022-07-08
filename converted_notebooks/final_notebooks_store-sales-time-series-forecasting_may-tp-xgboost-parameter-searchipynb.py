#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# In this notebook, I explore using optuna to perform hyperparameter searching. The underlying classification model is XGBoost. 
# 
# Preprocessing is done using a scikit-learn Pipeline, with custom written transformer classes to handle feature f_27. Feature f_27 (which consists of a string of 10 characters) is processed by splitting the string into 10 separate features (each containing a single character), and then one-hot encoding. All other features are left unchanged in preprocessing, I made this choice as the tree models used by XGBoost do not require normalization of numerical features, and can use categorical features directly.

# In[ ]:


import numpy as np 
import pandas as pd 
import random
from xgboost import XGBRegressor
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pickle
import optuna


# In[ ]:


class split_f27_transformer():
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        "create new table with column f_27 split into 10 columns"
        d={}
        for k in range(10):
            colname='f_27p%d'%k
            d[colname]=[x[k] for x in X['f_27']]
        df=pd.DataFrame(d)
        Xc=pd.concat((X,df),axis=1)
        Xc.drop('f_27',axis=1,inplace=True)
        return Xc            
    
class ohe_f27_transformer():
    def __init__(self):
        self.ohe=OneHotEncoder(handle_unknown='ignore',sparse=False)
        self.categorical_cols=list('f_27p%d'%k for k in range(10))
    def fit(self,X,y=None):
        self.ohe.fit(X[self.categorical_cols])
        return self
    def transform(self,X):
        df_cc=pd.DataFrame(self.ohe.transform(X[self.categorical_cols]),columns=self.ohe.get_feature_names_out())
        Xt=pd.concat((X.drop(self.categorical_cols,axis=1),df_cc),axis=1)
        return Xt


# # Loading / preprocessing 
# 
# Here, I load the training and test data, create and fit the preprocessing pipeline, and transform the training and test datasets. My pipeline was designed to keep the data as a pandas DataFrame all the way through, so the preprocessed X_train_c and X_test_c are pandas DataFrames.

# In[ ]:


X=pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
y=X['target']
X_train = X.drop(['id','target'],axis=1)
X_test=pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')

pipe=make_pipeline(split_f27_transformer(),ohe_f27_transformer())
pipe.fit(X_train)
X_train_c=pipe.transform(X_train)
X_test_c=pipe.transform(X_test)


# In[ ]:


# View some of preprocessed training data set, as sanity check
X_train_c.head()


# In[ ]:


# View some of preprocessed test data set, as sanity check
X_test_c.head()


# # training/validation split
# 
# I split the training data set into training/validation data. I also added code to allow selecting a smaller set of the training data, to allow faster training during debugging the code. For the final version, I set Ns to 900000 to select the whole training data set.
# 
# I could potentially have done n-fold cross validation, to assess the performance of set of hyperparameters, however as the training size is pretty large I did not do this, and am using a single train/validation split.

# In[ ]:


# Select training/validation set from Ns size subset of training data
Ns=900000
idx=list(range(X_train_c.shape[0]))
random.shuffle(idx)
X_t,X_v,y_t,y_v=train_test_split(X_train_c.iloc[idx[0:Ns],:],y[idx[0:Ns]],train_size=.8)


# I define here a function to evaluate how good a particular set of Hyperparameters are. This is done by training a XGBoost model on the training data, then computing the AUC metric with that trained model on the validation data.
# 
# I made the choice to have this function take a dictionary of parameters and pass it to the constructor for the XGBoost regression model. The code for this function does not use optuna, this lets me separate cleanly the code that uses optuna from the code that does not. (i.e. if I later wanted to evaluate a particular single hyperparameter set, I could do so without needing to use optuna)

# In[ ]:


def is_gpu_available():
    import subprocess
    try :
        t=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout
    except:
        t=''
    return t.find('CUDA')>0

def create_xgbr_model(params):
    local_params=params
    if is_gpu_available():
        local_params['tree_method']='gpu_hist'
    local_params['objective']='binary:logistic'
    model=XGBRegressor(**local_params)
    return model

# Define function to evaluate how good a particular set of hyperparameters are,
# by constructing and training a model using those hyperparameters, and then
# evaluating the auc score for those predictions (on the validation set)
def xgbr_model_auc_score(params):
    model=create_xgbr_model(params)
    model.fit(X_t,y_t)
    auc_v=roc_auc_score(y_v,model.predict(X_v))
    return auc_v


# # Initialize and run optuna study 
# 
# The objective function contains the code that defines the random distributions that are used to generate the parameter values. Here I am optimizing the hyperpameters 
# 
# * n_estimators
# * learning_rate
# * gamma
# * subsample
# * reg_lambda
# * reg_alpha
# 
# This is not all of the possible XGBoost regression parameters, but from what I have read it covers the most important ones.
# 
# I then run the study with a specified number of trials, and save the study object.

# In[ ]:


def objective(trial):
    params={'n_estimators':trial.suggest_int("n_estimators",50,600),
           'learning_rate':trial.suggest_float("learning_rate",.001,.5,log=True),
            'gamma':trial.suggest_float("gamma",0,1),
            'max_depth':trial.suggest_int('max_depth',5,9),
            'subsample':trial.suggest_float('subsample',.5,1),
            'reg_lambda':trial.suggest_float('reg_lambda',0.01,10,log=True),
            'reg_alpha':trial.suggest_float('reg_alpha',0,10)
           }
    return xgbr_model_auc_score(params)

study=optuna.create_study(direction='maximize')
n_trials=300
tstart=time.time()
r=study.optimize(objective,n_trials=n_trials)
elapsed_time=time.time()-tstart

# save study, in case I wish to look at it later
with open('hyperparameter_study.pkl','wb') as fid:
    pickle.dump(study,fid)
    
print('Training using %d data points, validating using %d data points'%(X_t.shape[0],X_v.shape[0]))
print('Completed %d trials in %f seconds, average %f seconds per trial'%(n_trials,elapsed_time,elapsed_time/n_trials))


# # Report / visualize study results 
# 
# I report the best parameters, and visualize the parameter importances and dependence of the quality on a few pairs of hyperparameters

# In[ ]:


study.best_params


# In[ ]:


optuna.visualization.plot_param_importances(study)


# In[ ]:


optuna.visualization.plot_contour(study,params=['learning_rate','gamma'])


# In[ ]:


optuna.visualization.plot_contour(study,params=['learning_rate','max_depth'])


# In[ ]:


optuna.visualization.plot_contour(study,params=['learning_rate','n_estimators'])


# # Retrain and generate predictions 
# 
# Lastly, I use the best identified parameters to create the submission. I retrain on the entire data set (X_train_c rather than X_t), calculate predictions, and save them.

# In[ ]:


##### train on all training data using best parameters, 
# predict on test data, and write submission
if True:
    model=create_xgbr_model(study.best_params)
    model.fit(X_train_c,y)
    preds=model.predict(X_test_c.drop('id',axis=1))
    out=pd.DataFrame({'id':X_test_c['id'],'target':preds})
    out.set_index('id')
    out.to_csv('submission.csv',index=False)

