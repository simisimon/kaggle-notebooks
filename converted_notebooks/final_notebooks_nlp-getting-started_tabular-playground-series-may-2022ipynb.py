#!/usr/bin/env python
# coding: utf-8

# ## Libraries Imported

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_theme(style="darkgrid")
import plotly.express as px

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings("ignore")


# ## Loading & Analyzing Data

# In[ ]:


pd.set_option("display.max_colwidth", 200)

train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv').drop('id', axis=1)
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv').drop('id', axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print('Shape of Train data:', train.shape)
print('Shape of Test data:', test.shape)


# In[ ]:


print('Null Values in Train data:', train.isnull().values.any())
print('Null Values in Test data:', test.isnull().values.any())


# In[ ]:


print('Duplicate Values in Train data:', train.duplicated().sum())
print('Duplicate Values in Test data:', test.duplicated().sum())


# In[ ]:


print('No. of Unique elements in Train data:')
print(train.nunique())
print('------------------------------------')
print('No. of Unique elements in Test data:')
print(test.nunique())


# In[ ]:


print('Checking the type of our data:')
train.dtypes


# In[ ]:


train['target'].value_counts(normalize=True)


# ## Data Visualization

# In[ ]:


px.pie(train,names='target',title='Target Distribution',hole=0.2)


# In[ ]:


fig, ax = plt.subplots(4,4, figsize = (30,25) , sharey= True)
ax = ax.ravel()

for i,col in enumerate(train.dtypes[train.dtypes =="float64"].index):
    train[col].plot(ax = ax[i], kind = "hist", bins = 100, color = "r")
    ax[i].set_title(f"{col}")
fig.suptitle("Histogram of Continous columns", fontsize=35)
plt.tight_layout()
plt.show()


# In[ ]:


fig, ax = plt.subplots(7,2, figsize = (20,15))
ax = ax.ravel()

for i,col in enumerate(train.dtypes[(train.dtypes =="int64") & (train.dtypes.index != "target") ].index):
    train[col].value_counts().plot(ax = ax[i], kind = "bar",color = "r")
    ax[i].set_title(f"{col}")
fig.suptitle("Histogram of Categorical Columns", fontsize=23)
plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(30, 2))
sns.heatmap(train.corr()[-1:],cmap="viridis",annot=True)

plt.title('Correlation with Target Feature')
plt.show()


# In[ ]:


test['target'] = -1
test.head()


# In[ ]:


df = pd.concat([train,test])
df.head()


# In[ ]:


df.nunique()


# ## Feature Engineering

# In[ ]:


df['f_27_engineered']=df['f_27'].apply(lambda x: len(set(x)))
df.head()


# ## Label Encoding

# In[ ]:


col_to_encode = ['f_07','f_08','f_09','f_10','f_11','f_12','f_13','f_14','f_15','f_16','f_17','f_18','f_27_engineered','f_29','f_30']

for col in col_to_encode:
    le = LabelEncoder()
    
    le.fit(df[col])
    
    df.loc[:, col] = le.transform(df[col])
    
df.head()


# In[ ]:


train = df.query("target != -1").reset_index(drop=True)
test = df.query("target == -1").reset_index(drop=True)


# In[ ]:


train.head()


# In[ ]:


test = test.drop(['f_27','target'], axis=1)

test.head()


# In[ ]:


train['target'].value_counts(normalize=True)


# ## Train-Test Split

# In[ ]:


X = train.drop(['f_27','target'], axis=1)
y = train['target']

X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state=42,test_size=0.2)


# In[ ]:


print(X_train.shape,y_train.shape)
print(X_valid.shape,y_valid.shape)


# ## XGBoost

# In[ ]:


get_ipython().run_cell_magic('time', '', "xgb = XGBClassifier(n_estimators=5000,tree_method='gpu_hist',objective='binary:logistic',eval_metric='auc',random_state=42)\nxgb.fit(X_train,y_train)\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "xgb_pred = xgb.predict_proba(X_valid)[:, 1]\nprint('XGBoost Model AUC :', roc_auc_score(y_valid,xgb_pred))\n")


# In[ ]:


fpr, tpr, _ = roc_curve(y_valid,xgb_pred)

plt.plot(fpr,tpr)
plt.title('ROC Curve for XGB Model')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## LightGBM

# In[ ]:


get_ipython().run_cell_magic('time', '', 'lgbm = lgb.LGBMClassifier(objective= \'binary\',\n                          metric= "auc",\n                          n_estimators = 5000,\n                          num_threads= -1,\n                          learning_rate= 0.18319492258552644,\n                          boosting=\'gbdt\',\n                          lambda_l1=0.00028648667113792726,\n                          lambda_l2=0.00026863027834978876,\n                          num_leaves=229,\n                          max_depth= 0,\n                          min_child_samples=80,\n                          device=\'gpu\',\n                          random_state=42\n                         )\nlgbm.fit(X_train, y_train, eval_set=[(X_valid,y_valid)],callbacks=[lgb.early_stopping(30)],eval_metric="auc")\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', "lgbm_pred = lgbm.predict_proba(X_valid)[:, 1]\nprint('LightGBM Model AUC :', roc_auc_score(y_valid,lgbm_pred))\n")


# In[ ]:


fpr, tpr, _ = roc_curve(y_valid,lgbm_pred)

plt.plot(fpr,tpr)
plt.title('ROC Curve for LGBM Model')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Cross-Validation

# In[ ]:


get_ipython().run_cell_magic('time', '', '# initialize kfold column to -1\ntrain[\'kfold\'] = -1\n\n# fetch labels\ny = train[\'target\']\n\n# initialize the KFold class from model_selection module\n# n_splits = number of folds = 5, don\'t forget to initialize random_state\nskf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)\n\n# fill the new kfold column\nfor f, (t_, v_) in enumerate(skf.split(X=train, y=y)):\n    train.loc[v_, \'kfold\'] = f\n    \n# save the new folds file\ntrain.to_csv("train_folds.csv", index=False)\n')


# In[ ]:


train['kfold'].value_counts()


# In[ ]:


for i in range(5):
    print(f"Fold: {i}")
    print(train[train['kfold'] == i].target.value_counts(normalize=True))
    print()


# In[ ]:


get_ipython().run_cell_magic('time', '', '# iterate over each fold\nscores = []\ntest_preds = []\n\nfor fold in tqdm(range(5)):\n\n    print("Getting df_train and df_valid")\n    df_train = train.query("kfold != @fold").reset_index(drop=True)\n    df_valid = train.query("kfold == @fold").reset_index(drop=True)\n\n    lgbm = lgb.LGBMClassifier(objective= \'binary\',\n                              metric= "auc",\n                              n_estimators = 5000,\n                              num_threads= -1,\n                              learning_rate= 0.18319492258552644,\n                              boosting=\'gbdt\',\n                              lambda_l1=0.00028648667113792726,\n                              lambda_l2=0.00026863027834978876,\n                              num_leaves=229,\n                              max_depth= 0,\n                              min_child_samples=80,\n                              device=\'gpu\',\n                              random_state=42\n                            )\n\n    print("Splitting into X_train and X_valid")\n    Xtrain = df_train.drop([\'f_27\',\'target\', \'kfold\'], axis=1)\n    Xvalid = df_valid.drop([\'f_27\',\'target\', \'kfold\'], axis=1)\n\n    ytrain = df_train[\'target\']\n    yvalid = df_valid[\'target\']\n\n    print("Fitting model")\n    lgbm.fit(Xtrain,ytrain,eval_set=[(Xvalid,yvalid)],callbacks=[lgb.early_stopping(30)],eval_metric="auc")\n\n    print("Getting predictions")\n    # we need probabilities of class 1 to calculate roc_auc_score\n    y_preds = lgbm.predict_proba(Xvalid)[:, 1]\n\n    test_pred = lgbm.predict_proba(test)[:, 1]\n\n    test_preds.append(test_pred)\n\n    auc = roc_auc_score(yvalid, y_preds)\n    \n    print("*"*50)\n    print(f"Fold = {fold}, AUC = {auc}")\n    print("*"*50)\n\n    scores.append(auc)\n\nprint(f"CV average: {np.mean(scores)}")\n')


# In[ ]:


fpr, tpr, _ = roc_curve(yvalid,y_preds)

plt.plot(fpr,tpr)
plt.title('ROC Curve for Cross Validated LGBM Model')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


pred = np.mean(np.array(test_preds).T, axis=1)


# ## Submission File

# In[ ]:


df_submit = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv', index_col=0)
df_submit['target'] = pred
df_submit.to_csv('submission.csv',index=True)

