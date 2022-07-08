#!/usr/bin/env python
# coding: utf-8

# # Red Wine Quality

# ## Import Packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna

from mlxtend.plotting import plot_learning_curves
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling  import RandomOverSampler
from sklearn.feature_selection import RFE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score ,confusion_matrix


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


os.chdir('/kaggle/input/red-wine-quality-cortez-et-al-2009/')
get_ipython().system('ls')


# In[ ]:


# path
df = pd.read_csv('winequality-red.csv')
print(f' Shape of  Dataset: {df.shape}')


# In[ ]:


df.head()


# # Exploratory Data Analysis
# 

# ## Descriptive Statistics
# 

# In[ ]:


df.describe(include='all')


# In[ ]:


df.columns


# ## Data Visualization
# 

# In[ ]:


cor = df.corr()
top_corr_fea = cor.index
plt.figure(figsize=(10,6))
sns.heatmap(df[top_corr_fea].corr(),annot=True)


# In[ ]:


plt.figure(figsize=(3,3))
sns.pairplot(df,hue='quality')


# ## Disturbtions

# In[ ]:


df['quality'].value_counts()


# In[ ]:


label  = ['quality_5','quality_6','quality_7','quality_4','quality_8','quality_3']
data = df['quality'].value_counts()
colors = sns.color_palette('pastel')[0:5]


# In[ ]:


plt.figure(figsize=(8,8))
plt.pie(data,labels=label,colors=colors,autopct='%.0f%%')
plt.show()


# In[ ]:


# histplot
sns.histplot(y=df['quality'])


# In[ ]:


# countplot
sns.countplot(y=df['quality'])


# In[ ]:


sns.relplot(x="density", y='fixed acidity', hue="quality" ,data=df)


# In[ ]:


sns.histplot(x='alcohol', data=df, kde=True, hue="quality")


# ## Missing value imputation
# 

# In[ ]:


print(df.isnull().sum().sort_values(ascending=False))


# In[ ]:


df.info()


# ## Examine numerical features

# In[ ]:


numeric_features = df.select_dtypes(include=[np.number])
print(numeric_features.columns)


# ## Data preprocessing
# 

# ##  Data Splitting

# In[ ]:


X = df.drop('quality',axis=1)
y = df['quality']

print(f' X_shape: {X.shape} \n y_shape: {y.shape}')


# # Feature Selection Technique

# ## Random Forest Importance

# In[ ]:


rf_clf = RandomForestClassifier(criterion='entropy',n_estimators=20)
rfe_selector = RFE(estimator=rf_clf, n_features_to_select=6, step=10, verbose=5)
rfe_selector.fit(X, y)


# In[ ]:


rfe_support = rfe_selector.get_support()

rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(rfe_feature)
print('*******************************************************************************************')
print(str(len(rfe_feature)), 'selected features')


# In[ ]:


X  = df[rfe_feature]
y = df['quality']

print(f' X_shape: {X.shape} \n y_shape: {y.shape}')


# In[ ]:


y.value_counts()


# ## Over-Sampling to deal with Imbalanced Data

# In[ ]:


rs = RandomOverSampler()
X_resample, y_resample = rs.fit_resample(X,y)

print(f' Shape of X after Oversampling: {X_resample.shape} \n Shape of y after Oversampling: {y_resample.shape}')


# In[ ]:


y_resample.value_counts()


# ## Data Standardization

# In[ ]:


from sklearn.preprocessing import StandardScaler
scl = StandardScaler()
X_rescale = scl.fit_transform(X_resample)


# ## Data Splitting into train and test

# In[ ]:


from sklearn.model_selection import train_test_split
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_rescale,y_resample,test_size=0.20,shuffle=True,random_state=0)
print(f' Shape of X_train: {X_train_res.shape} & Shape of X_test: {X_test_res.shape}')
print(f' Shape of y_train: {y_train_res.shape} & Shape of y_test: {y_test_res.shape}')


# # Model Selection
# 

# ## RandomForest Classifier

# In[ ]:


rf_clf = RandomForestClassifier()

rf_clf.fit(X_train_res, y_train_res)
y_pred_test_res = rf_clf.predict(X_test_res)
y_pred_train_res = rf_clf.predict(X_train_res)


# In[ ]:


print('Accuracy score of Train after Oversampling: {}'.format(accuracy_score(y_train_res,y_pred_train_res)*100))
print('Accuracy score of Test after Oversampling: {}'.format(accuracy_score(y_test_res,y_pred_test_res)*100))


# ## Hyperparameter Tuning using optuna

# In[ ]:


def objective(trial):
    param_grid = dict(
    criterion = trial.suggest_categorical('criterion', ['gini','entropy']),
    bootstrap = trial.suggest_categorical('bootstrap',['True','False']),
    max_depth = trial.suggest_int('max_depth', 1, 10000),
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt','log2']),
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1, 10000),
    n_estimators =  trial.suggest_int('n_estimators', 30, 1000)
    
                     )
    
    rf_clf = RandomForestClassifier(**param_grid)
    scores = cross_val_score(rf_clf, X_train_res, y_train_res, cv=5, scoring='accuracy')
    return scores.mean().round(5)

study = optuna.create_study(direction='maximize')


study.optimize(objective, n_trials=10, show_progress_bar=True)


# In[ ]:


best_params_rf = study.best_trial.params
print('Best parameters:', best_params_rf)
print('Best score: {:.2f}%'.format(study.best_value*100))


# In[ ]:


rf_clf = RandomForestClassifier(**best_params_rf).fit(X_train_res,y_train_res)
y_pred_rf = rf_clf.predict(X_test_res)


# In[ ]:


print('Accuracy score of Test: ', accuracy_score(y_test_res, y_pred_rf)*100)


# In[ ]:


print(confusion_matrix(y_test_res, y_pred_rf))


# In[ ]:


unique, counts = np.unique(y_pred_rf, return_counts=True)

print (np.asarray((unique, counts)))


# In[ ]:


disp = plot_confusion_matrix(rf_clf, X_test_res, y_test_res,
                              display_labels=unique,
                              cmap=plt.cm.Blues)


# In[ ]:


print(classification_report(y_test_res, y_pred_rf))


# ## XGB Classifier
# 

# #### i encoded the quality [3->0,4->1,5->2,6->3,7->4,8->5]

# In[ ]:


y_train_res  = np.unique(y_train_res, return_inverse = True)[1] 
y_train_res


# In[ ]:


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 0, 300)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
    learning_rate = trial.suggest_discrete_uniform('learning_rate', 0.1, 0.3, 0.01)
    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)
    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)

    xgboost_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42, 
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        subsample = subsample,
        learning_rate=learning_rate,
        colsample_bytree = colsample_bytree
    )

    scores = cross_val_score(xgboost_clf, X_train_res, y_train_res, cv=5, scoring='accuracy')
    return scores.mean().round(5)

study = optuna.create_study(direction='maximize')


study.optimize(objective, n_trials=10, show_progress_bar=True)


# In[ ]:


best_params_xgb = study.best_trial.params
print('Best parameters:', best_params_xgb)
print('Best score: {:.2f}%'.format(study.best_value*100))


# In[ ]:


xgb_clf = xgb.XGBClassifier(**best_params_xgb).fit(X_train_res,y_train_res)
y_pred_xgb = xgb_clf.predict(X_test_res)


# In[ ]:


y_test_res_xgb = np.unique(y_test_res, return_inverse = True)[1] # to encode quality


# In[ ]:


print('Accuracy score of Test: ', accuracy_score(y_test_res_xgb, y_pred_xgb)*100)


# In[ ]:


print(classification_report(y_test_res_xgb, y_pred_xgb))


# ## LGBM Classifier

# In[ ]:


def objective(trial):
    param_grid = dict(n_estimators=trial.suggest_int('n_estimators', 20, 1000, 10), 
                      learning_rate=trial.suggest_float('learning_rate', 0, 1), 
                      max_depth=trial.suggest_int('max_depth', 3, 12))
    clf = LGBMClassifier(**param_grid)
    scores = cross_val_score(clf, X_train_res, y_train_res, cv=5, scoring='accuracy')
    return scores.mean().round(5)

study = optuna.create_study(direction='maximize')


study.optimize(objective, n_trials=10, show_progress_bar=True)


# In[ ]:


best_params_lgb = study.best_trial.params
print('Best parameters:', best_params_lgb)
print('Best score: {:.2f}%'.format(study.best_value*100))


# In[ ]:


lgb_clf = LGBMClassifier(**best_params_lgb).fit(X_train_res,y_train_res)
y_pred_lgb = lgb_clf.predict(X_test_res)


# In[ ]:


y_test_res_lgb = np.unique(y_test_res, return_inverse = True)[1] 


# In[ ]:


print('Accuracy score of Test: ', accuracy_score(y_test_res_lgb, y_pred_lgb)*100)


# In[ ]:


print(classification_report(y_test_res_lgb, y_pred_lgb))


# ### Voting Classifier
# 

# In[ ]:


voting_clf = VotingClassifier([('rf',rf_clf),('xgb',xgb_clf),('lgb',lgb_clf)],verbose=True)
voting_clf.fit(X_train_res,y_train_res)
voting_clf_results = cross_val_score(estimator = voting_clf, X = X_train_res, y = y_train_res, cv = 5)
voting_clf_pred = voting_clf.predict(X_test_res)


# In[ ]:


y_test_res_voting = np.unique(y_test_res, return_inverse = True)[1] 


# In[ ]:


print("Validation score after Oversampling: %.5f%% (%.5f%%)" % (voting_clf_results.mean()*100.0, voting_clf_results.std()*100.0))
print('Accuracy score of Test: {}'.format(accuracy_score(y_test_res_voting,voting_clf_pred)*100))


# In[ ]:


print(confusion_matrix(y_test_res_voting, voting_clf_pred))


# In[ ]:


disp = plot_confusion_matrix(voting_clf, X_test_res, y_test_res_voting,
                              display_labels=np.unique(voting_clf_pred, return_counts=False),
                              cmap=plt.cm.Blues)


# In[ ]:


print(classification_report(y_test_res, voting_clf_pred))


# In[ ]:


rf = accuracy_score(y_test_res,y_pred_rf)*100
xgb = accuracy_score(y_test_res_xgb,y_pred_xgb)*100
lgb = accuracy_score(y_test_res_lgb,y_pred_lgb)*100
voting = accuracy_score(y_test_res_voting,voting_clf_pred)*100

scores = [rf,xgb,lgb,voting]
labels = ['Random Forest Classifier', 'XGB Classifier','LGB Classifier', 'Voting Classifier']
accs = {'Accuracy': scores}
df = pd.DataFrame(accs,index=labels)
df


# In[ ]:


ax = sns.heatmap(df,annot=True,fmt="1f")
for t in ax.texts: t.set_text(t.get_text() + " %")


# ####  Upvote it if you like it. Thank you

# In[ ]:




