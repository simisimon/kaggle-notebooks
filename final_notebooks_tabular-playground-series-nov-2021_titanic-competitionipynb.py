#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#Data analysis
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visualization
import matplotlib.pyplot as plt
plt.style.use('bmh')
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#Machine Learning
import sklearn
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df_train_copy = pd.read_csv('/kaggle/input/titanic/train.csv')
train_df = df_train_copy
train_df.head()


# In[ ]:


df_test_copy = pd.read_csv('/kaggle/input/titanic/test.csv')
test_df = df_test_copy
test_df.head()


# In[ ]:


print(f'Train data rows: {train_df.shape[0]}, train data columns: {train_df.shape[1]}')


# In[ ]:


print(f'Test data rows: {test_df.shape[0]}, test data columns: {test_df.shape[1]}')


# ### EDA

# __Checking the null and N/A values: features with null values - Age, Cabin, Embarked__
# 

# In[ ]:


#Null values
train_df.isnull().sum()


# In[ ]:


#N?A values
train_df.isna().sum()


# In[ ]:


#Percents of null values: 77% of feature Cabin are missing, this feature might be dropped later.

perc_age = round(len(train_df[train_df["Age"].isnull()])/len(train_df)*100,2)
perc_cabin = round(len(train_df[train_df["Cabin"].isnull()])/len(train_df)*100,2)
perc_Embarked = round(len(train_df[train_df["Embarked"].isnull()])/len(train_df)*100,2)

print(f"Percents of null values Column Age: {perc_age}%")
print(f"Percents of null values Column Cabin: {perc_cabin}%")
print(f"Percents of null values Column Embarked: {perc_Embarked}%")


# __Data Types__

# In[ ]:


#Train data types: Feature Name, Sex, Cabin and Embarked are object; Age is float
train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


#Some statistic insight
train_df.describe()


# - The mean Age is 29 years old
# - the maximum Age is 80 years old
# 

# In[ ]:


#Correlation Between Numerical feature and Target
train_df.corr()


# - Negative correlation between Pclass, Age, SibSp, Parch and Target, its means that, while those feature are increasing, the possibility to be survived are decreasing.
# - Positive correlation between feature Fare and Survived

# In[ ]:


#Categorical Feature
train_df.describe(include=['O'])


# - There are lots of unique values of tickets
# - The valid numbers of cabin is not enough to keep on the traning

# In[ ]:


#Checking the balance class
train_df['Survived'].value_counts()


# In[ ]:


print(f"Class Survived percents: \n{train_df['Survived'].value_counts(normalize=True)}")


# In[ ]:


#Class imbalanced
plt.figure(figsize=(8,5))
sns.countplot(x='Survived', data=train_df, palette='Blues')
plt.show()


# In[ ]:


# Analysis categorical features x Target
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).sum()


# In[ ]:


plt.figure(figsize=(6,4))
grid = sns.FacetGrid(train_df, row='Embarked', col='Sex', aspect=1.6)
grid.map(sns.countplot, 'Survived', alpha=.5)
grid.add_legend()


# In[ ]:


train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).sum()


# - The most people who survived were female and used Embark S and C, sth feature embark and Sex correlates with survival
# - The features Name, ticket and Cabin will be dropped later because they will not have a good contribuition to the model

# In[ ]:


#Feature Fare has lots of diferents numbers, so its wil be transformed to ordinal feature
plt.figure(figsize=(5,3))
data = train_df[['Fare']]
sns.displot(data)
plt.tight_layout()


# In[ ]:


train_df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).sum()


# In[ ]:


plt.figure(figsize=(6,4))
grid = sns.FacetGrid(train_df, row='Pclass', aspect=1.6)
grid.map(sns.countplot, 'Survived', alpha=.5)
grid.add_legend()


# - Feature Pclass correlates with survival, the class with the most survived people was class 1

# In[ ]:


#number of children and parents
train_df[["Parch","Survived"]].groupby(['Parch'], as_index=False).sum()


# In[ ]:


#Number of siblings ans spouses
train_df[["SibSp","Survived"]].groupby(['SibSp'], as_index=False).sum()


# ### Decision
# 
# - Drop features PassengerId, Name, Cabin and Ticket
# - Feature to encode: Embarked(fix to null values - with mode), Sex
# - Age: fix null values,
# - Fare, SibSp and Parch: transform to ordinal feature,

# ### Pre Processing

# In[ ]:


#Drop feature PassengerId, name, cabin and ticket, those features won't be in the model
def drop_features(df):
    features_to_drop = ['PassengerId','Name', 'Cabin', 'Ticket']
    df = df.drop(features_to_drop, inplace=True, axis=1)
    return df


# In[ ]:


drop_features(train_df)
drop_features(test_df)

print('Shape df train: ', train_df.shape)
print('Shape df test: ', test_df.shape)


# In[ ]:


#Fill null values and encode features embraked and Pclass
print('Null values Embarked: ', train_df['Embarked'].isnull().sum())


# In[ ]:


#the null values are survived example, then I decided use in the model and fill with mode
train_df.loc[train_df['Embarked'].isnull()]


# In[ ]:


embarked_mode = train_df.Embarked.dropna().mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(embarked_mode)


# In[ ]:


np.unique(train_df['Embarked'], return_counts=True)


# In[ ]:


#encode feature Sex and Embarked
def encode_features(df):
    dict_embarked = {'S': 0,
                     'C': 1,
                     'Q': 2}
    dict_sex = {'male': 0,
                'female': 1}
    df["Embarked"] = df["Embarked"].map(dict_embarked)
    df["Sex"] = df["Sex"].map(dict_sex)
    return df


# In[ ]:


train_df = encode_features(train_df)
test_df = encode_features(test_df)


# In[ ]:


def fillna_age(df1, df2):
    dataset = pd.concat([df1['Age'], df2['Age']], axis=0, ignore_index=True)
    mean_age = dataset.mean().astype(int)
    
    df1['Age'] = df1['Age'].fillna(mean_age).astype(int)
    df2['Age'] = df2['Age'].fillna(mean_age).astype(int)
    return df1, df2
    


# In[ ]:


df_train, df_test = fillna_age(train_df, test_df)


# In[ ]:


df_test.info()


# In[ ]:


fare_min = df_test['Fare'].min()
df_test['Fare'] = df_test['Fare'].fillna(fare_min)


# In[ ]:


df_test.info()


# In[ ]:


np.unique(df_train['SibSp'], return_counts=True)


# In[ ]:


np.unique(df_train['Parch'], return_counts=True)


# In[ ]:


#transform to ordinal feature
def transform_to_ordinal(df):
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    
    df.loc[df['SibSp'] == 0, 'SibSp'] = 0
    df.loc[(df['SibSp'] >= 1) & (df['SibSp'] <= 2), 'SibSp'] = 1
    df.loc[df['SibSp'] >= 3, 'SibSp'] = 2
    
    df.loc[df['Parch'] == 0, 'Parch'] = 0
    df.loc[(df['Parch'] >= 1) & (df['Parch'] <= 2), 'Parch'] = 1
    df.loc[df['Parch'] >= 3, 'Parch'] = 2
    
    return df


# In[ ]:


df_train = transform_to_ordinal(df_train)
df_test = transform_to_ordinal(df_test)


# In[ ]:


df_train.head()


# In[ ]:


Y_train = df_train['Survived'].values
X_train = df_train.drop('Survived', axis=1).values


# In[ ]:


X_train.shape, Y_train.shape


# In[ ]:


X_test = df_test.values
X_test.shape


# ### Models
# 

# ### XGBoost Model

# In[ ]:


model_xgboost = XGBClassifier(objective='multi:softmax',
                              num_class=2,
                              n_estimators = 600,
                              max_depth = 10, 
                              learning_rate = 0.05,
                              random_state=42)


# In[ ]:


model_xgboost.fit(X_train, Y_train)


# In[ ]:


y_predict = model_xgboost.predict(X_test)


# In[ ]:


#Model accuracy with train data
score = model_xgboost.score(X_train, Y_train)
score_xgboost = round(score * 100, 2)
print('Score train data XGBoost: %.2f%%' % score_xgboost)


# ### Feature importance to XGBoost Model

# In[ ]:


from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel

print(model_xgboost.feature_importances_)
# In[ ]:


# plot
plot_importance(model_xgboost)
plt.show()


# ### Using SelectFromModel to select features

# In[ ]:


thresholds = np.sort(model_xgboost.feature_importances_)
for thresh in thresholds:
    selection = SelectFromModel(model_xgboost, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, Y_train)
    
    select_X_test = selection.transform(X_test)
    
    score = selection_model.score(select_X_train, Y_train)
    score_selection_model = round(score * 100, 2)
    
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], score_selection_model))
    


# ### Its means that the best performance is using all features

# ### Decision Tree Model
# 

# In[ ]:


model_decision = DecisionTreeClassifier(random_state=42, max_depth=3)
model_decision.fit(X_train, Y_train)
Y_predict_decision = model_decision.predict(X_test)


# In[ ]:


score_decision = model_decision.score(X_train, Y_train)
score_decision = round(score_decision * 100, 2)
print('Score train data Decision Tree Classifier: %.2f%%' % score_decision)


# 
# ### Random Forest Classifier

# In[ ]:


model_random = RandomForestClassifier(random_state=42, max_depth=3)
model_random.fit(X_train, Y_train)
Y_predict_random = model_random.predict(X_test)


# In[ ]:


score_random = model_random.score(X_train, Y_train)
score_random = round(score_random * 100, 2)
print('Score train data Random Forest Classifier: %.2f%%' % score_random)


# ### LigthGBM Model

# In[ ]:


model_lgbm = LGBMClassifier(is_unbalance=True, random_state=42)
model_lgbm.fit(X_train, Y_train)
predict_lgbm = model_lgbm.predict(X_test)


# In[ ]:


score_lgbm = model_lgbm.score(X_train, Y_train)
score_lgbm = round(score_lgbm * 100, 2)
print('Score train data LGBM Classifier: %.2f%%' % score_lgbm)


# ### Cross Validation

# In[ ]:


class CFG:
    debug = False
    n_folds = 15
    seeds = [85, 12]


# In[ ]:


def get_preds_etr(X = X_train, y = Y_train,
                  nfolds = CFG.n_folds,
                  n_estimators = 600):
    
    seeds_avg = list()
    etr_models = []

    for seed in CFG.seeds:
        scores = list()
        kfold = KFold(n_splits = CFG.n_folds, shuffle=True, random_state = seed)
        
        for k, (train_idx,valid_idx) in enumerate(kfold.split(X, y)):
            model_xgb = XGBClassifier(objective='multi:softmax',
                                       n_estimators = n_estimators,
                                       num_class=2,
                                       random_state=seed)
            x_train, y_train = X[train_idx], y[train_idx]
            x_valid, y_valid = X[valid_idx], y[valid_idx]
            prediction = np.zeros((x_valid.shape[0]))
            model_xgb.fit(x_train, y_train) # train
            etr_models.append(model_xgb)
            prediction = model_xgb.predict(x_valid) # predict

            score = round(accuracy_score(prediction, y_valid), 4)
            print(f'Seed {seed} | Fold {k} | Accuracy score: {score}')
            scores.append(score)
            
        print(f"\nMean Accuracy for seed {seed} : {round(np.mean(scores), 4)}\n")
        seeds_avg.append(round(np.mean(scores), 4)) 
    
    print(f"Average score: {round(np.mean(seeds_avg), 4)}")
    return round(np.mean(seeds_avg), 4), etr_models, y_valid, np.array(prediction), x_train, y_train # for the last fold
        
   


# In[ ]:


score_etr, etr_models, y_valid, y_pred, x_train, y_train = get_preds_etr()


# ### Training with the best parameter after cross validation

# In[ ]:


etr_models[-1].get_params


# In[ ]:


model_cros_validation = XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
              importance_type=None, interaction_constraints='',
              learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
              monotone_constraints='()', n_estimators=600,
              n_jobs=0, num_class=2, num_parallel_tree=1,
              objective='multi:softmax', predictor='auto', random_state=12)


# In[ ]:


model_cros_validation.fit(x_train, y_train)


# In[ ]:


score_cv = model_cros_validation.score(x_train, y_train)
score_cv = round(score_cv * 100, 2)
print('Score train data XGBoost Classifier CV: %.2f%%' % score_cv)


# In[ ]:


predict_cross_validation = model_cros_validation.predict(X_test)
predict_cross_validation


# In[ ]:


submission_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": submission_df["PassengerId"],
        "Survived": predict_cross_validation
    })


# In[ ]:


submission_csv = submission.to_csv('submission.csv', index=False)


# In[ ]:




