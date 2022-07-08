#!/usr/bin/env python
# coding: utf-8

# ## Importing Modules 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


os.chdir('/kaggle/input/spaceship-titanic/')


# In[ ]:


get_ipython().system('ls')


# ## Importing Data
# 

# In[ ]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')


# In[ ]:


print(f'Train: {train.shape} \n Test: {test.shape} \n sample_submission: {sample_submission.shape}')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


sample_submission.head()


# # Exploratory Data Analysis

# ## Descriptive Statistics
# 

# In[ ]:


# Train Data
train.describe(include='all')


# In[ ]:


# Test Data
test.describe(include='all')


# ## Data Visualization

# In[ ]:


cor = train.corr()
top_corr_fea = cor.index
plt.figure(figsize=(10,6))
sns.heatmap(train[top_corr_fea].corr(),annot=True)


# ## Missing value imputation
# 

# In[ ]:


# Train Data
print(train.isnull().sum().sort_values(ascending=False))


# In[ ]:


# Test Data
print(test.isnull().sum().sort_values(ascending=False))


# ## Examine numerical features

# In[ ]:


# Train Data
numeric_features = train.select_dtypes(include=[np.number])
print(numeric_features.columns)


# In[ ]:


# Test Data
numeric_features = test.select_dtypes(include=[np.number])
print(numeric_features.columns)


# ## Examine categorical features

# In[ ]:


# Train Data
categorical_features = train.select_dtypes(include=[np.object_])
print(categorical_features.columns)


# In[ ]:


# Test Data
categorical_features = test.select_dtypes(include=[np.object_])
print(categorical_features.columns)


# ## Examine boolean features

# In[ ]:


# Train Data
categorical_features = train.select_dtypes(include=[np.bool_])
print(categorical_features.columns)


# ## Features Distribution Visulaization

# In[ ]:


# Home Planet
home = train['HomePlanet'].unique()
home


# In[ ]:


home_cnt = train['HomePlanet'].value_counts()
home_cnt


# In[ ]:


sns.countplot(x='HomePlanet',data=train)


# In[ ]:


sns.histplot(x='HomePlanet',data=train)


# In[ ]:


# Destination
des = train['Destination'].unique()
des


# In[ ]:


des_cnt = train['Destination'].value_counts()
des_cnt


# In[ ]:


sns.histplot(x='Destination',data=train)


# In[ ]:


# Transported
trans = train['Transported'] == True
no_trans = train['Transported'] == False

print('Transported: ', len(trans)) 
print('Not Transported:', len(no_trans))


# In[ ]:


classes = [len(trans),len(no_trans)]
labels = ['Transported','Not Transported'] 
clss = {'Transported':classes[0], 'Not Transported': classes[1]}
df = pd.DataFrame(clss,index=labels)
df


# In[ ]:


# Age
sns.histplot(x='Age', data=train, kde=True, hue="Transported")


# # Data preprocessing 

# ## Handle Missing Data

# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


# Train:

# Categorical Data
train['HomePlanet'].fillna(train['HomePlanet'].mode()[0],inplace=True)
train['CryoSleep'].fillna(train['CryoSleep'].mode()[0],inplace=True)
train['Cabin'].fillna(train['Cabin'].mode()[0],inplace=True)
train['Destination'].fillna(train['Destination'].mode()[0],inplace=True)
train['VIP'].fillna(train['VIP'].mode()[0],inplace=True)
train['Name'].fillna(train['Name'].mode()[0],inplace=True)

# numerical data
train['Age'].fillna(train['Age'].mean(),inplace=True)
train['RoomService'].fillna(train['RoomService'].mean(),inplace=True)
train['FoodCourt'].fillna(train['FoodCourt'].mean(),inplace=True)
train['ShoppingMall'].fillna(train['ShoppingMall'].mean(),inplace=True)
train['Spa'].fillna(train['Spa'].mean(),inplace=True)
train['VRDeck'].fillna(train['VRDeck'].mean(),inplace=True)


# Test: 

# Categorical Data
test['HomePlanet'].fillna(test['HomePlanet'].mode()[0],inplace=True)
test['CryoSleep'].fillna(test['CryoSleep'].mode()[0],inplace=True)
test['Cabin'].fillna(test['Cabin'].mode()[0],inplace=True)
test['Destination'].fillna(test['Destination'].mode()[0],inplace=True)
test['VIP'].fillna(test['VIP'].mode()[0],inplace=True)
test['Name'].fillna(test['Name'].mode()[0],inplace=True)


# numerical data
test['Age'].fillna(test['Age'].mean(),inplace=True)
test['RoomService'].fillna(test['RoomService'].mean(),inplace=True)
test['FoodCourt'].fillna(test['FoodCourt'].mean(),inplace=True)
test['ShoppingMall'].fillna(test['ShoppingMall'].mean(),inplace=True)
test['Spa'].fillna(test['Spa'].mean(),inplace=True)
test['VRDeck'].fillna(test['VRDeck'].mean(),inplace=True)


# In[ ]:


train.info()


# In[ ]:


test.info()


# ## Remove unnecessary columns
# 

# In[ ]:


# # Train Data
# train.drop('PassengerId',axis=1,inplace=True)

# # Test Data
# test.drop('PassengerId',axis=1,inplace=True)


# ## Handle object Datatype
# 

# In[ ]:


#Label Encoding for object to numeric

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Train Data
objList = train.select_dtypes(include = "object").columns
print(objList)

for fea in objList:
    train[fea] = le.fit_transform(train[fea])
    

# Test Data 
objList = test.select_dtypes(include = "object").columns
print(objList)

for fea in objList:
    test[fea] = le.fit_transform(test[fea])
    


# ## Handle boolean Datatype

# In[ ]:


# Train Data
boolList = train.select_dtypes(include = "bool").columns
print(boolList)

for fea in boolList:
    train[fea]  = pd.Series(np.where(train[fea] == True, 1, 0))
    
# Test Data
boolList = test.select_dtypes(include = "bool").columns
print(boolList)

for fea in boolList:
    test[fea]  = pd.Series(np.where(test[fea] == True, 1, 0))


# In[ ]:


train.info()


# In[ ]:


test.info()


# ## Splitting Data

# In[ ]:


X = train.drop('Transported',axis=1)
y = train['Transported']

print(f' X_shape: {X.shape} \n y_shape: {y.shape}')


# ## Feature Selection Techniques

# ## Univariate feature selection

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# find best scored 5 features
chi_selector = SelectKBest(chi2, k=10).fit(X, y)


# In[ ]:


chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns
print(chi_feature)


# ## Splitting Data after Chooseing best features

# In[ ]:


X = train[chi_feature]
y = train['Transported']

print(f' X_shape: {X.shape} \n y_shape: {y.shape}')


# In[ ]:


y.value_counts()


# ## Over-Sampling to deal with Imbalanced Data

# In[ ]:


from imblearn.over_sampling  import SMOTE
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X,y)

print(f' X_shape: {X_smote.shape} \n y_shape: {y_smote.shape}')


# In[ ]:


y_smote.value_counts()


# ## Data Standardization

# In[ ]:


# Without Oversampling
from sklearn.preprocessing import StandardScaler
scl = StandardScaler()
X = scl.fit_transform(X)


# In[ ]:


# with Oversampling
from sklearn.preprocessing import StandardScaler
scl = StandardScaler()
X_smote = scl.fit_transform(X_smote)


# ## data Splitting into train and test
# 

# In[ ]:


# Without Oversampling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,shuffle=True,random_state=0)
print(f' X_train: {X_train.shape} & X_test: {X_test.shape}')
print(f' y_train: {y_train.shape} & y_test: {y_test.shape}')


# In[ ]:


# With Oversampling
from sklearn.model_selection import train_test_split
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_smote,y_smote,test_size=0.20,shuffle=True,random_state=0)
print(f' X_train: {X_train_res.shape} & X_test: {X_test_res.shape}')
print(f' y_train: {y_train_res.shape} & y_test: {y_test_res.shape}')


# ## Model Selection
# 

# ### RandomForest Classifier
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Without Oversampling

rf_clf = RandomForestClassifier(criterion='entropy',n_estimators=50)

rf_clf.fit(X_train, y_train)
y_pred_test = rf_clf.predict(X_test)
y_pred_train = rf_clf.predict(X_train)


# In[ ]:


# With Oversampling

rf_clf_res = RandomForestClassifier(criterion='entropy',n_estimators=50)

rf_clf_res.fit(X_train_res, y_train_res)
y_pred_test_res = rf_clf_res.predict(X_test_res)
y_pred_train_res = rf_clf_res.predict(X_train_res)


# In[ ]:


print('Accuracy of Train befor Oversampling: {}'.format(accuracy_score(y_train, y_pred_train)*100))
print('Accuracy of  Test befor Oversampling: {}'.format(accuracy_score(y_test,y_pred_test)*100))
print('********************************************************')
print('Accuracy of Train after Oversampling: {}'.format(accuracy_score(y_train_res,y_pred_train_res)*100))
print('Accuracy of Test after Oversampling: {}'.format(accuracy_score(y_test_res,y_pred_test_res)*100))


# ## Uesing RandomizedSearchCV

# In[ ]:


rf_clf.get_params().keys()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# Without Oversampling

parameters= {'n_estimators':list(np.arange(100,300)),
             'criterion':['gini','entropy'],
             'max_depth':[3,4,5,6,7,8,9],
             'bootstrap' : [True, False],
             'random_state': [0]
            }

search = RandomizedSearchCV(estimator = rf_clf,        # The Classifer That we need its best Parameters 
                            param_distributions= parameters,       # It must Be Dictionary or List Of Dictionaries 
                           scoring = 'accuracy',          # The type of Evaluation Metric 
                           cv = 5,                       
                           n_jobs = 1,
                          verbose=0)

search = search.fit(X_train, y_train)
print("best accuracy is :" , search.best_score_ * 100)
search.best_params_ 


# In[ ]:


# With Oversampling

parameters= {'n_estimators':list(np.arange(100,300)),
             'criterion':['gini','entropy'],
             'max_depth':[3,4,5,6,7,8,9],
             'bootstrap' : [True, False],
             'random_state': [0]
            }

search_res = RandomizedSearchCV(estimator = rf_clf,        # The Classifer That we need its best Parameters 
                            param_distributions= parameters,       # It must Be Dictionary or List Of Dictionaries 
                           scoring = 'accuracy',          # The type of Evaluation Metric 
                           cv = 5,                       
                           n_jobs = 1,
                          verbose=0)

search_res = search_res.fit(X_train_res, y_train_res)
print("best accuracy is :" , search_res.best_score_ * 100)
search_res.best_params_ 


# ## Applying k-Fold Cross Validation

# In[ ]:


rf = search.best_estimator_
rf.fit(X_train, y_train)


# In[ ]:


rf_res = search_res.best_estimator_
rf_res.fit(X_train_res, y_train_res)


# In[ ]:


from sklearn.model_selection import cross_val_score
rf_clf_results = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = 5)
rf_clf_results_res = cross_val_score(estimator = rf_res, X = X_train_res, y = y_train_res, cv = 5)

rf_clf_pred = rf.predict(X_test)
rf_clf_res_pred = rf_res.predict(X_test_res)


print("Validation score befor Oversampling: %.5f%% (%.5f%%)" % (rf_clf_results.mean()*100.0, rf_clf_results.std()*100.0))
print("Validation score after Oversampling: %.5f%% (%.5f%%)" % (rf_clf_results_res.mean()*100.0, rf_clf_results_res.std()*100.0))
print('********************************************************')

print('Accuracy score of Test  befor Oversampling: {}'.format(accuracy_score(y_test,rf_clf_pred)*100))
print('Accuracy score of Test after Oversampling: {}'.format(accuracy_score(y_test_res,rf_clf_res_pred)*100))


# ## Model Evaluation

# In[ ]:


from sklearn.metrics import confusion_matrix

# After Oversampling
print(confusion_matrix(y_test_res, rf_clf_res_pred))


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
disp = plot_confusion_matrix(rf_res, X_test_res, y_test_res,
                              display_labels=['Transported','Not Transported'],
                              cmap=plt.cm.Blues)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test_res, rf_clf_res_pred))


# ## XGB Classifier

# In[ ]:


import xgboost as xgb
xgb_clf = xgb.XGBClassifier()


# In[ ]:


xgb_clf.get_params().keys()


# In[ ]:


# Without Oversampling

from scipy import stats
from scipy.stats import randint
parameters = {'n_estimators': stats.randint(150, 300),
              'learning_rate': stats.uniform(0.01, 0.3),
              'subsample': stats.uniform(0.3, 0.5),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.4),
              'min_child_weight': [1, 2, 3, 4]
             }
search = RandomizedSearchCV(estimator = xgb_clf,        # The Classifer That we need its best Parameters 
                            param_distributions= parameters,       # It must Be Dictionary or List Of Dictionaries 
                           scoring = 'accuracy',          # The type of Evaluation Metric 
                           cv = 5,                       
                           n_jobs = 1,
                          verbose=0)

search = search.fit(X_train, y_train)
print("best accuracy is :" , search.best_score_ * 100)
search.best_params_ 


# In[ ]:


# With Oversampling

parameters = {'n_estimators': stats.randint(150, 300),
              'learning_rate': stats.uniform(0.01, 0.3),
              'subsample': stats.uniform(0.3, 0.6),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.4),
              'min_child_weight': [1, 2, 3, 4]
             }
search_res = RandomizedSearchCV(estimator = xgb_clf,        # The Classifer That we need its best Parameters 
                            param_distributions= parameters,       # It must Be Dictionary or List Of Dictionaries 
                           scoring = 'accuracy',          # The type of Evaluation Metric 
                           cv = 5,                       
                           n_jobs = 1,
                          verbose=0)

search_res = search_res.fit(X_train_res, y_train_res)
print("best accuracy is :" , search_res.best_score_ * 100)
search_res.best_params_ 


# In[ ]:


xgb = search.best_estimator_
xgb.fit(X_train, y_train)


# In[ ]:


xgb_res = search_res.best_estimator_
xgb_res.fit(X_train_res, y_train_res)


# In[ ]:


xgb_clf_results = cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 5)
xgb_clf_results_res = cross_val_score(estimator = xgb_res, X = X_train_res, y = y_train_res, cv = 5)

xgb_clf_pred = xgb.predict(X_test)
xgb_clf_res_pred = xgb_res.predict(X_test_res)


print("Validation score befor Oversampling: %.5f%% (%.5f%%)" % (xgb_clf_results.mean()*100.0, xgb_clf_results.std()*100.0))
print("Validation score after Oversampling: %.5f%% (%.5f%%)" % (xgb_clf_results_res.mean()*100.0, xgb_clf_results_res.std()*100.0))
print('********************************************************')

print('Accuracy score of Test  befor Oversampling: {}'.format(accuracy_score(y_test,xgb_clf_pred)*100))
print('Accuracy score of Test after Oversampling: {}'.format(accuracy_score(y_test_res,xgb_clf_res_pred)*100))


# In[ ]:


# After Oversampling
print(confusion_matrix(y_test_res, xgb_clf_res_pred))


# In[ ]:


disp = plot_confusion_matrix(xgb_res, X_test_res, y_test_res,
                              display_labels=['Transported','Not Transported'],
                              cmap=plt.cm.Blues)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test_res, rf_clf_res_pred))


# ## LGBM Classifier

# In[ ]:


from lightgbm import LGBMClassifier
lgb_clf = LGBMClassifier()


# In[ ]:


lgb_clf.get_params().keys()


# In[ ]:


# With Oversampling

parameters = {        
      'bagging_fraction': (0.5, 0.8),
        'bagging_frequency': (5, 8),
        'feature_fraction': (0.5, 0.8),
        'max_depth': (10, 13),
        'min_data_in_leaf': (90, 120),
        'num_leaves': (1200, 1550),
        'n_estimators': stats.randint(150, 400),

        'learning_rate': stats.uniform(0.01, 0.3),
              'subsample': stats.uniform(0.3, 0.6),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.4),
              'min_child_weight': [1, 2, 3, 4]
             }
search_res = RandomizedSearchCV(estimator = lgb_clf,        # The Classifer That we need its best Parameters 
                            param_distributions= parameters,       # It must Be Dictionary or List Of Dictionaries 
                           scoring = 'accuracy',          # The type of Evaluation Metric 
                           cv = 5,                       
                           n_jobs = 1,
                          verbose=0)

search_res = search_res.fit(X_train_res, y_train_res)
print("best accuracy is :" , search_res.best_score_ * 100)
search_res.best_params_ 


# In[ ]:


lgb_clf = search_res.best_estimator_
lgb_clf.fit(X_train_res, y_train_res)


# In[ ]:


lgb_clf_results_res = cross_val_score(estimator = lgb_clf, X = X_train_res, y = y_train_res, cv = 5)

lgb_clf_res_pred = lgb_clf.predict(X_test_res)

print("Validation score after Oversampling: %.5f%% (%.5f%%)" % (lgb_clf_results_res.mean()*100.0, lgb_clf_results_res.std()*100.0))
print('********************************************************')
print('Accuracy score of Test after Oversampling: {}'.format(accuracy_score(y_test_res,lgb_clf_res_pred)*100))


# In[ ]:


disp = plot_confusion_matrix(lgb_clf, X_test_res, y_test_res,
                              display_labels=['Transported','Not Transported'],
                              cmap=plt.cm.Blues)


# In[ ]:


print(classification_report(y_test_res, lgb_clf_res_pred))


# ## Voting Classifier
# 

# In[ ]:


from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier([('rf',rf_res),('xgb',xgb_res),('lgb',lgb_clf)],verbose=True)
voting_clf.fit(X_train_res,y_train_res)
voting_clf_results = cross_val_score(estimator = voting_clf, X = X_train_res, y = y_train_res, cv = 5)
voting_clf_pred = voting_clf.predict(X_test_res)


# In[ ]:


print("Validation score after Oversampling: %.5f%% (%.5f%%)" % (voting_clf_results.mean()*100.0, voting_clf_results.std()*100.0))
print('Accuracy score of Test after Oversampling: {}'.format(accuracy_score(y_test_res,voting_clf_pred)*100))


# In[ ]:


disp = plot_confusion_matrix(voting_clf, X_test_res, y_test_res,
                              display_labels=['Transported','Not Transported'],
                              cmap=plt.cm.Blues)


# In[ ]:


print(classification_report(y_test_res, voting_clf_pred))


# In[ ]:


rf = accuracy_score(y_test_res,rf_clf_res_pred)*100
xgb = accuracy_score(y_test_res,xgb_clf_res_pred)*100
lgb = accuracy_score(y_test_res,lgb_clf_res_pred)*100
voting = accuracy_score(y_test_res,voting_clf_pred)*100

scores = [rf,xgb,lgb,voting]
labels = ['Random Forest Classifier', 'XGB Classifier','LGB Classifier', 'Voting Classifier']
accs = {'Accuracy': scores}
df = pd.DataFrame(accs,index=labels)
df


# ## Model Deployment
# 

# In[ ]:


test = test[chi_feature]


# In[ ]:


test_scl = scl.fit_transform(test)


# In[ ]:


predicted_xgb = xgb_res.predict(test_scl)
predicted_xgb


# In[ ]:


finalPreds = pd.DataFrame(predicted_xgb.astype(bool))
finalPreds.insert(0,"PassngerId", sample_submission.PassengerId)
finalPreds.columns = sample_submission.columns
finalPreds.to_csv('/kaggle/working/xgb_submission.csv', index = False)


# In[ ]:


predicted_lgb = lgb_clf.predict(test_scl)
predicted_lgb


# In[ ]:


finalPreds = pd.DataFrame(predicted_lgb.astype(bool))
finalPreds.insert(0,"PassngerId", sample_submission.PassengerId)
finalPreds.columns = sample_submission.columns
finalPreds.to_csv('/kaggle/working/lgb_submission.csv', index = False)


# In[ ]:




