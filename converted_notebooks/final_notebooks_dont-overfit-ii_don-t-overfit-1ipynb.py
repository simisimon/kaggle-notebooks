#!/usr/bin/env python
# coding: utf-8

# # Load data

# In[ ]:


import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
# Data processing, metrics and modeling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import RFE

from sklearn.metrics import roc_auc_score as aucroc
# load libraries
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import eli5
from eli5.sklearn import PermutationImportance
import shap


# In[ ]:


datapath = "../input/older-dataset-for-dont-overfit-ii-challenge"
train = pd.read_csv(os.path.join(datapath,"train.csv"))
test = pd.read_csv(os.path.join(datapath,"test.csv"))


# # EDA

# In[ ]:


train.head(10)


# In[ ]:


train.set_index('id')
test.set_index('id')
train['target'].describe()


# In[ ]:


labels = train.pop('target')
train = train.drop(["id"],axis=1)
test=test.drop(["id"],axis=1)


# # choosing data

# # **way one**

# **1- nulls**

# In[ ]:


missing_series = train.isnull().sum() / train.shape[0]


# In[ ]:


#to be sure there r no missing data in any column
train.isnull().sum().sum()


# there r no missing values in the data

# **2- unique**

# In[ ]:


# identify columns with single unique values
unique_counts = train.nunique()
unique_stats = pd.DataFrame(unique_counts).rename(columns =
                                                  {'index': 'feature', 0: 'nunique'})
unique_stats = unique_stats.sort_values('nunique', ascending = True)
record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(columns =
                                                                                            {'index': 'feature', 0: 'nunique'})


# In[ ]:


# drop column if it have unique values
to_drop = list(record_single_unique['feature'])
to_drop


# There r no unique column to drop -_-

# **3- collinear**

# In[ ]:


corr_matrix = train.corr()
# Extract the upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
upper


# In[ ]:


# Select the features with correlations above the threshold
# Need to use the absolute value
correlation_threshold = 0.5
to_drop = [column for column in upper.columns if any(upper[column].abs() >= correlation_threshold)]

# Dataframe to hold correlated pairs
record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])


# In[ ]:


for column in to_drop:

    # Find the correlated features
    corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

    # Find the correlated values
    corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
    drop_features = [column for _ in range(len(corr_features))]    

    # Record the information (need a temp df for now)
    temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                     'corr_feature': corr_features,
                                     'corr_value': corr_values})

    # Add to dataframe
    record_collinear = record_collinear.append(temp_df, ignore_index = True)

record_collinear


# it's seems no solution except using toolers for feature seliction and dimentionality reduction

# # **Way two**

# In[ ]:


# ANOVA feature selection for numeric input and categorical output
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# generate dataset
X, y =train, labels
# define feature selection
fs = SelectKBest(score_func=f_classif, k=30)
# apply feature selection
X_selected = fs.fit_transform(X, y)
train_df = pd.DataFrame(X_selected)
print(X_selected.shape)


# In[ ]:


train_df = pd.DataFrame(train_df)
test_df =(pd.DataFrame(fs.transform(test)))


# In[ ]:


train_df


# # Taining model

# In[ ]:


# std = StandardScaler()
# train_df = std.fit_transform(train_df)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_df, labels, test_size=0.20)


# In[ ]:


kfold = model_selection.KFold(n_splits=10, random_state=42)
best_parameters = {'C': 0.1, 'class_weight': 'balanced', 'penalty': 'l1'}

estimators = []
model1 = LogisticRegression(solver="liblinear"); est1 = RFE(model1, 25, step=1);estimators.append(('logistic', est1))
model2 = DecisionTreeClassifier(); est2 = RFE(model1, 25, step=1);estimators.append(('cart', est2))
model3 = SVC(); estimators.append(('svm', model3))
ensemble = VotingClassifier(estimators)
# selector = RFE(ensemble, 25, step=1)
results = model_selection.cross_val_score(ensemble, X_train, y_train, cv=kfold)
print(); 
print("Result Mean:{}".format(results.mean()))


# In[ ]:


ensemble.fit(X_train, y_train)
print("Score:{0}".format(ensemble.score(X_train, y_train)))


# In[ ]:


prediction = ensemble.predict(test_df)
submission = pd.read_csv("../input/older-dataset-for-dont-overfit-ii-challenge/sample_submission.csv")
submission['target'] = prediction
submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:


perm = PermutationImportance(ensemble, random_state=42).fit(train_df, labels)


# In[ ]:


submission = pd.read_csv("../input/older-dataset-for-dont-overfit-ii-challenge/sample_submission.csv")
submission['target'] = perm.predict(test_df)
submission.to_csv('submission_perm.csv', index=False)


# In[ ]:


submission.shape


# In[ ]:





# In[ ]:


model1 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear', tol=0.00001,dual=False)
est1 = RFE(model1, 25, step=1)
est1.fit(train_df,labels)
print("Score:{0}".format(est1.score(train_df,labels)))


# In[ ]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# In[ ]:


sfs1 = SFS(est1, k_features=(10, 15), forward=True, floating=False,verbose=1,scoring='roc_auc',cv=5,n_jobs=-1)
sfs1 = sfs1.fit(X_train, y_train)


# In[ ]:


plt.figure(figsize=(20,8))
fig1 = plot_sfs(sfs1.get_metric_dict(), color = "red",kind='std_dev', marker="p")
plt.ylim([0.8, 1])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()


# In[ ]:


sfseatures = list(sfs1.k_feature_names_)


# In[ ]:


train_1 = train_df[sfseatures]
test_1 = test_df[sfseatures]


# In[ ]:


model1 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear', tol=0.00001,dual=False)
est1 = RFE(model1, 25, step=1)
est1.fit(train_1,labels)
print("Score:{0}".format(est1.score(train_1,labels)))


# In[ ]:


submission = pd.read_csv("../input/older-dataset-for-dont-overfit-ii-challenge/sample_submission.csv")
submission['target'] = est1.predict(test_1)
submission.to_csv('submission_sfs.csv', index=False)


# In[ ]:




