#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import re as re
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 数据导入

# In[ ]:


train_df = pd.read_csv('../input/GiveMeSomeCredit/cs-training.csv')
test_df = pd.read_csv('../input/GiveMeSomeCredit/cs-test.csv')


# In[ ]:


print(train_df.info())
train_df.head(5)


# In[ ]:


print(test_df.info())
test_df.head(5)


# In[ ]:


print(test_df.info())
test_df.head(5)


# ## 数据清洗
# ### 数据检查

# In[ ]:


train_df.rename(columns={'Unnamed: 0':'ID'}, inplace=True)
test_df.rename(columns={'Unnamed: 0':'ID'}, inplace=True)


# In[ ]:


print(train_df.duplicated().value_counts())
print(test_df.duplicated().value_counts())


# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


# In[ ]:


import seaborn as sns
plt.figure()
sns.countplot('SeriousDlqin2yrs',data=train_df)


# In[ ]:


P = train_df.groupby('SeriousDlqin2yrs')['ID'].count().reset_index()
P['Percentage'] = 100 * P['ID'] / P['ID'].sum()
print(P)


# ### 数据整理

# In[ ]:


train_df.loc[train_df['age'] < 18]


# In[ ]:


train_df.loc[train_df['age'] == 0, 'age'] = train_df['age'].median()


# In[ ]:


working = train_df.loc[(train_df['age'] >= 18) & (train_df['age'] <= 60)]
senior = train_df.loc[(train_df['age'] > 60)]
working_income_mean = working['MonthlyIncome'].mean()
senior_income_mean = senior['MonthlyIncome'].mean()
print(working_income_mean)
print(senior_income_mean)


# In[ ]:


train_df['MonthlyIncome'] = train_df['MonthlyIncome'].replace(np.nan,train_df['MonthlyIncome'].mean())


# In[ ]:


train_df.info()


# In[ ]:


train_df['NumberOfDependents'].value_counts()


# In[ ]:


train_df['NumberOfDependents'].fillna(train_df['NumberOfDependents'].median(), inplace=True)


# In[ ]:


corr = train_df.corr()
plt.figure(figsize=(19, 15))
sns.heatmap(corr, annot=True, fmt='.2g')


# In[ ]:


plt.figure(figsize=(19, 12)) 
train_df[['NumberOfTime30-59DaysPastDueNotWorse', 
          'NumberOfTime60-89DaysPastDueNotWorse',
          'NumberOfTimes90DaysLate']].boxplot()
plt.show()


# In[ ]:


def replace98and96(column):
    new = []
    newval = column.median()
    for i in column:
        if (i == 96 or i == 98):
            new.append(newval)
        else:
            new.append(i)
    return new

train_df['NumberOfTime30-59DaysPastDueNotWorse'] = replace98and96(train_df['NumberOfTime30-59DaysPastDueNotWorse'])
train_df['NumberOfTimes90DaysLate'] = replace98and96(train_df['NumberOfTimes90DaysLate'])
train_df['NumberOfTime60-89DaysPastDueNotWorse'] = replace98and96(train_df['NumberOfTime60-89DaysPastDueNotWorse'])

test_df['NumberOfTime30-59DaysPastDueNotWorse'] = replace98and96(test_df['NumberOfTime30-59DaysPastDueNotWorse'])
test_df['NumberOfTimes90DaysLate'] = replace98and96(test_df['NumberOfTimes90DaysLate'])
test_df['NumberOfTime60-89DaysPastDueNotWorse'] = replace98and96(test_df['NumberOfTime60-89DaysPastDueNotWorse'])


# In[ ]:


corr = train_df.corr()
plt.figure(figsize=(19, 15))
sns.heatmap(corr, annot=True, fmt='.2g')


# In[ ]:


test_df.loc[test_df['age'] == 0, 'age'] = test_df['age'].median()
test_df['MonthlyIncome'] = test_df['MonthlyIncome'].replace(np.nan,test_df['MonthlyIncome'].mean())
test_df['NumberOfDependents'].fillna(test_df['NumberOfDependents'].median(), inplace=True)


# ## 数据分析
# ### 数据设定

# In[ ]:


X = train_df.drop(['SeriousDlqin2yrs', 'ID'],axis=1)
y = train_df['SeriousDlqin2yrs']
W = test_df.drop(['SeriousDlqin2yrs', 'ID'],axis=1)
z = test_df['SeriousDlqin2yrs']


# ### 线性回归分类

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=111)
logit = LogisticRegression(random_state=111, solver='saga', penalty='l1', class_weight='balanced', C=1.0, max_iter=500)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
logit.fit(X_train_scaled, y_train)
logit_scores_proba = logit.predict_proba(X_train_scaled)
logit_scores = logit_scores_proba[:,1]


# In[ ]:


def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(12,10))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1], "k--") # 画直线做参考
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive rate")


# In[ ]:


fpr_logit, tpr_logit, thresh_logit = roc_curve(y_train, logit_scores)

plot_roc_curve(fpr_logit,tpr_logit)
print('AUC Score : ', (roc_auc_score(y_train,logit_scores)))


# In[ ]:


logit_scores_proba_val = logit.predict_proba(X_test_scaled)

logit_scores_val = logit_scores_proba_val[:,1]

fpr_logit_val, tpr_logit_val, thresh_logit_val = roc_curve(y_test, logit_scores_val)

plot_roc_curve(fpr_logit_val,tpr_logit_val)
print('AUC Score :', (roc_auc_score(y_test,logit_scores_val)))


# In[ ]:


from sklearn.linear_model import LogisticRegressionCV
logit = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1, 10, 100], penalty='l1', solver='saga', max_iter=500, class_weight='balanced', random_state=111)

logit.fit(X_train_scaled, y_train)

print(logit.C_)


# In[ ]:


logit_scores_proba = logit.predict_proba(X_train_scaled)
logit_scores = logit_scores_proba[:,1]
fpr_logit, tpr_logit, thresh_logit = roc_curve(y_train, logit_scores)
plot_roc_curve(fpr_logit,tpr_logit)
print('AUC Score : ', (roc_auc_score(y_train,logit_scores)))


# ### 降采样处理

# In[ ]:


from imblearn.under_sampling import RandomUnderSampler

from collections import Counter
print('Original dataset shape :', Counter(y))


# In[ ]:


rus = RandomUnderSampler(random_state=111)

X_resampled, y_resampled = rus.fit_resample(X, y)
print('Resampled dataset shape:', Counter(y_resampled))


# In[ ]:


from sklearn.model_selection import train_test_split
X_train_rus, X_test_rus, y_train_rus, y_test_rus = train_test_split(X_resampled, y_resampled, random_state=111)
X_train_rus.shape, y_train_rus.shape


# In[ ]:


logit_resampled = LogisticRegression(random_state=111, solver='saga', penalty='l1', class_weight='balanced', C=1.0, max_iter=500)

logit_resampled.fit(X_resampled, y_resampled)
logit_resampled_proba_res = logit_resampled.predict_proba(X_resampled)
logit_resampled_scores = logit_resampled_proba_res[:, 1]
fpr_logit_resampled, tpr_logit_resampled, thresh_logit_resampled = roc_curve(y_resampled, logit_resampled_scores)
plot_roc_curve(fpr_logit_resampled, tpr_logit_resampled)
print('AUC score: ', roc_auc_score(y_resampled, logit_resampled_scores))


# ### 随机森林法分类

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
forest = RandomForestClassifier(n_estimators=300, random_state=111, max_depth=5, class_weight='balanced')
forest.fit(X_train_rus, y_train_rus)
y_scores_prob = forest.predict_proba(X_train_rus)
y_scores = y_scores_prob[:, 1]
fpr, tpr, thresh = roc_curve(y_train_rus, y_scores)
plot_roc_curve(fpr, tpr)
print('AUC score:', roc_auc_score(y_train_rus, y_scores))


# In[ ]:


y_test_proba = forest.predict_proba(X_test_rus)
y_scores_test = y_test_proba[:, 1]
fpr_test, tpr_test, thresh_test = roc_curve(y_test_rus, y_scores_test)
plot_roc_curve(fpr_test, tpr_test)
print('AUC Score:', roc_auc_score(y_test_rus, y_scores_test))


# In[ ]:


def plot_feature_importances(model):
    plt.figure(figsize=(10,8))
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)

plot_feature_importances(forest)


# ### 梯度提升法分类

# In[ ]:


gbc_clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=112)
gbc_clf.fit(X_train, y_train)
gbc_clf_proba = gbc_clf.predict_proba(X_train)
gbc_clf_scores = gbc_clf_proba[:, 1]
fpr_gbc, tpr_gbc, thresh_gbc = roc_curve(y_train, gbc_clf_scores)
plot_roc_curve(fpr_gbc, tpr_gbc)
print('AUC Score:', roc_auc_score(y_train, gbc_clf_scores))


# In[ ]:


gbc_val_proba = gbc_clf.predict_proba(X_test)
gbc_val_scores = gbc_val_proba[:, 1]
print('AUC score:', roc_auc_score(y_test, gbc_val_scores))


# In[ ]:


gbc_clf_submission = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05 ,max_depth=4,  random_state=42)
gbc_clf_submission.fit(X_train,y_train)
gbc_clf_proba = gbc_clf_submission.predict_proba(X_train)
gbc_clf_scores = gbc_clf_proba[:,1]
gbc_val_proba = gbc_clf_submission.predict_proba(X_test)
gbc_val_scores = gbc_val_proba[:,1]
fpr_gbc, tpr_gbc, thresh_gbc = roc_curve(y_train, gbc_clf_scores)
print('AUC Score :', roc_auc_score(y_train, gbc_clf_scores))
print('AUC Score :', roc_auc_score(y_test, gbc_val_scores))


# In[ ]:


plot_feature_importances(gbc_clf)


# ### 数据输出

# In[ ]:


submission_proba = gbc_clf_submission.predict_proba(W)
submission_scores = submission_proba[:, 1]
submission_scores.shape


# In[ ]:


W.shape


# In[ ]:


ids = np.arange(1, 101504)
submission = pd.DataFrame( {'Id': ids, 'Probability': submission_scores})
submission.to_csv('submission.csv', index=False)


# In[ ]:




