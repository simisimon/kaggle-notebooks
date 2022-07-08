#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get data
get_ipython().system('wget https://www.muratkoklu.com/datasets/vtdhnd06.php')
get_ipython().system('unzip /content/vtdhnd06.php')


# In[ ]:


# install packages
get_ipython().system('pip install statsmodels')
get_ipython().system('pip install flaml')
get_ipython().system('pip install openpyxl')


# In[ ]:


import warnings
warnings.simplefilter(action='ignore')


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


data = pd.read_excel('../input/date-fruit-datasets/Date_Fruit_Datasets/Date_Fruit_Datasets.xlsx')


# In[ ]:


# first look
data.head()


# In[ ]:


# shape
data.shape


# 898 samples & 35 features

# All features are numeric, target is categorical.

# In[ ]:


# data types
data.info()


# In[ ]:


classes = data['Class'].unique()


# # EDA

# In[ ]:


import plotly.figure_factory as ff
import plotly.express as px

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from scipy.stats import boxcox
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import boxcox, shapiro

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols


# ## Univariate analysis

# ### distribution of features

# In[ ]:


def draw_distribution(feature, column):
  fig, axes = plt.subplots(nrows=2, figsize=(12,5))
  axes[0].title.set_text(f'{column}')
  sns.boxplot(feature, ax=axes[0])
  sns.histplot(feature, kde=True, ax=axes[1])
  
  plt.show()


# In[ ]:


for column in data.columns[:-1]:
  draw_distribution(data[column].to_numpy(), column)


# Quick notes on observations:
# 
# - Non of the features are normal distributed. However, some of them are distributred normal-wise: bell-shaped, unimodal, symmetric.
# - Some normal-wise distributions experience outliers: extreme values on one or both sides.
# - Some features are exponentially distributed.
# - Scale is very different from feature to feature.
# - Some fatures contain outliers, which derivate from the majority of the observations very far across big gaps.
# 
# Distinctive characteristics of some particular features:
# - acpect ration has low variability: 897 samples has this feature equal to 0.
# - same for shapefactor_1
# - All Kurtosis* have exponential distribution and very long tails which make boxplots irrelevant to analyse. However, we can notice some extreme values on the ends of the tails across gaps in values which can be considered as outliers.

# ## Target EDA

# In[ ]:


target = data['Class'].value_counts()


# In[ ]:


plt.figure(figsize=(8, 8))
plt.pie(target.values, labels=target.index, autopct = '%0.0f%%')
plt.title('Classes proportion')
plt.show()


# Three out of seven classes are responsible for 63.4% of all the data. The largest class conaines 22.7% of the samples, the smallest -- only 7.24% which is 3 times less.
# 
# Some imbalance in the data is observed, however it is not critical. Nonetheless, some oversampling technique could be implemented to improve the final result.

# ## Features with respect to target.

# In[ ]:


def draw_distributions(features, column):
  plt.figure(figsize=(13, 4))
  plt.title(f'{column}')

  sns.histplot(features, x=column, kde=True, hue='Class')
  
  plt.show()


# In[ ]:


for column in data.columns[:-1]:
  draw_distributions(data[[column, 'Class']], column)


# From visual perspective, distributions of some features hue to diffrenet classes differ from each other clearly (e.g. area) but for other features the difference for classes is not observed. I intend to perform ANOVA test to identify usefull independent variables which show different distributions for different classes.
# If distribution of a features is the same for every class than it is not helpfull for future predictions, so we can eliminate it.

# ANOVA assumptions: we have already explored features' distributions and established that most of them do not meet some ANOVA requirements.
# 1. Normal distribution
# 2. No outliers
# 3. Variance is equal in groups.
# 
# Moreover, target analysis showed that there is disbalance in classes so the samples are od different sizes.
# 
# So, it is reqired to perform:
# 1. Data normalization
# 2. Upsampling
# 3. Outliers detection and removal.

# In[ ]:


target = data['Class']
data_eda = data.drop(['SHAPEFACTOR_1', 'ASPECT_RATIO', 'Class'], axis=1)


# In[ ]:


TARGET_MAP = {t: i for i, t in enumerate(target.unique())}
INVERSE_TARGET_MAP = {v:k for k,v in TARGET_MAP.items()}

class ANOVATransformer(BaseEstimator, TransformerMixin):
  def __init__(self, classes):
    self.classes = [TARGET_MAP[c] for c in classes]
    self.scaler = MinMaxScaler()

  def _devide_x_into_groups(self, x, y):
    x = [[x[col][y==t_class] for t_class in self.classes] for col in x]
    y = [[y[y==t_class] for t_class in self.classes] for col in x]
    return x, y

  def _remove_outliers(self, x, y):
    new_x = x.copy()
    new_y = y.copy()
    for i, f in enumerate(x):
      for k, group in enumerate(f):
        new_x[i][k] = group[(np.abs(stats.zscore(group)) < 3)]
        new_y[i][k] = y[i][k][(np.abs(stats.zscore(group)) < 3)]

    return new_x, new_y

  def _normalize_distribution(self, x):
    new_x = x.copy()
    for i, f in enumerate(x):
      for k, group in enumerate(f):
        group_norm = self.scaler.fit_transform(np.array(group).reshape(-1, 1))
        group_norm = [x[0]+1 for x in group_norm]
        x[i][k] = self.scaler.fit_transform(boxcox(group_norm, -1).reshape(-1, 1))
        x[i][k] = [x[0] for x in x[i][k]]
    return x


  def transform(self, x, y):
    y = pd.Series([*map(lambda x: TARGET_MAP[x], y)])
    self.features = x.columns
    x, y = self._devide_x_into_groups(x, y)
    x, y = self._remove_outliers(x, y)
    x = self._normalize_distribution(x)

    return x, y


# In[ ]:


t = ANOVATransformer(classes)
x, y = t.transform(data_eda, target)


# In[ ]:


for i, feature in enumerate(x):
  plt.figure(figsize=(12, 4))
  for group in feature:
    sns.distplot(group, hist=False, rug=True)
  plt.title(f'{t.features[i]}')  
  plt.legend([INVERSE_TARGET_MAP[c] for c in t.classes])
  plt.show()


# In[ ]:


# ANOVA
for i, feature in enumerate(x):
  F, p = stats.f_oneway(*feature)
  p = round(p, 5)
  if p < 0.01:
    print(f'+++ {t.features[i]}: p={p} --> reject H0, samples means are not equal')
  else:
    print(f'--- {t.features[i]}: p={p} --> fail to reject H0, samples means are equal')


# Features that failed to reject H0 wont be included in the final dataset.

# ## Correlation

# Correlation must be explored to avoid multicollinarity in the dataset.
# 
# It is important because if there are features correlated with each other, it might cause instability in model.

# In[ ]:


corr_data = data.drop(['SHAPEFACTOR_1', 'ASPECT_RATIO'], axis=1).corr()
plt.figure(figsize=(20, 12))
sns.heatmap(corr_data, annot=True)


# Multicollinearity is very high in the dataset. Some features are correlated with the coefficient equal to 1 (e.g. area and convex area)
# 

# Possible ways to deal with it:
# 1. Delete one of correlated features
# 2. Combine them
# 3. Implement DR
# 4. Use regularization in model.

# Conclusion:
# 1. Features are not normal-distributed --> Normalization required
# 2. Features are of different scales --> Scaling required
# 3.  Disbalance in classes --> Resampling reqired
# 4. Some features do not influence target distribution --> Feature removal required
# 5. Multicollinearity observed --> Feature removal required
# 6. No missing values observed
# 7. Outliers observed --> Outliers removal required
# 
# Features are considered to be deleted because of multicolliearity: CONVEX_AREA, MINOR_AXIS, PERIMETER, EQDIASQ, SHAPEFACTOR_2, SHAPEFACTOR_3, MeanRR, MeanRG, StdDevRR, SkewRR, SkewRG, KurtosisRR, KurtosisRG, EntropyRR, EntropyRG, ALLdaub4RR, ALLdaub4RG
# 
# + ECCENTRICITY, StdDevRB (ANOVA)

# # Data Preprocessing

# In[ ]:


from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import PowerTransformer


# In[ ]:


normalizer = PowerTransformer(method='box-cox')
scaler = MinMaxScaler(feature_range=(1, 2))
outlier_removal = FunctionTransformer(lambda x : x[(np.abs(stats.zscore(x)) < 3)])
over_sampler = SMOTE(random_state=0)
prepr_pipe = Pipeline([
                 ('scaler', scaler), 
                 ('normalizer', normalizer),
                 ('resampler', over_sampler)
                ])

test_pipe = Pipeline([
                 ('scaler', scaler), 
                 ('normalizer', normalizer)
                ])


# # Models fitting

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, randint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


def train_cv(model, X_train, y_train, params, n_splits=10):
  kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)

  cv = RandomizedSearchCV(model,
                        params,
                        cv=kf,
                        scoring='roc_auc',
                        return_train_score=True,
                        n_jobs=-1,
                        verbose=True,
                        random_state=1
                        )
  cv.fit(X_train, y_train)

  print('Best params', cv.best_params_)
  return cv


# ## With features removal

# In[ ]:


data_fr = data.drop(['CONVEX_AREA', 'MINOR_AXIS', 'MAJOR_AXIS', 'PERIMETER', 'EQDIASQ', 'SHAPEFACTOR_2',
               'SHAPEFACTOR_3', 'MeanRR', 'MeanRG', 'StdDevRR', 'SkewRR', 'SkewRG', 
               'KurtosisRR', 'KurtosisRG', 'EntropyRR', 'EntropyRG', 'ALLdaub4RR', 'ALLdaub4RB',
               'ALLdaub4RG', 'ECCENTRICITY', 'StdDevRB', 'SHAPEFACTOR_1', 'ASPECT_RATIO'], axis=1)
X = data_fr.drop(['Class'], axis=1)
y = data_fr['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=True)
plt.show()


# In[ ]:


sns.pairplot(data_fr, hue='Class')


# In[ ]:


X_train, y_train = prepr_pipe.fit_resample(X_train, y_train)
X_test = test_pipe.fit_transform(X_test, y_test)


# ## Logistic Regression

# In[ ]:


rs_parameters = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'C': uniform(scale=10),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
    'l1_ratio': uniform(scale=10)
    }
lr = LogisticRegression()
model_cv_lr = train_cv(lr, X_train, y_train, rs_parameters)

bestimator_lr = model_cv_lr.best_estimator_


# In[ ]:


ypred = bestimator_lr.predict(X_test)
ypred_proba = bestimator_lr.predict_proba(X_test)


# In[ ]:


print(roc_auc_score(y_test, ypred_proba, multi_class='ovr'))
print(classification_report(y_test, ypred))


# ## KNN

# In[ ]:


kf = KFold(n_splits=10, random_state=0, shuffle=True)
rs_parameters = {
    'n_neighbors': randint(10, 20),
    'weights': ['uniform', 'distance'],
    'leaf_size': randint(2, 10)
    }

knn = KNeighborsClassifier(n_jobs=-1)

model_cv_knn = train_cv(knn, X_train, y_train, rs_parameters)
bestimator_knn = model_cv_knn.best_estimator_


# In[ ]:


ypred = bestimator_knn.predict(X_test)
ypred_proba = bestimator_knn.predict_proba(X_test)


# In[ ]:


print(roc_auc_score(y_test, ypred_proba, multi_class='ovr'))
print(classification_report(y_test, ypred))


# ## Without features removal

# In[ ]:


X = data.drop(['SHAPEFACTOR_1', 'ASPECT_RATIO', 'Class'], axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0)


# In[ ]:


X_train, y_train = prepr_pipe.fit_resample(X_train, y_train)
X_test = test_pipe.fit_transform(X_test, y_test)


# ## Logistic Regression

# In[ ]:


rs_parameters = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'C': uniform(scale=10),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
    'l1_ratio': uniform(scale=10)
    }
lr = LogisticRegression()
model_cv_lr = train_cv(lr, X_train, y_train, rs_parameters)

bestimator_lr = model_cv_lr.best_estimator_


# In[ ]:


ypred = bestimator_lr.predict(X_test)
ypred_proba = bestimator_lr.predict_proba(X_test)


# In[ ]:


print(roc_auc_score(y_test, ypred_proba, multi_class='ovr'))
print(classification_report(y_test, ypred))


# ## KNN

# In[ ]:


kf = KFold(n_splits=10, random_state=0, shuffle=True)
rs_parameters = {
    'n_neighbors': randint(10, 20),
    'weights': ['uniform', 'distance'],
    'leaf_size': randint(2, 10)
    }

knn = KNeighborsClassifier(n_jobs=-1)

model_cv_knn = train_cv(knn, X_train, y_train, rs_parameters)
bestimator_knn = model_cv_knn.best_estimator_


# Best params {'leaf_size': 7, 'n_neighbors': 18, 'weights': 'distance'}

# In[ ]:


ypred = bestimator_knn.predict(X_test)
ypred_proba = bestimator_knn.predict_proba(X_test)


# In[ ]:


print(roc_auc_score(y_test, ypred_proba, multi_class='ovr'))
print(classification_report(y_test, ypred))


# ## with DR

# In[ ]:


from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA


# In[ ]:


normalizer = PowerTransformer(method='box-cox')
scaler = MinMaxScaler(feature_range=(2, 3))
outlier_removal = FunctionTransformer(lambda x : x[(np.abs(stats.zscore(x)) < 3)])
over_sampler = SMOTE(random_state=0)
pca = PCA(n_components=16)

prepr_pipe = Pipeline([
                 ('scaler', scaler), 
                 ('normalizer', normalizer),
                 ('DR', pca),
                ])


# In[ ]:


X = data.drop(['Class'], axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


X_train, y_train = over_sampler.fit_resample(X_train, y_train)
X_train = prepr_pipe.fit_transform(X_train)

X_test = prepr_pipe.transform(X_test)


# ### Logistic Regression

# In[ ]:


rs_parameters = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'C': uniform(scale=10),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
    'l1_ratio': uniform(scale=10)
    }
    
lr = LogisticRegression()
model_cv_lr = train_cv(lr, X_train, y_train, rs_parameters)

bestimator_lr = model_cv_lr.best_estimator_


# Best params {'C': 4.17022004702574, 'l1_ratio': 7.203244934421581, 'penalty': 'l1', 'solver': 'saga'}

# In[ ]:


ypred = bestimator_lr.predict(X_test)
ypred_proba = bestimator_lr.predict_proba(X_test)


# In[ ]:


print(roc_auc_score(y_test, ypred_proba, multi_class='ovr'))
print(classification_report(y_test, ypred))


# ### KNN

# In[ ]:


kf = KFold(n_splits=10, random_state=0, shuffle=True)
rs_parameters = {
    'n_neighbors': randint(10, 20),
    'weights': ['uniform', 'distance'],
    'leaf_size': randint(2, 10)
    }

knn = KNeighborsClassifier(n_jobs=-1)

model_cv_knn = train_cv(knn, X_train, y_train, rs_parameters)
bestimator_knn = model_cv_knn.best_estimator_


# Best params {'leaf_size': 7, 'n_neighbors': 18, 'weights': 'distance'}

# In[ ]:


ypred = bestimator_knn.predict(X_test)
ypred_proba = bestimator_knn.predict_proba(X_test)


# In[ ]:


print(roc_auc_score(y_test, ypred_proba, multi_class='ovr'))
print(classification_report(y_test, ypred))


# ### Random Forest

# In[ ]:


kf = KFold(n_splits=10, random_state=0, shuffle=True)
rs_parameters = {
    'n_estimators': randint(10, 100),
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': randint(2, 20),
    'min_samples_split': randint(2, 5),
    'min_samples_leaf': [1, 2, 3],

    }

rf = RandomForestClassifier(random_state=0, n_jobs=-1)

model_cv_rf = train_cv(rf, X_train, y_train, rs_parameters)
bestimator_rf = model_cv_knn.best_estimator_


# Best params {'criterion': 'entropy', 'max_depth': 13, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 19}

# In[ ]:


ypred = bestimator_rf.predict(X_test)
ypred_proba = bestimator_rf.predict_proba(X_test)


# In[ ]:


print(roc_auc_score(y_test, ypred_proba, multi_class='ovr'))
print(classification_report(y_test, ypred))


# ### LGBM

# In[ ]:


import lightgbm as lgb
from lightgbm import LGBMClassifier


# In[ ]:


train_data = lgb.Dataset(X_train, label=[*map(lambda x:TARGET_MAP[x], y_train)])
test_data = lgb.Dataset(X_test)


# In[ ]:


rs_parameters = {
    'learning_rate': [0.05, 0.1],
    'num_leaves': [35, 50, 100],
    'boosting_type' : ['gbdt'],
    'max_depth' : [5, 7, 10, 15],
    'random_state' : [501], 
    'colsample_bytree' : [0.5,0.7],
    'subsample' : [0.5, 0.7],
    'min_split_gain' : [0.01, 0.1],
    'min_data_in_leaf':[3, 5, 8]
    }

lgbm = LGBMClassifier()
model_cv_lgbm = train_cv(lgbm, X_train, y_train, rs_parameters)
bestimator_lgbm = model_cv_lgbm.best_estimator_


# Best params {'subsample': 0.7, 'random_state': 501, 'num_leaves': 50, 'min_split_gain': 0.1, 'min_data_in_leaf': 3, 'max_depth': 15, 'learning_rate': 0.1, 'colsample_bytree': 0.5, 'boosting_type': 'gbdt'}

# In[ ]:


ypred = bestimator_lgbm.predict(X_test)


# In[ ]:


print(classification_report(y_test, ypred))


# ## Automl

# In[ ]:


from flaml import AutoML


# In[ ]:


aml = AutoML()


# In[ ]:


aml.fit(X_train, y_train, task='classification', metric='log_loss', time_budget=250)


# In[ ]:


print(f'Best Model: {aml.best_estimator}')
print(f'Best hp config: {aml.best_config}')
print('Best log_loss: {0:.3g}'.format(aml.best_loss))


# In[ ]:


yt_pred = aml.predict(X_test)
print(classification_report(y_test, yt_pred))


# ## Overall results on test data from classification reports

# In[ ]:


RESULTS = pd.DataFrame(columns=['Model', 'DP', 'Prec', 'Recall', 'F1', 'Acc'])
RESULTS = RESULTS.append(pd.Series(['LR', 'feature removal', 0.85, 0.86, 0.84, 0.87], index=RESULTS.columns), ignore_index=True)
RESULTS = RESULTS.append(pd.Series(['KNN', 'feature removal', 0.78, 0.79, 0.77, 0.82], index=RESULTS.columns), ignore_index=True)
RESULTS = RESULTS.append(pd.Series(['LR', 'no feature removal', 0.89, 0.89, 0.88, 0.89], index=RESULTS.columns), ignore_index=True)
RESULTS = RESULTS.append(pd.Series(['KNN', 'no feature removal', 0.84, 0.85, 0.84, 0.87], index=RESULTS.columns), ignore_index=True)
RESULTS = RESULTS.append(pd.Series(['LR', 'PCA', 0.86, 0.87, 0.86, 0.89], index=RESULTS.columns), ignore_index=True)
RESULTS = RESULTS.append(pd.Series(['KNN', 'PCA', 0.89, 0.90, 0.89, 0.91], index=RESULTS.columns), ignore_index=True)
RESULTS = RESULTS.append(pd.Series(['RF', 'PCA', 0.89, 0.88, 0.88, 0.91], index=RESULTS.columns), ignore_index=True)
RESULTS = RESULTS.append(pd.Series(['LGBM', 'PCA', 0.87, 0.87, 0.86, 0.90], index=RESULTS.columns), ignore_index=True)
RESULTS = RESULTS.append(pd.Series(['AUTOML', 'PCA', 0.89, 0.88, 0.88, 0.90], index=RESULTS.columns), ignore_index=True)


# In[ ]:


by = ['Model', 'DP']
RESULTS.groupby(by).apply(lambda a: a.drop(by, axis=1)[:])


# In[ ]:


RESULTS.sort_values('Acc', ascending=False)


# In[ ]:


y_test


# In[ ]:


y_test = [*map(lambda x: TARGET_MAP[x], y_test)]
ypred_best = [*map(lambda x: TARGET_MAP[x],bestimator_knn.predict(X_test))]
labels = [INVERSE_TARGET_MAP[i] for i in range(7)]

plt.figure(figsize=(9, 7))
ax= plt.subplot()
sns.heatmap(confusion_matrix(y_test, ypred_best), annot=True, ax=ax)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.show()


# The confusiom matrix seems to be pretty nice. However, we can see that the most common mistake made our model is misclassification DOKOL as DEGLET. Other fruits are usually missclassified not more than 4 times.

# #### **Conclusion**
# 
# Dimensionality Reduction seems to be the best way to deal with multicollinearity as top results were obtained with it.
#  Interesting to observe how KNN improved after PCA transformation. At the same time, Linear Regression show nearly the same results among all the variants of DP.
# 
# Honestly, I expected better results from automl but still it shows one of the best results.
# 
# Overall, KNN + PCA + Oversampling is the best model with accuracy = 0.91 and f1 = 0.89.
# 
# 
# 

# In[ ]:




