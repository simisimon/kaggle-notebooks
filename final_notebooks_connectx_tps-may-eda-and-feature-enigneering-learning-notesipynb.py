#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
import seaborn as sns
from cycler import cycler
from IPython.display import display
import datetime
import scipy.stats

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, normalized_mutual_info_score
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibrationDisplay
from lightgbm import LGBMClassifier

import shap
from matplotlib import gridspec

import plotly.express as px
from sklearn.preprocessing import KBinsDiscretizer
import copy


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv', index_col='id')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv',index_col='id')


# # Introduction
# This notebook is used as the learning notes of the EDA and the feature engineering. Welcome to read and comment on it :)
# 
# # Table of Contents
# 1. [What is Feature Interaction?](#1.-What-is-Feature-Interaction?)
# 2. [How to measure the feature relevance?](#2.-How-to-measure-the-feature-relevance?)
#     1. [Pearson's Correlation Matrix](#2.1-Pearson's-Correlation-Matrix)
#     2. [Mutual Information Matrix](#2.2-Mutual-Information-Matrix)
# 3. [Feature types](#3.-Feature-types)
# 4. [Feature 27](#4.-Feature-27)
#     1. [Categorical feature?](#4.1-Categorical-feature?)
#     2. [Position of str](#4.2-Position-of-str)
#     3. [Unique value](#4.3-Unique-value)
#     4. [Interaction plot](#4.4-Interaction-plot)
# 5. [Modelling](#5.-Modelling)
# 6. [Analysis with SHAP](#6.-Analysis-with-SHAP)
#     1. [SHAP interaction values](#6.1-SHAP-interaction-values)
#     2. [Dependence plot](#6.2-Dependence-plot)

# # Acknowledgement
# * [AMBROSM](https://www.kaggle.com/code/ambrosm/tpsmay22-eda-which-makes-sense/notebook)
# * [CABAXIOM](https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model)
# * [WTI 200](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/discussion/323766)

# # 1. What is Feature Interaction?
# * Feature **Relevance**: the association between a feature and the target
# * Feature **Redundancy**: the association between a feature and another feature
# * Feature **Interaction**: the association between two features and the target, when the features appear together, e.g., XOR gate. Note: the two features may have small relevance with the target individually, but large interaction with the target when they are put together.

# # 2. How to measure the feature relevance?
# * Linear feature relevance: [Pearson's Correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
# * Nonlinear feature relevance: [Mutual Information](https://en.wikipedia.org/wiki/Mutual_information)
# 
# ## 2.1 Pearson's Correlation Matrix

# In[ ]:


r_matrix = train.drop('f_27', axis = 1).corr()


# In[ ]:


fig = px.imshow(r_matrix, text_auto=".2f", width=1000, height=1000)
fig.update_layout(title="Pearson's Correlation Heatmap", title_x=0.5)
fig.show()


# ## 2.2 Mutual Information Matrix
# Before computing mutual information matrix, the continues features should be discretized. 
# The features whose number of unique values are greater than 20 are regarded as continues features.

# In[ ]:


N_unique = train.nunique().to_frame(name = 'N_unique')
display(N_unique)


# In[ ]:


features = train.columns
continues_features = train[features[N_unique.N_unique > 20]].drop('f_27', axis = 1)
continues_features.head()


# Use K-Bins Discretizer to discretize the features into 5-value discrete features.

# In[ ]:


est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
est.fit(continues_features)
discretized_feature = est.transform(continues_features)


# In[ ]:


train_discrete = copy.deepcopy(train.drop('f_27', axis = 1))
train_discrete.update(pd.DataFrame(discretized_feature, columns = continues_features.columns))
train_discrete.head()


# Use `normalized_mutual_info_score` to compute normlized mutual information.
# It should be noted that the minimum of MI is 0, but the maximum of MI is not a fixed value.
# Normalized MI will transfer the maximum MI to 1.

# In[ ]:


n = train_discrete.shape[1]
mi_matrix = np.ones((n, n))
for i in range(n):
    for j in range(i+1, n):
        mi_matrix[i, j] = normalized_mutual_info_score(train_discrete.iloc[:, i], train_discrete.iloc[:, j])
        mi_matrix[j, i] = mi_matrix[i, j]


# In[ ]:


mi_df_matrix = pd.DataFrame(mi_matrix, index = train_discrete.columns, columns = train_discrete.columns)


# In[ ]:


mask = np.triu(np.ones_like(mi_df_matrix))
sns.set(rc={'figure.figsize':(30, 30)})
dataplot = sns.heatmap(mi_df_matrix, cmap="YlGnBu", annot=True, mask=mask, fmt='.2f')
dataplot.axes.set_title("Mutual Information Matrix", fontsize=50)
plt.show()


# # 3. Feature types
# * **Numerical** feature: values can be compared and can be used in algebraic operations
#     * **Continues** feature: variable interval between neighbour values
#     * **Discrete** feature: constant interval between neighbour values
# * **Categorical** feature: values cannot be compared, and data encoding methods are required.
# 
# Data Encoding Methods:
# * [Ordinal Encode](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder)
# * [One Hot (Dummy) Encode](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder)
#     * m Dummy Encode
#     * [m-1 Dummy Encode](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=Specifies%20a%20methodology%20to%20use%20to%20drop%20one%20of%20the%20categories%20per%20feature.): If One Hot Encode is adopted for a feature, the feature matrix will not be full rank after centring. If the full rank feature is required, m-1 Dummy Encode is preferred.
# 
# <table>
# <tr><th>One Hot Encoding Feature (rank = 3) </th><th>Centred Feature (rank = 2)</th></tr>
# <tr><td>
# 
# | Value | Encode 1 | Encode 2 | Encode 3 |
# | --- | --- | --- | --- |
# | A | 1 | 0 | 0 |
# | B | 0 | 1 | 0 |
# | C | 0 | 0 | 1 |
# | B | 0 | 1 | 0 |
# | C | 0 | 0 | 1 |
# 
# </td><td>
# 
# | Value | Encode 1 | Encode 2 | Encode 3 |
# | --- | --- | --- | --- |
# | A | 0.8 | -0.4 | -0.4 |
# | B | -0.2 | 0.6 | -0.4 |
# | C | -0.2 | -0.4 | 0.6 |
# | B | -0.2 | 0.6 | -0.4 |
# | C | -0.2 | -0.4 | 0.6 |
# 
# </td></tr> </table>
# 
# 
# <table>
# <tr><th>m-1 Dummy Encoding Feature (rank = 2) </th><th>Centred Feature (rank = 2)</th></tr>
# <tr><td>
# 
# | Value | Encode 1 | Encode 2 |
# | --- | --- | --- |
# | A | 0 | 0 |
# | B | 1 | 0 |
# | C | 0 | 1 |
# | B | 1 | 0 |
# | C | 0 | 1 |
# 
# </td><td>
# 
# | Value | Encode 1 | Encode 2 |
# | --- | --- | --- |
# | A | -0.4 | -0.4 |
# | B | 0.6 | -0.4 |
# | C | -0.4 | 0.6 |
# | B | 0.6 | -0.4 |
# | C | -0.4 | 0.6 |
# 
# </td></tr> </table>
# 
# 
# 

# # 4. Feature 27
# Examples:

# In[ ]:


train.f_27.head()


# ## 4.1 Categorical feature?
# > It is important to understand whether the f_27 strings in test are the same as in training. Unfortunately, test contains 1181880 - 741354 = 440526 strings which do not occur in training.
# 
# > Insight: We must not use this string as a categorical feature in a classifier. Otherwise, the model learns to rely on strings which never occur in the test data.

# In[ ]:


temp = pd.DataFrame({'Dataset': ['Train', 'Test'], 
                     'No. of Unique': [train.f_27.nunique(), test.f_27.nunique()],
                     'Percentage of length': [train.f_27.nunique()/len(train)*100, test.f_27.nunique()/len(test)*100]}).set_index('Dataset')
print(temp)


# ## 4.2 Position of str

# In[ ]:


temp = pd.DataFrame(columns=['Position', 'No. of Unique'])
for i in range(10):
    temp = temp.append({'Position': i, 'No. of Unique': train.f_27.str.get(i).nunique()}, ignore_index=True)
print(temp.set_index('Position'))


# ## 4.3 Unique value
# The table clearly shows that the target probability depends on the unique character count:

# In[ ]:


unique_characters = train.f_27.apply(lambda s: len(set(s))).rename('unique_characters')
tg = train.groupby(unique_characters)
temp = pd.DataFrame({'size': tg.size(), 'probability': tg.target.mean().round(2)})
print(temp)


# ## 4.4 Interaction plot
# The interaction between the features and the target can be viewed in the scatter plots.
# For example, in the first plot, it is shown that the single feature f_02 cannot discriminate between the yellow and blue dots, which means that f_02 has low relevance to the target.
# However, f_02 and f_21 together can discriminate between the yellow and blue dots, which implies that f_02 and f_21 have interaction with the target.

# In[ ]:


plt.rcParams['axes.facecolor'] = 'k'
plt.figure(figsize=(11, 5))
cmap = ListedColormap(["#ffd700", "#0057b8"])
# target == 0 → yellow; target == 1 → blue

ax = plt.subplot(1, 3, 1)
ax.scatter(train['f_02'], train['f_21'], s=1,
           c=train.target, cmap=cmap)
ax.set_xlabel('f_02')
ax.set_ylabel('f_21')
ax.set_aspect('equal')
ax0 = ax

ax = plt.subplot(1, 3, 2, sharex=ax0, sharey=ax0)
ax.scatter(train['f_05'], train['f_22'], s=1,
           c=train.target, cmap=cmap)
ax.set_xlabel('f_05')
ax.set_ylabel('f_22')
ax.set_aspect('equal')

ax = plt.subplot(1, 3, 3, sharex=ax0, sharey=ax0)
ax.scatter(train['f_00'] + train['f_01'], train['f_26'], s=1,
           c=train.target, cmap=cmap)
ax.set_xlabel('f_00 + f_01')
ax.set_ylabel('f_26')
ax.set_aspect('equal')

plt.tight_layout(w_pad=1.0)
plt.savefig('three-projections.png')
plt.show()
plt.rcParams['axes.facecolor'] = '#0057b8' # blue


# # 5. Modelling
# The model is built for SHAP analysis.
# 
# See [AMBROSM](https://www.kaggle.com/code/ambrosm/tpsmay22-gradient-boosting-quickstart/notebook)

# In[ ]:


for df in [train, test]:
    # Extract the 10 letters of f_27 into individual features
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
        
    # unique_characters feature is from https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))
    
    # Feature interactions: create three ternary features
    # Every ternary feature can have the values -1, 0 and +1
    df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
    df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
    
features = [f for f in test.columns if f != 'id' and f != 'f_27']
test[features].head()


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Cross-validation of the classifier\n\ndef my_booster(random_state=1):\n    return LGBMClassifier(n_estimators=10000, min_child_samples=80,\n                          max_bins=511, random_state=random_state)\n      \nprint(f"{len(features)} features")\nscore_list = []\nkf = KFold(n_splits=5)\nfor fold, (idx_tr, idx_va) in enumerate(kf.split(train)):\n    X_tr = train.iloc[idx_tr][features]\n    X_va = train.iloc[idx_va][features]\n    y_tr = train.iloc[idx_tr].target\n    y_va = train.iloc[idx_va].target\n    \n    model = my_booster()\n\n    if True or type(model) != XGBClassifier and type(model) != LGBMClassifier: # no early stopping except hgb\n        model.fit(X_tr.values, y_tr)\n    else: # early stopping\n        model.fit(X_tr.values, y_tr, eval_set = [(X_va.values, y_va)], \n                  early_stopping_rounds=30, verbose=100)\n    y_va_pred = model.predict_proba(X_va.values)[:,1]\n    score = roc_auc_score(y_va, y_va_pred)\n    try:\n        print(f"Fold {fold}: n_iter ={model.n_iter_:5d}    AUC = {score:.5f}")\n    except AttributeError:\n        print(f"Fold {fold}:                  AUC = {score:.5f}")\n    score_list.append(score)\n    break # we only need the first fold\n    \nprint(f"OOF AUC:                       {np.mean(score_list):.5f}")\n')


# # 6. Analysis with SHAP
# [Shapley value](https://en.wikipedia.org/wiki/Shapley_value#:~:text=Shapley%20values%20provide%20a%20natural%20way%20to%20compute%20which%20features%20contribute%20to%20a%20prediction) is a popular indicator for feature importance analysis. The feature importance is different from the feature relevance.
# * Feature relevance: the association (i.e. mutual info and correlation) of a feature is estimated isolatedly.
# * Feature importance: the association between a feature and the target consider other features' affect.
# 
# Note: as the feature relevance does not 

# In[ ]:


# The computation of SHAP interaction is quite time consuming, here we only take 200 samples as an example.
X_sampled = train[features].sample(200, random_state=1307)  
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sampled)
shap_interaction = explainer.shap_interaction_values(X_sampled)
print(np.shape(shap_interaction))


# ## 6.1 SHAP interaction values

# In[ ]:


# Get absolute mean of matrices
mean_shap = np.abs(shap_interaction).mean(0)
df = pd.DataFrame(mean_shap, index=features, columns=features)


# In[ ]:


fig = px.imshow(df, text_auto=".2f", width=1000, height=1000)
fig.update_layout(title="SHAP interaction values", title_x=0.5)
fig.show()


# ## 6.2 Dependence plot
# 
# > Plots the value of the feature on the x-axis and the SHAP value of the same feature on the y-axis. This shows how the model depends on the given feature, and is like a richer extenstion of the classical parital dependence plots. Vertical dispersion of the data points represents interaction effects.
# 
# See [ref1](https://shap-lrjball.readthedocs.io/en/latest/generated/shap.dependence_plot.html) and [ref2](https://slundberg.github.io/shap/notebooks/plots/dependence_plot.html)
# 
# * The color corresponds to the feature f_28. 
# * The x-axis corresponds to f_02. 
# * The y-axis corresponds to **shape value** of f_02 (Note: not shap interaction value but shap value).
# 
# The interaction of the two features can be viewed by the vertical dispersion of the plot.
# For example, when f_02 at around -1, the red dots are generally split from the blue dots, which suggests an interaction effect between f_02 and f_28.

# In[ ]:


shap.dependence_plot('f_02', shap_values[0], X_sampled[features], interaction_index='f_28')


# In[ ]:




