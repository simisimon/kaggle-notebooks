#!/usr/bin/env python
# coding: utf-8

# # Intro 
# 
# In this Notebook we fit a few decision tree ensemble models to the Spaceship Titanic data, with and without Boruta feature selection; we will look at the impact on model performance.
# 
# ![boruta](https://upload.wikimedia.org/wikipedia/commons/2/26/Leshy_%281906%29.jpg)

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


import numpy as np
import pandas as pd

from sklearn import set_config
set_config(display='diagram')

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.metrics import RocCurveDisplay
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from boruta import BorutaPy
import plotly.express as px


# I use 20% of the data for validation, the rest for training.

# In[ ]:


#X, y = df.drop('Transported', axis=1), df['Transported']
df = pd.read_csv(
    '../input/spaceship-titanic/train.csv', 
    dtype=dict(
        HomePlanet='category', Destination='category',
        ))

y = np.where(df.pop('Transported'), 1, 0)
X = df

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2)


# A tiny bit of feature engineering.

# In[ ]:


def df_feat_eng(df):
    engineered_df = (
        df.assign(cabin0 = lambda x: x['Cabin'].str.split('/').str[0].astype('category'))
        .assign(cabin2 = lambda x: x['Cabin'].str.split('/').str[2].astype('category'))
        .assign(psg_group = lambda x: x['PassengerId'].str.split('_').str[1].astype('category'))
        .assign(spend = lambda x: x[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1))
        .drop('PassengerId', axis=1)
    )
    return engineered_df


# Let's build a preprocessing pipeline to deal with missing data, and encode the categorical data.

# In[ ]:


cat_sel = make_column_selector(dtype_include=['category'])
num_sel = make_column_selector(dtype_include=np.number)

col_prep = ColumnTransformer([
    (
        'cat', 
        Pipeline([
            ('std_cat', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder())
        ]),
        cat_sel
    ),
    (
        'num',
        SimpleImputer(strategy='mean'),
        num_sel
    ),
    (   
        'bool',
        FunctionTransformer(lambda x: np.where(x, 1, 0)),
        ['CryoSleep', 'VIP']
    )
    ])


# In[ ]:


col_prep.fit(X)
print(col_prep.transform(X).shape)
print(X.shape)


# In[ ]:


pipe_all_features = Pipeline([
    ('prepipe', FunctionTransformer(df_feat_eng)),
    ('col_prep', col_prep), 
    ('xgb', XGBClassifier(n_estimators=1000, random_state=42))
    ])


# In[ ]:


pipe_all_features_rf = Pipeline([
    ('prepipe', FunctionTransformer(df_feat_eng)),
    ('col_prep', col_prep), 
    ('xgb', RandomForestClassifier(n_estimators=1000, random_state=42))
    ])


# In[ ]:


pipe_all_features[:-1].fit(X)
print(pipe_all_features[:-1].transform(X).shape)
print(X.shape)


# In[ ]:


get_ipython().run_line_magic('time', 'pipe_all_features.fit(X_train,y_train)')


# In[ ]:


get_ipython().run_line_magic('time', 'pipe_all_features_rf.fit(X_train,y_train)')


# In[ ]:


RocCurveDisplay.from_estimator(pipe_all_features, X_val, y_val)


# # Using Boruta for feature selection
# 
# Below I use Boruta first with RF (the default), and then with XGboost to compare the results.

# In[ ]:


rf_boruta = RandomForestClassifier(n_jobs=2, max_depth=3, n_estimators=100)
xgb_boruta = XGBClassifier(n_jobs=2, max_depth=3, n_estimators=100)

def pipe_boruta_sel(estimator):
    return Pipeline([
    ('prepipe', FunctionTransformer(df_feat_eng)),
    ('col_prep', col_prep), 
    ('boruta', BorutaPy(estimator, verbose=2, perc=100, max_iter=20))
    ])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pipe_boruta_sel_rf = pipe_boruta_sel(rf_boruta)\npipe_boruta_sel_rf.fit(X, y)\n')


# In[ ]:


#%%time
#pipe_boruta_sel_xgb = pipe_boruta_sel(xgb_boruta)
#pipe_boruta_sel_xgb.fit(X,y)


# In[ ]:


#pd.DataFrame({'rf_rank': pipe_boruta_sel_rf[-1].ranking_, 'xgb_rank': pipe_boruta_sel_xgb[-1].ranking_})


# In[ ]:


def boruta_selector(X):
    return X[:, pipe_boruta_sel_rf[-1].support_]
selector = FunctionTransformer(boruta_selector)


# In[ ]:


rf_params = dict(n_jobs=2, n_estimators=200, random_state=42)
xgb_params = dict(n_jobs=2, max_depth=3, n_estimators=200, random_state=42)
lgb_params = {'subsample': 0.8,
 'reg_lambda': 0,
 'reg_alpha': 0.1,
 'objective': 'binary',
 'num_leaves': 20,
 'n_estimators': 200,
 'max_depth': -1,
 'learning_rate': 0.1,
 'colsample_bytree': 0.8,
 'boosting_type': 'gbdt',
 'random_state': 42
             }
hist_params = {'random_state': 42}
cb_params = {'random_state': 42, 'verbose': 0}

estimators = [
    {'short_name': 'rf', 'name': 'Random Forest', 'estimator': RandomForestClassifier, 'params': rf_params},
    {'short_name': 'xgb', 'name': 'XGBoost', 'estimator': XGBClassifier, 'params': xgb_params},
    {'short_name': 'lgb', 'name': 'LGBM', 'estimator': LGBMClassifier, 'params': lgb_params},
    {'short_name': 'hgb', 'name': 'HistGB', 'estimator': HistGradientBoostingClassifier, 'params': hist_params},
    {'short_name': 'cb', 'name': 'CatBoost', 'estimator': CatBoostClassifier, 'params': cb_params}
]

preproc = Pipeline([
    ('prepipe', FunctionTransformer(df_feat_eng)),
    ('col_prep', col_prep),
    ])

sel = Pipeline([
    ('sel', selector)
])

pipelines_all_features = {}
pipelines_boruta_features = {}
for est in estimators:
    pipelines_all_features[est['short_name']] = {
            'short_name': est['short_name'] + '_all',
            'name': est['name'],
            'pipeline': Pipeline([
                ('preproc', preproc),
                (est['short_name'], est['estimator'](**est['params']))
                ])
        }
    pipelines_boruta_features[est['short_name']] = {
            'short_name': est['short_name'] + '_boruta',
            'name': est['name'],
            'pipeline': Pipeline([
                ('preproc', preproc),
                ('sel', sel),
                (est['short_name'], est['estimator'](**est['params']))
                ])
        }


# In[ ]:


pipelines_all_features['lgb']['pipeline'].fit(X,y)


# Quick check of the shape of the model input to make sure the feature selection step is working

# In[ ]:


print(pipelines_all_features['rf']['pipeline'][:-1].transform(X_train).shape)
print(pipelines_boruta_features['rf']['pipeline'][:-1].transform(X_train).shape)


# In[ ]:


fig, ax = plt.subplots(len(pipelines_all_features), 1, figsize=(8, 8*len(pipelines_all_features)))

for i, v in enumerate(pipelines_all_features.values()):
    v['pipeline'].fit(X_train, y_train)
    RocCurveDisplay.from_estimator(v['pipeline'], X_val, y_val, ax=ax[i], name=f"{v['short_name']}, all features")
    print(f"fit {v['name']}")

for i, v in enumerate(pipelines_boruta_features.values()):
    v['pipeline'].fit(X_train, y_train)
    RocCurveDisplay.from_estimator(v['pipeline'], X_val, y_val, ax=ax[i], name=f"{v['short_name']}, Boruta features")
    print(f"fit {v['name']}")


# In[ ]:


for v in pipelines_all_features.values():
    print(f"Accuracy of {v['short_name']} with all features: {v['pipeline'].score(X_val, y_val)}")
    
for v in pipelines_boruta_features.values():
    print(f"Accuracy of {v['short_name']} with Boruta features: {v['pipeline'].score(X_val, y_val)}")


# In[ ]:


X_test = pd.read_csv('../input/spaceship-titanic/test.csv')


final_model = StackingClassifier(
[(model['short_name'], model['pipeline']) for model in list(pipelines_all_features.values()) + list(pipelines_boruta_features.values())]
)

final_model.fit(X, y)

pd.DataFrame(
    {'PassengerId': X_test.PassengerId, 
     'Transported': final_model.predict(X_test).astype(bool)}
).to_csv('/kaggle/working/submission.csv', index=False)

