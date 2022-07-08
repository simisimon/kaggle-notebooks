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


# # 1. Import the relevant modules

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# # 2. Import the data

# In[ ]:


df = pd.read_csv('/kaggle/input/tabular-playground-series-jun-2021/train.csv')
df_test = pd.read_csv('/kaggle/input/tabular-playground-series-jun-2021/test.csv')


# In[ ]:


X_test = df_test.drop('id', axis=1)


# # 3. EDA

# In[ ]:


df.shape


# In[ ]:


pd.set_option('display.max_rows', 100)
df.isnull().sum(axis=0)


# In[ ]:


df.dtypes


# #### All numeric data, except for target variable! It needs to be transformed.

# In[ ]:


df.columns


# In[ ]:


df[['target_feature', 'target_number']] = df['target'].str.split("_", expand=True)
df['target_number'] = pd.to_numeric(df['target_number'])
df.drop(['target', 'target_feature', 'id'], axis=1, inplace=True)
df.rename(columns={'target_number': 'target'}, inplace=True)


# In[ ]:


# XGBoost starts counting at 0, so need to deduct 1 from target column
df['target'] = df['target'] - 1


# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# #### The target variable is now numeric!

# # 4. Split the data

# In[ ]:


# Generate X and y variables
X = df.drop('target', axis=1)
y = df['target'].copy()

# Train, test, split
X_train, X_eval, y_train, y_eval = train_test_split(X, y, random_state=42, test_size=0.1, stratify=y)


# # 5. Modelling

# In[ ]:


# Setting the parameters of the XGB Classifier
clf_xgb = xgb.XGBClassifier(objective='multi:softmax',
                            eval_metric='mlogloss',
                            seed=42,
                            num_class=9,
                            use_label_encoder=False)


# In[ ]:


# Training the Classifier on the train set
bst = clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=10,
            eval_metric='mlogloss',
            eval_set=[(X_eval, y_eval)])


# # 6. Submission 1

# In[ ]:


preds = bst.predict_proba(X_test)
y_test = pd.DataFrame(preds)
submission = y_test.copy()
# Add back column names
submission.columns = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7", "Class_8", "Class_9"]
#Add back the 'id' column to the submission file
submission['id'] = df_test['id'] 


# #### Results not sufficient, let's check the confusion matrix!

# In[ ]:


# Check the confusion matrix
plot_confusion_matrix(clf_xgb,
                      X_eval,
                      y_eval,
                      values_format='d');


# #### Only features 2, 6 and 8 seem to be predicted reliably (remember to add back 1...).

# ### 5.1 Optimising with GridSearchCV

# In[ ]:


param_grid = {
    'max_depth': [4],
    'num_class': [9],
    'learning_rate': [0.1, 0.5, 1.0],
    'gamma': [0.25],
    'reg_lambda': [10, 20, 100],
}

optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='multi:softmax',
                            eval_metric='mlogloss',
                            seed=42,
                            use_label_encoder=False),
    param_grid=param_grid,
    scoring='neg_log_loss',
    verbose=2,
    n_jobs=1,
    cv=3)


# In[ ]:


#optimal_params.fit(X_train,
 #                  y_train,
 #                  early_stopping_rounds=10,
 #                  eval_metric='mlogloss',
 #                 eval_set=[(X_eval, y_eval)],
 #                  verbose=False)

#print(optimal_params.best_params_)


# ### Best parameters: {'gamma': 0.25, 'learning_rate': 0.1, 'max_depth': 4, 'num_class': 9, 'reg_lambda': 100}

# # Submission 2

# In[ ]:


clf_xgb2 = xgb.XGBClassifier(objective='multi:softmax',
                        eval_metric="mlogloss",
                        gamma=0.25,
                        learning_rate=0.1,
                        max_depth=4,
                        num_class=9,
                        n_estimators=200,
                        reg_lambda=100,
                        use_label_encoder=False)
bst2 = clf_xgb2.fit(X_train, y_train)


# In[ ]:


preds = bst2.predict_proba(X_test)
y_test = pd.DataFrame(preds)
submission2 = y_test.copy()
# Add back column names
submission2.columns = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7", "Class_8", "Class_9"]
#Add back the 'id' column to the submission file
submission2['id'] = df_test['id'] 

