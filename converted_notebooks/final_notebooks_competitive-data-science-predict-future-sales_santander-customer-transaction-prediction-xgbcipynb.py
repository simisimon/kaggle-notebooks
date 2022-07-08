#!/usr/bin/env python
# coding: utf-8

# <div style="border-radius:10px;
#             border: #0b0265 solid;
#            background-color:#e8efff;
#            font-size:110%;
#            letter-spacing:0.5px;
#             text-align: center">
# 
# <center><h1 style="padding: 25px 0px; color:#0b0265; font-weight: bold; font-family: Cursive">
# Santander customer transaction prediction</h1></center>
# <center><h3 style="padding-bottom: 25px; color:#0b0265; font-weight: bold; font-style:italic; font-family: Cursive">
# (With XGBClassifier)</h3></center>     
# 
# </div>

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# # Define Functions

# In[ ]:


kfold = StratifiedKFold(n_splits=5, random_state=14, shuffle=True)

def grid_model(x, y , model , param):
    grid=GridSearchCV(estimator=model, param_grid=param, scoring="roc_auc", cv=kfold, n_jobs=-1)
    grid.fit(x, y)
    print("Best Score: %f use parameters: %s" % (grid.best_score_, grid.best_params_))
# -------------------------------------------------------
    
def cross_validation(x, y, model):
    result= cross_val_score(model, x, y, cv=kfold, scoring="roc_auc", n_jobs=-1)
    print("Score: %f" % result.mean())


# # Data Understanding & Preproccessing

# In[ ]:


np.random.seed(14)
df=pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")
df.head()


# In[ ]:


df[df.columns[df.isna().sum() > 0]].isna().mean()*100


# In[ ]:


df.drop('ID_code', axis=1, inplace=True)
df.shape


# In[ ]:


df.drop_duplicates(inplace=True)
df.shape


# In[ ]:


x=df.drop(['target'], axis=1)
y=df['target']


# In[ ]:


mmscaler=MinMaxScaler()
x_train=pd.DataFrame(mmscaler.fit_transform(x), columns=x.columns)
x_train.head()


# In[ ]:


sns.countplot(y, palette='dark')


# # Feature Selection

# In[ ]:


modelSelect = RandomForestClassifier(random_state=14, n_jobs=-1)
modelSelect.fit(x_train, y)
importance = modelSelect.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.6f' % (i, v))


# In[ ]:


list_features = []
for i,v in enumerate(importance):
    if(v > 0.0065):
        print('Feature: %0d, Score: %.6f' % (i, v))
        list_features.append(i)


# In[ ]:


x_select = x_train.iloc[:, list_features]
x_select.head()


# # Modeling with tuned hyperparameters

# <div style="padding: 5px 0px; font-family: Cursive; font-size:16px; background-color:#eaeefc;padding: 25px 10px">
# We used Grid search for tuning hyperparameters</div>

# In[ ]:


counter = Counter(y)
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)


# In[ ]:


modelXGB = XGBClassifier(n_jobs=-1, random_state=14, 
                         scale_pos_weight=estimate, 
                         n_estimators= 280, learning_rate=0.08, 
                         max_depth=3, subsample=0.5)

scores = cross_validation(x_select, y, modelXGB)
print(scores)


# # Prediction & Submission

# In[ ]:


test=pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")
test.head()


# In[ ]:


tmp_test = test.iloc[:,1:]
tmp_test.head()


# In[ ]:


test.drop_duplicates(inplace=True)
x_test=mmscaler.transform(tmp_test)
x_test=pd.DataFrame(x_test, columns=tmp_test.columns)
x_select_test = x_test.iloc[:, list_features]
x_select_test.head()


# In[ ]:


modelXGB.fit(x_select, y)
y_pred_XGB = modelXGB.predict(x_select)
y_pred_XGB_test = modelXGB.predict(x_select_test)


# In[ ]:


output = pd.DataFrame({'ID_code': test.ID_code, 
                       'target': y_pred_XGB_test})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# <div style="padding: 5px 0px; font-family: Cursive; font-size:16px; background-color:#eaeefc;padding: 25px 10px">
# Thank you very much, <a href="https://www.kaggle.com/arezoodahesh">Arezoo</a> for your cooperation. <br> You can see our other notebook related to this competition which is with algorithm LGBMClassifier <a href="https://www.kaggle.com/code/arezoodahesh/santander-customer-transaction-prediction-lgbm">here</a>.</div>

# <div style="border-radius:10px;
#             background-color:#ffffff;
#             border-style:solid;
#             border-color: #0b0265;
#             letter-spacing:0.5px;">
# 
# <center><h4 style="padding: 5px 0px; color:#0b0265; font-weight: bold; font-family: Cursive">
#     Thanks for your attention and for reviewing our notebook.🙌 <br><br>Please write your comments for us.📝</h4></center>
# <center><h4 style="padding: 5px 0px; color:#0b0265; font-weight: bold; font-family: Cursive">
# If you liked our work and found it useful, please upvote. Thank you🙏</h4></center>
# </div>
