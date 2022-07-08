#!/usr/bin/env python
# coding: utf-8

# <h1 style="color: #021f46"> Smart Home's Temperature - Time Series Forecasting </h1>

# ## Motivation
# This notebook is my contribution to the Smart Home's Temperature - Time Series Forecasting [competition](https://www.kaggle.com/competitions/smart-homes-temperature-time-series-forecasting/overview).
# I participate in this competition to improve my level in the field of Time Series Forecasting.
# 
# ## Methodology
# We will experiment with the performance of [EvalML](https://evalml.alteryx.com/en/stable/index.html) (an Automated Machine Learning (AutoML) Search algorithm) on this dataset.
# 
# 
# ## What you earn
# After reading this notebook :
# - You will be how to do Feature Engineering with Time Series data.
# - You will know how to use EvalML for regression tasks.
# - And much more...

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# # Load Data

# In[ ]:


train=pd.read_csv('../input/smart-homes-temperature-time-series-forecasting/train.csv')
test=pd.read_csv('../input/smart-homes-temperature-time-series-forecasting/test.csv')

train.shape, test.shape


# # Understand Data

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# # Feature Engineering

# FE credit: @gauravduttakiit [work](https://www.kaggle.com/code/gauravduttakiit/smart-home-s-temperature-flaml-mae)

# In[ ]:


get_ipython().system('pip install fast_ml')


# In[ ]:


from fast_ml.feature_engineering import FeatureEngineering_DateTime
dt_fe = FeatureEngineering_DateTime()

def features_eng(data):
    # Step 1
    data['Date']=data['Date'].copy() + ' ' + data['Time']
    data['Date']=pd.to_datetime(data['Date'])
    data=data.drop(['Time'],axis=1)
    
    # Step 2
    dt_fe.fit(data, datetime_variables=['Date'])
    data = dt_fe.transform(data)
    
    return data


# In[ ]:


train = features_eng(train)
test = features_eng(test)


# # Data Clean

# Copy and edit from : @gauravduttakiit [work](https://www.kaggle.com/code/gauravduttakiit/smart-home-s-temperature-flaml-mae)

# In[ ]:


nunique_train=train.nunique().reset_index()
remove_col=nunique_train[(nunique_train[0]==len(train)) | (nunique_train[0]==0) | (nunique_train[0]==1) ]['index'].tolist()
remove_col


# In[ ]:


def data_clean(data):
    # Step 1
    data=data.drop(remove_col,axis=1)
    
    # Step 2 (Data preprocessing)
    data['Date:is_weekend']=data['Date:is_weekend'].replace({False: 0, True: 1})
    data['Date:is_quarter_end']=data['Date:is_quarter_end'].replace({False: 0, True: 1})
    data['Date:is_month_end']=data['Date:is_month_end'].replace({False: 0, True: 1})
    
    # Step 3
    data=data.drop(['Date:time'],axis=1)
    
    # Step 4
    data['Date:day_part'] = data['Date:day_part'].fillna(value=np.nan)
    
    # Step 5
    data['Date:day_part'] = data['Date:day_part'].fillna(value='midnight')
    
    # Step 6
    data['Date:day_part'] = data['Date:day_part'].replace({'midnight':0, 'dawn':1, 'early morning':2, 
                                                               'late morning':3, 'noon':4, 'afternoon':5,
                                                               'evening':6, 'night':7})
    
    return data


# In[ ]:


train = data_clean(train)
test = data_clean(test)


# Function copy and edit from : @gauravduttakiit [work](https://www.kaggle.com/code/gauravduttakiit/smart-home-s-temperature-flaml-mae)

# In[ ]:


import re

train = train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
test = test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

train.head()


# # Split Data

# In[ ]:


X = train.copy()
y = X.pop('Indoor_temperature_room')


# In[ ]:


from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split( X, y, test_size=0.3, random_state=42)


# # Modeling with EvalML

# EvalML [docs here](https://evalml.alteryx.com/en/stable/user_guide/automl.html)

# - Installation

# In[ ]:


get_ipython().system('python3 -m pip install -q evalml==0.28.0')


# In[ ]:


from evalml.automl import AutoMLSearch


# ### Training

# In[ ]:


# run model
model_evalml = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', max_time=1200) # 20 minutes
model_evalml.search()


# ### Leaderboard

# In[ ]:


# check leaderboard
model_evalml.rankings


# ### Feature Importance

# ### Evaluate Model

# In[ ]:


import math
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluates_model(y_test, y_pred):
    print("*"*12, "Evaluations", "*"*12, '\n')
    
    print("MAE model :", mean_absolute_error(y_test, y_pred))
    
    print("MSE model :", mean_squared_error(y_test, y_pred))
    
    print("R2_Score model :", r2_score(y_test, y_pred))
    
    mse_1 = np.square(np.subtract(y_test,y_pred)).mean() 
    
    print("RMSE model :", math.sqrt(mse_1))


# In[ ]:


model_evalml.best_pipeline


# In[ ]:


pred = model_evalml.best_pipeline.predict(X_test)
evaluates_model(y_test, pred)


# # Make a Test prediction
# 

# In[ ]:


## generate predictions
y_predict = model_evalml.best_pipeline.predict(test)

len(y_predict)


# # Submission

# In[ ]:


sub=pd.read_csv('../input/smart-homes-temperature-time-series-forecasting/sample_submission.csv')
print(sub.shape)
sub.head()


# In[ ]:


get_ipython().system('pip install pandas -U')
import pandas as pd


# In[ ]:


sub['Indoor_temperature_room']=y_predict
sub.to_csv('submission.csv', index=False)
sub.head()


# In[ ]:





# # What Next ?
# 
# - Try to use a EvalML [Time Serie solution](https://evalml.alteryx.com/en/stable/user_guide/timeseries.html#)
# - Experiment with other Feature Engineering
# - Experiment with other AutoML algorithms (like H20, MLJAR)

# # Ressources
# 
# - Feature Engineering, Data Clean: @gauravduttakiit [work](https://www.kaggle.com/code/gauravduttakiit/smart-home-s-temperature-flaml-mae)
# - EvalML documentation : https://evalml.alteryx.com/en/stable/index.html

# <center>
#     <h1 style="color:#021f46"> I hope you learned something new today 🙃 
#         Thanks for reading  </h1>

# In[ ]:




