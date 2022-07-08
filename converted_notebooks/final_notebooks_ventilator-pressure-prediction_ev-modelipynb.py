#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import re
#### preprocessing data
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler 
#### regression algoritms   
from sklearn.model_selection import train_test_split 
#### accuracy scoring 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error 
import warnings 
warnings.filterwarnings('ignore')
# eval machine learning algoritm
get_ipython().system('python3 -m pip install -q evalml==0.28.0')
####################################
get_ipython().system('pip install fast_ml')
from fast_ml.feature_engineering import FeatureEngineering_DateTime
from evalml.automl import AutoMLSearch


# # load data

# In[ ]:


train_data =pd.read_csv('../input/smart-homes-temperature-time-series-forecasting/train.csv') 
test_data = pd.read_csv('../input/smart-homes-temperature-time-series-forecasting/test.csv')
sample = pd.read_csv('../input/smart-homes-temperature-time-series-forecasting/sample_submission.csv')


# In[ ]:


y =train_data['Indoor_temperature_room']
train_data.drop(['Id' ,'Indoor_temperature_room'] , axis=1 ,inplace=True) 
test_data.drop(['Id'] , axis=1 , inplace=True)


# # show data

# In[ ]:


#describe data
print(train_data.describe().T)
print(train_data.shape) 


# In[ ]:


#describe data
print(test_data.describe().T)
print(test_data.shape)


# In[ ]:


train_data.head()


# In[ ]:


test_data.tail(6)


# In[ ]:


### check if data not numerical features 
def ob_features(data):
    for col in data.columns:
        if (np.dtype(data[col]) != 'float'):
            print(col)
        else:
            print("not string")

print( ob_features(train_data) )
print( ob_features(test_data)  )


# In[ ]:


#show data type of Date and Time
print(np.dtype(train_data['Date'])) 
print(np.dtype(train_data['Time']))
print("*************************")
print(np.dtype(test_data['Date'])) 
print(np.dtype(test_data['Time']))


# In[ ]:


#check for nulls values 
print('nul values in train data',sum(train_data.isna().sum()))
print('nul values in test data ' ,sum(test_data.isna().sum()))


# # data processing

# credit [by](https://www.kaggle.com/code/gauravduttakiit/smart-home-s-temperature-flaml-mae)  // and edit by me

# In[ ]:


FE_model = FeatureEngineering_DateTime()

def features_eng(data):
    
    data['Date']=data['Date'].copy() + ' ' + data['Time']
    data['Date']=pd.to_datetime(data['Date'])
    data=data.drop(['Time'],axis=1)
    
    FE_model.fit(data, datetime_variables=['Date'])
    data = FE_model.transform(data)
    
    return data


# In[ ]:


train_data =features_eng(train_data) 
test_data =features_eng(test_data) 


# In[ ]:


train_data.head()


# unique features

# In[ ]:


def select(data):
    unique_train = data.nunique().reset_index() 
    select_features = unique_train[(unique_train[0]==len(data)) | (unique_train[0]==0) | (unique_train[0]==1)]['index'].to_list()
    return select_features


# In[ ]:


re_train_f = select(train_data) 
re_test_f = select(test_data)


# In[ ]:


re_train_f


# In[ ]:


re_test_f


# drop selected features up

# In[ ]:


train_data =train_data.drop(re_train_f,axis=1)
test_data =test_data.drop(re_train_f ,axis=1)


# In[ ]:


train_data.head()


# In[ ]:


#renam features
def renaming(data):
    data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)) 
    return data


# In[ ]:


train_data = renaming(train_data)
test_data= renaming(test_data)


# In[ ]:


# make data numerical
def Encod(data):
    label_model=LabelEncoder() 
    for col in data.columns:
        data[col]=label_model.fit_transform(data[col]) 
    return data

train_data =Encod(train_data) 
test_data =Encod(test_data)


# scale data

# In[ ]:


scal_model = StandardScaler()
train_data = scal_model.fit_transform(train_data) 
test_data =scal_model.fit_transform(test_data)


# In[ ]:


#spliting data
X=train_data
X_test = test_data 
y_test = sample['Indoor_temperature_room']
X_train,X_valid ,y_train , y_valid =train_test_split(X,y ,test_size=0.33 ,random_state=33,shuffle=True )


# # use algoritm

# In[ ]:


# fiting algoritms
EV_model = AutoMLSearch(X_train=X , y_train=y ,problem_type='regression' ,max_time=1200)
EV_model.search()


# In[ ]:


EV_model.rankings


# In[ ]:


EV_model.best_pipeline


# In[ ]:


y_predict=EV_model.best_pipeline.predict(X_valid) 
MAE = mean_absolute_error(y_valid,y_predict)
MSE = mean_squared_error(y_valid , y_predict) 
print("MAE---------->" ,MAE)
print("MSE---------->" ,MSE)


# # make submission

# In[ ]:


get_ipython().system('pip install pandas -U')
import pandas as pd


# In[ ]:


test_predict =EV_model.best_pipeline.predict(X_test) 
sample['Indoor_temperature_room'] =test_predict 
sample.to_csv('submission.csv' , index=False) 
sample.head()

