#!/usr/bin/env python
# coding: utf-8

# # About Dataset
# ## **Context**
# Since as a beginner in machine learning it would be a great opportunity to try some techniques to predict the outcome of the drugs that might be accurate for the patient.
# 
# ## **Content**
# The target feature is
# * Drug type
# The feature sets are:
# * Age
# * Sex
# * Blood Pressure Levels (BP)
# * Cholesterol Levels
# * Na to Potassium Ration
# 
# ## **Inspiration**
# The main problem here in not just the feature sets and target sets but also the approach that is taken in solving these types of problems as a beginner. So best of luck.

# # import 

# In[ ]:


import numpy as np

import pandas as pd

from pandas_profiling import ProfileReport

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,r2_score,classification_report


# # read data

# In[ ]:


data = pd.read_csv('../input/drug-classification/drug200.csv')


# # pandas_profiling

# In[ ]:


profile = ProfileReport(data, title="Drug Pandas Profiling Report")
profile


# In[ ]:


data.isna().sum()


# In[ ]:


data.info()


# In[ ]:


data.sample(2)


# In[ ]:


# x = data.drop('Drug',axis=1)
# y = data[['Drug']]


# # Preprocessing data

# In[ ]:


data_num = data.select_dtypes(['int64','float64'])
data_obj = data.select_dtypes(['object'])


# In[ ]:


col = list(data_obj.columns)
for i in col:
    print(i)
    print(data_obj[i].unique(),'\n\n')


# In[ ]:


### obj data

# nominal data = ['Sex']
# ordinal data = ['BP' , 'Cholesterol']


nominal_data = ['Sex']
ordinal_data = ['BP' , 'Cholesterol']


ordinal_encoder = OrdinalEncoder(categories=[['HIGH','NORMAL','LOW'] ,
                                             ['HIGH' ,'NORMAL']])
cat_encoded_ = ordinal_encoder.fit_transform(data_obj[ordinal_data])
data_obj_ord = pd.DataFrame(cat_encoded_,columns=ordinal_data)

def nom_data():
    
    pass

def nominal_data(df,i):
    cat_encoder = OneHotEncoder()
    x = cat_encoder.fit_transform(df[[i]])
    qw = [f'{i}{r}' for r in range(len(df[i].unique()))]
    df = pd.DataFrame(x.toarray(),dtype=np.float64,columns=qw)
    return df
Sex = nominal_data(data_obj,'Sex')


# In[ ]:


ALL_Data = pd.concat([data_num,data_obj_ord,Sex],axis=1)


# In[ ]:


def scaler(data):
    num_scaler=StandardScaler()
    scaler = num_scaler.fit_transform(data)
    data = pd.DataFrame(scaler,columns=data.columns,index=data.index)
    return data

train_ordinal_data_scaler = scaler(ALL_Data)


# In[ ]:


train_ordinal_data_scaler


# # splitting data

# In[ ]:


X = train_ordinal_data_scaler
y = data['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# # RandomForestClassifier

# In[ ]:


RFC = RandomForestClassifier()
RFC.fit(X_train,y_train)
ypred = RFC.predict(X_test)
print(RFC,":",accuracy_score(y_test,ypred)*100)


# # RandomForestClassifier() : 96.6

# # classification_report

# In[ ]:


print(classification_report(y_test,ypred))


# # Notes 😃😃😃😃
# * Thank for reading my analysis and my classification. 😃😃😃😃
# 
# * If you any questions or advice me please write in the comment . ❤️❤️❤️❤️
# 
# * If anyone has a model with a higher percentage, please tell me 🤝🤝🤝, its will support me .

# # Vote ❤️😃
# ## If you liked my work upvote me 

# # The End...
