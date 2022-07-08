#!/usr/bin/env python
# coding: utf-8

# # **Import pandas & Matplotlib Libraries**

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# # **Read train.csv using Pandas**

# In[ ]:


df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# # **Fill N.A values in your dataframe before using machine learning algorithms**

# In[ ]:


df.isna().sum()
x=df['LotFrontage'].fillna(0)
l=df.drop(columns=['LotFrontage'])
N=pd.concat([l,x],axis=1)
df=N


# # **Remove all objects before preprocessing LabelEncoder**

# In[ ]:


df_exobj=df.select_dtypes(exclude=['object'])


# In[ ]:


df_exobj.isna().sum()
w=df_exobj.fillna(0)
df_exobj=w
w.isnull().sum()


# # **Remove all integer & float before preprocessing LabelEncoder**

# In[ ]:


exc = {'int64','float64'}
df_exnum=df.select_dtypes(exclude=exc)


# In[ ]:


df_exnum


# # **Import Sklearn LabelEncoder**

# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
encode=df_exnum.apply(LabelEncoder().fit_transform)


# # **Concat Two Tables**

# In[ ]:


final_data=pd.concat([df_exobj,encode], axis=1)


# In[ ]:


final_data


# # **Define x and y for train_test_split**

# In[ ]:


x = final_data.drop(columns=['SalePrice'])
x.shape


# In[ ]:


y=final_data['SalePrice']
y.shape


# In[ ]:


y=final_data['SalePrice']
y.shape


# In[ ]:


from sklearn.model_selection import train_test_split

x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.30,random_state=0)


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# # **Import Random Forest from Sklearn and fit training data**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
le = RandomForestClassifier(n_estimators=10)
le.fit(x.iloc[:1168],y.iloc[:1168])


# # **Test Score of your Model**

# In[ ]:


le.score(x,y)


# # **Import Test.csv to predict SalesPrice**

# In[ ]:


test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


test


# # **Fill N.A in Test dataset before predicting**

# In[ ]:


test.isnull().sum()
test=test.fillna(0)


# In[ ]:


ex={'int64','float64'}
test_num=test.select_dtypes(exclude='object')
test_exnum=test.select_dtypes(exclude=ex)
ll=test_exnum.astype('|S')


# # **Preprocess Data before prediction**

# In[ ]:


ls = LabelEncoder()
enco=ll.apply(LabelEncoder().fit_transform)


# In[ ]:


test_final=pd.concat([test_num,enco],axis=1)


# In[ ]:


test_final


# # **Insert Preprocessed Data into Predict**

# In[ ]:


pre=le.predict(test_final)


# In[ ]:


sales_pre=pd.DataFrame({'SalePrice':pre})
output=pd.concat([test_final['Id'],sales_pre],axis=1)


# # **Output export to CSV format**

# In[ ]:


output.to_csv('output.csv')

