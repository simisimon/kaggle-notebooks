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


train_data = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
train_data.head()


# In[ ]:


test_data = pd.read_csv('/kaggle/input/DontGetKicked/test.csv')
test_data.head()


# In[ ]:


print("Length of train data " +str(len(train_data)))
print("Length of test data " +str(len(test_data)))


# In[ ]:


train_data.info()


# In[ ]:





# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


train_data['IsBadBuy'].value_counts()


# In[ ]:


train_data['Model'].value_counts()


# In[ ]:


train_data.drop('Model', axis =1, inplace=True)
test_data.drop('Model', axis =1, inplace=True)


# In[ ]:


train_data.drop('Trim', axis =1, inplace=True)
test_data.drop('Trim', axis =1, inplace=True)


# In[ ]:


train_data.drop('SubModel' ,axis =1, inplace=True)
test_data.drop('SubModel' , axis =1, inplace=True)


# In[ ]:


train_data.Color.value_counts()


# In[ ]:


test_data.Color.value_counts()


# In[ ]:


train_data.Color.fillna(value='Color_Unknown', inplace=True)
test_data.Color.fillna(value='Color_Unknown', inplace=True)


# In[ ]:


print(train_data.Color.isnull().sum())
print(test_data.Color.isnull().sum())


# In[ ]:


train_data.Transmission.value_counts()


# In[ ]:


test_data.Transmission.value_counts()
train_data[train_data.Transmission =='Manual']
train_data['Transmission'].replace("Manual","MANUAL",inplace=True)


# In[ ]:


train_data.drop('WheelTypeID', axis =1, inplace=True)
test_data.drop('WheelTypeID', axis =1, inplace=True)


# In[ ]:


train_data.WheelType.fillna(value='WheelType_unk', inplace=True)
test_data.WheelType.fillna(value='WheelType_unk', inplace=True)


# In[ ]:


train_data.Nationality.fillna(value='Nationality_Unk', inplace=True)
test_data.Nationality.fillna(value='Nationality_Unk', inplace=True)


# In[ ]:


train_data.Size.fillna(value='Size_Unk', inplace=True)
test_data.Size.fillna(value='Size_Unk', inplace=True)


# In[ ]:


train_data.TopThreeAmericanName.fillna(value='Top_Unk', inplace=True)
test_data.TopThreeAmericanName.fillna(value='top_Unk', inplace=True)


# In[ ]:





# In[ ]:


train_data.PRIMEUNIT.fillna(value='Prime_Unk', inplace=True)
test_data.PRIMEUNIT.fillna(value='Prime_Unk', inplace=True)


# In[ ]:


train_data.AUCGUART.replace("AGREEN", "GREEN", inplace=True)
test_data.AUCGUART.replace("ARED", "RED", inplace=True)


# In[ ]:


train_data.AUCGUART.fillna(value='AUC_Unk', inplace=True)
test_data.AUCGUART.fillna(value='AUC_Unk', inplace=True)


# In[ ]:


train_data.drop(['MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',
                'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
                'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice',
                'MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice'],
               inplace=True,axis=1)
test_data.drop(['MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',
                'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
                'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice',
                'MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice'],
               inplace=True,axis=1)


# In[ ]:


train_data.drop('PurchDate', axis=1, inplace=True)
test_data.drop('PurchDate', axis=1, inplace=True)


# In[ ]:


train_data.shape


# In[ ]:


test_data.shape


# In[ ]:


train_data.columns


# In[ ]:


train_data.dtypes


# In[ ]:


train_data.columns


# In[ ]:


train_data.drop(['RefId','IsBadBuy'],axis=1).dtypes!='object'


# In[ ]:


not_categorical=train_data.drop(['RefId','IsBadBuy'],axis=1).columns[train_data.drop(['RefId','IsBadBuy'],axis=1).dtypes!='object']


# In[ ]:


for i in not_categorical:
    maximum=np.max(train_data[i])
    train_data[i]=train_data[i]/maximum
    maximum_test=np.max(test_data[i])
    test_data[i]=test_data[i]/maximum_test


# In[ ]:


train_data[not_categorical].head()


# In[ ]:


test_data[not_categorical].head()


# In[ ]:


categorical=train_data.drop(['RefId','IsBadBuy'],axis=1).columns[train_data.drop(['RefId','IsBadBuy'],axis=1).dtypes=='object']


# In[ ]:


train_data[categorical[0]]


# In[ ]:


pd.get_dummies(train_data[categorical[0]])


# In[ ]:


for i in categorical:
    dummies=pd.get_dummies(train_data[i])
    dummies.columns=str(i)+'_'+dummies.columns
    train_data=pd.concat([train_data,dummies],axis=1)
    train_data.drop(i,inplace=True,axis=1)
    dummies=pd.get_dummies(test_data[i])
    dummies.columns=str(i)+'_'+dummies.columns
    test_data=pd.concat([test_data,dummies],axis=1)
    test_data.drop(i,inplace=True,axis=1)


# In[ ]:


train_data.head()


# In[ ]:


train_data.shape


# In[ ]:


for i in train_data.drop('IsBadBuy',axis=1).columns:
    if i not in test_data.columns:
        test_data[i]=np.zeros(len(test_data))


# In[ ]:


for i in test_data.columns:
    if i not in train_data.columns:
        train_data[i]=np.zeros(len(train_data))


# In[ ]:


train_data.shape


# In[ ]:


test_data.shape


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


test_data= test_data[train_data.drop("IsBadBuy",axis=1).columns]


# In[ ]:


print(train_data.shape)
print(test_data.shape)


# In[ ]:


train_data.columns


# In[ ]:


from sklearn.model_selection import train_test_split
X = train_data.drop(['RefId', 'IsBadBuy'], axis =1)
y = train_data['IsBadBuy']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)


# In[ ]:


print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)


# In[ ]:


knn.score(X_test, y_test)


# In[ ]:


predict = knn.predict(test_data.drop('RefId', axis = 1))


# In[ ]:


Submission=pd.DataFrame(data=predict,columns=['IsBadBuy'])
Submission.head()


# In[ ]:


Submission['RefId']=test_data['RefId']
Submission.set_index('RefId',inplace=True)


# In[ ]:


Submission.head()
Submission.to_csv('Submission.csv')


# In[ ]:




