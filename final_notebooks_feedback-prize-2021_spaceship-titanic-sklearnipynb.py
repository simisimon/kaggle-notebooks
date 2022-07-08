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


import numpy as np
import pandas as pd 
import sklearn
from sklearn.metrics  import f1_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
#
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.metrics import *
from sklearn.model_selection import *
# from catboost import CatBoost,CatBoostRegressor
from xgboost import XGBRegressor,XGBRFRegressor
np.random.seed(55)
sklearn.random.seed(55)


# In[ ]:


data = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')


# In[ ]:


data = data.sample(frac=1)
test_data = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')


# In[ ]:


from tqdm import tqdm


# In[ ]:


data['HomePlanet'] = data['HomePlanet'].replace({"Earth":1,"Europa":2,"Mars":3})
home_plant_labels = {"Earth":1,"Europa":2,"Mars":3}
data['CryoSleep'] = data['CryoSleep'].replace({True:1,False:2})
cryo_sleep_labels = {"Earth":1,"Europa":2,"Mars":3}
cabins_1 = []
cabins_2 = []
cabins_3 = []
for i in tqdm(range(len(data['Cabin']))):
    cabin = str(data.iloc[i]['Cabin']).split("/")
    if cabin[0] != "nan":
        cabins_1.append(cabin[0])
        cabins_2.append(int(cabin[1]))
        cabins_3.append(cabin[2])
    else:
        cabins_1.append(None)
        cabins_2.append(None)
        cabins_3.append(None)
data['Cabins_1'] = cabins_1
data['Cabins_2'] = cabins_2
data['Cabins_3'] = cabins_3


# In[ ]:


data['Transported'] =data['Transported'].astype('category').cat.codes
data['VIP'] =data['VIP'].astype('category').cat.codes
data['Cabins_1'] =data['Cabins_1'].astype('category').cat.codes
data['Cabins_3'] =data['Cabins_3'].astype('category').cat.codes
data.drop('Cabin',axis=1,inplace=True)
data.drop('Destination',axis=1,inplace=True)
data.drop('Name',axis=1,inplace=True)


# In[ ]:


data['PassengerId'] = data['PassengerId'].astype(int)


# In[ ]:


data['HomePlanet'] = data['HomePlanet'].fillna(data['HomePlanet'].median())
data['CryoSleep'] = data['CryoSleep'].fillna(data['CryoSleep'].median())
data['Age'] = data['Age'].fillna(data['Age'].median())
data['RoomService'] = data['RoomService'].fillna(data['RoomService'].median())
data['FoodCourt'] = data['FoodCourt'].fillna(data['FoodCourt'].median())
data['ShoppingMall'] = data['ShoppingMall'].fillna(data['ShoppingMall'].median())
data['Spa'] = data['Spa'].fillna(data['Spa'].median())
data['VRDeck'] = data['VRDeck'].fillna(data['VRDeck'].median())
data['Cabins_2'] = data['Cabins_2'].fillna(data['Cabins_2'].median())


# In[ ]:


X = data.drop('Transported',axis=1)
y = data['Transported']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.125,shuffle=True)


# In[ ]:


num_columns = list(X.columns)
ct = make_column_transformer(
    (MinMaxScaler(), num_columns),
    (StandardScaler(), num_columns),
    remainder='passthrough'
)

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier


# In[ ]:


model = SVC()
model.fit(X_train,y_train)


# In[ ]:


model.score(X_test,y_test)
preds = model.predict(X_test)
accuracy_score(preds,y_test),precision_score(preds,y_test),f1_score(preds,y_test)
np.mean(cross_val_score(model,X_test,y_test))


# In[ ]:


test_data = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
ids = test_data["PassengerId"]
test_data['HomePlanet'] = test_data['HomePlanet'].replace({"Earth":1,"Europa":2,"Mars":3})
home_plant_labels = {"Earth":1,"Europa":2,"Mars":3}
test_data['CryoSleep'] = test_data['CryoSleep'].replace({True:1,False:2})
cryo_sleep_labels = {"Earth":1,"Europa":2,"Mars":3}
cabins_1 = []
cabins_2 = []
cabins_3 = []
for i in tqdm(range(len(test_data['Cabin']))):
    cabin = str(test_data.iloc[i]['Cabin']).split("/")
    if cabin[0] != "nan":
        cabins_1.append(cabin[0])
        cabins_2.append(int(cabin[1]))
        cabins_3.append(cabin[2])
    else:
        cabins_1.append(None)
        cabins_2.append(None)
        cabins_3.append(None)
test_data['Cabins_1'] = cabins_1
test_data['Cabins_2'] = cabins_2
test_data['Cabins_3'] = cabins_3
test_data['VIP'] =test_data['VIP'].astype('category').cat.codes
test_data['Cabins_1'] =test_data['Cabins_1'].astype('category').cat.codes
test_data['Cabins_3'] =test_data['Cabins_3'].astype('category').cat.codes
test_data.drop('Cabin',axis=1,inplace=True)
test_data.drop('Destination',axis=1,inplace=True)
test_data.drop('Name',axis=1,inplace=True)
test_data['PassengerId'] = test_data['PassengerId'].astype(int)
test_data['HomePlanet'] = test_data['HomePlanet'].fillna(test_data['HomePlanet'].median())
test_data['CryoSleep'] = test_data['CryoSleep'].fillna(test_data['CryoSleep'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['RoomService'] = test_data['RoomService'].fillna(test_data['RoomService'].median())
test_data['FoodCourt'] = test_data['FoodCourt'].fillna(test_data['FoodCourt'].median())
test_data['ShoppingMall'] = test_data['ShoppingMall'].fillna(test_data['ShoppingMall'].median())
test_data['Spa'] = test_data['Spa'].fillna(test_data['Spa'].median())
test_data['VRDeck'] = test_data['VRDeck'].fillna(test_data['VRDeck'].median())
test_data['Cabins_2'] = test_data['Cabins_2'].fillna(test_data['Cabins_2'].median())


# In[ ]:


test_data = ct.transform(test_data)


# In[ ]:


preds = model.predict(np.array(test_data))


# In[ ]:


preds


# In[ ]:


predictions = {
    "PassengerId":[],
    "Transported":[]
}


# In[ ]:


labels = {1:True,0:False}


# In[ ]:


for id,pred in zip(ids,preds):
    predictions['PassengerId'].append(id)
    predictions['Transported'].append(labels[pred])


# In[ ]:


predictions = pd.DataFrame(predictions)


# In[ ]:


predictions.to_csv('./final.csv',index=False)


# In[ ]:




