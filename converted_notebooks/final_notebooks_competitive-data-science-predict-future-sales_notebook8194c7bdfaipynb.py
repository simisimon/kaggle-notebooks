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


# load data
items=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
shops=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
cats=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
train=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")


# In[ ]:


train.head(3)


# In[ ]:


temp2=train.groupby('date')['item_id'].count()
df=pd.DataFrame(temp2)
df.columns.map({'item_id':'item_id_cnt'})
df.rename(columns={'item_id':'item_id_cnt'},inplace=True)
df.plot()


# In[ ]:


temp=train[train['date']=='02.01.2013']


# In[ ]:


temp.head(3)


# In[ ]:


temp2=train[train['item_id']==5037][train['shop_id']==5][['date','item_cnt_day']]
#Convert datetime type
import datetime
def str_to_datetime(s):
    split=s.split('.')
    d,m,y=int(split[0]),int(split[1]),int(split[2])
    return datetime.datetime(year=y,month=m,day=d)
temp2['date']=temp2['date'].apply(str_to_datetime).sort_values()
temp2#Now proper format


# The aim is to predict the monthly sales of items in specific shops, given historical data. The sale counts are clipped between 0 and 20.

# In[ ]:


train.head(3)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
f,ax=plt.subplots(figsize=(8,6))
sns.boxplot(x=train['item_price'])


# In[ ]:


f,ax=plt.subplots(figsize=(8,6))
sns.boxplot(x=train['item_cnt_day'])


# In[ ]:


train = train[(train.item_price < 250000 )& (train.item_cnt_day < 1000)]


# In[ ]:


train = train[train.item_price > 0].reset_index(drop = True)
train.loc[train.item_cnt_day < 1, "item_cnt_day"] = 0


# In[ ]:


shops['city']=shops.shop_name.str.split(" ").map(lambda x:x[0])
shops['category']=shops.shop_name.str.split(" ").map(lambda x:x[1])
shops.head(3)


# In[ ]:


category=[]
for cat in shops['category'].unique():
    if len(shops[shops['category']==cat])>=5:
        category.append(cat)
shops.category=shops.category.apply(lambda x: x if (x in category) else 'others')
shops.head(3)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
shops['category']=LabelEncoder().fit_transform(shops.category)
shops['city']=LabelEncoder().fit_transform(shops.city)


# In[ ]:


shops=shops[['shop_id','city','category']]


# In[ ]:


cats['type']=cats.item_category_name.apply(lambda x: x.split(' ')[0]).astype(str)


# In[ ]:


categ=[]
for cat in cats.type.unique():
    if len(cats[cats.type==cat])>=5:
        categ.append(cat)
cats.type=cats.type.apply(lambda x: x if (x in categ) else 'others')
cats.head(3)


# In[ ]:


cats['type_code']=LabelEncoder().fit_transform(cats.type)


# In[ ]:


cats['split']=cats.item_category_name.apply(lambda x: x.split('-'))
cats['sub_type']=cats.split.apply(lambda x: x[1].strip() if len(x)>1 else x[0].strip())


# In[ ]:


cats['sub_type_code']=LabelEncoder().fit_transform(cats.sub_type)
cats=cats[['item_category_id','type_code','sub_type_code']]


# In[ ]:


items.head(10)


# In[ ]:


import re
def name_correction(x):
    x = x.lower() # all letters lower case
    x = x.partition('[')[0] # partition by square brackets
    x = x.partition('(')[0] # partition by curly brackets
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x) # remove special characters
    x = x.replace('  ', ' ') # replace double spaces with single spaces
    x = x.strip() # remove leading and trailing white space
    return x


# In[ ]:


# split item names by first bracket
items["name1"], items["name2"] = items.item_name.str.split("[", 1).str
items["name1"], items["name3"] = items.item_name.str.split("(", 1).str

# replace special characters and turn to lower case
items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
items = items.fillna('0')


# In[ ]:


items["item_name"] = items["item_name"].apply(lambda x: name_correction(x))
items.head(3)


# In[ ]:


items["type"] = items.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )
items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"
items.loc[ items.type == "", "type"] = "mac"
items.type = items.type.apply( lambda x: x.replace(" ", "") )
items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc"
items.loc[ items.type == 'рs3' , "type"] = "ps3"


# In[ ]:


group_sum = items.groupby(["type"]).agg({"item_id": "count"})
group_sum = group_sum.reset_index()
drop_cols = []
for cat in group_sum.type.unique():
    if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40:
        drop_cols.append(cat)
items.name2 = items.name2.apply( lambda x: "other" if (x in drop_cols) else x )
items = items.drop(["type"], axis = 1)


# In[ ]:


items.name2 = LabelEncoder().fit_transform(items.name2)
items.name3 = LabelEncoder().fit_transform(items.name3)

items.drop(["item_name", "name1"],axis = 1, inplace= True)
items.head()


# In[ ]:


train.head(3)


# In[ ]:


train.date_block_num.nunique()


# In[ ]:


from itertools import product
matrix=[]
cols  = ["date_block_num", "shop_id", "item_id"]
for i in range(34):
    sales=train[train.date_block_num==i]
    matrix.append(np.array(list(product( [i], sales.shop_id.unique(), sales.item_id.unique()))))
matrix=pd.DataFrame(np.vstack(matrix),columns=cols)
matrix.head(3)


# In[ ]:


matrix['date_block_num']=matrix['date_block_num'].astype(int)
matrix['shop_id']=matrix['shop_id'].astype(np.int8)
matrix['item_id']=matrix['item_id'].astype(np.int16)
matrix.sort_values(cols,inplace=True)


# In[ ]:


train['revenue']=train['item_cnt_day']*train['item_price']


# In[ ]:


group=train.groupby(cols).agg({'item_cnt_day':['sum']})
group.columns=['item_cnt_month']
group.reset_index(inplace=True)
group


# In[ ]:


matrix=pd.merge(matrix,group,on=cols,how='left')
matrix['item_cnt_month']=matrix['item_cnt_month'].fillna(0).astype(np.float16)


# In[ ]:


test['date_block_num']=34
test["date_block_num"] = test["date_block_num"].astype(np.int8)
test['shop_id']=test['shop_id'].astype(np.int8)
test['item_id']=test['item_id'].astype(np.int16)


# In[ ]:


test.drop('ID',inplace=True,axis=1)


# In[ ]:


test.shape


# In[ ]:


matrix=pd.concat([matrix,test],ignore_index=True,sort=False,keys=cols)
matrix.fillna(0,inplace=True)
matrix.tail(5)


# In[ ]:


matrix[matrix['date_block_num']==34].shape


# In[ ]:


shops.head(2)


# In[ ]:


matrix=pd.merge(matrix,shops,on=['shop_id'],how='left')
matrix.head(4)


# In[ ]:


items.head(2)


# In[ ]:


matrix=pd.merge(matrix,items,on='item_id',how='left')
matrix.head(3)


# In[ ]:


cats.head(2)


# In[ ]:


matrix=pd.merge(matrix,cats,on='item_category_id',how='left')
matrix.head(3)


# In[ ]:


matrix.tail(3)


# In[ ]:


matrix["city"] = matrix["city"].astype(np.int8)
matrix["category"] = matrix["category"].astype(np.int8)
matrix["item_category_id"] = matrix["item_category_id"].astype(np.int8)
matrix["sub_type_code"] = matrix["sub_type_code"].astype(np.int8)
matrix["name2"] = matrix["name2"].astype(np.int8)
matrix["name3"] = matrix["name3"].astype(np.int16)
matrix["type_code"] = matrix["type_code"].astype(np.int8)


# In[ ]:


matrix.head(3)


# In[ ]:


matrix[matrix['date_block_num']==34].shape


# In[ ]:


matrix.head(3)


# In[ ]:


def lag_feature(df,lags,cols ):
    for col in cols:
        print('Adding lag feature in ',col)
        tmp=df[['date_block_num','shop_id','item_id',col]]
        for i in lags:
            shifted=tmp.copy()
            shifted.columns=['date_block_num','shop_id','item_id',col+'_shifted_'+str(i)]
            shifted.date_block_num=shifted.date_block_num+i
            df=pd.merge(df,shifted,on=['date_block_num','shop_id','item_id'],how='left')
    return df


# In[ ]:


matrix=lag_feature(matrix,[1,2,3],["item_cnt_month"])


# In[ ]:


group=matrix.groupby(['date_block_num','item_category_id']).agg({'item_cnt_month':['mean']})
group.columns=['date_item_cat_avg']
group.reset_index(inplace=True)
matrix=pd.merge(matrix,group,on=['date_block_num','item_category_id'],how='left')
matrix['date_item_cat_avg']=matrix['date_item_cat_avg'].astype(np.float16)


# In[ ]:


matrix=lag_feature(matrix,[1,2,3],['date_item_cat_avg'])
matrix.drop(['date_item_cat_avg'],axis=1,inplace=True)


# In[ ]:


group=matrix.groupby(['date_block_num','category']).agg({'item_cnt_month':['mean']})
group.columns=['date_cat_avg']
group.reset_index(inplace=True)
matrix=pd.merge(matrix,group,on=['date_block_num','category'],how='left')
matrix['date_cat_avg']=matrix['date_cat_avg'].astype(np.float16)


# In[ ]:


matrix=lag_feature(matrix,[1,2,3],['date_cat_avg'])
matrix.drop(['date_cat_avg'],axis=1,inplace=True)


# In[ ]:


group=matrix.groupby(['date_block_num']).agg({'item_cnt_month':['mean']})
group.columns=['date_avg_item_cnt']
group.reset_index(inplace=True)
matrix=pd.merge(matrix,group,on='date_block_num',how='left')
matrix.date_avg_item_cnt = matrix["date_avg_item_cnt"].astype(np.float16)


# In[ ]:


matrix=lag_feature(matrix,[1,2,3],["date_avg_item_cnt"])
matrix.drop(['date_avg_item_cnt'],inplace=True,axis=1)


# In[ ]:


group=matrix.groupby(['date_block_num','item_id']).agg({'item_cnt_month':['mean']})
group.columns=['date_item_avg_item_cnt']
group.reset_index(inplace=True)
matrix=pd.merge(matrix,group,on=['date_block_num','item_id'],how='left')
matrix.date_item_avg_item_cnt=matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix.head(3)


# In[ ]:


matrix=lag_feature(matrix,[1,2,3],['date_item_avg_item_cnt'])
matrix.drop(['date_item_avg_item_cnt'],inplace=True,axis=1)


# In[ ]:


group=matrix.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_month':['mean']})
group.columns=['date_shop_item_avg_item_cnt']
group.reset_index(inplace=True)
matrix=pd.merge(matrix,group,on=['date_block_num','shop_id','shop_id'],how='left')
matrix.date_shop_item_avg_item_cnt=matrix['date_shop_item_avg_item_cnt'].astype(np.float16)
matrix.head(3)
matrix=lag_feature(matrix,[1,2,3],['date_shop_item_avg_item_cnt'])
matrix.drop(['date_shop_item_avg_item_cnt'],inplace=True,axis=1)


# In[ ]:


group=matrix.groupby(['date_block_num','city','item_id']).agg({'item_cnt_month':['mean']})
group.columns=['date_city_item_avg_item_cnt']
group.reset_index(inplace=True)
matrix=pd.merge(matrix,group,on=['date_block_num','city','shop_id'],how='left')
matrix.date_city_item_avg_item_cnt=matrix['date_city_item_avg_item_cnt'].astype(np.float16)
matrix.head(3)
matrix=lag_feature(matrix,[1,2,3],['date_city_item_avg_item_cnt'])
matrix.drop(['date_city_item_avg_item_cnt'],inplace=True,axis=1)


# In[ ]:


group=train.groupby(['item_id']).agg({'item_price':['mean']})
group.columns=['item_id_price_avg']
group.reset_index(inplace=True)
matrix=pd.merge(matrix,group,on=['item_id'],how='left')
matrix['item_id_price_avg']=matrix['item_id_price_avg'].astype(np.float16)


# In[ ]:


group=train.groupby(['date_block_num','item_id']).agg({'item_price':['mean']})
group.columns=['date_item_id_price_avg']
group.reset_index(inplace=True)
matrix=pd.merge(matrix,group,on=['date_block_num','item_id'],how='left')
matrix['date_item_id_price_avg']=matrix['date_item_id_price_avg'].astype(np.float16)


# In[ ]:


matrix=lag_feature(matrix,[1,2,3],['date_item_id_price_avg'])


# In[ ]:


matrix.head(2)


# In[ ]:


for i in [1,2,3]:
    matrix['delta_price_shifted_'+str(i)]=(matrix['date_item_id_price_avg_shifted_'+str(i)]-matrix['item_id_price_avg'])/matrix['item_id_price_avg']
features_to_drop = ["item_id_price_avg", "date_item_id_price_avg"]
matrix.drop(features_to_drop, axis = 1, inplace = True)


# In[ ]:


matrix = matrix[matrix["date_block_num"] > 3]


# In[ ]:


from xgboost import XGBRegressor
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[ ]:


X_train = matrix[matrix.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = matrix[matrix.date_block_num < 33]['item_cnt_month']
X_valid = matrix[matrix.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = matrix[matrix.date_block_num == 33]['item_cnt_month']
X_test = matrix[matrix.date_block_num == 34].drop(['item_cnt_month'], axis=1)


# In[ ]:


Y_train = Y_train.clip(0, 20)
Y_valid = Y_valid.clip(0, 20)


# In[ ]:


model = XGBRegressor(
    max_depth=10,
    n_estimators=1000,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1,
    seed=42)
model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 20)


# In[ ]:


Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('submission.csv', index=False)


# In[ ]:


from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

plot_features(model, (10,14))


# In[ ]:




