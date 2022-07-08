#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Lets import the library avaliable to analysis

import pandas as pd # data processing, pd.read_csv
import numpy as np # algebra
import matplotlib.pyplot as plt
import seaborn as sns

color=sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment=None
pd.options.display.max_columns = 999


# ##### listing out avaliable files under all folder

# In[ ]:


ls - all


# In[ ]:


train_df=pd.read_csv("../input/train_2016_v2.csv",parse_dates=['transactiondate'])

# train_df=pd.read_csv("C:/Users/raundral/MyPythonLab/Case study/Zillow price estimates/all/train_2016_v2.csv",parse_dates=['transactiondate'])


# In[ ]:


train_df.head(10)
train_df.dtypes
train_df.shape


# In[ ]:


train_df.head(10)


# In[ ]:


# our target variable is logerror , let analise the bit more

# plt.figure(figsize=(10,10))

plt.scatter(range(train_df.shape[0]),np.sort(train_df.logerror.values))

plt.xlabel('Index',fontsize=12)
plt.ylabel('Logerr', fontsize=12)

plt.show()


# In[ ]:





# In[ ]:


ulimit=np.percentile(train_df.logerror.values,99)
llimit=np.percentile(train_df.logerror.values,1)

train_df['logerror'].loc[train_df['logerror']>ulimit]=ulimit
train_df['logerror'].loc[train_df['logerror']<llimit]=llimit

#plt.figure(figsize=(20,10))

sns.distplot(train_df.logerror.values,bins=50,kde=False)

plt.xlabel('Logerror',fontsize=20)

plt.show()



# * Nice Normal distribution

# In[ ]:


## Understand how TransactionDate on this 

train_df['transactiondate_month']=train_df['transactiondate'].dt.month

cnt_srs=train_df['transactiondate_month'].value_counts()

sns.barplot(cnt_srs.index,cnt_srs.values)

plt.xticks(rotation='vertical')

plt.xlabel('Month of Transaction', fontsize=12)
plt.ylabel('Number of occures',fontsize=12)

plt.show()


# In[ ]:


(train_df['parcelid'].value_counts().reset_index())['parcelid'].value_counts()


# ## properties 2016

# In[ ]:


prop_df=pd.read_csv('../input/properties_2016.csv')


# In[ ]:


prop_df.shape


# In[ ]:


prop_df.head(5)


# In[ ]:





# In[ ]:


missing_df=prop_df.isnull().sum(axis=0).reset_index()


# In[ ]:


missing_df.columns=['column_name','missing_count']


# In[ ]:


missing_df=missing_df.loc[missing_df['missing_count']>0] 
missing_df=missing_df.sort_values(by='missing_count')

ind=missing_df.shape[0]




# In[ ]:


fig, ax = plt.subplots(figsize=(12,18))

ax.barh(np.arange(ind),missing_df.missing_count.values)

ax.set_yticks(np.arange(ind))
ax.set_yticklabels(missing_df.missing_count.values,rotation='horizontal')
ax.set_xlabel("Number of missing values")
ax.set_title("Number of missing values in each columns")

plt.show()


# In[ ]:


sns.jointplot(x=prop_df.latitude.values,y=prop_df.longitude.values)

plt.xlabel('Longitude',fontsize=12)
plt.ylabel('Latitude',fontsize=12)
plt.show()


# From the data page, *we are provided with a full list of real estate properties in three counties (Los Angeles, Orange and Ventura, California) data in 2016.*
# 
# We have about 90,811 rows in train but we have about 2,985,217 rows in properties file. So let us merge the two files and then carry out our analysis. 

# In[ ]:


train_df=pd.merge(train_df,prop_df,on='parcelid',how='left')

train_df.shape


# In[ ]:


train_df.shape


# Checking data types different types od data

# In[ ]:


pd.options.display.max_rows=50

dtype_df=train_df.dtypes.reset_index()

dtype_df.columns=['Count','Column_type']

dtype_df


# In[ ]:


dtype_df.groupby('Column_type').aggregate('count').reset_index()


# Lets understand the missing values in merged dataframe

# In[ ]:


missing_df=train_df.isnull().sum(axis=0).reset_index()
missing_df.columns=['column_name', 'missing_count']

missing_df['missing_ratio']=missing_df['missing_count'] / train_df.shape[0]

missing_df.loc[missing_df['missing_count']>0.999]



# Four columns has 99.9% of missing Values 

# We have more float type columns so lets look into how these are correlated to target variables (logerror )

# Let impute the missing values using mean for missing values

# In[ ]:


train_df_mean=train_df.mean(axis=0)

train_df_new=train_df.fillna(train_df_mean)

# Now look at how correlation Coff correlated with of each of these variables

# x_cols=[col for col in train_df_new.columns if col not in ['logerror'] if train_df_new[col].dtype=='float64']

x_cols = [col for col in train_df_new.columns if col not in ['logerror'] if train_df_new[col].dtype=='float64']

labels=[]
values=[]
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(train_df_new[col].values,train_df_new.logerror.values)[0,1])
corr_df=pd.DataFrame({'col_labels':labels,'corr_values':values})
corr_df=corr_df.sort_values(by='corr_values')

ind=np.arange(len(labels))

fig, ax = plt.subplots(figsize=(12,20))

ax.barh(ind,np.array(corr_df.corr_values.values),color='y')

ax.set_yticks(ind)

ax.set_yticklabels(corr_df.col_labels.values,rotation='horizontal')

ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
#autolabel(rects)
plt.show()


# In[ ]:


corr_zero_cols = ['assessmentyear', 'storytypeid', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'poolcnt', 'decktypeid', 'buildingclasstypeid']
for col in corr_zero_cols:
    print(col, len(train_df_new[col].unique()))


# Let us take high correlated values and do some analysis

# In[ ]:


corr_df_sel=corr_df.loc[(corr_df['corr_values']>0.02) | (corr_df['corr_values']<-0.01)]

corr_df_sel


# In[ ]:


cols_to_use=corr_df_sel.col_labels.tolist()

tem_df = train_df[cols_to_use]

corrmat=tem_df.corr(method='spearman')

fig,ax=plt.subplots(figsize=(8,8))

sns.heatmap(corrmat,vmax=1.,square=True)

plt.title("Important variable corrlationMap",fontsize=20)
plt.show()


# Important variables themself have highly corrlated !! Lets understand each variable and analyse

# ### FinishedSquarefeet12

# Let us seee how the finished square feet 12 varies with the log error.

# In[ ]:


col = "finishedsquarefeet12"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=train_df.finishedsquarefeet12.values, y=train_df.logerror.values, size=10, color=color[4])
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Finished Square Feet 12', fontsize=12)
plt.title("Finished square feet 12 Vs Log error", fontsize=15)
plt.show()


# Seems the range of logerror narrows down with increase in finished square feet 12 variable. Probably larger houses are easy to predict?
# 
# **Calculated finished square feet:**

# In[ ]:


col = "calculatedfinishedsquarefeet"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=train_df.finishedsquarefeet12.values, y=train_df.logerror.values, size=10, color=color[4])
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('calculatedfinishedsquarefeet', fontsize=12)
plt.title("calculatedfinishedsquarefeet Vs Log error", fontsize=15)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="bathroomcnt", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Bathroom', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Bathroom count", fontsize=15)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x="bathroomcnt", y="logerror", data=train_df)
plt.ylabel('Log error', fontsize=12)
plt.xlabel('Bathroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("How log error changes with bathroom count?", fontsize=15)
plt.show()


# In[ ]:


train_df_latest=train_df_new


# In[ ]:


train_df_new.columns


# We had an understanding of important variables from the univariate analysis. But this is on a stand alone basis and also we have linearity assumption. Now let us build a non-linear model to get the important variables by building Extra Trees model.

# In[ ]:





# In[ ]:


train_y=train_df_new['logerror'].values

cat_cols=["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]

train_df=train_df_new.drop(['parcelid', 'logerror', 'transactiondate', 'transactiondate_month'] + cat_cols,axis=1)

feat_names=train_df.columns.values


# In[ ]:


from sklearn import ensemble

model=ensemble.ExtraTreesRegressor(n_estimators=25,max_depth=30,max_features=0.3,n_jobs=-1,random_state=0)

model.fit(train_df,train_y)


# plotting the importance features

importance = model.feature_importances_
std=np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
indices=np.argsort(importance)[::-1][:20]

plt.bar(range(len(indices)), importance[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()


# Seems "tax amount" is the most importanct variable followed by "structure tax value dollar count" and "land tax value dollor count"

# In[ ]:





# In[ ]:


import xgboost as xgb
xgb_params={
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}

dtrain=xgb.DMatrix(train_df,train_y,feature_names=train_df.columns.values)

model=xgb.train(dict(xgb_params,silent=0),dtrain,num_boost_round=50)

# plotting

fig,ax=plt.subplots(figsize=(20,20))
xgb.plot_importance(model,max_num_features=50,height=.8,ax=ax)

plt.show()

