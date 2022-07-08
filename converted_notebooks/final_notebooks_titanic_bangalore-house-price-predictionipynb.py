#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/bangalore/banglore.csv')
df


# In[ ]:


df.isnull().sum()


# In[ ]:


## check the null values in the dataset
plt.figure(figsize=(15,10))
sns.heatmap(df.isna(),yticklabels=False,cmap='viridis')


# In[ ]:


### checking value counts in the 'area_type'
df.area_type.value_counts().plot.bar(figsize=(12,10),title='Count by area_type')
plt.xlabel('area_type')
plt.ylabel('value_counts')
plt.xticks(rotation=0)
plt.grid()


# In[ ]:


### checking value counts in the 'bath'
plt.figure(figsize=(10,10))
df1=df['bath'].value_counts()
keys=df1.keys().to_list()
count=df1.to_list()
plt.pie(x=count,labels=keys,autopct='%1.1f%%')
plt.show()


# In[ ]:


### checking value counts in the 'area_type'
plt.figure(figsize=(10,10))
explode=(.1,0,0,.1)
df1=df['area_type'].value_counts()
keys=df1.keys().to_list()
count=df1.to_list()
plt.pie(x=count,labels=keys,autopct='%1.1f%%',explode=explode)
plt.show()


# In[ ]:


## checking the value counts of 'size'
plt.figure(figsize = (10,10))
df['size'].value_counts().plot(kind='bar')
plt.xlabel('size')
plt.ylabel('value_count')
plt.grid()


# In[ ]:


df['society'].value_counts()


# In[ ]:


df['total_sqft'].isnull().sum()


# In[ ]:


# Getting the data that are float only


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[ ]:


bool_ts=df["total_sqft"].apply(is_float)
bool_ts


# To extract data which are not float from the given column "total_sqft"

# In[ ]:


# selecting the Non-Float data 

df[bool_ts==False].head(30)


# In[ ]:


# converting the Non-Float range data to Float type

def range_to_sqft(x):
    new=x.split('-')
    if len(new)==2:
        return(float(new[0])+float(new[-1])/2)
    try:
        return float(x)
    except:
        return x


# In[ ]:


# applying the function to  column name 'total_sqft'

df['total_sqft']=df['total_sqft'].apply(range_to_sqft)


# In[ ]:


# Again checking for float

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[ ]:


# checking if float then true else False

bool_ts=df["total_sqft"].apply(is_float)
bool_ts


# In[ ]:


# Selecting only the float

bool_new=df[bool_ts==False]
bool_new


# In[ ]:


# user define function to convert all the measurements to square feets

def str_num(x):
    for i in df['total_sqft']:
        num=str(x)
        if num.endswith('Sq. Meter'):
            return float(i)*10.764
        elif num.endswith('Sq. Yards'):
            return float(i)*9
        elif num.endswith('Cents'):
            return float(i)*435.56
        elif num.endswith('Acres'):
            return float(i)*43560
        elif num.endswith('Perch'):
            return float(i)*272.25
        elif num.endswith('Guntha'):
            return float(i)*1089
        elif num.endswith('Grounds'):
            return float(i)*2400
        else:
            return x   
        


# In[ ]:


# applying the function to the 'total_sqft' column 

df['total_sqft'] = df['total_sqft'].apply(lambda x: str_num(x))

df


# In[ ]:


# converting the datatype of 'total_sqft' from object to float

df['total_sqft']=df['total_sqft'].astype('float64')


# In[ ]:


df.dtypes


# In[ ]:


# 'Society' column can be dropped since it will not put a big impact on the house price since its Secondary to look at.

df.drop('society',axis=1,inplace=True)


# # Checking for Null_ values 

# In[ ]:


null_val=df.isnull().sum()
null_val=pd.DataFrame(null_val)
null_val['percent']=round((null_val/df.shape[0])*100,2)
null_val.sort_values('percent',ascending=False)


# ### Treating null values in 'size'

# 

# In[ ]:


df['size'].unique()


# In[ ]:


df.isnull().sum()


# In[ ]:


# converting all alphabets in string to 'BHK'
def bhk(x):
    i=str(x)
    if i.endswith('Bedroom'):
        return i.replace('Bedroom','BHK')
    elif i.endswith('RK'):
        return i.replace('RK','BHK')
    else:
        return x


# In[ ]:


df['size']=df['size'].apply(lambda x:bhk(x))


# In[ ]:


df['size'].unique()


# In[ ]:


# Replacing all the 'nan' with '0'
df['size'].replace(to_replace = np.nan, value =0,inplace=True)


# In[ ]:


# replacing all the '0' with '0 BHK'
df['size'].replace(to_replace = 0, value ="0 BHK",inplace=True)


# In[ ]:


# all the data is in 'BHK' now without any 'nan'
df['size'].unique()


# In[ ]:


df['size'].isnull().sum()


# In[ ]:


df['size'][0].split(" ")[0]


# ### Creating a Copy as 'df_new'

# In[ ]:


# copy 
df_new=df.copy(deep=True)


# In[ ]:


# converting the old column 'size' to new numerical column 'BHK'
df_new['BHK']=df_new['size'].str.split().apply(lambda x: x[0])


# In[ ]:


# Dropping the column 'size' since not required anymore
df_new.drop('size',axis=1,inplace=True)


# In[ ]:


df_new


# In[ ]:


df_new.isnull().sum()


# In[ ]:


## checking outlier on 'price' with respect to 'balcony'
plt.figure(figsize=(15,10))
sns.boxplot(data=df_new,x='BHK',y='price')
plt.xlabel('BHK')
plt.ylabel('price')
plt.show()


# ### Treating null value in 'location'

# In[ ]:


df_new[df_new.location.isnull()]


# In[ ]:


df_new['location'].mode()


# In[ ]:


df_new['location'].fillna('Whitefield',inplace=True)


# In[ ]:


df_new.location.isnull().sum()


# In[ ]:


df_new.location.value_counts()


# In[ ]:


df_new['location'] = df_new['location'].apply(lambda x: x.strip())   
df_new['location']


# In[ ]:


location_count=df_new.location.value_counts()
location_count


# In[ ]:


location_count_30 = location_count[location_count < 30]
location_count_30


# In[ ]:


df_new['location'] = df_new['location'].apply(lambda x: 'other' if x in location_count_30 else x)


# In[ ]:


df_new['location'].value_counts()


# ### Treating Null Values in 'balcony'

# In[ ]:


df_bal=df_new.pivot_table(values='balcony',index='area_type',aggfunc=np.median)
df_bal


# In[ ]:


bool_bal=df_new.balcony.isnull()
bool_bal


# In[ ]:


df_new.loc[bool_bal,'balcony']=df_new.loc[bool_bal,'area_type'].apply(lambda x: df_bal.loc[x])


# In[ ]:


df_new.balcony.isnull().sum()


# In[ ]:


## checking outlier on 'price' with respect to 'balcony'
plt.figure(figsize=(15,10))
sns.boxplot(data=df_new,x='balcony',y='price')
plt.xlabel('balcony')
plt.ylabel('price')
plt.show()


# ### Treating Null values in "bath"

# In[ ]:


df_bath=df_new.pivot_table(values='bath',index='area_type',aggfunc=np.median)
df_bath


# In[ ]:


bool_bath=df_new.bath.isnull()
bool_bath


# In[ ]:


df_new.loc[bool_bath,'bath']=df_new.loc[bool_bath,'area_type'].apply(lambda x: df_bath.loc[x])


# In[ ]:


df_new.isnull().sum()


# In[ ]:


## checking outlier on 'price' with respect to 'bath'
plt.figure(figsize=(15,10))
sns.boxplot(data=df_new,x='bath',y='price')
plt.xlabel('bath')
plt.ylabel('price')
plt.show()


# ### Converting all the dates to 'Will Be Available' in 'availability' column

# In[ ]:


df_new.availability.unique()


# In[ ]:


df_new['availability']=df_new['availability'].replace('Immediate Possession','Ready To Move')


# In[ ]:


df_new.availability.unique()


# In[ ]:


def changeitem(x):
    if x!='Ready To Move':
        x='will be available'
        return x
    else:
        return 'Ready To Move'


# In[ ]:


df_new['availability']=df_new['availability'].apply(lambda x: changeitem(x))


# In[ ]:


df_new['availability'].unique()


# In[ ]:


# Replacing the unique values to 0 and 1 in 'availability' columns

df_new['availability']=df_new['availability'].replace(['will be available', 'Ready To Move'],[0,1])


# In[ ]:


df_new['availability'].unique()


# 

#  
# 
# count=0
# for i in df_new['location'].unique():
#     df_new.replace(i,count,inplace=True)
#     count+=1

# In[ ]:


df_new


# In[ ]:


df_new.describe()


# 

# In[ ]:


df_new.describe()


# In[ ]:


df_new.dtypes


# In[ ]:


df_new['BHK']=df_new['BHK'].astype('float64')


# In[ ]:


df_new.dtypes


# In[ ]:


## To check the outliers in 'price' with respect to 'bath'
plt.figure(figsize=(8,8))
plt.scatter(x='bath',y='price',data=df_new,color="green")
plt.xlabel('bath')
plt.ylabel('price')
plt.grid()
plt.show()


# In[ ]:


## To check the outliers in 'price' with respect to 'area_type'
plt.figure(figsize=(15,8))
plt.scatter(x='area_type',y='price',data=df_new,color="green")
plt.xlabel('area_type')
plt.ylabel('price')
plt.grid()
plt.show()


# In[ ]:


## To check the outliers in 'price' with respect to 'area_type'
plt.figure(figsize=(15,8))
sns.boxplot(data=df,x='area_type',y='price')
plt.xlabel('area_type')
plt.ylabel('price')
plt.show()


# #### Creating a new column 'Per_sqft_price'

# In[ ]:


df_new['Per_sqft_price']=round((df_new['price']/df_new['total_sqft'])*100000,2)


# ### Treating the Outliers using Standard Deviation  and mean

# In[ ]:


def rmv_outlierBy_std(x):
    out_df = pd.DataFrame()
    for key, subdf in x.groupby('location'):
        mean = np.mean(subdf.Per_sqft_price)
        std = np.std(subdf.Per_sqft_price)
        reduced_df = subdf[(subdf.Per_sqft_price > (mean - std)) & (subdf.Per_sqft_price < (mean + std))]
        out_df = pd.concat([out_df, reduced_df], ignore_index = True)
    return out_df

df4 = rmv_outlierBy_std(df_new)


# ### Now we can remove those 2 BHK apartments whose Per_sqft_price is less than mean Per_sqft_price of 1 BHK apartment

# In[ ]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.Per_sqft_price),
                'std': np.std(bhk_df.Per_sqft_price),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.Per_sqft_price<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df5 = remove_bhk_outliers(df4)


# In[ ]:


df5


# In[ ]:


## to check ouliers in 'price'
sns.boxplot(df.price)
plt.show()


# In[ ]:


## to check ouliers in 'price'
sns.boxplot(df5.price)
plt.show()


# In[ ]:


### value count of 'area_type' with respect to price
plt.figure(figsize=(8,8))
sns.barplot(x='area_type',y='price',data=df5)
plt.grid()
plt.show()


# In[ ]:


### value count of 'location' with respect to 'price'
plt.figure(figsize=(50,25))
sns.barplot(x='location',y='price',data=df5)
plt.xlabel('location')
plt.ylabel('price')
plt.xticks(rotation = 90)
plt.grid()
plt.show()


# In[ ]:


## 
plt.figure(figsize=(15,7)) 
sns.boxplot(data=df5,x=df5['bath'],y=df5['price']) 
plt.xlabel('bath')
plt.ylabel('price')
plt.show()


# In[ ]:


df5.describe()


# In[ ]:


df5.shape


# ## Applying "one hot encoding" on all the categorical columns

# In[ ]:


df_dummies=pd.get_dummies(df5)


# In[ ]:


df_dummies


# In[ ]:


df_dummies.dtypes


# #### Creating a Copy of old variable as Df1_new

# In[ ]:


df1_new=df_dummies.copy(deep=True)


# In[ ]:


df1_new


# ## Model Building

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


# In[ ]:


x=df1_new.drop(['price'],axis=1)
y=df1_new['price']


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)


# In[ ]:


lr=LinearRegression()


# In[ ]:


lr.fit(x_train,y_train)


# In[ ]:


train_pred=lr.predict(x_train)


# In[ ]:


test_pred=lr.predict(x_test)


# In[ ]:


print("RMSE for train",np.sqrt(mean_squared_error(y_train,train_pred)))
print("RMSE for test",np.sqrt(mean_squared_error(y_test,test_pred)))


# In[ ]:


print('r2_score train',r2_score(y_train,train_pred))
print('r2_score test',r2_score(y_test,test_pred))


# #### Standard Scaling 

# In[ ]:


x=df1_new.drop('price',axis=1)
sc=StandardScaler()
x_sc=sc.fit_transform(x)
x_sc=pd.DataFrame(x_sc)
x_sc


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

x=x_sc
y=df1_new['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

RFC=RandomForestRegressor(n_estimators=50,n_jobs=-1,oob_score=True,random_state=1)

RFC.fit(x_train,y_train)

train_pred=RFC.predict(x_train)

test_pred=RFC.predict(x_test)

print('RMSE for train:',np.sqrt(mean_squared_error(y_train,train_pred)))


print('r2_score for train:',r2_score(y_train,train_pred))
print('r2_score for test:',r2_score(y_test,test_pred))


# ## GridSearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV

x=x_sc
y=df1_new['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

param={'max_depth':[5,10,15,20,30],'min_samples_leaf':[10,15,20,50,100]}

gs=GridSearchCV(estimator=RFC,param_grid=param,cv=4,n_jobs=-1,scoring='r2')

gs.fit(x_train,y_train)

gs.best_estimator_
gs_best=gs.best_estimator_

gs_best.fit(x_train,y_train)

train_pred=gs_best.predict(x_train)

test_pred=gs_best.predict(x_test)


print('RMSE for train:',np.sqrt(mean_squared_error(y_train,train_pred)))
print('RMSE for train:',np.sqrt(mean_squared_error(y_test,test_pred)))


print('r2_score for train:',r2_score(y_train,train_pred))
print('r2_score for test:',r2_score(y_test,test_pred))


# ### From all the models that we have performed we can conclude that "Random Forest" is the best model for Bangalore_House_Pricing prediction compared to others and having RMSE=7.83 and Accuracy of training data=99%.

# In[ ]:




