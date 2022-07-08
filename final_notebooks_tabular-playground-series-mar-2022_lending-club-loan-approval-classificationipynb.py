#!/usr/bin/env python
# coding: utf-8

# # This project is part of a case study to improve Exploratory Data Analysis, Missing Data handling, Feature Engineering, Deep Learning and Model Creation/Comparison.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Load Data Information and create module to retrieve description in case there is doubts while executing work.
data_info = pd.read_csv('../input/lendingclub-data-sets/lending_club_info.csv', index_col='LoanStatNew')


# In[ ]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[ ]:


feat_info('loan_amnt')


# In[ ]:


#Import data
df = pd.read_csv('../input/lendingclub-data-sets/lending_club_loan_two.csv')


# In[ ]:


#Get an understanding of the dataset and if there is any null cells
df.isnull().sum()


# In[ ]:


df.describe().transpose()


# # Start of Exploratory Data Analysis and find important variables for possible use of Feature Engineering
# 
# ### The goal of project is to predict Loan_Status

# In[ ]:


sns.countplot(x=df['loan_status'])

#Unballanced database. This could possibly make harder to predict if load will be fully paid.


# In[ ]:


#Check out the loan amount
plt.figure(figsize=(16,6))
sns.histplot(data=df,
             x='loan_amnt',
             bins=50,
             alpha=0.5,
             edgecolor=None) 


# In[ ]:


#Look for correlation between variables (Loan_Status)
plt.figure(figsize=(12,8))
sns.heatmap(np.abs(df.corr()),
            annot=True,
            cmap='coolwarm')


# In[ ]:


#Correlation of 0.95 between installment and loan_amnt.
#Probably because installments ammount might increase as you get higher loans. 
#Lets explore that

feat_info('installment')
print('\n')
feat_info('loan_amnt')


# In[ ]:


sns.scatterplot(x='loan_amnt',
                y='installment',
                data=df)

#Feels like a linear correlation with a lot of noise as you increase the Loan Amount.


# In[ ]:


#Let's move on to Loan Status and Loan Amount

sns.boxplot(x='loan_status', y='loan_amnt', data=df)

#Other tha a slightly higher average, there is no strong correlation that the amount borrowed would lead to default.


# In[ ]:


df.groupby('loan_status')['loan_amnt'].describe()
#Data complementing chart above.


# In[ ]:


#Moving on to Grade and Subgrade
#Not a great explanation...

feat_info('grade')
feat_info('sub_grade') 


# In[ ]:


#Let's check out unique values
df['grade'].unique()


# In[ ]:


#Same as above
df['sub_grade'].unique()


# In[ ]:


sns.countplot(x='grade',
              data=df,
              hue='loan_status',
              order = sorted(df['grade'].unique()),
             palette='viridis')

#Very little information on F and G categories.


# In[ ]:


plt.figure(figsize=(14,6))
sns.countplot(x='sub_grade',
              data=df,
              hue='loan_status',
              order=sorted(df['sub_grade'].unique()),
              palette='viridis')
plt.tight_layout()

#Fully paied and Charged Off ration for grades E, F and G are low. Let's dive a little into it.


# In[ ]:


GF_df = df[(df['grade'] == 'G') | (df['grade'] == 'F') | (df['grade'] == 'E')]

sns.countplot(x='sub_grade',
              data=GF_df,
              hue='loan_status',
              order=sorted(GF_df['sub_grade'].unique()))

#If there was more explanation between classes we could look futher into the feature. Trying to understand the reasons why it behaves like it does.


# ### Feature Engineering of Response

# In[ ]:


#Checking featueres under categorical response
df['loan_status'].unique()


# In[ ]:


#Reponse is categorical, so we will create a new column and change the strings to zero or one. 
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1, 'Charged Off':0})


# In[ ]:


#Did if work?
df[['loan_status','loan_repaid']]


# In[ ]:


#Now that we have a categorical column as response, let's check how the features correlate with that.
df.corr()['loan_repaid'].sort_values().iloc[:-1].plot(kind='bar')


# ### That's it for the EDA 

# # Section 2 - Data PreProcessing and Feature Engineering

# In[ ]:


df.head()


# ## Missing Data

# In[ ]:


len(df)


# In[ ]:


df.isnull().sum()


# In[ ]:


100*df.isnull().sum()/len(df)
#See missing data as percentage


# In[ ]:


feat_info('emp_title')
print('\n')
feat_info('emp_length')


# In[ ]:


df['emp_title'].nunique()
#That's far too many job titles to easily fill missing data, or create dummy variables.


# In[ ]:


df['emp_length'].nunique()


# In[ ]:


df['emp_length'].value_counts().plot(kind='bar')


# In[ ]:


order = ['< 1 year',
         '1 year',
         '2 years',
         '3 years',
         '4 years',
         '5 years',
         '6 years',
         '7 years',
         '8 years',
         '9 years',
         '10+ years'
        ]


# In[ ]:


#Let's see if there is a trend between length of employment and loan status.
plt.figure(figsize=(12,5))
sns.countplot(x=df['emp_length'], hue=df['loan_status'], order = order)


# In[ ]:


emp_npaid = df[df['loan_status'] == 'Charged Off'].groupby('emp_length')['loan_status'].count()
emp_paid = df[df['loan_status'] == 'Fully Paid'].groupby('emp_length')['loan_status'].count()

ratio = emp_npaid/(emp_paid + emp_npaid)
ratio.loc[order].plot(kind = 'bar')

#Pretty even across the board. Apparently length of employnment does not correlate with paying the loan


# In[ ]:


#Ratio is similar across length of emplynment, column will be dropped.
df = df.drop(columns=['emp_length','emp_title'], axis=1)


# In[ ]:


df.isna().sum()


# In[ ]:


df[['title','purpose']].value_counts()

#Title column is the same as purpose, but with extra unecessary information. It will be dropped.


# In[ ]:


df = df.drop('title', axis =1)


# In[ ]:


feat_info('revol_util')

100* df['revol_util'].isna().sum()/len(df['revol_util'])

#Revol_Util missing data is less than .1% of total dataset. Column will be dropped later.


# In[ ]:


feat_info('mort_acc')
print('\n')
df['mort_acc'].value_counts()


# In[ ]:


df.corr()['mort_acc'].sort_values(ascending=False)[1:]
#To fill Mort_Acc we could use the median or remove the column. We will try to find a feature that has the most correlation and use that and group by to do it.


# In[ ]:


#Correlation between mort_acc and total_acc is very low, 38%. That means that there is low correlation and can be ignored.
#The study groups the data and fill the NA with the average of that.
#For learning purposes I will follow the process, but the correlation is very low and does not hold on its own.

acc_fill = df.groupby(by='total_acc').mean()['mort_acc']


# In[ ]:


acc_fill


# In[ ]:


def fill_na(total_acc, mort_acc):
    
    if np.isnan(mort_acc):
        return acc_fill[total_acc]
    else:
        return mort_acc


# In[ ]:


df['mort_acc'] = df.apply(lambda x: fill_na(x['total_acc'], x['mort_acc']), axis=1)


# In[ ]:


df.isna().sum()


# In[ ]:


df = df.dropna(axis=0)
#Dropping NA of all rows that are empty. Less than 1% empty.


# In[ ]:


df.isna().sum()


# # Feature Engineering 
# 
# ## Dealing with Strings and Categorical Columns

# In[ ]:


df.select_dtypes(include=[object]).columns


# In[ ]:


df['term'].value_counts()


# In[ ]:


df['term'] = df['term'].apply(lambda term: int(term[:3]))
#Let's turn that into integer


# In[ ]:


df.select_dtypes(include=[object]).columns


# In[ ]:


#Drop Grade because it's part of sub_grade

df.drop(columns=('grade'), axis=1, inplace=True)


# In[ ]:


df.select_dtypes(include=[object]).columns


# ### Get those Dummies. 
# Turn some features into dummie categorical column.

# In[ ]:


grade_dummies = pd.get_dummies(df['sub_grade'], drop_first=True)


# In[ ]:


df = pd.concat([df.drop('sub_grade', axis=1), grade_dummies], axis=1)


# In[ ]:


df.columns


# In[ ]:


other_dummies = pd.get_dummies(df[['verification_status','application_type','initial_list_status', 'purpose']], drop_first=True)


# In[ ]:


df = df.drop(columns=['verification_status','application_type','initial_list_status', 'purpose'], axis=1)
df = pd.concat([df,other_dummies], axis=1)


# In[ ]:


df.head()


# In[ ]:


df['home_ownership'].value_counts()


# In[ ]:


df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'], 'OTHER')


# In[ ]:


df['home_ownership'].value_counts()


# In[ ]:


house_dummies = pd.get_dummies(df['home_ownership'], drop_first=True)


# In[ ]:


df = df.drop(columns=['home_ownership'], axis=1)
df = pd.concat([df,house_dummies], axis=1)


# In[ ]:


df.head()


# In[ ]:


df['zip_code'] = df['address'].str[-5:]
#Extract the ZIP Code


# In[ ]:


df.drop(columns=['address'], inplace=True)


# In[ ]:


df.info()


# In[ ]:


zip_dummies = pd.get_dummies(df['zip_code'], drop_first=True)


# In[ ]:


df = df.drop(columns=['zip_code'], axis=1)
df = pd.concat([df,zip_dummies], axis=1)


# In[ ]:


feat_info('issue_d')

#We would not know if the loan was funded or not, so this feature will be dropped to prevent data leakage.


# In[ ]:


df = df.drop(columns=['issue_d'], axis=1)


# In[ ]:


feat_info('earliest_cr_line')


# In[ ]:


df['earliest_cr_line'].value_counts()


# In[ ]:


df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
#Extract year from earliest credit line


# In[ ]:


df.drop(columns=['earliest_cr_line'], axis=1, inplace=True)


# In[ ]:


df.info()


# # Model Creation
# ## Train Test Split
# 
# The work will be done comparing a NN with early stopping and one with Droupout feature.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


df.drop(columns=['loan_status'], axis=1, inplace=True)


# In[ ]:


X = df.drop(columns=['loan_repaid'], axis=1).values
y = df['loan_repaid'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


# Normalize data


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


X_train = scaler.fit_transform(X_train)


# In[ ]:


X_test = scaler.transform(X_test)


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping


# ## Will do early stopping and later on add drop out to compare results

# In[ ]:


df.shape


# In[ ]:


model = Sequential()
earlystopping = EarlyStopping()


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)


# In[ ]:


# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

model.add(Dense(units=79,activation='relu'))

model.add(Dense(units=39,activation='relu'))

model.add(Dense(units=19,activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))

# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[ ]:


# https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
# https://datascience.stackexchange.com/questions/18414/are-there-any-rules-for-choosing-the-size-of-a-mini-batch

model.fit(x=X_train, 
          y=y_train, 
          epochs=50,
          validation_data=(X_test, y_test),
          batch_size = 256,
          callbacks=[early_stop]
          )


# In[ ]:


model.save('Loan_EarlyStoppingOnly')


# In[ ]:


model_loss = pd.DataFrame(model.history.history)


# In[ ]:


model_loss[['loss', 'val_loss']].plot()


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


pred_ES = (model.predict(X_test) > 0.5).astype("int32")


# In[ ]:


print(confusion_matrix(y_test, pred_ES))
print('\n')
print(classification_report(y_test, pred_ES))


# ## Dropout

# In[ ]:


model = Sequential()

model.add(Dense(units=79,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=39,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=19,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))


# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[ ]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=100,
          validation_data=(X_test, y_test),
          batch_size = 256,
          callbacks=[early_stop]
          )


# In[ ]:


model.save('Loan_Dropout')


# In[ ]:


model_loss_do = pd.DataFrame(model.history.history)


# In[ ]:


model_loss_do.plot()


# In[ ]:


pred_DO = (model.predict(X_test) > 0.5).astype("int32")


# In[ ]:


print(confusion_matrix(y_test, pred_DO))
print('\n')
print(classification_report(y_test, pred_DO))


# ## Test

# In[ ]:


import random
random.seed(100)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]

new_customer


# In[ ]:


(model.predict(new_customer.values.reshape(1,78)) > 0.5).astype("int32")


# In[ ]:


df.iloc[random_ind]['loan_repaid']

