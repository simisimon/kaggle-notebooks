#!/usr/bin/env python
# coding: utf-8

# # 🚜 Predicting the sale price of bulldozers using machine learning 🚜
# 
# In this notebook, we´re going to go through an example of machine learning project with the goal of predicting the sale price of bulldozers.
# 
# ## 1. Problem definition
# 
#  > How well can we predict the future sale price of a bulldozer, given it's characteristics and previous examples of how much similar bulldozers have been sold for?
#  
#  
# 
# ## 2. Data
# 
# The data is downloaded from the kaggle Bluebook for Bulldozers competition:https://www.kaggle.com/c/bluebook-for-bulldozers
# 
# There are 3 data sets:
# 
# * Train.csv is the training set, which contains data through the end of 2011.
# * Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
# * Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.
# 
# 
# ## 3. Evaluation
# 
# The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.
# 
# For more on the evaluation of this project check: https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation
# 
# Note: The goal for most regression evaluation metrics is to minimize the error. For example, our goal will be to build a machine learning model which minimises RMLSE.
# 
# ## 4. Features
# 
# Kaggle provides a data dictionary detailing all of the features of the dataset.
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[ ]:


# import data

df = pd.read_csv('../input/bluebook-for-bulldozers/TrainAndValid.csv', low_memory=False)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,8))
ax.scatter(df['saledate'][:1000], df['SalePrice'][:1000])


# In[ ]:


df['SalePrice'].plot.hist()


# ### Parsing dates
# 
# When we work with time series data, we want to enrich the time & date components as much as possible.
# 
# 
# We can do that by telling pandas which of our columns has dates in it using `parse_dates` parameter

# In[ ]:


# import data again but this time parse dates
df = pd.read_csv('../input/bluebook-for-bulldozers/TrainAndValid.csv', parse_dates=['saledate'], low_memory=False)


# In[ ]:


df['saledate'].dtype


# In[ ]:


fig, ax = plt.subplots(figsize=(15,8))
ax.scatter(df['saledate'][:1000], df['SalePrice'][:1000]);


# In[ ]:


df.head().T


# In[ ]:


df['saledate'].head(20)


# ### Sort dataframe by saledate
# 
# When working with time series data, it's a good idea to sort it by date.
# 

# In[ ]:


df.sort_values(by=['saledate'], inplace=True, ascending=True)


# ### Make a copy of the original DataFrame
# 
# We've make a copy of the original dataframe so when we manipulate the copy, we've still got our original data.

# In[ ]:


df_tmp = df.copy()


# In[ ]:


### Add datetime parameters for `saledate` parameter 


# In[ ]:


df_tmp['saleYear'] = df_tmp['saledate'].dt.year
df_tmp['saleMonth'] = df_tmp['saledate'].dt.month
df_tmp['saleDay'] = df_tmp['saledate'].dt.day
df_tmp['saleDayOfWeek'] = df_tmp['saledate'].dt.dayofweek
df_tmp['saleDayOfYear'] = df_tmp['saledate'].dt.dayofyear


# In[ ]:


df_tmp.head().T


# In[ ]:


# Now we're enrich our dataframe with date time features, can drop saletime

df_tmp.drop('saledate', axis=1, inplace=True)


# In[ ]:


df_tmp['state'].value_counts()


# ## 5. Modelling 
# 
# We've done enough EDA (we could always do more) but let's start to do some model-driven EDA

# In[ ]:


# # Let's build a machine learning model 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1, random_state=42)

# X = df_tmp.drop('SalePrice', axis=1)
# y = df_tmp['SalePrice']

# model.fit(X, y)


# In[ ]:


df['UsageBand'].head()


# ### Filling missing values
# 
# #### Fill numeric missing values first

# In[ ]:


for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[ ]:


for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content) and pd.isnull(content).sum():
        print(label)


# In[ ]:


# fill with median
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content) and pd.isnull(content).sum():
        df_tmp[label + '_is_missing'] = pd.isnull(content)
        df_tmp[label] = content.fillna(content.median())


# In[ ]:


for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content) and pd.isnull(content).sum():
        print(label)


# ### convert string to categories
# 
# One way can turn all of our data into numbers is by converting them into pandas categories.

# In[ ]:


for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content) and pd.isnull(content).sum():
        print(label)


# In[ ]:


# turn strings to category
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype('category').cat.as_ordered()


# In[ ]:


df_tmp.info()


# In[ ]:


df_tmp.isna().sum()


# ### save

# In[ ]:


df_tmp.to_csv('train_tmp.csv', index=False)


# In[ ]:


df_tmp = pd.read_csv('train_tmp.csv', low_memory=False)
df_tmp.head().T


# In[ ]:


# turn strings to category
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_tmp[label] = content.astype('category').cat.as_ordered()


# In[ ]:


df_tmp.info()


# ### Filling and turning categorical variables into numbers

# In[ ]:


for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[ ]:


for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[ ]:


# Turn categorical variables into numbers and fill missing
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to indicate whether sample had missing value
        df_tmp[label+'_is_missing']  = pd.isnull(content)
        # Turn categories into numbers and add + 1
        df_tmp[label] = pd.Categorical(content).codes + 1
        #print(label)


# In[ ]:


pd.Categorical(df_tmp['state']).codes


# In[ ]:


df_tmp.info()


# In[ ]:


df_tmp.isna().sum()


# In[ ]:


df_tmp.head()


# In[ ]:


df_tmp['state'].value_counts()


# Now that all of data is numeric as well as our dataframe has no missing values, we should be able to build a machine learning model.

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = RandomForestRegressor(n_jobs=-1, random_state=42)\nX = df_tmp.drop('SalePrice', axis=1)\ny = df_tmp['SalePrice']\nmodel.fit(X, y)\n")


# In[ ]:


# Score the model

model.score(X, y)


# ### Splitting data into train/validation sets

# In[ ]:


df_tmp['saleYear']


# In[ ]:


df_tmp['saleYear'].value_counts()


# In[ ]:


df_val = df_tmp[df_tmp['saleYear'] == 2012]

df_train = df_tmp[df_tmp['saleYear'] != 2012]

len(df_val), len(df_train)


# In[ ]:


df_val


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Split data and X & y\nX_train, y_train = df_train.drop('SalePrice', axis=1), df_train['SalePrice']\nX_valid, y_valid = df_val.drop('SalePrice', axis=1), df_val['SalePrice']\n\nmodel.fit(X_train, y_train)\n\nmodel.score(X_valid, y_valid)\n")


# In[ ]:


# Create own evaluation function
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rlmse(y_test, y_preds):
    
    """
    Calculates root mean squared log error between predictions and true labels
    """
    
    return np.sqrt(mean_squared_log_error(y_test, y_preds))
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {
        "Training MAE":mean_absolute_error(y_train, train_preds),
        "Valid MAE" : mean_absolute_error(y_valid, val_preds),
        "training RLMSE" : rlmse(y_train, train_preds),
        'Valid RLMSE' : rlmse(y_valid, val_preds),
        'Training R^2' : r2_score(y_train, train_preds),
        'Valid R^2' : r2_score(y_valid, val_preds)
        
    }
    return scores


# In[ ]:


show_scores(model)


# ## Testing our model on a subset (to tune the hyperparameters)

# In[ ]:


# %%time
# model = RandomForestRegressor(n_jobs=-1, random_state=42)

# model.fit(X_train, y_train)


# In[ ]:


model = RandomForestRegressor(n_jobs=-1 , random_state=42, max_samples=10000)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit(X_train, y_train)\n')


# In[ ]:


show_scores(model)


# ### Hyperparameter tunning with RandomizedSearchCV

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import RandomizedSearchCV\n\nrf_grid = {\n   'n_estimators':np.arange(10, 100, 10),\n    'max_depth' : [None, 3, 5, 10],\n    'min_samples_split' : np.arange(2,20,2),\n    'min_samples_leaf' : np.arange(2,20,2),\n    'max_features' : [0.5, 1, 'sqrt', 'auto'],\n    'max_samples' : [10000]\n}\n\n\nrs_rf = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1, random_state=42), \n                           param_distributions=rf_grid, \n                           n_iter=5, \n                           cv=5, \n                           verbose=True)\n\nrs_rf.fit(X_train, y_train)\n")


# In[ ]:


rs_rf.score(X_valid, y_valid)


# In[ ]:


rs_rf.best_params_


# In[ ]:


show_scores(rs_rf)


# ### Train a model with the best hyperparameters 
# 
# **Note:** These was found after 100 iterations of `RandomizedSearchCV`

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Most ideal hyperparameters\n\nideal_model = RandomForestRegressor(n_estimators=40,\n                                    min_samples_leaf=1,\n                                    min_samples_split=14,\n                                    max_features=0.5,\n                                    n_jobs=1,\n                                    max_samples=None,\n                                    random_state=42\n                                   )\n\nideal_model.fit(X_train, y_train)\n')


# In[ ]:


show_scores(ideal_model)


# ## Make predicitons on test data

# In[ ]:


df_test = pd.read_csv('../input/bluebook-for-bulldozers/Test.csv', low_memory=False, parse_dates=['saledate'])
df_test


# # Preprocessing the data (TEST FORMAT)

# In[ ]:


def preprocess_data(df):
    """ 
    Performs transformations on df and returns transformed df
    """
    df['saleYear'] = df['saledate'].dt.year
    df['saleMonth'] = df['saledate'].dt.month
    df['saleDay'] = df['saledate'].dt.day
    df['saleDayOfWeek'] = df['saledate'].dt.dayofweek
    df['saleDayOfYear'] = df['saledate'].dt.dayofyear
    df.drop('saledate', axis=1, inplace=True)
    
    #numeric data missing values
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content) and pd.isnull(content).sum():
            df[label + '_is_missing'] = pd.isnull(content)
            df[label] = content.fillna(content.median())
            
    # Turn categorical variables into numbers and fill missing
    for label, content in df.items():
        if not pd.api.types.is_numeric_dtype(content):
            # Add binary column to indicate whether sample had missing value
            df[label+'_is_missing']  = pd.isnull(content)
            # Turn categories into numbers and add + 1
            df[label] = pd.Categorical(content).codes + 1
            #print(label)
    return df


# In[ ]:


df_test = preprocess_data(df_test)

df_test.head().T


# In[ ]:


df_test.isna().sum()


# In[ ]:


# We can find how the colums differ using sets

set(X_train.columns) - set(df_test.columns)


# In[ ]:


df_test['auctioneerID_is_missing'] = False


# In[ ]:


test_preds = ideal_model.predict(df_test)


# In[ ]:


test_preds


# # Format predictions Kaggle

# In[ ]:


df_preds = pd.DataFrame()
df_preds['SalesID'] = df_test['SalesID']
df_preds['SalesPrice'] = test_preds
df_preds.head()


# In[ ]:


#Export data 
df_preds.to_csv('test_predictions.csv', index=False)


# ## Feature importance

# In[ ]:


ideal_model.feature_importances_


# In[ ]:


def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({'features':columns, 'features_importances': importances})
          .sort_values('features_importances', ascending=False)
          .reset_index(drop=True))
    
    # Plot
    fig, ax = plt.subplots()
    ax.barh(df['features'][:n], df['features_importances'][:n])
    ax.set_ylabel('Features')
    ax.set_xlabel('Feature importance')
    ax.invert_yaxis()
    
    


# In[ ]:


plot_features(X_train.columns, ideal_model.feature_importances_, n=5)

