#!/usr/bin/env python
# coding: utf-8

# Content
# 
# Import Libraries
# Load data
# Data Preparation
#   Missing values imputation
#   Feature Engineering
# Modeling
#   Build the model
# Evaluation
#   Model performance
#   Feature importance
#   Who gets the best performing model?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Modelling Algorithms

from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model


# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


# In[ ]:


# get TMDB Box Office Prediction train & test csv files as a DataFrame
train = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/train.csv")
test  = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/test.csv")


# In[ ]:


def plot_correlation_map( df ):
    corr = train.corr()
    _ , ax = plt.subplots( figsize =( 23 , 22 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()


# **Visualization**

# In[ ]:


train.corr()


# In[ ]:


np.count_nonzero(train.budget)


# In[ ]:


train.describe()


# In[ ]:


data=pd.concat([train['budget'],train['revenue']],axis=1)
data.plot.scatter(x='budget',y='revenue',xlim=(0,1e7),ylim=(0,1e8))


# **Training**

# In[ ]:


# Splitting into Test and validation data and feature selection

# Selecting features Budget and Popularity
train_mod = train[{"budget","popularity"}]

# Selecting the first 2001 indices of the training data for training
train_train = train_mod[0:2000]
# Selecting the rest of the training data for validation
train_val= train_mod[2001:2999]

# Obtain labels
train_mod_y = train[{"revenue"}]
train_train_y = train_mod_y[0:2000]
train_val_y= train_mod_y[2001:2999]
train_val_title = train["original_title"][2001:2999]


# In[ ]:


# Check for NaN
if(train_mod.isnull().values.any()):
    print("Too bad, Nan found...")
else :
    print("All right!!! Data ok!")


# In[ ]:


# Initialize and train a linear regression (Lasso) model
model = linear_model.Lasso(alpha=0.1)
model.fit(train_train,train_train_y.values.ravel())


# In[ ]:


# Evaluate on the training data
res = model.predict( train_val)

res2 = res + 1

prediction_vector = [res, res2]


# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluateModels(predictions, ground_truth):
    for prediction in predictions:
        r2 = r2_score(ground_truth, prediction)
        rms = np.sqrt(mean_squared_error(ground_truth, prediction))
        mae = mean_absolute_error(ground_truth, prediction)
        print("R2: ", r2, "RMS: ", rms, "MAE: ", mae)
    
    # Create error array
    prediction_error = ground_truth - predictions
    print(type(prediction_error))
    ax = sns.boxplot(data = np.transpose(prediction_error), orient = 'h')
    #return [r2, rms, mae]
                   


# In[ ]:


# Obtain R2 score (ordinary least square)
evaluateModels(prediction_vector, train_val_y.values.ravel())


# In[ ]:


# Display best predictions
res.shape


# In[ ]:


# Create the table for comparing predictions with labels
absolute_error =  np.abs(res - train_val_y.values.ravel())
relative_error = absolute_error/train_val_y.values.ravel()

evaluation = pd.DataFrame({'Title': train_val_title.values.ravel(), 'budget': train_val['budget'].values.ravel(), 'popularity': train_val['popularity'].values.ravel(),'Prediction': res.round(), 'Actual revenue': train_val_y.values.ravel(), 'Absolute error 1': absolute_error, 'Relative error 1': relative_error})

evaluation.sort_values(by=['Relative error 1'])


# In[ ]:




