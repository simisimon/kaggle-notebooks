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


from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
import torch

import time

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#processing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[ ]:


train_data = pd.read_csv('../input/tabular-playground-series-aug-2021/train.csv')
train_data.head()


# In[ ]:


test_data = pd.read_csv('../input/tabular-playground-series-aug-2021/test.csv')
test_data.head()


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# In[ ]:


train_data.isnull()


# In[ ]:


train_data.describe()


# In[ ]:


test_data.describe()


# In[ ]:


features = train_data.columns[1:-1]

info_train_data = train_data.dtypes
info_test_data = test_data.dtypes

int_features = list(filter(lambda x: (x[1]=='int64'), zip(train_data.columns, info_train_data)))[1:]

print(int_features)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train_data[features], train_data.loss, test_size=0.3, random_state=0)

scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_dtest = scaler.transform(test_data.drop(["id"], axis=1))


# In[ ]:


#PCA

def reduce_dimension(array, dim=2):
    """
    Defining the outpout size for pca
    """
    
    #Dimension Reduction
    pca = PCA(n_components=dim)
    
    #fit to the train set
    pca.fit(array)
    
    #return the pca object
    return pca

#Call reduce dimension on fatures for reduction to 2 features
pca = reduce_dimension(x_train)
x_train_pca = pca.transform(x_train)

#validation set
y_pca = pca.transform(x_test)

#Actual test set
xtest_pca = pca.transform(x_dtest)


# In[ ]:


#Build the dataframe from pca arrays
dtrain_pca = pd.DataFrame(np.column_stack((x_train_pca, y_train)), columns=["x_pca", "y_pca", "loss"])
dtest_pca = pd.DataFrame(xtest_pca, columns=["test_x", "test_y"])
dtrain_pca["loss"]=dtrain_pca["loss"].astype(int)
print(dtrain_pca.head())
print(dtest_pca.head())


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

rows_to_plot = 10000

#Visualizing the 2D data obtaibed through PCA
fig = plt.figure(figsize=(24, 8))
fig.suptitle("Training and test set distributions")

ax = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2) ,fig.add_subplot(1, 3, 3, projection="3d")]

#using sequential colomap for loss
ax[2].scatter(dtrain_pca["x_pca"][:rows_to_plot], dtrain_pca["y_pca"][:rows_to_plot], c=dtrain_pca["loss"][:rows_to_plot], cmap="inferno")
ax[2].set_title("3D Plot")

g1 = sns.kdeplot(x="x_pca",y="y_pca",data=dtrain_pca[:rows_to_plot],palette="hls", ax=ax[0])
ax[0].set_title("Train Data")

g2 = sns.kdeplot(x=xtest_pca[:, :1].reshape(1, -1)[0][:rows_to_plot],y=xtest_pca[:, 1:].reshape(1, -1)[0][:rows_to_plot], ax=ax[1])
ax[1].set_title("Test Data");


plt.show()


# In[ ]:


#Visualizing the 2D data obtaibed through PCA
g3 = sns.FacetGrid(dtrain_pca,hue="loss", palette="hls",height=8)
g3.map(sns.scatterplot, "x_pca","y_pca").add_legend();


# In[ ]:


g3 = sns.FacetGrid(dtrain_pca, palette="hls",height=8, aspect=2)
g3.map(sns.kdeplot, "x_pca", color="g")
g3.map(sns.kdeplot, "y_pca", color="b")
g3.map(sns.kdeplot, "loss", color="r")
g3.set(xticks=range(-10, 20, 2))


# In[ ]:


#Invovking linear model
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression().fit(x_train, y_train)


# In[ ]:


#Cross Validation
predicted_test_loss = linear_model.predict(x_test)
predicted_test_loss = predicted_test_loss.astype(int)


# In[ ]:


from sklearn.metrics import mean_squared_error as RMSE

def rmse_plot(y_true, y_pred):
    rmse = np.sqrt(RMSE(y_true,y_pred))
    return rmse

def rmse_plots(y_test, predicted_test_loss):
    rmse = np.sqrt((1/len(y_test))*(sum(y_test-predicted_test_loss)))
    fig = plt.figure()
    plt.plot(y_test, predicted_test_loss)


# In[ ]:


#RMSE ERROR

print(f"The Error for the test set during cross vaidation is observed to be {rmse_plot(y_test, predicted_test_loss)}")


# In[ ]:


#Now fit the data on PCA
linear_model_pca = LinearRegression().fit(x_train_pca, y_train)
predicted_test_loss_pca = linear_model_pca.predict(y_pca)
print(f"The Error for the test set during cross vaidation on pca is observed to be {rmse_plot(y_test, predicted_test_loss_pca)}")


# In[ ]:


#Invovking linear model Ridge
from sklearn.linear_model import BayesianRidge
bayesridge = BayesianRidge(verbose=True).fit(x_train, y_train)
predicted_loss_ridge = bayesridge.predict(x_test)
print(f"The Error for the test set during cross vaidation with Bayesian Ridge is observed to be {rmse_plot(y_test, predicted_loss_ridge)}")


# In[ ]:


#Predicting on the unseen value we don't know loss here
y_predicted = bayesridge.predict(x_dtest)
result_f0 = pd.DataFrame({"id":test_data["id"],"loss":y_predicted})
dtest_pca["loss"] = y_predicted


# In[ ]:


result_f0.to_csv("./submit_bayes.csv", index=False)


# In[ ]:


y_predicted


# In[ ]:




