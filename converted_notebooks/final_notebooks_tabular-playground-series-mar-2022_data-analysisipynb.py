#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import os
print('successful')


# In[ ]:


#Declaring data path
Data = '../input/virtual-betting-dataset/dataset.csv'


# In[ ]:


#reading the data 
df = pd.read_csv(Data)
df


# In[ ]:


nRowsRead = None # specify 'None' if want to read whole file
# root2tail.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df = pd.read_csv('../input/virtual-betting-dataset/dataset.csv', delimiter=',', nrows = nRowsRead)
df.dataframeName = '../input/virtual-betting-dataset/dataset.csv'
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


#Displaying the first five rows
df.head(5)


# In[ ]:


#showing dataset info
df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()

