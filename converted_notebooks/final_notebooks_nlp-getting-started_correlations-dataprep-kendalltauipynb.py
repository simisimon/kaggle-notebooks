#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ![](https://image.slidesharecdn.com/whatisakendallstau-141002162706-phpapp02/85/what-is-a-kendalls-tau-22-320.jpg?cb=1412267289)pt.slideshare.net

# In[ ]:


get_ipython().system('pip install dataprep')


# In[ ]:


get_ipython().system(' python -m pip install "dask[dataframe]" --upgrade  # or python -m pip install')


# In[ ]:


from dataprep.eda import plot, plot_correlation, create_report, plot_missing


# In[ ]:


df = pd.read_csv('../input/AI4Code/train_orders.csv',delimiter=',', encoding='ISO-8859-2')
df.head()


# In[ ]:


df1 = pd.read_csv('../input/AI4Code/train_ancestors.csv',delimiter=',', encoding='ISO-8859-2')
df1.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

#fill in mean for floats
for c in df.columns:
    if df[c].dtype=='float16' or  df[c].dtype=='float32' or  df[c].dtype=='float64':
        df[c].fillna(df[c].mean())

#fill in -999 for categoricals
df = df.fillna(-999)
# Label Encoding
for f in df.columns:
    if df[f].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(list(df[f].values))
        df[f] = lbl.transform(list(df[f].values))
        
print('Labelling done.')


# ![](https://i.ytimg.com/vi/gDNmhEBZAO8/maxresdefault.jpg)

# In[ ]:


#API Correlation
plot_correlation(df)


# #Kendall's Tau, Pearson and Spearman's
# 
# "In this video, they briefly reviewed the Pearson correlation coefficient. Of course, that's the most popular measure of correlation, but mostly just so they have a baseline to compare to the two measures of rank correlations. Specifically, they looked at the Spearman's rank correlation and Kendall's tau rank correlation."
# 
# "These two are measures of ordinal correlation as opposed to a measure of cardinal correlation. In computing Kendall's tau, which has a formula that's easier to remember, they had to count up the number of concordant versus discordant pairs."
# 
# https://www.youtube.com/watch?v=gDNmhEBZAO8

# In[ ]:


from sklearn.preprocessing import LabelEncoder

#fill in mean for floats
for c in df1.columns:
    if df1[c].dtype=='float16' or  df1[c].dtype=='float32' or  df1[c].dtype=='float64':
        df1[c].fillna(df[c].mean())

#fill in -999 for categoricals
df1 = df1.fillna(-999)
# Label Encoding
for f in df1.columns:
    if df1[f].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(list(df1[f].values))
        df1[f] = lbl.transform(list(df1[f].values))
        
print('Labelling done.')


# In[ ]:


plot_correlation(df1)


# In[ ]:


#Code by Sadegh Jalalian https://www.kaggle.com/code/sadeghjalalian/prediction-of-stock-index-using-xgboost-rf-and-svm

plt.figure(figsize=(12,8))
sns.heatmap(df.describe()[1:].transpose(),
            annot=True,linecolor="w",
            linewidth=2,cmap=sns.color_palette("Set2"))
plt.title("Data summary")
plt.show()


# In[ ]:


#Code by Sadegh Jalalian https://www.kaggle.com/code/sadeghjalalian/prediction-of-stock-index-using-xgboost-rf-and-svm

plt.figure(figsize=(12,8))
sns.heatmap(df1.describe()[1:].transpose(),
            annot=True,linecolor="w",
            linewidth=2,cmap=sns.color_palette("Set1"))
plt.title("Data summary")
plt.show()


# #Acknowledgements:
# 
# "DataPrep is free, open-source software released under the MIT license. Anyone can reuse DataPrep code for any purpose."
# 
# "DataPrep is built using Pandas/Dask DataFrame and can be seamlessly integrated with other Python libraries."
# 
# "DataPrep is designed for computational notebooks, the most popular environment among data scientists."
# 
# "Dataprepare is an initiative by SFU Data Science Research Group to speed up Data Science. Dataprep.eda attempts to simplify the entire EDA process with very minimal lines of code. EDA is a very essential and time-consuming part of the data science pipeline, having a tool that eases the process is a boon."
# 
# https://dataprep.ai/
# 
# Sadegh Jalalian https://www.kaggle.com/code/sadeghjalalian/prediction-of-stock-index-using-xgboost-rf-and-svm
