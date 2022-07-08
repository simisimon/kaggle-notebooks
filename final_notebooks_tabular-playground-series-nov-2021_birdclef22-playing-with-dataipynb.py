#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

import plotly.express as px

import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon

import librosa
import librosa.display
import IPython.display as ipd

import sklearn

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_csv = pd.read_csv("../input/birdclef-2022/train_metadata.csv")
test_csv = pd.read_csv("../input/birdclef-2022/test.csv")
print("There are {:,} unique bird species in the dataset.".format(len(train_csv['common_name'].unique())))


# In[ ]:


train_csv[['hour', 'minute']] = train_csv['time'].str.split(':', 1, expand=True)


# In[ ]:


plt.figure(figsize=(16, 6))
ax = sns.countplot(train_csv['hour'], palette="hls")


plt.title("Audio Files Registration at specific times", fontsize=16)
plt.xticks(rotation=90, fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel("Frequency", fontsize=14)
plt.xlabel("");


# In[ ]:


plt.figure(figsize=(16, 6))
ax = sns.countplot(train_csv['rating'], palette="hls")


plt.title("Audio Files Registration with regards to Ratings", fontsize=16)
plt.xticks(rotation=90, fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel("Frequency", fontsize=14)
plt.xlabel("");


# In[ ]:


adjusted_type = train_csv['type'].apply(lambda x: x[2:-2].split("', '")).reset_index().explode("type")

# Strip of white spaces and convert to lower chars
adjusted_type = adjusted_type['type'].apply(lambda x: x.strip().lower()).reset_index()
adjusted_type['type'] = adjusted_type['type'].replace('calls', 'call')

# Create Top 15 list with song types
top_15 = list(adjusted_type['type'].value_counts().head(15).reset_index()['index'])
data = adjusted_type[adjusted_type['type'].isin(top_15)]

# === PLOT ===

plt.figure(figsize=(16, 6))
ax = sns.countplot(data['type'], palette="hls", order = data['type'].value_counts().index)

plt.title("Top 15 Song Types", fontsize=16)
plt.ylabel("Frequency", fontsize=14)
plt.yticks(fontsize=13)
plt.xticks(rotation=90, fontsize=13)
plt.xlabel("");


# In[ ]:


top20 = train_csv['primary_label'].value_counts()[:20].index.tolist()


# In[ ]:


#plt.figure(figsize=(15, 9))
world_map = gpd.read_file("../input/world-shapefile/world_shapefile.shp")
world_map.plot(figsize = (15,9))
sns.scatterplot("longitude", "latitude", 
                data=train_csv[train_csv['primary_label'].isin(top20)], hue=train_csv[train_csv['primary_label'].isin(top20)]['primary_label'], 
                palette="hls", legend="full")
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.legend() 
plt.show()


# ### Work in Progress

# In[ ]:




