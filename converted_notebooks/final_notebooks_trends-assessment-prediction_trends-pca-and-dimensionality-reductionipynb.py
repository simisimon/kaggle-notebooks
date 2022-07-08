#!/usr/bin/env python
# coding: utf-8

# # <center>TRENDS: PCA and Dimensionality Reduction</center>
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1537731%2Fa5fdbe17ca91e6713d2880887232c81a%2FScreen%20Shot%202019-12-09%20at%2011.25.31%20AM.png?generation=1575920121028151&alt=media)
# 
# ---
# 
# *Author: **Nikhil Praveen***

# ## Table of contents
# 
# * 1. Introduction to the data
# ```
#     + Imports
#     + Data description
# ```
# * 2. Overview of Methods
# * 3. Simple EDA
#    ```
#    + Starting the EDA
#    + Some fancy Plotly visuals
#    + Dimensionality reduction techniques
#        + PCA
#            + Normal PCA
#            + Incremental PCA
#            + Kernel PCA
#        + TruncatedSVD
#        + NMF
#        + Selection
#    + What to infer?
#    ```

# # 1. Introduction

# In this competition, we are asked to predict multiple assessments as well as age from many different features given in the data. Let's look at our data files:

# ## Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import seaborn as sns
import plotly.io as pio

pio.templates.default = "plotly_dark"

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold
from typing import List, Tuple
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=1, cols=2)


# In[ ]:


PATH = '../input/trends-assessment-prediction/'
fnc = pd.read_csv(PATH+'fnc.csv')
ts = pd.read_csv(PATH+'train_scores.csv')
loading = pd.read_csv(PATH+'loading.csv')


# ## Data description

# From the official page, the descriptions we get for each of the data files are as follows:
# 
# + **fnc.csv** (corresponding to neuron networks)
# ```
#     + SCN - Sub-cortical Network
#     + ADN - Auditory Network
#     + SMN - Sensorimotor Network
#     + VSN - Visual Network
#     + CON - Cognitive-control Network    
#     + DMN - Default-mode Network
#     + CBN - Cerebellar Network
# ```
# + **loading.csv** (parts of the brain)
# ```
#     + IC_01 - Cerebellum
#     + IC_07 - Precuneus+PCC
#     + IC_05 - Calcarine
#     + IC_16 - Middle Occipital?
#     + IC_26 - Inf+Mid Frontal
#     + IC_06 - Calcarine
#     + IC_10 - MTG
#     + IC_09 - IPL+AG
#     + IC_18 - Cerebellum
#     + IC_12 - SMA
#     + IC_24 - IPL+Postcentral
#     + IC_15 - STG
#     + IC_13 - Temporal Pole
#     + IC_17 - Cerebellum
#     + IC_02 - ACC+mpfc
#     + IC_08 - Frontal
#     + IC_03 - Caudate
#     + IC_21 - Temporal Pole + Cerebellum
#     + IC_28 - Calcarine
#     + IC_11 - Frontal
#     + IC_20 - MCC
#     + IC_30 - Inf Frontal
#     + IC_22 - Insula + Caudate
#     + IC_29 - MTG
#     + IC_14 - Temporal Pole + Fusiform
# ```

# # 2. Overview of methods
# 
# In this notebook, I will use the following methods:
# * **Dimensionality reduction**:
#     We have high dimensionality here, so I will try to apply: 
#         * PCA
#         * TruncatedSVD
#         
# *NOTE: THE FOLLOWING WILL COME IN A LATER KERNEL.*
# * **Feature maker**<br>
#     I shall use sklearn's `BaseEstimator` class and add some dummy methods to make it work in the pipeline. Here's some sample code for the feature maker:
#     ```
#     class FeatureMaker(BaseEstimator, TransformerMixin):
#         def __init__(self, df):
#             '''And so on and so forth and what have'''
#         def fit(self):
#             pass
#         def transform():
#             # and so on and so forth
#     ```
# * **Imputer**<br>
#     After performing the feature engineering with our simple function, the Imputer from `sklearn` will be able to handle the many missing values.
# * **K-Fold**<br>
#     It is pretty obvious why one would use CV in a Kaggle competition.
# * **HistGradient Boosting Regressor**<br>
#     OK, this is an experimental `sklearn` regressor which I have decided to attempt in this competition.

# # 3. Simple EDA

# ## Starting the EDA

# First, we look at the dataset:

# In[ ]:


id1 = fnc["Id"]
fnc = fnc.drop(["Id"], axis=1)
fnc.head(3)


# In[ ]:


ts.head(3)


# In[ ]:


loading.head(3)


# ## Fancy Plotly
# 
# If we want to look at building models, we will have to look at fancy Plotly instead of good old-fashioned Seaborn and Matplotlib.
# 
# Our first 3d plot explores the relationships between `age` and the two `domain1_var` variables.

# In[ ]:


import plotly.graph_objects as go

x, y, z = np.array(ts["age"]), np.array(ts["domain1_var1"]), np.array(ts["domain1_var2"])

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()


# Now let's look at the domain2 variables:

# In[ ]:


x, y, z = np.array(ts["age"]), np.array(ts["domain2_var1"]), np.array(ts["domain2_var2"])

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()


# What about the relation between domain1 and domain2 variables? That would be interesting.

# In[ ]:


from plotly.subplots import make_subplots

x, y, z = np.array(ts["domain1_var1"]), np.array(ts["domain2_var1"]), np.array(ts["domain2_var2"])
x2, y2, z2 = np.array(ts["domain1_var2"]), np.array(ts["domain2_var1"]), np.array(ts["domain2_var2"])

fig = make_subplots(rows=1, cols=1)

fig.add_trace(go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
))

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()


# In[ ]:


fig = make_subplots(rows=1, cols=1)

fig.add_trace(go.Scatter3d(
    x=x2,
    y=y2,
    z=z2,
    mode='markers',
    marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    ),
))


# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()


# ## Dimensionality reduction
# 
# The main one we have to reduce is the data from `fnc.csv` as well as the data from `loading.csv`. For this purpose, we shall try PCA and TruncatedSVD on this dataset and pick which one is better.

# ### PCA

# #### 1. Normal PCA

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(fnc)
transformed_pca = pca.fit_transform(fnc)


# In[ ]:


pd.DataFrame(transformed_pca)


# One one hand, there is a lot of variance between components 0 and 1. What's the explained variance ratio?

# In[ ]:


pca.explained_variance_ratio_


# So, not off to a great start. Let's look at the other PCA methods.

# #### 2. Incremental PCA

# In[ ]:


from sklearn.decomposition import IncrementalPCA
pca = IncrementalPCA(n_components=5)
pca.fit(fnc)
transformed_pca = pca.fit_transform(fnc)


# In[ ]:


pd.DataFrame(transformed_pca)


# There is a very minimal difference between our first attempt at PCA and this one. Honestly, I didn't expect the difference to be too much between these two.

# In[ ]:


pca.explained_variance_ratio_


# There is virtually **no variance** between these PCA and IncrementalPCA. Looks like it's time to move on to our good old friend TruncatedSVD.

# ### Truncated SVD

# In[ ]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd.fit(fnc)
x = pd.DataFrame(svd.fit_transform(fnc))
x


# It seems that the first component is carrying most of the weight in both PCA and SVD. This leads to the absence of a well-balanced dataset. Let's check for skew:

# In[ ]:


skewValue = x.skew(axis=1)
skewValue


# It seems like Incremental PCA is the best choice here for dimensionality reduction. Let's perform PCA on the full dataset:

# In[ ]:


from sklearn.decomposition import IncrementalPCA
pca = IncrementalPCA(n_components=5)
pca.fit(fnc)
transformed_pca = pca.fit_transform(fnc)
pd.DataFrame(transformed_pca).to_csv('PCAData.csv', index=False)


# ---
# 
# # Please upvote if you liked this kernel!
# 
# ---
