#!/usr/bin/env python
# coding: utf-8

# # Zathura Project -  Classification of Stellar Objects

# ## 1. Importing Useful Libraries and Dataset

# In[ ]:


import pandas as pd
import numpy as np

from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import h2o
import time

from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators import H2OPrincipalComponentAnalysisEstimator

get_ipython().run_line_magic('matplotlib', 'inline')

zat_des= pd.read_csv('../input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv')


# ## 2. Getting familiar with Data

# In[ ]:


#info() and describe() to get to know our database...

zat_des.info()
zat_des.describe()


# #### We have discovered that the database consists of 18 columns per 10³ rows of information, all containing non-null values. There are 17 columns containing numeric data, and the qualitative one, "class", is the classification between Stars, Galaxies and Quasars; we will train our model on top of it. 

# In[ ]:


#plotting distribution per class...

zat_des['class'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.savefig('DIST.png')


# #### Above, the distribution of our dataset.
# #### The correlation of our predictors is shown below:

# In[ ]:


#reordering columns...
column_names = ['class','objid', 'ra', 'dec', 'run', 'rerun', 'camcol', 'field', 'specobjid', 'redshift', 'plate', 'mjd', 'fiberid', 'u', 'g', 'r', 'i', 'z']
zat_des = zat_des.reindex(columns=column_names)

#getting our correlation...
corr=zat_des.corr()

#plotting it...

sns.set_theme(style="white")
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.savefig("correlation.png")


# #### Accessing available literature on stellar objects classification, we've found that the five SDSS bands, UGRIZ magnitude, are main factors to solve our classification problem, altogether with the "redshift" factor. Let's look at it's matrix!

# In[ ]:


#plotting matrix of UGRIZ magnitude with redshift, grouped by 'class'...

zat_graph= zat_des[['u', 'g', 'r', 'i', 'z', 'redshift', 'class']]
sns.set_theme(style="white")
sns.pairplot(zat_graph, hue='class')
plt.savefig("UGRIZ.png")


# #### Looks promising so far. Looking at the database information, we've discovered that 'objid' and 'specobjid' are only indexes; the 'ra' and 'dec' determine astronomical coordinates; 'run', 'rerun', 'camcol' and 'field' describe the image taken by the SDSS station; 'plate' corresponds to a serial number of one material of the telescope; 'mjd' refers to the date; 'fiberid' refers to the SDSS spectrograph optical fiber ID. 
# 
# #### And 'redshift', 'u', 'g', 'r', 'i', 'z', as the literature pointed, are the spectral references we should focus on to predict the object's class.

# #### Considering available literature, we've decided to include PCA composition to our analysis as well! Below, the graphs and explained variance by the PCA with UGRIZ and adding "redshift" as a factor afterwards.

# In[ ]:


features = ['u', 'g', 'r', 'i', 'z']
x = zat_des.loc[:, features].values
y = zat_des.loc[:,['class']].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=3)
components = pca.fit_transform(x)

total_var = pca.explained_variance_ratio_.sum() * 100

fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=zat_des['class'],
    title=f'Total Explained Variance: {total_var:.2f}%',
    size_max=0,
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig.show()


# In[ ]:


features = ['u', 'g', 'r', 'i', 'z', 'redshift']
x = zat_des.loc[:, features].values
y = zat_des.loc[:,['class']].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=3)
components = pca.fit_transform(x)

total_var = pca.explained_variance_ratio_.sum() * 100

fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=zat_des['class'],
    title=f'Total Explained Variance: {total_var:.2f}%',
    size_max=0,
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'})

fig.show()


# #### Looks promising! Let's train our data and build our algorithm on our next chapter.

# ## Training Dataset

# In[ ]:


#initializing library...
h2o.init()


# In[ ]:


#importing file...
zat = h2o.import_file('../input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv', destination_frame="zat")
zat_pca=zat[['redshift', 'u', 'g', 'r', 'i', 'z', 'class']]


# In[ ]:


#describing data to double-check it 
zat.describe()


# ##### Training Model - Random Forest "1.0"

# In[ ]:


# classifying response variable as a factor so H2O algorithm treat the problem as classification problem
zat['class']=zat['class'].asfactor()

# predictors=['objid', 'ra', 'dec', 'run', 'rerun', 'camcol', 'field', 'specobjid', 'redshift', 'plate', 'mjd', 'fiberid', 'u', 'g', 'r', 'i', 'z']
predictors=['redshift', 'u', 'g', 'r', 'i', 'z']
response='class'

train, valid, test = zat.split_frame(ratios=[.7, .2], seed=2002)

zat_drf = H2ORandomForestEstimator(ntrees=50,
                                    max_depth=20,
                                    nfolds=10)

zat_drf.train(x=predictors, y=response, training_frame=train)


# ##### Training Model - Random Forest "2.0"

# In[ ]:


# classifying response variable as a factor so H2O algorithm treat the problem as classification problem
zat['class']=zat['class'].asfactor()

# predictors=['objid', 'ra', 'dec', 'run', 'rerun', 'camcol', 'field', 'specobjid', 'redshift', 'plate', 'mjd', 'fiberid', 'u', 'g', 'r', 'i', 'z']
predictors=['redshift', 'u', 'g', 'r', 'i', 'z']
response='class'

train, valid, test = zat.split_frame(ratios=[.7, .2], seed=2002)

zat_drf = H2ORandomForestEstimator(ntrees=100,
                                    max_depth=30,
                                    nfolds=10)

zat_drf.train(x=predictors, y=response, training_frame=train)


# ##### Training Model - Random Forest "2.0" + PCA

# In[ ]:


#splitting data in training data and test data
train, test = zat_pca.split_frame(ratios = [.7], seed = 2002)

#PCA application
PCAzat = H2OPrincipalComponentAnalysisEstimator(k = 3,model_id='pca_zat',
                                                   use_all_factor_levels = True,
                                                   pca_method = "glrm",
                                                   transform = "standardize",
                                                   impute_missing = True)
PCAzat.train(training_frame = train)
pred = PCAzat.predict(zat_pca)
#pred contains the predictions of PC1, 2 and 3


# In[ ]:


#combining dataframes
df = pred.cbind(zat_pca)
df.head()
df['class']=df['class'].asfactor()

#defining predictors for our new random forest
predictors=['PC1', 'PC2', 'PC3']
response='class'

#defining 2.0 training and testing datasets - with PC1, 2 and 3 analysis
train2, test2 = df.split_frame(ratios=[.7], seed=2002)

df_final = H2ORandomForestEstimator(ntrees=100,
                                    max_depth=30,
                                    nfolds=10)

df_final.train(x=predictors, y=response, training_frame=train2)


# In[ ]:


"""References:

https://medium.com/tech-vision/random-forest-classification-with-h2o-python-for-beginners-b31f6e4ccf3c

https://www.kdnuggets.com/2015/12/beyond-one-hot-exploration-categorical-variables.html

https://www.sdss.org/dr12/algorithms/redshifts/

https://www.youtube.com/watch?v=HMOI_lkzW08&t=210s

https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/tutorial/astronomy/dimensionality_reduction.html
"""

