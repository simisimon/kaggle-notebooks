#!/usr/bin/env python
# coding: utf-8

# ## Clustering on wine dataset

# ##### Import necessary library

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly as py
import plotly.graph_objs as go

py.offline.init_notebook_mode(connected=True)


# ##### Import Wine Dataset

# In[ ]:


df = pd.read_csv('../input/wine-dataset-for-clustering/wine-clustering.csv')
df.head()


# ## Data-Preprocessing

# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# Great! We don't have any null value

# ## EDA and Visualization

# In[ ]:


plt.rcParams["figure.figsize"] = (20, 20)
df.hist(bins=20)
plt.show()


# In[ ]:


sns.pairplot(df)
plt.show()


# In[ ]:


# Plot a histogram on Age

fig = plt.figure(figsize = (8, 4))

plt.hist(df['Proline'], color='purple')


# ### Apply Corelation Technique

# In[ ]:


df.head()


# In[ ]:


# now, plot the data

plt.figure(figsize=(12,8))
ax = sns.heatmap(df.corr(), annot=True)
plt.show()


# In[ ]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# ## Making Clusters

# ##### 1. K_Means

# In[ ]:


from sklearn.cluster import KMeans

kmeans = KMeans(3)
kmeans


# In[ ]:


pred = kmeans.fit_predict(df)
pred


# In[ ]:


df['K_means_Cluster'] = pred
df.head()


# In[ ]:


df1 = df[df['K_means_Cluster']==0]
df2 = df[df['K_means_Cluster']==1]
df3 = df[df['K_means_Cluster']==2]

plt.figure(figsize=(8, 6))

sns.scatterplot(x='Alcohol', y='Ash_Alcanity', data=df1, color='green')
sns.scatterplot(x='Alcohol', y='Ash_Alcanity', data=df2, color='red')
sns.scatterplot(x='Alcohol', y='Ash_Alcanity', data=df3, color='blue')

# sns.scatterplot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='purple', marker='*')


plt.xlabel('Alcohol')
plt.ylabel('Ash_Alcanity')

plt.show()


# In[ ]:


kmeans.cluster_centers_


# ### Using Elbow Method to Find Appropriate number of clusters

# In[ ]:


k_rng = range(1,10)
sse = []

for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(df[['Alcohol', 'Ash_Alcanity']])
    sse.append(km.inertia_)


# In[ ]:


sse


# In[ ]:


# Elbow Plot

fig = plt.figure(figsize=(8,6))

plt.xlabel('K')
plt.ylabel('Sum of Squred Error')
plt.plot(k_rng,sse)


# From Elbow Method it is Clear that we have 3 Clusters

# ##### 2. Hierarchial (Agglomerative Clustering)

# In[ ]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[ ]:


points = df[['Alcohol', 'Ash_Alcanity']]


# In[ ]:


# Create a dendrogram

fig = plt.figure(figsize=(15,10))

dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))


# In[ ]:


#  Perform a clstering

hc = AgglomerativeClustering(3)
hc


# In[ ]:


pred = hc.fit_predict(df)
pred


# In[ ]:


df['Hierarchical_Cluster'] = pred
df.head()


# In[ ]:


df1 = df[df['Hierarchical_Cluster']==0]
df2 = df[df['Hierarchical_Cluster']==1]
df3 = df[df['Hierarchical_Cluster']==2]


# In[ ]:


plt.figure(figsize=(8, 6))

sns.scatterplot(x='Alcohol', y='Ash_Alcanity', data=df1, color='green')
sns.scatterplot(x='Alcohol', y='Ash_Alcanity', data=df2, color='red')
sns.scatterplot(x='Alcohol', y='Ash_Alcanity', data=df3, color='blue')

plt.xlabel('Width')
plt.ylabel('Length')

plt.show()


# ##### 3. DBSCAN (Density - Based)

# In[ ]:


from sklearn.cluster import DBSCAN


# In[ ]:


#  Perform a clstering

dbscan = DBSCAN(3)
dbscan


# In[ ]:


model = dbscan.fit_predict(df)
model


# In[ ]:


df['DBSCAN_Cluster'] = model
df.head()


# In[ ]:


df1 = df[df['DBSCAN_Cluster']==0]
df2 = df[df['DBSCAN_Cluster']==1]
df3 = df[df['DBSCAN_Cluster']==-1]


# In[ ]:


plt.figure(figsize=(8, 6))

sns.scatterplot(x='Alcohol', y='Ash_Alcanity', data=df1, color='green')
sns.scatterplot(x='Alcohol', y='Ash_Alcanity', data=df2, color='red')
sns.scatterplot(x='Alcohol', y='Ash_Alcanity', data=df3, color='blue')

plt.xlabel('Width')
plt.ylabel('Length')

plt.show()


# ## Visualizing Cluster

# In[ ]:


sb = df[['Alcohol', 'Ash_Alcanity', 'K_means_Cluster', 'Hierarchical_Cluster', 'DBSCAN_Cluster']]
sb.head()


# In[ ]:


sb.shape


# In[ ]:


km = sb['K_means_Cluster'].value_counts()
hc = sb['Hierarchical_Cluster'].value_counts()
db = sb['DBSCAN_Cluster'].value_counts()
db


# In[ ]:


data1 = pd.DataFrame({
    'KM_Cluster' : km.index,
    'K_Means' : km.values
}, columns=['KM_Cluster', 'K_Means'])

data1.set_index('KM_Cluster', inplace=True)

data1


# In[ ]:


data2 = pd.DataFrame({
    'HC_Cluster' : hc.index,
    'Hierarchial' : hc.values
}, columns=['HC_Cluster', 'Hierarchial'])

data2.set_index('HC_Cluster', inplace=True)
data2


# In[ ]:


data3 = pd.DataFrame({
    'DB_Cluster': db.index,
    'DBSCAN' : db.values
}, columns=['DB_Cluster', 'DBSCAN'])


data3.set_index('DB_Cluster', inplace=True)
data3


# In[ ]:


result = pd.concat([data1, data2, data3], axis=1)
result.fillna(0, inplace=True)
result


# In[ ]:


trace1 = go.Bar(
    x=result.index,
    y=result['K_Means'],
    marker_color='orange',
    name='K_Means'
)

trace2 = go.Bar(
    x=result.index,
    y=result['Hierarchial'],
    marker_color='purple',
    name='Hierarchical'
)

trace3 = go.Bar(
    x=result.index,
    y=result['DBSCAN'],
    marker_color='green',
    name='DBSCAN'
)

data = [trace1, trace2, trace3]
layout = go.Layout(
    barmode='stack',
    title=" ________________",
    xaxis={
        'title':"Cluster",
    },
    yaxis={
        'title':"Cluster NUmber",
    }
)
figure=go.Figure(data=data,layout=layout)
py.offline.iplot(figure)

