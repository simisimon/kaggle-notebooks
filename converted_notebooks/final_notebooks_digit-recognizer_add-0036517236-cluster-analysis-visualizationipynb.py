#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler, RobustScaler
import seaborn as sb


# ### Preprocessing

# In[ ]:


#loading the data
df_train = pd.read_csv('../input/dapprojekt22/train.csv')
df_test = pd.read_csv('../input/dapprojekt22/test.csv')

df_train.head()


# In[ ]:


# dropping missing values
df_missing = df_train.isna().sum()[df_train.isna().sum() != 0]

df_train.drop(columns=df_missing.index.values, inplace=True)
df_test.drop(columns=df_missing.index.values, inplace=True)


# In[ ]:


# removing constant features and monotonous ones
df_nunique1 = df_train.nunique()[df_train.nunique() == 1]
df_nunique3000 = df_train.nunique()[df_train.nunique() >= 3000]

df_train.drop(columns=df_nunique1.index.values, inplace=True)
df_test.drop(columns=df_nunique1.index.values, inplace=True)

df_train.drop(columns=df_nunique3000.index.values, inplace=True)
df_test.drop(columns=df_nunique3000.index.values, inplace=True)


# In[ ]:


# change min_home and min_away from categorical to integer
df_train['MIN_HOME'] = np.int32(df_train['MIN_HOME'].str.split(':').str[0])
df_train['MIN_AWAY'] = np.int32(df_train['MIN_AWAY'].str.split(':').str[0])

df_test['MIN_HOME'] = np.int32(df_test['MIN_HOME'].str.split(':').str[0])
df_test['MIN_AWAY'] = np.int32(df_test['MIN_AWAY'].str.split(':').str[0])


# In[ ]:


# save and drop categoricals from train
df_categorical_train = df_train.select_dtypes(include=['object']).copy()
df_categorical_test = df_test.select_dtypes(include=['object']).copy()

df_train.drop(columns=df_categorical_train.columns.values.tolist(), inplace=True)
df_test.drop(columns=df_categorical_test.columns.values.tolist(), inplace=True)


# In[ ]:


# drop next weight
df_train.drop(columns=['NEXT_WEIGHT'], inplace=True)


# In[ ]:


# detect outliers
d = {}
df_train_wo_nw = df_train.loc[:, df_train.columns != 'NEXT_WINNER']
for i in range(1, 20):
    lof = IsolationForest(n_estimators=i)
    X_out = lof.fit_predict(df_train_wo_nw)
    d[i] = np.sum(X_out == 1)/len(X_out)

plt.plot(list(d.keys()), list(d.values()), '-o')
plt.xticks(list(d.keys()))
plt.xlabel('n_estimators')
plt.ylabel('% of the original dataset that won\'t be removed')
plt.title('n_estimators hyperparameter compared to the dataset size in %')
plt.axhline(y=0.9, c='k', linestyle='dashed', label='90% threshold')
plt.legend()
plt.show()


# In[ ]:


# remove outliers
lof = IsolationForest(n_estimators=6)
X_out = lof.fit_predict(df_train_wo_nw)
rows_to_drop = np.where(X_out == -1)[0]

df_train.drop(df_train.index[list(rows_to_drop)], inplace=True)
df_categorical_train.drop(df_categorical_train.index[list(rows_to_drop)], inplace=True)


# In[ ]:


# remove id variables
df_train.drop(columns=['TEAM_ID_AWAY', 'TEAM_ID_HOME'], inplace=True)
df_test.drop(columns=['TEAM_ID_AWAY', 'TEAM_ID_HOME'], inplace=True)


# In[ ]:


df_train = pd.concat((df_categorical_train[['TEAM_ABBREVIATION_AWAY', 'TEAM_ABBREVIATION_HOME']], df_train), axis=1)
df_test = pd.concat((df_categorical_test[['TEAM_ABBREVIATION_AWAY', 'TEAM_ABBREVIATION_HOME']], df_test), axis=1)

df_train.head()


# In[ ]:


# df_train = pd.concat((df_train, df_categorical_train[['TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY']]), axis=1)
df_train.reset_index(drop=True, inplace=True)
df_train.head()


# In[ ]:


df_train.columns.tolist()


# In[ ]:


print(df_categorical_train.shape)
print(df_train.shape)


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

def calculate_features_for_set(df, d, start=1):
    home, away = 'TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY'
    home_col = [i for i in filter(lambda x: x.endswith('HOME'), df.columns.tolist())]
    away_col = [i for i in filter(lambda x: x.endswith('AWAY'), df.columns.tolist())]
    
    print('Calculating features for dataset . . .')
    for i in range(start, df.shape[0]):
        if i % 400 == 399:
            print(f'Iteration {i+1}')
        
        rows_before = df.loc[0:i-1]
        row_curr = df.loc[i]
        
        curr_home, curr_away = row_curr[home], row_curr[away]
        
        mask1 = (rows_before[home] == curr_home) & (rows_before[away] == curr_away)
        mask2 = (rows_before[away] == curr_home) & (rows_before[home] == curr_away)
        
        rows_home = rows_before[mask1]
        rows_away = rows_before[mask2]
        
        if rows_home.empty and rows_away.empty:
            [d[k].append(0) for k in d.keys()]
        
        elif rows_home.empty:
            res = rows_away.mean()
            for i, k in enumerate(d.keys()):
                k2 = k.replace('AWAY', 'HOME') if 'AWAY' in k else k.replace('HOME', 'AWAY')
                d[k2].append(res[k])
                    
        elif rows_away.empty:
            res = rows_home.mean()
            [d[k].append(res[k]) for i, k in enumerate(d.keys())]
        
        else:
            res1 = rows_away.mean()
            res2 = rows_home.mean()
            for i, k in enumerate(d.keys()):
                k2 = k.replace('AWAY', 'HOME') if 'AWAY' in k else k.replace('HOME', 'AWAY')
                d[k].append((res1[k2] + res2[k])/2)
            
    return pd.DataFrame.from_dict(d)

def define_features(df_train, df_test):
    data_train = {}
    data_test = {}
    
    # shifting makes calculating the features easier
    y_train = df_train['NEXT_WINNER'][:-1]
    x_train = df_train[1:].reset_index(drop=True)
    x_train.drop(columns=['NEXT_WINNER'], inplace=True)
    train_shifted = pd.concat((x_train, y_train), axis=1)
    merged = pd.concat((train_shifted, df_test)).drop(columns=['id']).reset_index(drop=True)

    home, away = 'TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY'
   
    cols = [i for i in train_shifted.columns.tolist() if i != home and i != away and i != 'NEXT_WINNER']
    for c in cols:
        data_train[c] = []
        data_test[c] = []
    
    df_train_features = calculate_features_for_set(train_shifted, data_train)
    df_test_features = calculate_features_for_set(merged, data_test, start=len(train_shifted))
    
    return df_train_features, df_test_features


# In[ ]:


import time
start = time.time()
df_train_features, df_test_features = define_features(df_train, df_test)
end = time.time()
print(f'Time for executing the function {end-start}s')


# In[ ]:


df_train_features.tail()


# In[ ]:


df_test_features.tail()


# ### Clustering
# 
# Clustering algorithms transform the data into sets that consist of similar examples. On this particular dataset this could be used to create new features for classification by grouping games by certain features, grouping similar features to lower the dimensionality or for exploratory analysis. 
# 
# Some well known algorithms for this are:
# 1. KMeans
# 2. DBSCAN
# 3. Gaussian mixtures

# The hyperparameter we need to determine for using KMeans is the number of groups $k$. We can determine this by calculating the inertia for each of the models and determining where the "elbow" is on the plot. This point can be intuitively explained because it doesn't make much sense to add more components if the inertia doesn't get significantly smaller.
# 
# Inertia is defined as:
# $$\sum_{i=0}^{n} \min_{\mu_j \in C}{(||x_i - \mu_j||^2)}$$
# In this formula $x_i$ represents the data point $i$ and $\mu_j$ represents the centroid point for group $j$.

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def do_elbow_plot(data, range_ = range(2, 40)):
    inertia = []
    for i in range_:
        pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('kmeans', KMeans(i))
        ])
        pipeline.fit_transform(data)
        inertia.append(pipeline['kmeans'].inertia_)
    plt.plot(range_, inertia)
    plt.xlabel('Number of components')
    plt.ylabel('Inertia')
    plt.title('Elbow plot for KMeans')
    
do_elbow_plot(df_train_features)


# 
# For this particular graph dataset I would set the number of components $k$ to 5.

# In[ ]:


pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('kmeans', KMeans(5))
])
kmeans_labels = pipeline.fit_predict(df_train_features)
kmeans_labels


# A better visualization tool for evaluating these clusters is a silhouette score plot. The dotted red line represents the average silhouette score. Each of the clusters should have at least one silhouette coefficient value higher than the average score. 

# In[ ]:


from yellowbrick.cluster import SilhouetteVisualizer
visualizer = SilhouetteVisualizer(pipeline['kmeans'], colors='yellowbrick')

visualizer.fit(df_train_features)
visualizer.show()
plt.show()


# Next up is DBSCAN algorithm. This is a different approach compared to KMeans as it uses the high density found in natural clusters and also finds outliers as a separate cluster. KMeans tends to focus on spherical groups while DBSCAN can easily find unnatural looking groups. 
# 
# This algorithm uses 2 most important hyperparameters:
# 1. eps - the maximum distance between two samples for one to be in the others' neighborhood
# 2. min_samples - the number of samples in a neighborhood for a point to be considered a core point
# 
# The end result consists of clusters which consist of so called core points and reachable points. Points that are neither core points nor directly-reachable are considered as outliers and form a separate cluster.
# 
# For determining hyperparameter values we can use Davies Bouldin index where the validation of how well the clustering has been done is made using quantities and features inherent to the dataset. Lower values indicate better clusters.

# In[ ]:


from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

dbs_dict = {}
min_s_range = range(2, 20, 1)
eps_range = range(1, 50)

for min_s in min_s_range:
    print(f'Iteration {min_s}')
    for i in eps_range:
        pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('dbscan', DBSCAN(eps=i*0.05, min_samples=min_s))
        ])
        y = pipeline.fit_predict(df_train_features)
        dbs = davies_bouldin_score(df_train_features, y)
        dbs_dict[(min_s, i)] = dbs


# In[ ]:


print(f'Minimum value: {min(dbs_dict.values())} for i, j: {min(dbs_dict, key=dbs_dict.get)}')


# In[ ]:


pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('dbscan', DBSCAN(eps=0.05, min_samples=2))
])
dbscan_labels = pipeline.fit_predict(df_train_features)
dbscan_labels


# Finally, the third algorithm we'll be looking at is Gaussian mixture models. This is a probabilistic generative model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. It estimates probabilities but we can easily obtain the clusters for these probabilities. 
# 
# Important hyperparameters:
# 1. number of components - the number of assumed Gaussian distributions with unknown parameters
# 1. covariance_type - the type of covariance matrix for each of these distributions
# 
# In order to find the we can use BIC or AIC. Both of these criterions take into account the log likelihood of the model and the complexity. Models with lower BIC are preferred.
# 
# This algorithm also works best with spherical groups, but it's a lot more powerful than the KMeans algorithm because it's a generalized version of KMeans. KMeans assumes that the covariances are isotropic while gaussian mixture models do not.

# In[ ]:


from sklearn.mixture import GaussianMixture
import itertools as it

bic = []
n_components_range = range(2, 10)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    print(f'Calculating {cv_type} . . .')
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
        gmm.fit(df_train_features)
        bic.append(gmm.bic(df_train_features))

bic = np.array(bic)
color_iter = it.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
bars = []

plt.figure(figsize=(10, 8))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + i*0.1 + 0.05
    bars.append(
        plt.bar(
            xpos,
            bic[i*len(n_components_range):(i + 1)*len(n_components_range)],
            width=0.1,
            color=color,
            label=cv_type
        )
    )
plt.title('BIC scores for combinations of covariance and number of components')
plt.legend()
plt.show()


# The full covariance matrix and 3 components seems to be the winner here.

# In[ ]:


gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm_labels = gmm.fit_predict(df_train_features)
gmm_labels


# ### Cluster visualization using dimensionality reduction
# 
# There is a problem that the dataset consists of a 164 features but we can visualize data only only up to 3 dimensions. Here we can use dimensionality reduction techniques such as:
# 
# 1. PCA
# 2. t-SNE
# 
# Both of these algorithms use a lot of hyperparameters, but for visualization the most important one is the number of components. We can visualize data in 3-dimensional or 2-dimensional space so it will be set to 2 or 3.

# In[ ]:


from sklearn.decomposition import PCA

pca = Pipeline([
    ('scaler', StandardScaler()), 
    ('pca', PCA(n_components=2))
])

df_train_features_pca = pd.DataFrame(pca.fit_transform(df_train_features))


# In[ ]:


df_train_features_pca.head()


# In[ ]:


def do_dim_red_plot(labels, alg_name, df, dim_red_name, cmap='viridis'):
    plt.figure(figsize=(10, 10))
    plt.scatter(df[0], df[1], c=labels, cmap=cmap)
    plt.xlabel('Feature 0')
    plt.xlabel('Feature 1')
    plt.title(f'{alg_name} visualized using {dim_red_name}')
    plt.show()
    
do_dim_red_plot(kmeans_labels, 'KMeans', df_train_features_pca, 'PCA')


# In[ ]:


do_dim_red_plot(dbscan_labels, 'DBSCAN', df_train_features_pca, 'PCA')


# In[ ]:


do_dim_red_plot(gmm_labels, 'GMM', df_train_features_pca, 'PCA')


# We can see that the DBSCAN PCA plot makes most sense, but in reality the situation is probably different. The problem with our data is that the first few examples aren't really representative and consist of all zeros. These methods might give completely different results if removed the first 500 examples. 

# In[ ]:


from sklearn.manifold import TSNE

tsne = Pipeline([
    ('scaler', StandardScaler()), 
    ('tsne', TSNE(n_components=2))
])

df_train_features_tsne = pd.DataFrame(tsne.fit_transform(df_train_features))


# In[ ]:


df_train_features_tsne.head()


# In[ ]:


do_dim_red_plot(kmeans_labels, 'KMeans', df_train_features_tsne, 'TSNE')


# In[ ]:


do_dim_red_plot(dbscan_labels, 'DBSCAN', df_train_features_tsne, 'TSNE')


# In[ ]:


do_dim_red_plot(gmm_labels, 'GMM', df_train_features_tsne, 'TSNE')


# We can see some very interesting results using the TSNE method. The DBSCAN clustering here makes the most sense.
