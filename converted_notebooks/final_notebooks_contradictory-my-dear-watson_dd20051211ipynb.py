#!/usr/bin/env python
# coding: utf-8

# # K-means Homework
# ## 杜承豫 20051211

# ## Import modules requiered

# #### First of all, we need to import the required module.

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')


# In[ ]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns


# For this particular project, we’ll work with two scikit-learn modules: Kmeans and PCA. They will allow us to perform a clustering algorithm and dimensionality reduction.

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.io as pio


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


# DEFINE A EXTRA_GRAPHS

# In[ ]:


def biplot(score, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    plt.scatter(xs*scalex,ys*scaley, color="#c7e9c0", edgecolor="#006d2c", alpha=0.5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color='#253494',alpha=0.5,lw=2)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color="#000000", ha="center", va="center")
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color="#000000", ha="center", va="center")
    plt.xlim(-.75,1)
    plt.ylim(-0.5,1)
    plt.grid(False)
    plt.xticks(np.arange(0, 1, 0.5), size=12)
    plt.yticks(np.arange(-0.75, 1, 0.5), size=12)
    plt.xlabel("Component 1", size=14)
    plt.ylabel("Component 2", size=14)
    plt.gca().spines["top"].set_visible(False);
    plt.gca().spines["right"].set_visible(False);


# ### Read data into a DataFrame

# We read the basic data stored in the TRAIN file into a `DataFrame` using pandas.

# In[ ]:


train = pd.read_csv("../input/ml2021-2022-2-kmeans/train.csv", encoding='utf-8')


# We check the first five rows of the DataFrame. We can see that we have: id,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality,style.

# In[ ]:


train=train.drop('style',1)
train.head()


# ### Exploring the data

# Now, it's time to explore the data to check the quality of the data and the distribution of the variables.

# First, we check that if there is any missing value in the dataset. K-means algorithm is not able to deal with missing values.

# In[ ]:


print(f"Missing values in each variable: \n{train.isnull().sum()}")


# Fortunately, there is no missing data. We can also check if there are duplicated rows.

# In[ ]:


print(f"Duplicated rows: {train.duplicated().sum()}")


# Finally, we check how each variable is presented in the DataFrame. Categorical variables cannot be handled directly. K-means is based on distances. The approach for converting those variables depend on the type of categorical variables.

# In[ ]:


print(f"Variable:                  Type: \n{train.dtypes}")


# After that, we can start observing the distribution of the variables. Here, we'll define two functions. The first one will retrieve descriptive statistics of the variables. The second one will help us graph the variable distribution.

# #### Descriptive statistics and Distribution.

# For the descriptive statistcs, we'll get mean, standard deviation, median and variance. If the variable is not numeric, we'll get the counts in each category.

# In[ ]:


def statistics(variable):
    if variable.dtype == "int64" or variable.dtype == "float64":
        return pd.DataFrame([[variable.name, np.mean(variable), np.std(variable), np.median(variable), np.var(variable)]],
                            columns = ["Variable", "Mean", "Standard Deviation", "Median", "Variance"]).set_index("Variable")
    else:
        return pd.DataFrame(variable.value_counts())


# In[ ]:


def graph_histo(x):
    if x.dtype == "int64" or x.dtype == "float64":
        # Select size of bins by getting maximum and minimum and divide the substraction by 10
        size_bins = 10
        # Get the title by getting the name of the column
        title = x.name
        #Assign random colors to each graph
        color_kde = list(map(float, np.random.rand(3,)))
        color_bar = list(map(float, np.random.rand(3,)))

        # Plot the displot
        sns.distplot(x, bins=size_bins, kde_kws={"lw": 1.5, "alpha":0.8, "color":color_kde},
                       hist_kws={"linewidth": 1.5, "edgecolor": "grey",
                                "alpha": 0.4, "color":color_bar})
        # Customize ticks and labels
        plt.xticks(size=14)
        plt.yticks(size=14);
        plt.ylabel("Frequency", size=16, labelpad=15);
        # Customize title
        plt.title(title, size=18)
        # Customize grid and axes visibility
        plt.grid(False);
        plt.gca().spines["top"].set_visible(False);
        plt.gca().spines["right"].set_visible(False);
        plt.gca().spines["bottom"].set_visible(False);
        plt.gca().spines["left"].set_visible(False);
    else:
        x = pd.DataFrame(x)
        # Plot
        sns.catplot(x=x.columns[0], kind="count", palette="spring", data=x)
        # Customize title
        title = x.columns[0]
        plt.title(title, size=18)
        # Customize ticks and labels
        plt.xticks(size=14)
        plt.yticks(size=14);
        plt.xlabel("")
        plt.ylabel("Counts", size=16, labelpad=15);
        # Customize grid and axes visibility
        plt.gca().spines["top"].set_visible(False);
        plt.gca().spines["right"].set_visible(False);
        plt.gca().spines["bottom"].set_visible(False);
        plt.gca().spines["left"].set_visible(False);


# We'll start by the **Spending Score**.

# In[ ]:


fixed_acidity = train["fixed_acidity"]


# In[ ]:


statistics(fixed_acidity)


# In[ ]:


graph_histo(fixed_acidity)


# Then, we'll check **volatile_acidity**.

# In[ ]:


volatile_acidity = train["volatile_acidity"]


# In[ ]:


statistics(volatile_acidity)


# In[ ]:


graph_histo(volatile_acidity)


# Finally, we'll explore **total_sulfur_dioxide** variable.

# In[ ]:


total_sulfur_dioxide = train["total_sulfur_dioxide"]


# In[ ]:


statistics(total_sulfur_dioxide)


# In[ ]:


graph_histo(total_sulfur_dioxide)


# In[ ]:


residual_sugar = train["residual_sugar"]


# In[ ]:


statistics(residual_sugar)


# In[ ]:


graph_histo(residual_sugar)


# #### Correlation between parameteres

# Also, we will analyze the correlation between the numeric parameters. For that aim, we'll use the `pairplot` seaborn function. We want to see whether there is a difference between gender. So, we are going to set the `hue` parameter to get different colors for points belonging to female or customers.

# In[ ]:


sns.pairplot(train, x_vars = ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol","quality"
],
               y_vars = ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol","quality"
],

               hue = "quality",
               kind= "scatter",
               palette = "YlGnBu",
               height = 4,
               plot_kws={"s": 35, "alpha": 0.8});


# #### Why is it important to look into the descriptive statistics, distribution and correlation between variables?

# In order to apply K-means, we need to meet the algorithm assumptions.
# 
# K-means assumes:
# 
# - **Cluster's shape**: The variance of the distribution is spherical meaning that clusters have a spherical shape. In order for this to be true, all variables should be normally distributed and have the same variance.
# - **Clusters' Size**: All clusters have the same number of observations.
# - **Relationship between variables**: There is little or no correlation between the variables.

# In our dataset, our variables are normally distributed. Variances are quite close to each other. Except for age that has a lower variance that the rest of the variables. We could find a proper transformation to solve this issue. We could apply the logarithm or Box-Cox transformation.  Box-Cox is a family of transformations which allows us to correct non-normal distributed variables or non-equal variances.

# ### Dimensionality reduction

# After we checked that we can apply k-means, we can apply Principal Component Analysis (PCA) to discover which dimensions best maximize the variance of features involved.

# #### Principal Component Analysis (PCA)

# First, we'll transform the categorical variable into two binary variables.

# In[ ]:





# In[ ]:


# train["red"] = train["style"].apply(lambda x: 0.0 if x == "red" else 1.0)


# In[ ]:


# train["white"] = train["style"].apply(lambda x: 0.0 if x == "white" else 1.0)


# In[ ]:


# train=train.drop('style',1)


# Then, we are going to select from the dataset all the useful columns.  style will split it into two binaries categories. It should not appear in the final dataset

# In[ ]:





# In order to apply PCA, we are going to use the `PCA` function from sklearn module.

# In[ ]:


def standardization(x): # 对连续数据标准化
    x_mean=np.mean(x,axis=0)
    x_std=np.std(x.astype('float'),axis=0)
    x_new=(x-x_mean)/x_std
    return x_new
def normalization(x):  # 对离散数据归一化
    x_max=np.max(x,axis=0)
    x_min=np.min(x,axis=0)
    x_new=(x-x_min)/(x_max-x_min)
    return x_new


# In[ ]:


X=standardization(train)
X.head()


# In[ ]:


# Apply PCA and fit the features selected
pca = PCA(n_components=2).fit(X)


# During the fitting process, the model learns some quantities from the data: the "components" and "explained variance".

# In[ ]:


print(pca.components_)


# In[ ]:


print(pca.explained_variance_)


# These numbers that appear to be abstract define vectors. The components define the direction of the vector while the explained variance define the squared-length of the vector.
# 
# The vectors represent the principal axes of the data. The length of the vector indicates the importance of that axis in describing the distribution of the data. The projection of each data point onto the principal axes are the principal components of the data.

# In[ ]:


# Transform samples using the PCA fit
pca_2d = pca.transform(X)


# We can represent this using a type of scatter plot called biplot. Each point is represented by its score regarding the principal components.
# It is helpful to understand the reduced dimensions of the data. It also helps us discover relationships between the principal components and the original variables.

# In[ ]:


# Biplot
import matplotlib.pyplot as plt
import numpy as np

def biplot(score, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    plt.scatter(xs*scalex,ys*scaley, color="#c7e9c0", edgecolor="#006d2c", alpha=0.5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color='#253494',alpha=0.5,lw=2) 
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color="#000000", ha="center", va="center")
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color="#000000", ha="center", va="center")
    plt.xlim(-.75,1)
    plt.ylim(-0.5,1)
    plt.grid(False)
    plt.xticks(np.arange(0, 1, 0.5), size=12)
    plt.yticks(np.arange(-0.75, 1, 0.5), size=12)
    plt.xlabel("Component 1", size=14)
    plt.ylabel("Component 2", size=14)
    plt.gca().spines["top"].set_visible(False);
    plt.gca().spines["right"].set_visible(False);  
 


# In[ ]:


biplot(pca_2d[:,0:2], np.transpose(pca.components_[0:2, :]), labels=X.columns)


# We can observe that Annual Income as well as Spending Score at the two most important components.

# ### K-means clustering

# 
# Time for clustering!

# In order to cluster data, we need to determine how to tell if two data points are similar. A proximity measure characterizes the similarity or dissimilarity that exists between objects.
# 
# We can choose to determine if two points are similar. So if the value is large, the points are very similar.
# Or choose to determine if they are dissimilar. If the value is small, the points are similar. This is what we know as "distance".
# 
# There are various distances that a clustering algorithm can use: Manhattan distance, Minkowski distance, Euclidean distance, among others.

# ${\sqrt{\sum_{i=1}^n (x_i-y_i)^2}}$

# K-means typically uses Euclidean distance to determine how similar (or dissimilar) two points are.

# First, we need to fix the numbers of clusters to use.

# There are several direct methods to perform this. Among them, we find the elbow and silhouette methods.

# We'll consider the total intra-cluster variation (or total within-cluster sum of square (WSS)). The goal is to minimize WSS.

# The Elbow method looks at how the total WSS varies with the number of clusters. 
# For that, we'll compute k-means for a range of different values of k. Then, we calculate the total WSS. We plot the curve WSS vs. number of clusters. 
# Finally, we locate the elbow or bend of the plot. This point is considered to be the appropriate number of clusters.

# 

# In[ ]:


wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit(X)
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss, c="#c51b7d")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.title('Elbow Method', size=14)
plt.xlabel('Number of clusters', size=12)
plt.ylabel('wcss', size=14)
plt.show()


# How does k-means clustering works? The main idea is to select k centers, one for each cluster. There are several ways to initialize those centers. We can do it randomly, pass certain points that we believe are the center or place them in a smart way (e.g. as far away from each other as possible).
# Then, we calculate the Euclidean distance between each point and the cluster centers. We assign the points to the cluster center where the distance is minimum.
# After that, we recalculate the new cluster center. We select the point that is in the middle of each cluster as the new center. 
# And we start again, calculate distance, assign to cluster, calculate new centers. When do we stop? When the centers do not move anymore.

# In[ ]:


# Kmeans algorithm
# n_clusters: Number of clusters. In our case 5
# init: k-means++. Smart initialization
# max_iter: Maximum number of iterations of the k-means algorithm for a single run
# n_init: Number of time the k-means algorithm will be run with different centroid seeds.
# random_state: Determines random number generation for centroid initialization.
kmeans = KMeans(n_clusters=2)

# Fit and predict
y_means = kmeans.fit_predict(X)


# In[ ]:





# Now, let's check how our clusters look like:

# In[ ]:


fig, ax = plt.subplots(figsize = (8, 6))

plt.scatter(pca_2d[:, 0], pca_2d[:, 1],
            c=y_means,
            edgecolor="none",
            cmap=plt.cm.get_cmap("Spectral_r", 2),
            alpha=0.5)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)

plt.xticks(size=12)
plt.yticks(size=12)

plt.xlabel("Component 1", size = 14, labelpad=10)
plt.ylabel("Component 2", size = 14, labelpad=10)

plt.title('Dominios agrupados en 2 clusters', size=16)


plt.colorbar(ticks=[0, 1]);

plt.show()


# In[ ]:


centroids = pd.DataFrame(kmeans.cluster_centers_, columns = ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol","quality"])


# In[ ]:


centroids.index_name = "ClusterID"


# In[ ]:


centroids["ClusterID"] = centroids.index
centroids = centroids.reset_index(drop=True)


# In[ ]:


centroids


# The most important features appear to be Annual Income and Spending score. 
# We have people whose income is low but spend in the same range - segment 0. People whose earnings a high and spend a lot - segment 1. Customers whose income is middle range but also spend at the same level - segment 2. 
# Then we have customers whose income is very high but they have most spendings - segment 4. And last, people whose earnings are little but they spend a lot- segment 5.

# Imagine that tomorrow we have a new member. And we want to know which segment that person belongs. We can predict this.
# 1 == red
# 0 == white

# In[ ]:


# train_data = pd.read_csv("./data/train.csv").values
# model1 = KMeans1(num_k=2, label=True)
# test = pd.read_csv("./data/test.csv").values
#
# model1.fit(train_data)
# result = model1.predict(test)


# In[ ]:


# train_data = X
# model1 = KMeans(n_clusters=2, max_iter=300, n_init=10, random_state=1)
# test = pd.read_csv("./data/test.csv").values
#
# model1.fit(train_data)
# result = model1.predict(test)
#
# result


# In[ ]:


test = pd.read_csv("../input/ml2021-2022-2-kmeans/test.csv", encoding='utf-8')


# In[ ]:


test=standardization(test)


# In[ ]:





# In[ ]:


tmp=[]
tmp1=[]


# In[ ]:


j=0
for i in range(len(test)):

    print(j)
    X_new = np.array(test[j:j+1])
    new_customer = kmeans.predict(X_new)
    print(f"The new customer belongs to segment {new_customer[0]}")
    tmp.append(new_customer[0])
    j=j+1


# In[ ]:


len(tmp)
for i in range(len(tmp)):
    if(tmp[i] == 0):
        tmp1.append("red")
    else:
        tmp1.append("white")


# In[ ]:


len(tmp1)


# In[ ]:


out_dict = {
    'id':list(np.arange(len(test))),
    'style':list(tmp1)
}
out = pd.DataFrame(out_dict)
out.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:




