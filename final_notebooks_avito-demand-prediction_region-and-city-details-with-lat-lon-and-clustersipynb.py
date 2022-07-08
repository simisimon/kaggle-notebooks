#!/usr/bin/env python
# coding: utf-8

# **You can download the resulting file at the bottom of the notebook or under the [code](https://www.kaggle.com/frankherfert/region-and-city-details-with-lat-lon-and-clusters/code) section of the Kernel.**

# The Avito dataset provides information about the region and city an ad was placed in.
# 
# While we can label encode those, the labels will not have any relationship to each other. Different machine learning algorithms can learn some similarities between the labels but this will likely only work if there are enough similar ads in each city.
# 
# With this preprocessing notebook we will add latitude, longitude and clusters for each city with the aim of helping our models to better group nearby cities.

# # imports

# In[49]:


import pandas as pd

import time

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import preprocessing

import string


# In[50]:


pd.set_option('display.max_columns', 300)
pd.set_option('max_colwidth',400)
pd.set_option('display.max_rows', 300)

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>")) # using this in your offline notebooks, will get rid of most of the side space
display(HTML("<style>table {float:left}</style>")) # makes the changelog table nicer


# # loading data

# In[41]:


train = pd.read_csv('../input/avito-demand-prediction/train.csv')


# In[42]:


test = pd.read_csv('../input/avito-demand-prediction/test.csv')


# In[43]:


full = pd.concat([train, test], axis=0)


# In[44]:


full.info(memory_usage="deep")


# # combine city and region

# In[12]:


get_ipython().run_cell_magic('time', '', 'full["city_region"] = full.loc[:, ["city", "region"]].apply(lambda l: " ".join(l), axis=1)\n')


# In[13]:


full.sample(5)


# In[14]:


print(full.shape)
city_region_unique = full.drop_duplicates(subset="city_region").reset_index(drop=True)
print(city_region_unique.shape)


# # geocoder

# We can use the geocoders class from the geopy package to search through different services for  additional data

# In[18]:


from geopy import geocoders

# api_key = "" # place your API key here if you want to access the API as often as you like.
# g = geocoders.GoogleV3(api_key=api_key)

g = geocoders.GoogleV3()


# Calling the .geocode() function will return a search results from the Google Maps API in the specified language. Depending on the country, this includes different levels of geographical information.
# 
# AS the Kaggle server has likely hit the API limit already, this part is commented out.

# In[20]:


# geocode = g.geocode("Самара Самарская область", timeout=10, language="en")


# In[21]:


# geocode.raw


# In[22]:


# print(geocode.address)
# print(geocode.latitude)
# print(geocode.longitude)


# Loop through all cities and extract the latitude and longitude from the geocode results.
# 
# Due to API limits, we will load the resulting file from a local run.

# In[23]:


city_region_unique = pd.read_csv("../input/avito-russian-region-cities/avito_region_city_features.csv")
print(city_region_unique.shape)


# In[24]:


# city_region_unique["latitude"] = np.nan


# In[25]:


# %%time
# print("searching", len(city_region_unique), "entries")

# for index, row in city_region_unique.loc[city_region_unique["latitude"].isnull(), 
#                                          :].iterrows():
    
#     search = city_region_unique.loc[index, "city_region"]
#     try:
#         geocode = g.geocode(search, timeout=10, language="en")

#         city_region_unique.loc[index, 'latitude'] = geocode.latitude
#         city_region_unique.loc[index, 'longitude'] = geocode.longitude
#     except:
#         city_region_unique.loc[index, 'latitude'] = -999
#         city_region_unique.loc[index, 'longitude'] = -999
        
#     time.sleep(.1)
    
#     if index%10==0:
#         print(str(index).ljust(6), end=" ")
#     if (index+1)%120==0:
#         print("")

# print("\n done")


# # Clusters

# A kmeans cluster will not do much good here because it will create equally sized area clusters, which doesn't reflect the real world distribution of cities.
# 
# A density based cluster will provide better groups and also detec outliers (cities that are too far away from everything else).

# [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) is a great clustering algorithm and has sklearn integration as well. 
# 
# It allows cluster creation with a minimum entry count per cluster compared to a specified number of clusters as in kmeans. 
# 
# It can be installed with `conda install -c conda-forge hdbscan`

# In[27]:


# The library in the Kaggle kernel seems to be miscompiled
# import hdbscan


# ## cluster size 5

# In[28]:


# city_region_unique["lat_lon_hdbscan_cluster_05_03"] = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3
#                                                                      ).fit_predict(city_region_unique.loc[:, ["latitude", "longitude"]])


# In[29]:


print("number of clusters:", len(city_region_unique["lat_lon_hdbscan_cluster_05_03"].unique()))


# A lot of cities are located in the west of Russia with a big orange cluster around the Moscow region, a big blue cluster towards the the Georgian border, a single pink cluster at 55,20 around Kaliningrad and some scattered cities in the north-east

# In[48]:


sns.lmplot(data=city_region_unique, x="longitude", y="latitude", hue="lat_lon_hdbscan_cluster_05_03", 
           size=10, legend=False, fit_reg=False)


# ## cluster size 10

# Let's add a few more clusters with different parameters

# In[31]:


# city_region_unique["lat_lon_hdbscan_cluster_10_03"] = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3
#                                                                      ).fit_predict(city_region_unique.loc[:, ["latitude", "longitude"]])
print("number of clusters:", len(city_region_unique["lat_lon_hdbscan_cluster_10_03"].unique()))


# In[32]:


sns.lmplot(data=city_region_unique, x="longitude", y="latitude", hue="lat_lon_hdbscan_cluster_10_03", 
           size=10, legend=False, fit_reg=False)


# ## cluster size 20

# In[33]:


# city_region_unique["lat_lon_hdbscan_cluster_20_03"] = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=3
#                                                                      ).fit_predict(city_region_unique.loc[:, ["latitude", "longitude"]])
print("number of clusters:", len(city_region_unique["lat_lon_hdbscan_cluster_20_03"].unique()))


# In[34]:


sns.lmplot(data=city_region_unique, x="longitude", y="latitude", hue="lat_lon_hdbscan_cluster_20_03", 
           size=10, legend=False, fit_reg=False)


# # label encoding

# The cluster IDs will likely be more accurate than simple label-encoded city-IDs but just in case we will add in regular label encodings as well.

# In[35]:


city_region_unique["region_id"]      = preprocessing.LabelEncoder().fit_transform(city_region_unique["region"].values)
city_region_unique["city_region_id"] = preprocessing.LabelEncoder().fit_transform(city_region_unique["city_region"].values)


# # Saving features

# In[36]:


city_region_unique.head()


# In[37]:


city_region_unique.to_csv("avito_region_city_features.csv", index=False)


# As most people don't have access to a Google Maps API key, the generated features will be saved in a csv file for everyone to download.
# 
# You can download it from the Kernel or from the Kaggle datasets site: https://www.kaggle.com/frankherfert/avito-russian-region-cities

# You can add these features to your regular data this way:

# In[45]:


print("before:", full.shape)
full = pd.merge(left=full, right=city_region_unique, how="left", on=["region", "city"])
print("after :", full.shape)


# In[46]:


full.loc[:, ["item_id", "user_id", "region", "city", "latitude", "longitude", "lat_lon_hdbscan_cluster_05_03"]].sample(5)

