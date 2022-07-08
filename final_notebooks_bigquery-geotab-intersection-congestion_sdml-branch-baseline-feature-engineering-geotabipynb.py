#!/usr/bin/env python
# coding: utf-8

# This notebook was branched in order to make public the experiments done by the San Diego Machine Learning meetup group
# * Forked from : https://www.kaggle.com/pulkitmehtawork1985/beating-benchmark
# * Copies feature code over from my other kernel; https://www.kaggle.com/danofer/basic-features-geotab-intersections
# 
# * V6 - try  a multitask model in addition to a model per target. Likely to have worse performance, but will be faster

# In[ ]:


import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression, LassoLarsCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Load Data

train = pd.read_csv("../input/bigquery-geotab-intersection-congestion/train.csv").sample(frac=0.15,random_state=42)#,nrows=123456)
test = pd.read_csv("../input/bigquery-geotab-intersection-congestion/test.csv")


# ## Data Cleaning
# 

# 0 Street 2286 47.67 %
# 1 Avenue 1620 33.79 %
# 2 Boulevard 278 5.80 %
# 3 Road 258 5.38 %
# 4 Drive 200 4.17 %
# 5 Parkway 60 1.25 %
# 6 Highway 22 0.46 %
# 7 Pkwy 6 0.13 %
# 8 St 5 0.10 %
# 9 Ave 3 0.06 %
# 10 Overpass 1 0.02 %
# 11 Bypass 1 0.02 %
# 12 Expressway 1 0.02 %

# In[ ]:





# In[ ]:


kind_of_roads = ["Street", "Avenue", "Boulevard", "Road", "Drive", "Parkway", "Highway", "Pkwy", " St", "Ave", "Overpass", "Bypass", "Expressway"]
kind_of_road_dict = {road:[] for road in kind_of_roads}


# In[ ]:


for row in train["EntryStreetName"]:
    for kind_of_road in kind_of_roads:
        try:
            if kind_of_road in row.split(" "):
                kind_of_road_dict[kind_of_road].append(1)
            else:
                kind_of_road_dict[kind_of_road].append(0)
        except:
            kind_of_road_dict[kind_of_road].append(0)


# In[ ]:


len(kind_of_road_dict["Street"])


# In[ ]:


len(train)


# In[ ]:


for road_type in kind_of_roads:
    train[road_type] = kind_of_road_dict[road_type]


# In[ ]:


train[kind_of_roads]


# In[ ]:


train[train[kind_of_roads].sum(axis = 1) > 1]["EntryStreetName"].unique()


# In[ ]:


train.nunique()


# In[ ]:


kind_of_roads = ["Street", "Avenue", "Boulevard", "Road", "Drive", "Parkway", "Highway", "Pkwy", " St", "Ave", "Overpass", "Bypass", "Expressway"]
kind_of_road_dict = {road:[] for road in kind_of_roads}


# In[ ]:


for row in test["EntryStreetName"]:
    for kind_of_road in kind_of_roads:
        try:
            if kind_of_road in row.split(" "):
                kind_of_road_dict[kind_of_road].append(1)
            else:
                kind_of_road_dict[kind_of_road].append(0)
        except:
            kind_of_road_dict[kind_of_road].append(0)


# In[ ]:


for road_type in kind_of_roads:
    test[road_type] = kind_of_road_dict[road_type]


# In[ ]:


print(train["City"].unique())
print(test["City"].unique())


# In[ ]:


# test.groupby(["City"]).apply(np.unique)
test.groupby(["City"]).nunique()


# In[ ]:


train.isna().sum(axis=0)


# In[ ]:


test.isna().sum(axis=0)


# ## Add features
# 
# ##### turn direction: 
# The cardinal directions can be expressed using the equation: $$ \frac{\theta}{\pi} $$
# 
# Where $\theta$ is the angle between the direction we want to encode and the north compass direction, measured clockwise.
# 
# * This is an **important** feature, as shown by janlauge here : https://www.kaggle.com/janlauge/intersection-congestion-eda
# 
# * We can fill in this code in python (e.g. based on: https://www.analytics-link.com/single-post/2018/08/21/Calculating-the-compass-direction-between-two-points-in-Python , https://rosettacode.org/wiki/Angle_difference_between_two_bearings#Python , https://gist.github.com/RobertSudwarts/acf8df23a16afdb5837f ) 
# 
# * TODO: circularize / use angles

# In[ ]:


directions = {
    'N': 0,
    'NE': 1/4,
    'E': 1/2,
    'SE': 3/4,
    'S': 1,
    'SW': 5/4,
    'W': 3/2,
    'NW': 7/4
}


# In[ ]:


train['EntryHeading'] = train['EntryHeading'].map(directions)
train['ExitHeading'] = train['ExitHeading'].map(directions)

test['EntryHeading'] = test['EntryHeading'].map(directions)
test['ExitHeading'] = test['ExitHeading'].map(directions)


# In[ ]:


train['diffHeading'] = train['EntryHeading']-train['ExitHeading']  # TODO - check if this is right. For now, it's a silly approximation without the angles being taken into consideration

test['diffHeading'] = test['EntryHeading']-test['ExitHeading']  # TODO - check if this is right. For now, it's a silly approximation without the angles being taken into consideration

train[['ExitHeading','EntryHeading','diffHeading']].drop_duplicates().head(10)


# In[ ]:


### code if we wanted the diffs, without changing the raw variables:

# train['diffHeading'] = train['ExitHeading'].map(directions) - train['EntryHeading'].map(directions)
# test['diffHeading'] = test['ExitHeading'].map(directions) - test['EntryHeading'].map(directions)


# In[ ]:


train.head()


# * entering and exiting on same street
# * todo: clean text, check if on same boulevard, etc' 

# In[ ]:


train["same_street_exact"] = (train["EntryStreetName"] ==  train["ExitStreetName"]).astype(int)
test["same_street_exact"] = (test["EntryStreetName"] ==  test["ExitStreetName"]).astype(int)


# ### Skip OHE intersections for now - memory issues
# * Intersection IDs aren't unique  etween cities - so we'll make new ones
# 
# * Running fit on just train reveals that **the test data has a "novel" city + intersection!** ( '3Atlanta'!) (We will fix this)
#      * Means we need to be careful when OHEing the data
#      
#  * There are 2,796 intersections, more if we count unique by city (~4K) = many, many columns. gave me memory issues when doing one hot encoding
#      * Could try count or target mean encoding. 
#      
# * For now - ordinal encoding

# In[ ]:


le = preprocessing.LabelEncoder()
# le = preprocessing.OneHotEncoder(handle_unknown="ignore") # will have all zeros for novel categoricals, [can't do drop first due to nans issue , otherwise we'd  drop first value to avoid colinearity


# In[ ]:


train["Intersection"] = train["IntersectionId"].astype(str) + train["City"]
test["Intersection"] = test["IntersectionId"].astype(str) + test["City"]

print(train["Intersection"].sample(6).values)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# pd.concat([train,le.transform(train["Intersection"].values.reshape(-1,1)).toarray()],axis=1).head()


# #### with ordinal encoder - ideally we'd encode all the "new" cols with a single missing value, but it doesn't really matter given that they're Out of Distribution anyway (no such values in train). 
# * So we'll fit on train+Test in order to avoid encoding errors - when using the ordinal encoder! (LEss of a n issue with OHE)

# In[ ]:


pd.concat([train["Intersection"],test["Intersection"]],axis=0).drop_duplicates().values


# In[ ]:


le.fit(pd.concat([train["Intersection"],test["Intersection"]]).drop_duplicates().values)
train["Intersection"] = le.transform(train["Intersection"])
test["Intersection"] = le.transform(test["Intersection"])


# In[ ]:


train.head()


# ### ORIG  OneHotEncode
# ##### We could Create one hot encoding for entry , exit direction fields - but may make more sense to leave them as continous
# 
# 
# * Intersection ID is only unique within a city

# In[ ]:


pd.get_dummies(train["City"],dummy_na=False, drop_first=False).head()


# In[ ]:


# pd.get_dummies(train[["EntryHeading","ExitHeading","City"]].head(),prefix = {"EntryHeading":'en',"ExitHeading":"ex","City":"city"})


# In[ ]:


train = pd.concat([train,pd.get_dummies(train["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)
test = pd.concat([test,pd.get_dummies(test["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)


# In[ ]:


train.shape,test.shape


# In[ ]:


test.head()


# In[ ]:


train.columns


#  #### Approach: We will make 6 predictions based on features we derived - IntersectionId , Hour , Weekend , Month , entry & exit directions .
#  * Target variables will be TotalTimeStopped_p20 ,TotalTimeStopped_p50,TotalTimeStopped_p80,DistanceToFirstStop_p20,DistanceToFirstStop_p50,DistanceToFirstStop_p80 .
#  
#  * I leave in the original IntersectionId just in case there's meaning accidentally encoded in the numbers

# In[ ]:


FEAT_COLS = ["IntersectionId",
             'Intersection',
           'diffHeading',  'same_street_exact',
           "Hour","Weekend","Month",
          'Latitude', 'Longitude',
          'EntryHeading', 'ExitHeading',
            'Atlanta', 'Boston', 'Chicago',
       'Philadelphia']


# In[ ]:


train.head()


# In[ ]:


train.columns


# In[ ]:


X = train[FEAT_COLS]
y1 = train["TotalTimeStopped_p20"]
y2 = train["TotalTimeStopped_p50"]
y3 = train["TotalTimeStopped_p80"]
y4 = train["DistanceToFirstStop_p20"]
y5 = train["DistanceToFirstStop_p50"]
y6 = train["DistanceToFirstStop_p80"]


# In[ ]:


y = train[['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80',
        'DistanceToFirstStop_p20', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']]


# In[ ]:


testX = test[FEAT_COLS]


# In[ ]:


## kaggle kernel performance can be very unstable when trying to use miltuiprocessing
# lr = LinearRegression()
lr = RandomForestRegressor(n_estimators=100,min_samples_split=3)#,n_jobs=3) #different default hyperparams, not necessarily any better


# In[ ]:


## Original: model + prediction per target
#############

lr.fit(X,y1)
pred1 = lr.predict(testX)
lr.fit(X,y2)
pred2 = lr.predict(testX)
lr.fit(X,y3)
pred3 = lr.predict(testX)
lr.fit(X,y4)
pred4 = lr.predict(testX)
lr.fit(X,y5)
pred5 = lr.predict(testX)
lr.fit(X,y6)
pred6 = lr.predict(testX)


# Appending all predictions
all_preds = []
for i in range(len(pred1)):
    for j in [pred1,pred2,pred3,pred4,pred5,pred6]:
        all_preds.append(j[i])   

sub  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")
sub["Target"] = all_preds
sub.to_csv("benchmark_beat_rfr_multimodels.csv",index = False)

print(len(all_preds))


# * ALT : multitask model

# In[ ]:


## New/Alt: multitask -  model for all targets

lr.fit(X,y)
print("fitted")

all_preds = lr.predict(testX)


# In[ ]:


## convert list of lists to format required for submissions
print(all_preds[0])

s = pd.Series(list(all_preds) )
all_preds = pd.Series.explode(s)

print(len(all_preds))
print(all_preds[0])


# In[ ]:


sub  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")
print(sub.shape)
sub.head()


# In[ ]:


sub["Target"] = all_preds.values
sub.sample(5)


# In[ ]:


sub.to_csv("benchmark_beat_rfr_multitask.csv",index = False)


# # Export featurized data
# 
# * Uncomment this to get the features exported for further use. 

# In[ ]:


train.drop("Path",axis=1).to_csv("train_danFeatsV1.csv.gz",index = False,compression="gzip")
test.drop("Path",axis=1).to_csv("test_danFeatsV1.csv.gz",index = False,compression="gzip")

