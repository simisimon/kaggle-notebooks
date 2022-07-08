#!/usr/bin/env python
# coding: utf-8

# ## This Featurewiz+SimpleXGBModel Template has Received Top Scores in Multiple Hackathons - Current Ranks in the following Hackathons are:
# AnalyticsVidhya Hackathons: https://datahack.analyticsvidhya.com/contest/all/
# 1.  Big_Mart Sales Prediction Score: 1147  -- Rank 250 out of 41,361 = That's a Top <1% Rank!!
# 1.  Loan Status Predictions Score 0.791  -- Rank 850 out of 67,424 - Top 1.25% Rank
# 
# Machine Hack Hackathons: https://www.machinehack.com/hackathon
# 1.  Machine Hack Flight Ticket Score 0.9389 -- Rank 165 out of 2723 - Top 6% Rank!
# 1.  Machine Hack Data Scientist Salary class Score 0.417 -- Rank 58 out of 1547 - Top 3.7% Rank! (Autoviml Score was 0.329 -- less than 0.417 of Featurewiz+Simple even though an NLP problem!)
# 1.  MCHACK Book Price NLP Score 0.7336 -- Rank 104 Autoviml NLP problem and should have done better
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.dates as md
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from mpl_toolkits.mplot3d import Axes3D

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from load_kaggle import load_kaggle


# In[ ]:


subm, train, test = load_kaggle()


# In[ ]:


print(train.shape)
train.head()


# In[ ]:


print(test.shape)
test.head()


# ## Remember to Restart your Kernel immediately after installing featurewiz

# In[ ]:


get_ipython().system('pip install featurewiz --ignore-installed --no-deps')
#!python3 -m pip install git+https://github.com/AutoViML/featurewiz.git


# ## After you install Featurewiz, you must install xlrd

# In[ ]:


get_ipython().system('pip install xlrd')


# In[ ]:


import featurewiz as FW


# In[ ]:


target = 'Cover_Type'
idcols = ['Id']


# In[ ]:


### In regression problems, you might want to convert target into log(target)
#df[target] = (df[target] - np.mean(df[target]))/np.std(df[target])
#train[target] = np.log(train[target].values)


# ## Let's create some TPS specific features and then select the best using featurewiz
# ###  Thanks to @Luca Massaron for condensing the features:
# https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/293612#1620087

# In[ ]:


# extra feature engineering
def r(x):
    if x+180>360:
        return x-180
    else:
        return x+180

def fe(df):
    df['EHiElv'] = df['Horizontal_Distance_To_Roadways'] * df['Elevation']
    df['EViElv'] = df['Vertical_Distance_To_Hydrology'] * df['Elevation']
    df['Aspect2'] = df.Aspect.map(r)
    ### source: https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/293373
    df["Aspect"][df["Aspect"] < 0] += 360
    df["Aspect"][df["Aspect"] > 359] -= 360
    df.loc[df["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
    df.loc[df["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
    df.loc[df["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
    df.loc[df["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
    df.loc[df["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
    df.loc[df["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255
    ########
    df['Highwater'] = (df.Vertical_Distance_To_Hydrology < 0).astype(int)
    df['EVDtH'] = df.Elevation - df.Vertical_Distance_To_Hydrology
    df['EHDtH'] = df.Elevation - df.Horizontal_Distance_To_Hydrology * 0.2
    df['Euclidean_Distance_to_Hydrolody'] = (df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)**0.5
    df['Manhattan_Distance_to_Hydrolody'] = df['Horizontal_Distance_To_Hydrology'] + df['Vertical_Distance_To_Hydrology']
    df['Hydro_Fire_1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
    df['Hydro_Fire_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
    df['Hydro_Road_1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])
    df['Hydro_Road_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_1'] = abs(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_2'] = abs(df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])
    df['Hillshade_3pm_is_zero'] = (df.Hillshade_3pm == 0).astype(int)
    return df

train = fe(train)
test = fe(test)

# Summed features pointed out by @craigmthomas (https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/292823)
soil_features = [x for x in train.columns if x.startswith("Soil_Type")]
wilderness_features = [x for x in train.columns if x.startswith("Wilderness_Area")]

train["soil_type_count"] = train[soil_features].sum(axis=1)
test["soil_type_count"] = test[soil_features].sum(axis=1)

train["wilderness_area_count"] = train[wilderness_features].sum(axis=1)
test["wilderness_area_count"] = test[wilderness_features].sum(axis=1)
print(train.shape)


# ## There are 73 columns now - we have added 18 columns
# Let us run featurewiz with just 1million rows

# In[ ]:


### We first run it with default setting 1000 nrows. Then change it to all nrows=None
train_best, test_best = FW.featurewiz(train,  target, test_data=test, nrows=1000000)


# # Now let us select the best variables using featurewiz 

# In[ ]:


print(train_best.shape)
train_best.head()


# In[ ]:


print(test_best.shape)
test_best.head(2)


# ## Featurewiz has found 10 best features out of 56 - let's build a model with those best features

# In[ ]:


preds = test_best.columns.tolist()
preds


# ## This simple LightGBM model works wonders since it is highly effective in many competitions

# In[ ]:


outputs = FW.simple_lightgbm_model(X_XGB=train_best[preds], Y_XGB=train_best[target],
                               X_XGB_test=test_best[preds], modeltype='Classification')


# In[ ]:


preds = test_best.columns.tolist()
len(preds)


# In[ ]:


outputs[0]


# In[ ]:


y_preds = outputs[0]
y_preds


# In[ ]:


y_preds.mean()


# In[ ]:


subm[target] = y_preds


# In[ ]:


### if multi-class, do this  ###
inverse_dict = {0: 2, 1: 1, 2: 3, 3: 7, 4: 6, 5: 4}
y_preds = subm[target].map(inverse_dict).values.astype(int)
y_preds


# In[ ]:


subm = test[idcols]
subm[target] = y_preds.astype(int)
subm.head()


# In[ ]:


print(subm[target].mean())
subm[target].hist()


# In[ ]:


train[target].mean()


# In[ ]:


subm.to_csv('submission.csv',index=False)


# In[ ]:




