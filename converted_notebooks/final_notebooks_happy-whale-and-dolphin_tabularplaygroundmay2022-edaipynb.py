#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import learning_curve


# # Table of Contents
# 
# 
# .
# 
# 
# 
# 
# 
# 
# * [Exploration](#exploration)
#     - [Encoding for visualization](#visualizatio_encoding)
#     - [Basic correlation](#visualization_basic_correlation)
#     - [Create some variable f_27](#f_27_maker)
#     - [Visualize float](#visualize_float)
#     - [Visualize int](#Visualize_int)
# 
# 
# .
# 
# 
# 
# 
# 
# * [Prepare Training Data](#PrepareTrainingData)
#     - [Features ingeneering](#featuresengeneering)
#     - [Categorical Features](#categoricalfeatures)
#     - [Int Features](#intfeatures)
#     - [Float Features](#floatfeatures)
#     - [Reassemble Train Set](#buildtrain)
#     - [Split Train Set](#splittrain)
#     

# <a id="exploration"></a>
# # Exploration

# In[ ]:


train = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
test  = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')

train = train.drop('id', axis = 1)


# Objectif: 
# 
# For this challenge, you are given (simulated) manufacturing control data and are tasked to predict whether the machine is in state 0 or state 1. The data has various feature interactions that may be important in determining the machine state.
# 

# FLOAT COLUMNS
# 
# col: f_00 ------- min: -4.599855946898881 ------- max: 4.749300580789184  ------- std: 0.9988878389440928 
# 
# col: f_01 ------- min: -4.682198839295782 ------- max: 4.81569922916624  ------- std: 0.9991926727254712 
# 
# col: f_02 ------- min: -4.642676261581941 ------- max: 4.961982157037259  ------- std: 1.0005144690470156 
# 
# col: f_03 ------- min: -4.658816423702026 ------- max: 4.454920087577203  ------- std: 1.000174900967013 
# 
# col: f_04 ------- min: -4.7485008094997285 ------- max: 4.948982867422749  ------- std: 1.0001665118150118 
# 
# col: f_05 ------- min: -4.750214349359606 ------- max: 4.9718811207257465  ------- std: 0.9998749002981248 
# 
# col: f_06 ------- min: -4.84291865900587 ------- max: 4.82266765638437  ------- std: 0.9999419519724667 
# 
# col: f_19 ------- min: -11.280941430768523 ------- max: 12.079667330851391  ------- std: 2.3160259192433807 
# 
# col: f_20 ------- min: -11.257916931897675 ------- max: 11.47532524994984  ------- std: 2.400494067416627 
# 
# col: f_21 ------- min: -13.310145923543772 ------- max: 14.455426205132513  ------- std: 2.4847059911491693 
# 
# col: f_22 ------- min: -11.85352955078141 ------- max: 11.344080226183534  ------- std: 2.450797464299929 
# 
# col: f_23 ------- min: -12.301097278095106 ------- max: 12.247100149129777  ------- std: 2.4534050736556265 
# 
# col: f_24 ------- min: -11.416188952158356 ------- max: 12.389843507506578  ------- std: 2.3869412057119423 
# 
# col: f_25 ------- min: -11.918306113601654 ------- max: 12.529178899213573  ------- std: 2.4169588222033807 
# 
# col: f_26 ------- min: -14.300576720342912 ------- max: 12.913041459864832  ------- std: 2.4760201301290983 
# 
# col: f_28 ------- min: -1229.7530519319007 ------- max: 1229.5625773272975  ------- std: 238.77305399789742 

# INT COLUMNS
# 
# col: f_07 ------- min: 0 ------- max: 15  ------- std: 1.656172288564021 
# 
# col: f_08 ------- min: 0 ------- max: 16  ------- std: 1.590954697005824 
# 
# col: f_09 ------- min: 0 ------- max: 14  ------- std: 1.637706391678827 
# 
# col: f_10 ------- min: 0 ------- max: 14  ------- std: 1.645952747516177 
# 
# col: f_11 ------- min: 0 ------- max: 13  ------- std: 1.5374872022637933 
# 
# col: f_12 ------- min: 0 ------- max: 16  ------- std: 1.7628346989204557 
# 
# col: f_13 ------- min: 0 ------- max: 12  ------- std: 1.5384257185365562 
# 
# col: f_14 ------- min: 0 ------- max: 14  ------- std: 1.359212904781433 
# 
# col: f_15 ------- min: 0 ------- max: 14  ------- std: 1.5690933363355812 
# 
# col: f_16 ------- min: 0 ------- max: 15  ------- std: 1.5601689054454666 
# 
# col: f_17 ------- min: 0 ------- max: 14  ------- std: 1.4676749022348898 
# 
# col: f_18 ------- min: 0 ------- max: 13  ------- std: 1.5647834756413543 
# 
# col: f_29 ------- min: 0 ------- max: 1  ------- std: 0.4755835980007996 
# 
# col: f_30 ------- min: 0 ------- max: 2  ------- std: 0.8189894377801633 
# 
# col: target ------- min: 0 ------- max: 1  ------- std: 0.49981766417948514 
# 

# OneHotEncoding
# 
# target
# 
# f_29
# 
# f_30

# MinMaxScaler        
# 
# f_00
# 
# f_01
# 
# f_02
# 
# f_03
# 
# f_04
# 
# f_05
# 
# f_06
# 
# f_19
# 
# f_20
# 
# f_21
# 
# f_22
# 
# f_23
# 
# f_24
# 
# f_25
# 
# f_26
# 
# f_28

# Normalization
# 
# f_07
# 
# f_08
# 
# f_09
# 
# f_10
# 
# f_11
# 
# f_12
# 
# f_13
# 
# f_14
# 
# f_15
# 
# f_16
# 
# f_17
# 
# f_18
# 

# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


# here I see float, int and object


# In[ ]:


train.select_dtypes('int').shape


# In[ ]:


for col in train.select_dtypes('int'):
    print(f'col: {col} ------- min: {train[col].min()} ------- max: {train[col].max()}  ------- std: {train[col].std()} ')


# In[ ]:


train.select_dtypes('float').shape


# In[ ]:


# let's try to find out if some columns in float are already normalized
for col in train.select_dtypes('float'):
    print(f'col: {col} ------- min: {train[col].min()} ------- max: {train[col].max()}  ------- std: {train[col].std()} ')


# In[ ]:


train.select_dtypes('object').shape


# In[ ]:


for col in train.select_dtypes('object'):
    print(f'col: {col}')


# In[ ]:


train['f_27'].value_counts()


# In[ ]:


plt.figure(figsize = (30, 40))

columns = 4
i = 0

for col in train.select_dtypes('int'):
    plt.subplot(int(train.shape[1] / columns + 1), columns, i + 1)
    sns.distplot(train[col])
    i += 1


# In[ ]:


# here f_29 and f_30 look closely to the target so I think I will OneHotEncode them as a category


# In[ ]:


plt.figure(figsize = (30, 40))

columns = 4
i = 0

for col in train.select_dtypes('float'):
    plt.subplot(int(train.shape[1] / columns + 1), columns, i + 1)
    sns.distplot(train[col])
    i += 1


# In[ ]:


train['target'].value_counts()


# In[ ]:


def create_pie(df, target_variable, figsize=(10, 10)):
    print(df[target_variable].value_counts())
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(df[target_variable].value_counts().values, labels=df[target_variable].value_counts().index, autopct='%1.2f%%', textprops={'fontsize': 10})
    ax.axis('equal')
    plt.title(target_variable)
    plt.show()
    
    

create_pie(train, 'target')


# In[ ]:


def bar_chart(feature, title):
    survived = train[train['target']==1][feature].value_counts()
    dead = train[train['target']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['on','off']
    df.plot(kind='bar',stacked=True, figsize=(15,10)).set(title=f'{title}')
    
    
    


bar_chart('target', 'target state')


# In[ ]:


'''cat_f27 = train['f_27'].value_counts().reset_index()
cat_f27 = cat_f27['index'].to_list()

def categorize_f27(text):
    text = str(text)
    if text in cat_f27:
        index = cat_f27.index(text)
        return int(index)


train['f_27_b'] = train['f_27'].apply(categorize_f27)
train.head()

this code work but took infinite time so it's replace by the next cell

'''


# <a id="visualizatio_encoding"></a>
# # Encoding for visualization

# In[ ]:


numerical_encoder = OrdinalEncoder()
f = np.array(train['f_27']).reshape(-1, 1)

train['f_27_b'] = numerical_encoder.fit_transform(f)
train['f_27_b'] = train['f_27_b'].astype(int)


# In[ ]:


train.head()


# In[ ]:


corr_matrix = train.select_dtypes(np.number).corr()
corr_matrix[(corr_matrix < 0.01) & (corr_matrix > -0.01)] = 0
corr = corr_matrix["target"].sort_values(ascending = False)
print(corr)
indexNames = corr[abs(corr.values) < 0.4].index.values
indexNames = np.setdiff1d(indexNames, ['state','target'])


# <a id="visualization_basic_correlation"></a>
# # Basic correlation

# In[ ]:


plt.figure(figsize = (30, 30))
sns.heatmap(corr_matrix, annot=True, fmt = '0.1');


# In[ ]:


# let's take a closer look to our correlations


training = train.head(1000)
training.head()

# create a sub df
a = training[['f_21', 'f_24', 'f_23', 'f_09', 'f_22','f_26'
              , 'f_00', 'f_01', 'f_29', 'f_15'
              ,'f_08', 'f_28', 'f_05', 'f_02', 'f_30',
              'f_18', 'f_14', 'f_25', 'f_13', 'f_10','f_20'
              , 'f_16', 'f_11', 'f_19', 'f_27_b']]


g = sns.PairGrid(a)
g = g.map_upper(sns.regplot, scatter_kws={'alpha':0.15}, line_kws={'color':'red'})


# <a id="f_27_maker"></a>
# # Create some variable f_27

# In[ ]:


# here I could take first letter and his first 3 letter has header
train['f_27_l'] = train['f_27'].str[0]
train['f_27_h'] = train['f_27'].str[:3]
train['f_27_h']


# In[ ]:


numerical_encoder = OrdinalEncoder()
f = np.array(train['f_27_l']).reshape(-1, 1)

train['f_27_ln'] = numerical_encoder.fit_transform(f)
train['f_27_ln'] = train['f_27_ln'].astype(int)
sns.distplot(train['f_27_ln']);


# In[ ]:


train['f_27_l'].value_counts()


# In[ ]:


numerical_encoder = OrdinalEncoder()
f = np.array(train['f_27_h']).reshape(-1, 1)

train['f_27_hn'] = numerical_encoder.fit_transform(f)
train['f_27_hn'] = train['f_27_hn'].astype(int)
sns.distplot(train['f_27_hn']);


# In[ ]:


train['f_27_h'].value_counts()


# In[ ]:


from sklearn.compose import make_column_selector

numerical_features = make_column_selector(dtype_include = np.integer)
float_features = make_column_selector(dtype_include = np.float)



# In[ ]:


train.head()


# <a id="visualize_float"></a>
# # Visualize float

# Transform features by scaling each feature to a given range.
# 
# 
# This estimator scales and translates each feature individually such that it is in the given range on the 
# training set, e.g. between zero and one.
# 
# 
# The transformation is given by:
# 
# 
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
# 
# 
# where min, max = feature_range.
# 
# 
# This transformation is often used as an alternative to zero mean, unit variance scaling.

# In[ ]:


# Well maybe I need to normalize my float value
from sklearn.compose import make_column_transformer



negpos_encoder = MinMaxScaler()
float_transformer = make_column_transformer((negpos_encoder,
                                                 float_features))
float_encoded = float_transformer.fit_transform(train)


float_encoded.shape


# In[ ]:


tmp = pd.DataFrame(float_encoded)
tmp.head()


# In[ ]:


plt.figure(figsize = (30, 40))
tmp = pd.DataFrame(float_encoded)
columns = 4
i = 0

for col in tmp.select_dtypes('float'):
    plt.subplot(int(tmp.shape[1] / columns + 1), columns, i + 1)
    sns.distplot(tmp[col])
    i += 1


# In[ ]:


tmp = pd.DataFrame(float_encoded)
tmp['target'] = train['target']

corr_matrix = tmp.select_dtypes(np.number).corr()
corr_matrix[(corr_matrix < 0.01) & (corr_matrix > -0.01)] = 0

corr = corr_matrix["target"].sort_values(ascending = False)
print(corr)
indexNames = corr[abs(corr.values) < 0.4].index.values
indexNames = np.setdiff1d(indexNames, ['state','target'])


# In[ ]:


plt.figure(figsize = (30, 30))
sns.heatmap(corr_matrix, annot=True, fmt = '0.1');
del corr_matrix


# In[ ]:


tmpp = tmp.head(1000)

a = tmpp[[9, 12, 11, 10, 14,0, 1, 15,5,2,13,8,7]]
g = sns.PairGrid(a)
g = g.map_upper(sns.regplot, scatter_kws={'alpha':0.15}, line_kws={'color':'red'})

del tmpp



# <a id="Visualize_int"></a>
# # Visualize int

# In[ ]:


# normalization of integer value

tmpp = train
int_features = make_column_selector(dtype_include = np.integer)
# I need to remove f_27_b from here because I think it's a category

int_encoder = MinMaxScaler()
int_transformer = make_column_transformer((int_encoder,
                                           int_features))
int_encoded = int_transformer.fit_transform(tmpp)

del tmpp
int_encoded.shape


# In[ ]:


plt.figure(figsize = (30, 40))
tmp_int = pd.DataFrame(int_encoded)
columns = 4
i = 0

for col in tmp_int.select_dtypes('float'):
    plt.subplot(int(tmp_int.shape[1] / columns + 1), columns, i + 1)
    sns.distplot(tmp_int[col])
    i += 1


# In[ ]:


tmp_int.head()


# In[ ]:


tmp_int = pd.DataFrame(int_encoded)
tmp_int['target'] = train['target']

corr_matrix = tmp_int.select_dtypes(np.number).corr()
corr_matrix[(corr_matrix < 0.01) & (corr_matrix > -0.01)] = 0

corr = corr_matrix["target"].sort_values(ascending = False)
print(corr)
indexNames = corr[abs(corr.values) < 0.4].index.values
indexNames = np.setdiff1d(indexNames, ['state','target'])


# In[ ]:


plt.figure(figsize = (30, 30))
sns.heatmap(corr_matrix, annot=True, fmt = '0.1');
del corr_matrix


# In[ ]:


tmpp = tmp_int.head(1000)

a = tmpp[[14, 12, 2, 8, 13,11, 1, 7,6,3,9]]
g = sns.PairGrid(a)
g = g.map_upper(sns.regplot, scatter_kws={'alpha':0.15}, line_kws={'color':'red'})

del tmpp


# <a id="PrepareTrainingData"></a>
# # Prepare Training Data

# In[ ]:


train = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
train = train.drop('id', axis = 1)


# <a id="featuresengeneering"></a>
# # Features ingeneering

# In[ ]:


# here I could take first letter and his first 3 letter has header
train['f_27_l'] = train['f_27'].str[0]
train['f_27_h'] = train['f_27'].str[:3]


# <a id="categoricalfeatures"></a>
# # Categorical Features

# OneHotEncoding
# 
# target
# 
# f_29
# 
# f_30
# 
# f_27
# 
# f_27_l
# 
# f_27_h

# In[ ]:


# Label Encoder
f = train['target']
label_encoder = LabelBinarizer()
train['target_bin'] =  label_encoder.fit_transform(f)
target_bin = label_encoder.fit_transform(f)


# In[ ]:


# Features Encoder
f_29 = np.array(train['f_29']).reshape(-1, 1)
f_30 = np.array(train['f_30']).reshape(-1, 1)
f_27 = np.array(train['f_27']).reshape(-1, 1)
f_27_l = np.array(train['f_27_l']).reshape(-1, 1)
f_27_h = np.array(train['f_27_h']).reshape(-1, 1)


ohe_encoder = OneHotEncoder()
f_29_ohe = ohe_encoder.fit_transform(f_29)
f_30_ohe = ohe_encoder.fit_transform(f_30)
f_27_ohe = ohe_encoder.fit_transform(f_27)
f_27_l_ohe = ohe_encoder.fit_transform(f_27_l)
f_27_h_ohe = ohe_encoder.fit_transform(f_27_h)


# In[ ]:


cat_df = pd.DataFrame()
cat_df = cat_df.sparse.from_spmatrix(f_29_ohe, index=None, columns=None)
cat_df = cat_df.sparse.from_spmatrix(f_30_ohe, index=None, columns=None)
cat_df = cat_df.sparse.from_spmatrix(f_27_ohe, index=None, columns=None)
cat_df = cat_df.sparse.from_spmatrix(f_27_l_ohe, index=None, columns=None)
cat_df = cat_df.sparse.from_spmatrix(f_27_h_ohe, index=None, columns=None)
cat_df.head(10)


# <a id="floatfeatures"></a>
# # Float Features

# MinMaxScaler        
# 
# f_00
# 
# f_01
# 
# f_02
# 
# f_05
# 
# f_19
# 
# f_20
# 
# f_21
# 
# f_22
# 
# f_23
# 
# f_24
# 
# f_25
# 
# f_26
# 
# f_28

# In[ ]:


f_00 = np.array(train['f_00']).reshape(-1, 1)
f_01 = np.array(train['f_01']).reshape(-1, 1)
f_02 = np.array(train['f_02']).reshape(-1, 1)
f_05 = np.array(train['f_05']).reshape(-1, 1)
f_19 = np.array(train['f_19']).reshape(-1, 1)
f_20 = np.array(train['f_20']).reshape(-1, 1)
f_21 = np.array(train['f_21']).reshape(-1, 1)
f_22 = np.array(train['f_22']).reshape(-1, 1)
f_23 = np.array(train['f_23']).reshape(-1, 1)
f_24 = np.array(train['f_24']).reshape(-1, 1)
f_25 = np.array(train['f_25']).reshape(-1, 1)
f_26 = np.array(train['f_26']).reshape(-1, 1)
f_28 = np.array(train['f_28']).reshape(-1, 1)

float_encoder = MinMaxScaler()
f_00_mne = float_encoder.fit_transform(f_00)
f_01_mne = float_encoder.fit_transform(f_01)
f_02_mne = float_encoder.fit_transform(f_02)
f_05_mne = float_encoder.fit_transform(f_05)
f_19_mne = float_encoder.fit_transform(f_19)
f_20_mne = float_encoder.fit_transform(f_20)
f_21_mne = float_encoder.fit_transform(f_21)
f_22_mne = float_encoder.fit_transform(f_22)
f_23_mne = float_encoder.fit_transform(f_23)
f_24_mne = float_encoder.fit_transform(f_24)
f_25_mne = float_encoder.fit_transform(f_25)
f_26_mne = float_encoder.fit_transform(f_26)
f_28_mne = float_encoder.fit_transform(f_28)

# np to series
f_00_mne_s = pd.Series(f_00_mne.reshape(-1))
f_01_mne_s = pd.Series(f_01_mne.reshape(-1))
f_02_mne_s = pd.Series(f_02_mne.reshape(-1))
f_05_mne_s = pd.Series(f_05_mne.reshape(-1))
f_19_mne_s = pd.Series(f_19_mne.reshape(-1))
f_20_mne_s = pd.Series(f_20_mne.reshape(-1))
f_21_mne_s = pd.Series(f_21_mne.reshape(-1))
f_22_mne_s = pd.Series(f_22_mne.reshape(-1))
f_23_mne_s = pd.Series(f_23_mne.reshape(-1))
f_24_mne_s = pd.Series(f_24_mne.reshape(-1))
f_25_mne_s = pd.Series(f_25_mne.reshape(-1))
f_26_mne_s = pd.Series(f_26_mne.reshape(-1))
f_28_mne_s = pd.Series(f_28_mne.reshape(-1))

float_result = pd.concat([f_00_mne_s, f_01_mne_s, f_02_mne_s,
                          f_05_mne_s, f_19_mne_s, f_20_mne_s,
                          f_21_mne_s, f_22_mne_s, f_23_mne_s,
                          f_24_mne_s, f_25_mne_s, f_26_mne_s,
                          f_28_mne_s], axis=1)


# In[ ]:


tmp_int = pd.DataFrame(float_result)
tmp_int['target'] = train['target']

corr_matrix = tmp_int.select_dtypes(np.number).corr()
corr_matrix[(corr_matrix < 0.04) & (corr_matrix > -0.04)] = 0

corr = corr_matrix["target"].sort_values(ascending = False)
print(corr)
indexNames = corr[abs(corr.values) < 0.4].index.values
indexNames = np.setdiff1d(indexNames, ['state','target'])


# In[ ]:


plt.figure(figsize = (30, 30))
sns.heatmap(corr_matrix, annot=True, fmt = '0.1');
del corr_matrix


# In[ ]:


float_frames = np.hstack((f_00_mne, f_01_mne, f_19_mne,
                          f_21_mne, f_22_mne, f_23_mne,
                          f_24_mne, f_26_mne))
float_frames.shape

float_pandas = pd.concat([f_00_mne_s, f_01_mne_s, f_19_mne_s,
                          f_21_mne_s, f_22_mne_s, f_23_mne_s,
                          f_24_mne_s, f_26_mne_s],
                          axis=1)


# <a id="intfeatures"></a>
# # Int Features

# Normalization
# 
# 
# f_08
# 
# f_09
# 
# f_10
# 
# f_11
# 
# f_13
# 
# f_14
# 
# f_15
# 
# f_16
# 
# f_18
# 

# In[ ]:


f_08 = train['f_08'] / train['f_08'].max()
f_09 = train['f_09'] / train['f_09'].max()
f_10 = train['f_10'] / train['f_10'].max()
f_11 = train['f_11'] / train['f_11'].max()
f_13 = train['f_13'] / train['f_13'].max()
f_14 = train['f_14'] / train['f_14'].max()
f_15 = train['f_15'] / train['f_15'].max()
f_16 = train['f_16'] / train['f_16'].max()
f_18 = train['f_18'] / train['f_18'].max()

int_result = pd.concat([f_08, f_09, f_10,
                        f_11, f_13, f_14,
                        f_15, f_16, f_18,], axis=1)


# In[ ]:


tmp_int = pd.DataFrame(int_result)
tmp_int['target'] = train['target']

corr_matrix = tmp_int.select_dtypes(np.number).corr()
corr_matrix[(corr_matrix < 0.04) & (corr_matrix > -0.04)] = 0

corr = corr_matrix["target"].sort_values(ascending = False)
print(corr)
indexNames = corr[abs(corr.values) < 0.4].index.values
indexNames = np.setdiff1d(indexNames, ['state','target'])


# In[ ]:


plt.figure(figsize = (30, 30))
sns.heatmap(corr_matrix, annot=True, fmt = '0.1');
del corr_matrix


# In[ ]:


int_frames = np.vstack((f_09, f_15, f_11)).reshape(-1, 3)
int_frames.shape


int_pandas = pd.concat([f_09, f_15, f_11], axis=1)


# <a id="buildtrain"></a>
# # Reassemble Train Set

# In[ ]:


train_pandas = pd.concat([int_pandas, float_pandas, cat_df], axis=1)
train_pandas.head()


# <a id="splittrain"></a>
# # Split Train Set

# In[ ]:


y = target_bin
X = train_pandas
X.shape


# In[ ]:


# split our data into train and test
X_train, X_test, y_train, y_test = train_test_split( X, y,
                                                    test_size=0.3,
                                                    random_state=42)

# and we also need a validation set 
X_test, X_val, y_test, y_val = train_test_split( X_test, y_test,
                                                    test_size=0.3,
                                                    random_state=42)


# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


lasso = Lasso(random_state=0, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30,)

#alphas = (0.0001, 0.00001, 0.001)

print(alphas)

tuned_parameters = [{"alpha": alphas}]
n_folds = 3

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=True)
clf.fit(X_train, y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]


# In[ ]:


print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

print(clf.best_score_)
print(clf.best_params_)

#0.10555359896683603
#0.10207233723245313
#0.10547133235777102
#{'alpha': 0.0001}


# In[ ]:


plt.figure(figsize = (30, 15))
plt.semilogx(alphas, scores)

std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, "b--")
plt.semilogx(alphas, scores - std_error, "b--")

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel("CV score +/- std error")
plt.xlabel("alpha")
plt.axhline(np.max(scores), linestyle="--", color=".5")
plt.xlim([alphas[0], alphas[-1]])


# In[ ]:


# To answer this question we use the LassoCV object that sets its alpha
# parameter automatically from the data by internal cross-validation (i.e. it
# performs cross-validation on the training data it receives).
# We use external cross-validation to see how much the automatically obtained
# alphas differ across different cross-validation folds.

from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

lasso_cv = LassoCV(alphas=alphas, random_state=0, max_iter=10000)
k_fold = KFold(9)

print("Answer to the bonus question:", "how much can you trust the selection of alpha?")
print()
print("Alpha parameters maximising the generalization score on different")
print("subsets of the data:")
for k, (train, test) in enumerate(k_fold.split(X_train, y_train)):
    lasso_cv.fit(X_train, y_train)
    print(
        "[fold {0}] alpha: {1:.5f}, score: {2:.5f}".format(
            k, lasso_cv.alpha_, lasso_cv.score(X_test, y_test)
        )
    )
print()
'''print("Answer: Not very much since we obtained different alphas for different")
print("subsets of the data and moreover, the scores for these alphas differ")
print("quite substantially.")'''
print('relatively confident since the score seems to be regular')

plt.show()


# [fold 0] alpha: 0.00010, score: 0.10207
# 
# [fold 1] alpha: 0.00010, score: 0.10207
# 
# [fold 2] alpha: 0.00010, score: 0.10207
# 
# [fold 3] alpha: 0.00010, score: 0.10207
# 
# [fold 4] alpha: 0.00010, score: 0.10207
# 
# [fold 5] alpha: 0.00010, score: 0.10207
# 
# [fold 6] alpha: 0.00010, score: 0.10207
# 
# [fold 7] alpha: 0.00010, score: 0.10207
# 
# [fold 8] alpha: 0.00010, score: 0.10207

# In[ ]:


coefficients = clf.best_estimator_.coef_
importance = np.abs(coefficients)
importance.shape


# In[ ]:





# In[ ]:


np.array(X_train.columns)[importance > 1]


# In[ ]:


np.array(X_train.columns)[importance < 1]


# In[ ]:


X.shape


# In[ ]:


X_train = X_train[:5000]
y_train = y_train[:5000]


# In[ ]:


# Perceptron

from sklearn.neural_network import MLPClassifier

CLF = MLPClassifier(max_iter=1500)
CLF.fit(X_train, y_train)
print(CLF.score(X_train, y_train))
print(CLF.score(X_test, y_test))


train_score = f'NN -- train -- {CLF.score(X_train, y_train)}'
test_score = f'NN -- test -- {CLF.score(X_test, y_test)}'
#0.522
#0.5159047619047619

# 1000
#0.86
#0.5878677248677249

# 5000


# In[ ]:


# 0.6071909194315833
# 1000
# 0.9380597254041236

# 5000


roc_auc_score(y_train, CLF.predict_proba(X_train)[:, 1])


# In[ ]:


from sklearn.metrics import roc_curve, roc_auc_score

preds = CLF.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, preds, pos_label=1)

auc = roc_auc_score(y_test, preds)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='random')
plt.title(f'AUC: {auc}')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')


# In[ ]:


index = 1
x_pred = CLF.predict(X_test)
print('prediction: \n', x_pred)

X_prediction = CLF.predict(X_test)
print(f'test accuracy: \t{accuracy_score(y_test,X_prediction)}')
#print(f'{classification_report(y_test, X_prediction)}')
print(classification_report(y_test, X_prediction))

# Plot Actual vs. Predicted 
y_test = y_test
predicted = CLF.predict(X_test)

plt.figure(figsize=(28, 8))
plt.title("5000", fontsize=20)
plt.scatter(y_test, predicted,
            color="purple", marker="o", facecolors="none")

plt.plot([0, 2], [0, 2], "darkorange", lw=2)
plt.xlim(0, 2)
plt.ylim(0, 2)
plt.xlabel("Y test", fontsize=16)
plt.ylabel("Predicted Y", fontsize=16)
plt.show()


# In[ ]:


# https://www.kaggle.com/code/dlaststark/tps-may22-what-tf-again
import itertools
def plot_confusion_matrix(cm, classes):

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix', fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')
    plt.tight_layout()


# In[ ]:


y_pred = (predicted > 0.5).astype(int)
print(classification_report(y_test, X_prediction))

cnf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
np.set_printoptions(precision=2)
plt.figure(figsize=(12, 5))
plot_confusion_matrix(cnf_matrix, classes=[0, 1])


# In[ ]:


x_pred = CLF.predict(X_test)
print('prediction: \n', x_pred)

X_prediction = CLF.predict(X_test)
print(f'test accuracy: \t{accuracy_score(y_test,X_prediction)}')
#print(f'{classification_report(y_test, X_prediction)}')
print(classification_report(y_test, X_prediction))

# Plot Actual vs. Predicted 
y_test = y_test
predicted = CLF.predict(X_test)

plt.figure(figsize=(28, 8))
plt.title("all images -- Actual vs. Predicted SVM model -- normalized ", fontsize=20)
plt.scatter(y_test, predicted,
            color="purple", marker="o", facecolors="none")

plt.plot([0, 9], [0, 9], "darkorange", lw=2)
plt.xlim(0, 9)
plt.ylim(0, 9)
plt.xlabel("Y test", fontsize=16)
plt.ylabel("Predicted Y", fontsize=16)
plt.show()


N,train_score, val_score = learning_curve(CLF, X_train, y_train,
                                   train_sizes=np.linspace(0.2,1.0,10 ),cv=3)

plt.figure(figsize=(28, 8))
plt.title("5000 ", fontsize=20)
plt.plot(N, train_score.mean(axis = 1), label ='Train')
plt.plot(N, val_score.mean(axis = 1), label ='Validation')
plt.xlabel('train sizes')
plt.legend();


confusion_matrix(y_test, CLF.predict(X_test))


# In[ ]:


confusion_matrix(y_test, CLF.predict(X_test))


# In[ ]:


# split our data into train and test
X_train, X_test, y_train, y_test = train_test_split( X, y,
                                                    test_size=0.3,
                                                    random_state=42)

X_train = X_train[:15000]
y_train = y_train[:15000]

CLF = MLPClassifier(max_iter=1500)
CLF.fit(X_train, y_train)
print('train score', CLF.score(X_train, y_train))
print('test score', CLF.score(X_test, y_test))
print('roc train score', roc_auc_score(y_train, CLF.predict_proba(X_train)[:, 1]))
print('roc test score', roc_auc_score(y_test, CLF.predict_proba(X_test)[:, 1]))

preds = CLF.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, preds, pos_label=1)

auc = roc_auc_score(y_test, preds)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='random')
plt.title(f'AUC: {auc}')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')


# In[ ]:


x_pred = CLF.predict(X_test)
print('prediction: \n', x_pred)

X_prediction = CLF.predict(X_test)
print(f'test accuracy: \t{accuracy_score(y_test,X_prediction)}')
#print(f'{classification_report(y_test, X_prediction)}')
print(classification_report(y_test, X_prediction))

# Plot Actual vs. Predicted 
y_test = y_test
predicted = CLF.predict(X_test)

plt.figure(figsize=(28, 8))
plt.title("all images -- Actual vs. Predicted SVM model -- normalized ", fontsize=20)
plt.scatter(y_test, predicted,
            color="purple", marker="o", facecolors="none")

plt.plot([0, 9], [0, 9], "darkorange", lw=2)
plt.xlim(0, 9)
plt.ylim(0, 9)
plt.xlabel("Y test", fontsize=16)
plt.ylabel("Predicted Y", fontsize=16)
plt.show()


N,train_score, val_score = learning_curve(CLF, X_train, y_train,
                                   train_sizes=np.linspace(0.2,1.0,10 ),cv=3)

plt.figure(figsize=(28, 8))
plt.title("15000 ", fontsize=20)
plt.plot(N, train_score.mean(axis = 1), label ='Train')
plt.plot(N, val_score.mean(axis = 1), label ='Validation')
plt.xlabel('train sizes')
plt.legend();


confusion_matrix(y_test, CLF.predict(X_test))


# In[ ]:


# split our data into train and test
X_train, X_test, y_train, y_test = train_test_split( X, y,
                                                    test_size=0.3,
                                                    random_state=42)


CLF = MLPClassifier(max_iter=1500)
CLF.fit(X_train, y_train)
print('train score', CLF.score(X_train, y_train))
print('test score', CLF.score(X_test, y_test))
print('roc train score', roc_auc_score(y_train, CLF.predict_proba(X_train)[:, 1]))
print('roc test score', roc_auc_score(y_test, CLF.predict_proba(X_test)[:, 1]))

preds = CLF.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, preds, pos_label=1)

auc = roc_auc_score(y_test, preds)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='random')
plt.title(f'AUC: {auc}')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')


# In[ ]:


x_pred = CLF.predict(X_test)
print('prediction: \n', x_pred)

X_prediction = CLF.predict(X_test)
print(f'test accuracy: \t{accuracy_score(y_test,X_prediction)}')
#print(f'{classification_report(y_test, X_prediction)}')
print(classification_report(y_test, X_prediction))

# Plot Actual vs. Predicted 
y_test = y_test
predicted = CLF.predict(X_test)

plt.figure(figsize=(28, 8))
plt.title("all images -- Actual vs. Predicted SVM model -- normalized ", fontsize=20)
plt.scatter(y_test, predicted,
            color="purple", marker="o", facecolors="none")

plt.plot([0, 9], [0, 9], "darkorange", lw=2)
plt.xlim(0, 9)
plt.ylim(0, 9)
plt.xlabel("Y test", fontsize=16)
plt.ylabel("Predicted Y", fontsize=16)
plt.show()


confusion_matrix(y_test, CLF.predict(X_test))

