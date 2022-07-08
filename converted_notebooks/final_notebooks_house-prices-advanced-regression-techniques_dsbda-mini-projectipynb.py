#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction

# > - 1.0 Introduction
#     - 1.1 Importing libraries
#     - 1.2 Collecting the data
#   
# > - 2.0 Preprocessing
#     - 2.1 Dropping irrelevant features
#     - 2.2 Null Value Removal
#         - 2.2.1 Null Values
#         - 2.2.2 Legit
#             - 2.2.2.1 Numeric
#             - 2.2.2.2 Object
#             - 2.2.2.3 Complex
#     - 2.3 Data Encoding
#         - 2.3.1 One Hot Encoding
#     - 2.4 Feature Selection
#         - 2.4.1 High Correlation Filter (Resolved the Dummy Variable Trap)
#         - 2.4.2 Correlation of the target variable with all the features
#     - 2.5 Dimensionality Reduction
#         - 2.5.1 Low Variance Filter
#    
# > - 3.0 Model Training
#      - 3.0.1 Splitting the data
#      - 3.0.2 Standardizing the Data
# - 3.1 Multiple Linear Regression
# - 3.2 Decidion Tree
# - 3.3 Random Forest
# - 3.4 Support Vector Machine
# - 3.5 Gradient Boosting
# - 3.6 Ada Boosting
# - 3.7 Light GBM
# 
# > - 4.0 Final Result
# 
# > - 5.0 Submit

# # 1.0 Introduction
# 

# > In this project we intend to predict the price of the houses with the various given features

# # 1.1 Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score


# In[ ]:


# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm


# In[ ]:


import time
from collections import Counter


# # 1.2 Collecting the data

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


df = pd.concat([train, test], ignore_index=True)
df.head(10)


# In[ ]:


df.head(10)


# In[ ]:


# copying for future purposes
data_0 = df.copy()


# In[ ]:


train.info()


# > There a lot of features with non-numeric data which will be required to be Encoded for our code to parse it.

# > - To Drop:
#     - Id (Irrelevant Data)

# # 2.0 Preprocessing
# # 2.1 Dropping irrelevant features

# In[ ]:


df = df.drop(['Id'], axis=1)


# In[ ]:


df.info()


# # 2.2 Null Value Removal:

# > - Null:	
# 	- MSZoning, Utilities, Exterior1st, Exterior2nd, Electrical, BsmtFullBath(No Bsmt), BsmtHalfBath(No Bsmt), KitchenQual, Functional, GarageYrBlt(Not all), GarageFinish(Not all), GarageQual(Not all), GarageCond(Not all), SaleType
# 
# 
# > - Legit:
# 	- Numeric: LotFrontage, MasVnrArea, BsmtFinSF1, BsmtFinSF2, 
# 		BsmtUnfSF, TotalBsmtSF, GarageYrBlt(Not all), 
# 		GarageFinish(Not all), GarageCars, GarageArea, 
# 		GarageQual(Not all), GarageCond(Not all) 
#     - Object: Alley, MasVnrType, GarageType, MiscFeature, 
# 	- Obj-Num: BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, FireplaceQu, PoolQC, Fence, 

# In[ ]:


i = 0
for x in range(len(df.columns)):
    if df.iloc[:,x].isnull().sum() > 0:
        i += 1
print(i)


# ## 2.2.1 Null Values:
# > Values which are missing thereby have to be filled with median/mode

# In[ ]:


null_num = ['BsmtFullBath', 'BsmtHalfBath']
null_com = ['GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']
null_obj = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'KitchenQual', 'Functional', 'SaleType']


# In[ ]:


for x in null_num:
    df[x].fillna(df[x].median(), inplace = True)

for x in null_obj:
    df[x].fillna(df[x].mode()[0], inplace = True)


# In[ ]:


for x in null_com:
    df[x].fillna(0, inplace=True)


# ## 2.2.2 Legit:
# > Values are not missing just to be replaced with some value
# ### 2.2.2.1 Numeric:
# > Here null values are directly replaced by 0

# In[ ]:


df["LotFrontage"].fillna(0, inplace = True)
df["MasVnrArea"].fillna(0, inplace=True)
df["BsmtFinSF1"].fillna(0, inplace=True)
df["BsmtFinSF2"].fillna(0, inplace=True)
df["BsmtUnfSF"].fillna(0, inplace=True)
df["TotalBsmtSF"].fillna(0, inplace=True)
df["GarageCars"].fillna(0, inplace=True)
df["GarageArea"].fillna(0, inplace=True)
df["LotFrontage"].value_counts()


# ### 2.2.2.2 Object:
# > Here null values are directly replaced by 'No'

# In[ ]:


df["Alley"].fillna('No', inplace = True)
df["MasVnrType"].fillna('No', inplace = True)
df["GarageType"].fillna('No', inplace = True)
df["MiscFeature"].fillna('No', inplace = True)
df["BsmtQual"].fillna('No', inplace = True)
df["BsmtCond"].fillna('No', inplace = True)
df["BsmtExposure"].fillna('No', inplace = True)
df["BsmtFinType1"].fillna('No', inplace = True)
df["BsmtFinType2"].fillna('No', inplace = True)
df["FireplaceQu"].fillna('No', inplace = True)
df["PoolQC"].fillna('No', inplace = True)
df["Fence"].fillna('No', inplace = True)


# ### 2.2.2.3 Complex
# > For the Values with complexity

# In[ ]:


for x in range(df.shape[0]):
    for y in null_com:
        if df.iloc[x,df.columns.get_loc("GarageType")] == 'No':
            df.iloc[x,df.columns.get_loc(y)] = 0
        elif df.iloc[x,df.columns.get_loc("GarageType")] != 'No' and df.iloc[x,df.columns.get_loc(y)] == 'No':
            df.iloc[x,df.columns.get_loc(y)] = df[y].median()


# # 2.3 Data Encoding:
# > Used OneHotEncoder over the whole the object features
# 
# > Total Columns: 303

# In[ ]:


columns_numeric = list(df.dtypes[(df.dtypes=='int64') | (df.dtypes=='float64') ].index)
columns_object = list(df.dtypes[df.dtypes=='object'].index)
print(f"numeric columns: {len(columns_numeric)} \nobject columns: {len(columns_object)}")


# ## 2.3.1 One Hot Encoder

# In[ ]:


df2 = df.copy()
for x in columns_object:
    temp = pd.get_dummies(df2[x],prefix=x)
    df2 = pd.concat([df2,temp],axis=1)
    df2.drop(x,axis=1,inplace=True)
df2.shape


# #### Splitting Target and Feature Variables

# In[ ]:


X = df2.drop(['SalePrice'], axis=1)
y = df2['SalePrice']
train = df2.iloc[:1460,:]


# # 2.4 Feature Selection:

# *Not able to interpret anything from the heatmap as too many features therefore not using*
# > plt.subplots(figsize = (25,20))
# sns.heatmap(df2.corr(method='pearson'), annot=False, linewidths=0.2)

# ## 2.4.1 High Correlation Filter (Resolved the Dummy Variable Trap)
# > Calculated the correlation of all the feature variables with each other and then removed those having correlation above 0.9
# 
# > Removed 10, Total Columns Remaining: 293

# In[ ]:


corr = X.corr(method='pearson')
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = X.columns[columns]
X2 = X[selected_columns]


# ## 2.4.2 Correlation of the target variable with all the features 
# > Calculated the correlation of all the feature variables with the target variable and removed those with absolute val less than 0.05
# 
# > Removed 110, Total Columns Remaining: 183

# In[ ]:


df3 = X2.copy()
df3['SalePrice'] = y
corr = df3.corr(method='pearson')['SalePrice']


# In[ ]:


flag = 0
for x in range(len(corr)):
    if corr[x] < 0.05 and corr[x] > -0.05:
        flag += 1
        # print(f"Dropping column: {df2.columns[x]}: {corr[x]}")
        df3 = df3.drop([X2.columns[x]], axis=1)
        # print()
print(f"Columns dropped: {flag}")


# # 2.5 Dimensionality Reduction
# 
# ## 2.5.1 Low Variance Filter
# 
# > Calculating the variance of all feature columns and removing those with value less than 0.05
# 
# > Removed 11, Total Columns Remaining: 172

# In[ ]:


df11 = df3.copy()
var = df11.var()
i = 0
for x in range(len(var)):
    if var[x] < 0.005:
        i += 1
        df11 = df11.drop([df3.columns[x]], axis=1)
        # print(f"dropping: {df3.columns[x]}")
print(f'Columns dropped: {i}')


# # Outliers Handling
# > Wasn't able to find a model yet to remove the outliers

# # 3.0 Model Training:

# ## 3.0.1 Splitting the Data

# In[ ]:


dataset = df11


# In[ ]:


X = dataset.drop(['SalePrice'], axis=1)
y = dataset['SalePrice']
X_t = X.iloc[:1460,:]
y_t = y.iloc[:1460]
X_test = X.iloc[1460:,:]
y_test = y.iloc[1460:]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size = 0.3, random_state = 0)


# ## 3.0.2 Standardizing the Data

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# # 3.1 Multiple Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
start = time.time()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_val)
time_ML = time.time() - start
acc01 = round(r2_score(y_val, y_pred),4)
print('Linear regression accuracy : ' ,acc01)


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
print('Root Mean Log Squared Error:', np.sqrt(mean_squared_log_error(y_val, y_pred)))
RMLSE_ML = np.sqrt(mean_squared_log_error(y_val, y_pred))


# # 3.2 Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
regr = DecisionTreeRegressor(max_depth=2, random_state=0, max_leaf_nodes=2)
start = time.time()
regr.fit(X_train, y_train)
y_pred01 = regr.predict(X_val)
time_DT = time.time() - start
acc02 = round(r2_score(y_val, y_pred01),4)
print('Decision tree regression accuracy : ' ,acc02)


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred01))
print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred01))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred01)))
print('Root Mean Log Squared Error:', np.sqrt(mean_squared_log_error(y_val, y_pred01)))
RMLSE_DT = np.sqrt(mean_squared_log_error(y_val, y_pred01))


# # 3.3 Random Forest

# In[ ]:


# randomforest = RandomForestRegressor(n_estimators=200, random_state=2)
randomforest = RandomForestRegressor(n_estimators=400, random_state=2, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=None, bootstrap=False)
# randomforest = RandomForestRegressor(n_estimators=110, random_state=2, min_samples_split=6, min_samples_leaf=2, max_features='auto', max_depth=20, bootstrap=True)
start = time.time()
randomforest.fit(X_train, y_train)
y_pred02= randomforest.predict(X_val)
time_RF = time.time() - start
acc03 = round(r2_score(y_val, y_pred02),4)
print('Random Forest Regression accuracy : ' ,acc03)


# ### Hyperparameter Tuning(RF)

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
rf_random.best_params_


# In[ ]:


tuple(rf_random.best_params_)


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred02))  
print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred02))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred02)))
print('Root Mean Log Squared Error:', np.sqrt(mean_squared_log_error(y_val, y_pred02)))
RMLSE_RF = np.sqrt(mean_squared_log_error(y_val, y_pred02))


# # 3.4 Support Vector

# In[ ]:


from sklearn.svm import SVR

regr01 = SVR(kernel='linear')
start = time.time()
regr01.fit(X_train, y_train)
y_pred03 = regr01.predict(X_val)
time_SV = time.time() - start
acc04 = round(r2_score(y_val, y_pred03),4)
print('SVR accuracy : ' ,acc04)


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred03))  
print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred03))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred03)))
print('Root Mean Log Squared Error:', np.sqrt(mean_squared_log_error(y_val, y_pred03)))
RMLSE_SV = np.sqrt(mean_squared_log_error(y_val, y_pred03))


# # 3.5 Gradient Boosting
# > Model with highest accuracy

# In[ ]:


gb = GradientBoostingRegressor(n_estimators=1400, random_state=4, min_samples_split=10, min_samples_leaf=1, max_features='sqrt', max_depth=20, learning_rate=0.01)
start = time.time()
gb.fit(X_train, y_train)
y_pred04= gb.predict(X_val)
time_GB = time.time() - start
acc05 = round(r2_score(y_val, y_pred04),4)
print('Gradient Boosting accuracy : ' ,acc05)


# ### Hyperparameter Tuning(GB)

# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Learning rate
learning_rate = [1, 0.5, 0.25, 0.1, 0.05, 0.01]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate}
print(random_grid)

gb = GradientBoostingRegressor()
gb_random = RandomizedSearchCV(estimator = gb, param_distributions = random_grid, n_iter = 1, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
gb_random.fit(X_train, y_train)
gb_random.best_params_


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred04))  
print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred04))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred04)))
print('Root Mean Log Squared Error:', np.sqrt(mean_squared_log_error(y_val, y_pred04)))
RMLSE_GB = np.sqrt(mean_squared_log_error(y_val, y_pred04))


# # 4.0 Final Result

# In[ ]:


models= pd.DataFrame({ 
"Model" : ["MultipleLinearRegression", "DecisionTreeRegression", "RandomForestRegression", "SVR","Gradient Boosting"],
"Accuracy" : [acc01, acc02, acc03, acc04, acc05],
"Time" : [time_ML, time_DT, time_RF, time_SV, time_GB],
"RMLSE" : [RMLSE_ML, RMLSE_DT, RMLSE_RF, RMLSE_SV, RMLSE_GB]
})
model_notime = pd.DataFrame({ 
"Model" : ["MultipleLinearRegression", "DecisionTreeRegression", "RandomForestRegression", "SVR","Gradient Boosting"],
"Accuracy" : [acc01, acc02, acc03, acc04, acc05]
})
model_time = pd.DataFrame({ 
"Model" : ["MultipleLinearRegression", "DecisionTreeRegression", "RandomForestRegression", "SVR","Gradient Boosting"],
"Time" : [time_ML, time_DT, time_RF, time_SV, time_GB]
})
models


# In[ ]:


models.sort_values(by="RMLSE")


# # 5.0 Submissions
# > Used Gradient Boostiong as it is giving the highest accuracy

# In[ ]:


gb = GradientBoostingRegressor(n_estimators=1400, random_state=2, min_samples_split=10, min_samples_leaf=1, max_features='sqrt', max_depth=20, learning_rate=0.01)
gb.fit(X_t, y_t)
y_pred = gb.predict(X_test)
y_t


# In[ ]:


submit = data_0.iloc[1460:,0]
submit = pd.DataFrame(submit)


# In[ ]:


submit['SalePrice'] = y_pred


# In[ ]:


submit


# In[ ]:


submit.to_csv('Submission.csv', index=False)


# In[ ]:


submit.shape


# In[ ]:




