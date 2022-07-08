#!/usr/bin/env python
# coding: utf-8

# # Iowa House Price Prediction
# ---

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="dark")

import math


# In[ ]:


# Load in both datasets and merge them
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

df = pd.concat([df_train, df_test], axis=0)
df.head()


# In[ ]:


df.info()


# ## 1 - Data Pre-processing
# ---

# In[ ]:


# Look at the total missing data values
plt.figure(figsize=(24,9))
sns.heatmap(df.isnull());


# In[ ]:


# Delete columns with a lot of data missing
drop_features = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']
df = df.drop(drop_features, axis=1)


# In[ ]:


# Look at all of the empty features
empty_features = []

for col in df.columns:
    if df[col].isnull().nunique() == 2:
        empty_features.append(col)

# Look at how many features are missing
empty_features = [x for x in empty_features if x not in drop_features]
empty_features.remove('SalePrice')

# Loop through the features that only have a couple of missing values and set them based on the mean
removed_features = []
for feature in empty_features:
    if df[feature].isnull().sum() <= 5:
        df[feature] = df[feature].fillna(value=df[feature].mode()[0])
        removed_features.append(feature)

# See what is left
empty_features = [x for x in empty_features if x not in removed_features]
df[empty_features].isnull().sum()


# In[ ]:


# Look at the correlation matrix for the remaining missing data values
plt.figure(figsize=(24,9))
sns.heatmap(df[empty_features].isnull().corr(), annot=True);


# In[ ]:


# Do the remaining values by hand

# Set the following values to their respective means
df['LotFrontage'] = df['LotFrontage'].fillna(value=df['LotFrontage'].mean())
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(value=df['GarageYrBlt'].mean())

# Set the following values to zero
df['MasVnrArea'] = df['MasVnrArea'].fillna(value=0)

# Set the following values to 'None'
to_none = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

for feature in to_none:
    df[feature] = df[feature].fillna(value='None')


# In[ ]:


# All the missing data has been filled in appropriately
sns.heatmap(df.isnull());


# In[ ]:


# See how many missing values are left in total, excluding the target variable
df.drop('SalePrice', axis=1).isnull().sum().sum()


# ## 2 - Exploratory Data Analysis
# ---

# ### 2.1 Eliminating Skew

# In[ ]:


# Start by looking at the distributions of the housing prices

# Compare the skew of the price distribution to the skew of the log-transform
df['LogSalePrice'] = df['SalePrice'].map(lambda x : math.log(x))

pd.Series({
    'LogPriceSkew' : df.LogSalePrice.skew(),
    'SalePrice' : df.SalePrice.skew()
})


# In[ ]:


# We seem to have eleminated the skew of the price data
fig, ax = plt.subplots(1,2,figsize=(12,8))
sns.histplot(x='SalePrice', data=df, kde=True, ax=ax[0])
sns.histplot(x='LogSalePrice', data=df, kde=True, ax=ax[1]);


# ### 2.2 Separating Columns

# In[ ]:


# Separate the columns
ordinal_cols= list(df.columns[df.columns.str.contains('Yr|Year')])
print('ordinal/temporal columns:\n',ordinal_cols)
nominal_cols= list(set(df.select_dtypes(include=['object']).columns) - set(ordinal_cols))
print('nominal columns:\n', nominal_cols)
numeric_cols= list(set(df.select_dtypes(exclude=['object']).columns) - set(ordinal_cols))
print('numeric columns:\n',numeric_cols)


# ### 2.3 Correlation Matrix

# In[ ]:


plt.figure(figsize=(16,9))
sns.heatmap(df[list(set(numeric_cols) - {'SalePrice'})].corr());


# ### 2.4 Ordinal Feature Analysis

# In[ ]:


# Look at the distributions of different ordinal features
fig, ax = plt.subplots(1,4,figsize=(24, 9))

for i in range(len(ordinal_cols)):
    sns.histplot(x=ordinal_cols[i], y='LogSalePrice', data=df, ax=ax[i], cmap='mako', bins=35)


# ### 2.5 Nominal Feature Engineering

# In[ ]:


# Since nominal features are in the form of text, split them into number classes to make them easier to interpret
# and then put the classes in order to maximize the correlation between the nominal feature and the log-price
nominal_df = pd.DataFrame()
for nominal_col in nominal_cols:
    categories = df[nominal_col].unique()
    averages = dict()
    for category in categories:
        averages[category] = df[df[nominal_col] == category]['LogSalePrice'].mean()

    averages = dict(sorted(averages.items(), key=lambda item: item[1]))

    i = 0
    for average in averages.keys():
        averages[average] = i
        i+=1

    nominal_df = pd.concat([nominal_df, df[nominal_col].replace(averages)], axis=1)

nominal_df = pd.concat([nominal_df, df['LogSalePrice']], axis=1)
nominal_df.corr()['LogSalePrice']


# In[ ]:


# Stitch together a new dataframe with the new nominal features (we want to keep the old one just in case)
final_df = pd.concat([df.drop(nominal_cols, axis=1), nominal_df.drop('LogSalePrice', axis=1)], axis=1)
final_df.head()


# In[ ]:


final_df.drop('SalePrice', axis=1).corr()['LogSalePrice']


# ## 3 - Model Construction
# ---
# We will focus on constructing a few different models in this section, and evaluating their respective performances:
# 
# - Linear Regression
# - Random Forest
# - Decision Trees with Gradient Boosting
# 
# For the sake of the competition, we will always get the metrics of the logged predictions, as these give us a rough idea of what score we will attain in the competition.
# 
# **Finally, to avoid bias in the models below, we will choose a fixed random state.**

# In[ ]:


fixed_state = 42


# ### 3.1 Train-Test-Validation Split & Metrics

# In[ ]:


# Import some sklearn metrics
from sklearn.metrics import r2_score

# Define some metrics functions for later
def get_results(actual, predictions):
    res_df = pd.DataFrame([(x[0], x[1]) for x in zip(actual, predictions)], columns=['LogActual', 'LogPredicted']).sort_values(by='LogActual').reset_index().drop('index', axis=1).rename_axis('index')
    res_df['LogResiduals'] = res_df['LogActual'] - res_df['LogPredicted']

    res_df['Actual'] = res_df['LogActual'].map(lambda x : math.exp(x))
    res_df['Predicted'] = res_df['LogPredicted'].map(lambda x : math.exp(x))
    res_df['Residuals'] = res_df['Actual'] - res_df['Predicted']
    return res_df

def get_metrics(actual, predicted):
    residuals = actual - predicted
    return pd.Series({
        'MSE' : sum(residuals**2) / len(actual),
        'RMSE' : math.sqrt(sum(residuals**2) / len(actual)),
        'MAPE' : 100 * sum(abs(residuals / actual)) / len(actual),
        'MPE' : 100 * sum(residuals / actual) / len(actual),
        'R2' : r2_score(actual, predicted)
    })


# In[ ]:


# Import some modules
from sklearn.model_selection import train_test_split

# Split the dataset that we are concerned with into the test, validation and training data
def split_dataframe(df):
    return (
        df.iloc[:1460],
        df.iloc[1460:]
    )

X_features = df.drop(['SalePrice', 'LogSalePrice'], axis=1).columns
y_features = 'LogSalePrice'

final_train_val, final_test = split_dataframe(final_df)

# Set the validation set to be 0.07, which is 100 of the training values
X_train_val, y_train_val = final_train_val[X_features].values, final_train_val[y_features].values
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.07, random_state=fixed_state)
X_test = final_test[X_features].values

X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape # y_test is the data which is unknown


# ### 3.2 Linear Regression
# 
# The linear regression model does not have any hyper-parameters, so we will keep the model that we get.
# 
# **Expected Score : 0.102**

# In[ ]:


from sklearn.linear_model import LinearRegression

# Fit the model and look at the loggged results
lin_model = LinearRegression().fit(X_train, y_train)
lin_predictions = lin_model.predict(X_val)

lin_results = get_results(y_val, lin_predictions)
lin_metrics = get_metrics(lin_results.LogActual, lin_results.LogPredicted)

print(lin_metrics)


# In[ ]:


sns.displot(lin_results.Residuals, kde=True);


# In[ ]:


plt.figure(figsize=(16,9))
plt.plot(lin_results.index, lin_results.Actual, ls='', marker='.', label="Actual")
plt.plot(lin_results.index, lin_results.Predicted, ls='', marker='.', label="Predicted")
for x in lin_results.index:
    plt.plot([x,x], [lin_results.loc[x].Actual, lin_results.loc[x].Predicted], alpha=0.5, c='yellow')
plt.legend();


# ### 3.3 Random Forest
# 
# We will use the validation set to tune the hyper-parameters for this model.
# 
# **Expected Score : 0.112**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Construct a forest model
for_model_base = RandomForestRegressor(random_state=fixed_state)
for_model_base.fit(X_train, y_train)

for_predictions_base = for_model_base.predict(X_val)

for_results_base = get_results(y_val, for_predictions_base)
for_metrics_base = get_metrics(for_results_base.LogActual, for_results_base.LogPredicted)

print(for_metrics_base)


# The first model did not yield anything particularly exciting. Let this be our baseline, and try a grid search.

# In[ ]:


# from sklearn.model_selection import RandomizedSearchCV

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 600, stop = 1000, num = 5)]

# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']

# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
# max_depth.append(None)

# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]

# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]

# # Method of selecting samples for training each tree
# bootstrap = [True, False]

# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# random_grid


# In[ ]:


# for_model = RandomForestRegressor()
# for_random = RandomizedSearchCV(estimator = for_model, param_distributions = random_grid, n_iter = 15, cv = 2, verbose=2, random_state=fixed_state, n_jobs = -1)
# for_random.fit(X_train, y_train)
# for_random.best_params_


# In[ ]:


# for_model_best = for_random.best_estimator_

# # Construct the best forest model
# for_model_best.fit(X_train, y_train)

# for_predictions_best = for_model_best.predict(X_val)

# for_results_best = get_results(y_val, for_predictions_best)
# for_metrics_best = get_metrics(for_results_best.LogActual, for_results_best.LogPredicted)

# print(for_metrics_best)


# Grid search did not reveal any interesting developments. The code has been left here anyway. Let us try a decision tree with gradient boosting.
# 
# ### 3.4 Decision Tree with Gradient Boosting
# 
# **Expected Score : 0.102**

# In[ ]:


# Import the gradient booster
from sklearn.ensemble import GradientBoostingRegressor

gra_model = GradientBoostingRegressor(random_state=fixed_state, n_estimators=200)
gra_model.fit(X_train, y_train)

gra_predictions = gra_model.predict(X_val)

gra_results = get_results(y_val, gra_predictions)
gra_metrics = get_metrics(gra_results.LogActual, gra_results.LogPredicted)

print(gra_metrics)


# In[ ]:


# As there are a large number of estimators, let us try to determine where to stop so that we can formulate a more efficient model

# From the plot below, it looks as if anywhere between 25 and 50 estimators is appropriate before we start overfitting.

# Get a list of the test scores at each iteration
test_score = []
for i, pred in enumerate(gra_model.staged_predict(X_val)):
    test_score.append(gra_model.loss_(y_val, pred))

n_estimators = np.arange(len(test_score))

plt.figure(figsize=(16,8))
plt.title('Deviance in Scores')
plt.xlabel('Boosting Iterations')
plt.ylabel('Score')

plt.plot(n_estimators, test_score, label='Test Score')
plt.plot(n_estimators, gra_model.train_score_, label='Training Score')

plt.legend();


# In[ ]:


# Compare the importances of features in the model to the correlations that we predicted before
feature_importances = {x:y for x, y in zip(X_features, gra_model.feature_importances_)}
feature_importances = pd.Series(dict(sorted(feature_importances.items(), key=lambda item: item[1])))

corr_values = pd.Series(final_df.drop('SalePrice', axis=1).corr()['LogSalePrice']).drop('LogSalePrice')

df_imp_corr = pd.concat([feature_importances, corr_values], axis=1).rename_axis('index').set_axis(['Importance', 'LogSalePrice'], axis='columns')

# Plot on a double barplot
fig, ax = plt.subplots(1, 2, figsize=(18, 18), sharex=True)

ax[1].axes.yaxis.set_visible(False)

sns.barplot(x=df_imp_corr.Importance, y=df_imp_corr.index, ax=ax[0])
sns.barplot(x=df_imp_corr.LogSalePrice, y=df_imp_corr.index, ax=ax[1]);

ax[0].set_ylabel("Feature")
ax[1].set_xlabel("Correlation Value")

plt.tight_layout(h_pad=2)


# ### 3.5 Model Conclusion
# 
# Compare and contrast model performances. It looks as if the linear model wins this round with the lowest RMSE of logged residuals.

# In[ ]:


final_metrics = pd.concat([lin_metrics, for_metrics_base, gra_metrics], axis=1).set_axis(['Linear', 'RandomForest', 'GradientBoost'], axis='columns')
final_metrics


# ## 4 - Test Submissions
# 
# In this section we will submit the results of the linear model.

# In[ ]:


# Fit the entire train-validation set to the model
lin_model_final = LinearRegression().fit(X_train_val, y_train_val)
lin_predictions_final = lin_model_final.predict(X_test)
lin_predictions_final = [math.exp(x) for x in lin_predictions_final]

df_predictions = pd.DataFrame([(x,y) for x,y in zip(df_test.index, lin_predictions_final)]).set_axis(['Id', 'SalePrice'], axis='columns').set_index('Id')
df_predictions.to_csv('submission.csv')

