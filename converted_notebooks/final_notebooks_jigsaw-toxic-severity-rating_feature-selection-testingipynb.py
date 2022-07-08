#!/usr/bin/env python
# coding: utf-8

# **[Feature Engineering Home Page](https://www.kaggle.com/learn/feature-engineering)**
# 
# ---
# 

# # Introduction
# 
# Often you'll have dozens or even hundreds of features after various encodings and feature generation. This can lead to two problems. First, the more features you have, the more likely you are to overfit to the training and validation sets. This will cause your model to perform worse at generalizing to new data.
# 
# Secondly, the more features you have, the longer it will take to train your model and optimize hyperparameters. Also, when building user-facing products, you'll want to make inference as fast as possible. Using fewer features will speed up inference at the cost of performance.
# 
# To help with these issues, you'll want to use feature selection techniques to keep the most informative features for your model.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.preprocessing import LabelEncoder

ks = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv',
                 parse_dates=['deadline', 'launched'])

# Drop live projects
ks = ks.query('state != "live"')

# Add outcome column, "successful" == 1, others are 0
ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))

# Timestamp features
ks = ks.assign(hour=ks.launched.dt.hour,
               day=ks.launched.dt.day,
               month=ks.launched.dt.month,
               year=ks.launched.dt.year)

# Label encoding
cat_features = ['category', 'currency', 'country']
encoder = LabelEncoder()
encoded = ks[cat_features].apply(encoder.fit_transform)

data_cols = ['goal', 'hour', 'day', 'month', 'year', 'outcome']
baseline_data = ks[data_cols].join(encoded)


# Add generated features from tutorial 3.

# In[ ]:


cat_features = ['category', 'currency', 'country']
interactions = pd.DataFrame(index=ks.index)
for col1, col2 in itertools.combinations(cat_features, 2):
    new_col_name = '_'.join([col1, col2])
    # Convert to strings and combine
    new_values = ks[col1].map(str) + "_" + ks[col2].map(str)
    label_enc = LabelEncoder()
    interactions[new_col_name] = label_enc.fit_transform(new_values)
baseline_data = baseline_data.join(interactions)

launched = pd.Series(ks.index, index=ks.launched, name="count_7_days").sort_index()
count_7_days = launched.rolling('7d').count() - 1
count_7_days.index = launched.values
count_7_days = count_7_days.reindex(ks.index)

baseline_data = baseline_data.join(count_7_days)

def time_since_last_project(series):
    # Return the time in hours
    return series.diff().dt.total_seconds() / 3600.

df = ks[['category', 'launched']].sort_values('launched')
timedeltas = df.groupby('category').transform(time_since_last_project)
timedeltas = timedeltas.fillna(timedeltas.max())

baseline_data = baseline_data.join(timedeltas.rename({'launched': 'time_since_last_project'}, axis=1))


# In[ ]:


from sklearn import metrics

def get_data_splits(dataframe, valid_fraction=0.1):
    valid_fraction = 0.1
    valid_size = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_size * 2]
    # valid size == test size, last two sections of the data
    valid = dataframe[-valid_size * 2:-valid_size]
    test = dataframe[-valid_size:]
    
    return train, valid, test

def train_model(train, valid):
    feature_cols = train.columns.drop('outcome')

    dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])

    param = {'num_leaves': 64, 'objective': 'binary', 
             'metric': 'auc', 'seed': 7}
    print("Training model!")
    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 
                    early_stopping_rounds=10, verbose_eval=False)

    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid['outcome'], valid_pred)
    print(f"Validation AUC score: {valid_score:.4f}")
    return bst


# # Univariate Feature Selection
# 
# The simplest and fastest methods are based on univariate statistical tests. For each feature, measure how strongly the target depends on the feature using a statistical test like $\chi^2$ or ANOVA.
# 
# From the scikit-learn feature selection module, `feature_selection.SelectKBest` returns the K best features given some scoring function. For our classification problem, the module provides three different scoring functions: $\chi^2$, ANOVA F-value, and the mutual information score. The F-value measures the linear dependency between the feature variable and the target. This means the score might underestimate the relation between a feature and the target if the relationship is nonlinear. The mutual information score is nonparametric and so can capture nonlinear relationships.
# 
# With `SelectKBest`, we define the number of features to keep, based on the score from the scoring function. Using `.fit_transform(features, target)` we get back an array with only the selected features.

# In[ ]:


from sklearn.feature_selection import SelectKBest, f_classif

feature_cols = baseline_data.columns.drop('outcome')

# Keep 5 features
selector = SelectKBest(f_classif, k=5)

X_new = selector.fit_transform(baseline_data[feature_cols], baseline_data['outcome'])


# In[ ]:


X_new


# However, I've done something wrong here. The statistical tests are calculated using all of the data. This means information from the validation and test sets could influence the features we keep, introducing a source of leakage. This means we should select features using only a training set.

# In[ ]:


feature_cols = baseline_data.columns.drop('outcome')
train, valid, _ = get_data_splits(baseline_data)

# Keep 5 features
selector = SelectKBest(f_classif, k=5)

X_new = selector.fit_transform(train[feature_cols], train['outcome'])
X_new


# You should notice that the selected features are different than when I used the entire dataset. Now we have our selected features, but it's only the feature values for the training set. To drop the rejected features from the validation and test sets, we need to figure out which columns in the dataset were kept with `SelectKBest`. To do this, we can use `.inverse_transform` to get back an array with the shape of the original data.

# In[ ]:


# Get back the features we've kept, zero out all other features
selected_features = pd.DataFrame(selector.inverse_transform(X_new), 
                                 index=train.index, 
                                 columns=feature_cols)
selected_features.head()


# This returns a DataFrame with the same index and columns as the training set, but all the dropped columns are filled with zeros. We can find the selected columns by choosing features where the variance is non-zero.

# In[ ]:


# Dropped columns have values of all 0s, so var is 0, drop them
selected_columns = selected_features.columns[selected_features.var() != 0]

# Get the valid dataset with the selected features.
valid[selected_columns].head()


# # L1 regularization
# 
# Univariate methods consider only one feature at a time when making a selection decision. Instead, we can make our selection using all of the features by including them in a linear model with L1 regularization. This type of regularization (sometimes called Lasso) penalizes the absolute magnitude of the coefficients, as compared to L2 (Ridge) regression which penalizes the square of the coefficients.
# 
# As the strength of regularization is increased, features which are less important for predicting the target are set to 0. This allows us to perform feature selection by adjusting the regularization parameter. We choose the parameter by finding the best performance on a hold-out set, or decide ahead of time how many features to keep.
# 
# For regression problems you can use `sklearn.linear_model.Lasso`, or `sklearn.linear_model.LogisticRegression` for classification. These can be used along with `sklearn.feature_selection.SelectFromModel` to select the non-zero coefficients. Otherwise, the code is similar to the univariate tests.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

train, valid, _ = get_data_splits(baseline_data)

X, y = train[train.columns.drop("outcome")], train['outcome']

# Set the regularization parameter C=0.01
logistic = LogisticRegression(C=0.1, penalty="l1", random_state=7).fit(X, y)
model = SelectFromModel(logistic, prefit=True)

X_new = model.transform(X)
X_new


# Similar to the univariate tests, we get back an array with the selected features. Again, we will want to convert these to a DataFrame so we can get the selected columns.

# In[ ]:


# Get back the kept features as a DataFrame with dropped columns as all 0s
selected_features = pd.DataFrame(model.inverse_transform(X_new), 
                                 index=X.index,
                                 columns=X.columns)

# Dropped columns have values of all 0s, keep other columns 
selected_columns = selected_features.columns[selected_features.var() != 0]


# # Principle Component Analysis
# 
# When you have hundreds of numerical features you'll typically want to perform some sort of dimensionality reduction process to represent the data whith fewer features. Principle Component Analysis (PCA) learns a new representation for your data that reduces the number of features while retaining information. 
# 
# PCA learns a set of linearly uncorrelated component vectors, the number of components defined by the user. The first component points in the direction of the greatest variance in the original data. The second component points in the direction of the next highest variance, with the constraint that it is orthogonal to the first component. Each higher component follows similarly, they point in the direction of highest variance given the constraint of orthogonality.
# 
# Scikit-learn provides this as `sklearn.decomposition.PCA`. Our data here has only 3 numerical features so this isn't a great example, but I can still show how it works. Fitting the components is straightforward, it follows the sklearn API. However, PCA is sensitive to the scale of the input features, so you'll typically want to use `StandardScaler` to standardize before fitting.

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

train, valid, _ = get_data_splits(baseline_data)
feature_cols = ['goal', 'count_7_days', 'time_since_last_project']

scaler = StandardScaler()
pca = PCA(n_components=2, random_state=7)

pca.fit(scaler.fit_transform(train[feature_cols]))


# Here we've learned two components

# In[ ]:


pca.components_


# With the components learned, we want to replace the existing features with transformed values. This transformation is really just the matrix product of the original variables with the (transposed) components.
# 
# First, transform the original features with `pca.transform`. Remember that you need to scale the data the same as when fitting the components. Then, drop the original features and add the new features.

# In[ ]:


pca_features = pd.DataFrame(pca.transform(scaler.transform(train[feature_cols])),
                            index=train.index).add_prefix('pca_')
train_pca = train.drop(feature_cols, axis=1).join(pca_features)


# In[ ]:


train_pca.head()


# # Boruta feature selection
# 
# The final method I'll discuss is Boruta. This method is the slowest, but also attempts to keep all the relevant features.
# 
# Boruta firsts creates a clone of the original dataset then shuffles the values in each feature creating "shadow features". Then it trains a tree-based model on the merged original and shadow datasets to get importances for each feature.
# 
# The features that have a higher importance than the best shadow feature are recorded and the process is repeated. Features that are doing better than the shadow features, above random chance, are selected and removed from the dataset. Features that go 15 iterations without being better than the shadow features are removed. This process continues until all of the features are selected or rejected.
# 
# We can use the BorutaPy library to implement this in Python. It conforms to the scikit-learn API, so it's easy to use as well.

# In[ ]:


from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

train, valid, _ = get_data_splits(baseline_data)

feature_cols = train.columns.drop("outcome")

# BorutaPy doesn't work with DataFrames, so need to get the Numpy arrays.
X = train[feature_cols].values
y = train['outcome'].values
    
# Define random forest classifier, utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(class_weight='balanced', max_depth=5, 
                            n_jobs=-1, random_state=7)

# define Boruta feature selection method
selector = BorutaPy(rf, n_estimators='auto', random_state=7, verbose=2)

# Fit the Boruta selector
selector.fit(X, y)


# BorutaPy gives us an attribute `.support_` which we can use to get the selected features directly.

# In[ ]:


print(selector.support_)

# Get the selected columns
selected_columns = feature_cols[selector.support_]
selected_columns


# In this case, we're keeping all the columns. We have relatively few features here so this is not a typical case.
# 
# Next up, you'll be using these methods to select the best features for the TalkingData dataset.

# ---
# **[Feature Engineering Home Page](https://www.kaggle.com/learn/feature-engineering)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
