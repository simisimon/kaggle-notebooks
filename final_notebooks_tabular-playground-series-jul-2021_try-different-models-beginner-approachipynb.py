#!/usr/bin/env python
# coding: utf-8

# **MUST READ**
# 
# I Triend to folllow the advices given by the Kaggle grandmaster. A great blog post -
# [Progressively approaching Kaggle](https://towardsdatascience.com/progressively-approaching-kaggle-f58db71a42a9?gi=8278e4053c97)

# ## Importing Packages

# In[ ]:


import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rcParams["figure.figsize"] = 8, 5

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold

from xgboost import XGBRegressor, XGBRFRegressor

import eli5
from eli5.sklearn import PermutationImportance

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


# ## Setting Random seed

# In[ ]:


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# ## Defining File Paths

# In[ ]:


DATASET_DIR = '/kaggle/input/tabular-playground-series-jul-2021/'
TRAIN_CSV = os.path.join(DATASET_DIR, 'train.csv')
TEST_CSV = os.path.join(DATASET_DIR, 'test.csv')
SAMPLE_SUBMISSION_CSV = os.path.join(DATASET_DIR, 'sample_submission.csv')


# ## Exploring Dataset

# In[ ]:


dataset = pd.read_csv(TRAIN_CSV)
dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


p = sns.displot(dataset, x="deg_C", kind="kde");


# In[ ]:


sns.displot(dataset, x="relative_humidity", kind="kde");


# In[ ]:


sns.displot(dataset, x="absolute_humidity", kind="kde");


# In[ ]:


sensors = ["sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5"]

sns.pairplot(dataset[sensors]);


# In[ ]:


other_feature = ["deg_C", "relative_humidity", "absolute_humidity"]

sns.pairplot(dataset[other_feature]);


# In[ ]:


target_columns = ['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides']

sns.pairplot(dataset[target_columns]);


# ## Data Processing

# > Extracting Day, Month, DayOfWeek feature from `date_time` column

# In[ ]:


def get_processed_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset_copy = dataset.copy()
    dataset_copy['date_time'] = pd.to_datetime(dataset_copy['date_time'])
    dataset_copy['month'] = dataset_copy['date_time'].dt.month
    dataset_copy['day'] = dataset_copy['date_time'].dt.day
    dataset_copy['day_of_week'] = dataset_copy['date_time'].dt.dayofweek    
    
    # Winter – December, January and February. 
    # Spring – March, April and May. 
    # Summer – June, July and August.
    # Autumn – September, October and November.
    dataset_copy['winter_season'] = dataset_copy['month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
    dataset_copy['spring_season'] = dataset_copy['month'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    dataset_copy['summer_season'] = dataset_copy['month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
    dataset_copy['autumn_season'] = dataset_copy['month'].apply(lambda x: 1 if x in [9, 10, 11] else 0)
        
    dataset_copy = dataset_copy.drop('month', axis=1)    
    
    return dataset_copy


# In[ ]:


dataset_copy = get_processed_dataset(dataset)
dataset_copy.head()


# In[ ]:


dataset_copy.columns


# ## Preparing Dataset

# In[ ]:


excluded_feature_columns = ['date_time', 'target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides']

feature_columns = [column_name for column_name in dataset_copy.columns if column_name not in excluded_feature_columns]

target_columns = ['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides']


# In[ ]:


shuffled_dataset = dataset_copy.sample(frac=1).reset_index(drop=True)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(dataset_copy[feature_columns], 
                                                    dataset_copy[target_columns], 
                                                    test_size=0.20, 
                                                    random_state=RANDOM_STATE)

print("Training InputSet Shape: ", X_train.shape)
print("Training Target Shape: ", y_train.shape)
print("Testing InputSet Shape: ", X_test.shape)
print("Testing Target Shape: ", y_test.shape)


# ## Model Training

# In[ ]:


import enum

class ModelName(enum.Enum):
    LinearRegression = "LinearRegression"
    Ridge = "Ridge"
    DecisionTreeRegressor = "DecisionTreeRegressor"
    RandomForestRegressor = "RandomForestRegressor"
    XGBRegressor = "XGBRegressor"
    XGBRFRegressor = "XGBRFRegressor"


# In[ ]:


def get_trained_model(model,
                      X_train: pd.DataFrame,
                      y_true: pd.Series,
                      target_name: str):
    model.fit(X_train, y_true)
    
    y_pred = model.predict(X_train)
    
    mse = mean_squared_error(y_true, y_pred)
    
    rmsle = np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))
    
    score = model.score(X_train, y_true)
    
    print("Model Score: {0} || MSE: {1:.4f} || RMSLE: {2:.4f}".format(score, mse, rmsle))        
    return model, rmsle


# In[ ]:


def validate_trained_model(model,
                          X_test: pd.DataFrame,
                          y_true: pd.Series,
                          target_name: str) -> float:
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_true, y_pred)
    
    rmsle = np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))
    
    print("Model for {0},  MSE: {1:.4f} || RMSLE: {2:.4f}".format(target_name, mse, rmsle))
    return rmsle


# In[ ]:


def get_model_from_factory(model_name: ModelName):
    if model_name == ModelName.LinearRegression:
        return LinearRegression(normalize=True)
    elif model_name == ModelName.Ridge:
        return Ridge(normalize=True)
    elif model_name == ModelName.DecisionTreeRegressor:
        return DecisionTreeRegressor(random_state=RANDOM_STATE)            
    elif model_name == ModelName.RandomForestRegressor:
        MAX_SAMPLES = None
        MAX_DEPTH = 12
        return RandomForestRegressor(max_samples=MAX_SAMPLES,
                                     max_depth=MAX_DEPTH,
                                     random_state=RANDOM_STATE)
    elif model_name == ModelName.XGBRegressor:
        return XGBRegressor(random_state=RANDOM_STATE, eval_metric="rmsle")
    elif model_name == ModelName.XGBRFRegressor:
        MAX_SAMPLES = None
        MAX_DEPTH = 24
        LEARNING_RATE = 1.0
        return XGBRFRegressor(learning_rate=LEARNING_RATE,
                              max_samples=MAX_SAMPLES,
                              max_depth=MAX_DEPTH,
                              random_state=RANDOM_STATE,
                              eval_metric="rmsle")
    else:
        raise ValueError("Not a valid model name.")


# In[ ]:


k_fold = KFold(n_splits=10)

def k_fold_cross_validation(model_name: ModelName,
                           X: pd.DataFrame,
                           y: pd.Series,
                           target_column_name: str) -> None:    
    print("#"*20 + f"\033[1mTraining for Target - {target_column_name}\033[0m" + "#"*20)
    fold_num = 1
    trained_rmsles = []
    validation_rmsles = []
    for train_index, test_index in k_fold.split(X):
        print("-"*15 + f"Fold Number: {fold_num}" + "-"*15)
        # training dataset
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        
        # testing dataset
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        
        model = get_model_from_factory(model_name)
        
        print("*"*10 + f"Training" + "*"*10)
        trained_model, train_rmsle = get_trained_model(model, X_train, y_train, target_column_name)        
        trained_rmsles.append(train_rmsle)
        print("*"*10 + f"Validating" + "*"*10)
        validation_rmsle = validate_trained_model(trained_model, X_test, y_test, target_column_name)
        validation_rmsles.append(validation_rmsle)        
        
        fold_num += 1
        
    avg_trained_rmsle = sum(trained_rmsles)/len(trained_rmsles)
    avg_validation_rmsle = sum(validation_rmsles)/len(validation_rmsles)
    
    print("##"*20)
    print(f"## \033[1mTraining Average RMSLE: {avg_trained_rmsle:.4f}\033[0m ##")
    print(f"## \033[1mValidation Average RMSLE: {avg_validation_rmsle:.4f}\033[0m ##")
    print("##"*20)


# ### LinearRegression Model Training & Validating

# #### K-Fold Validation

# In[ ]:


k_fold_cross_validation(ModelName.LinearRegression,
                           shuffled_dataset[feature_columns],
                           shuffled_dataset["target_carbon_monoxide"],
                           "target_carbon_monoxide")

k_fold_cross_validation(ModelName.LinearRegression,
                           shuffled_dataset[feature_columns],
                           shuffled_dataset["target_benzene"],
                           "target_benzene")

k_fold_cross_validation(ModelName.LinearRegression,
                           shuffled_dataset[feature_columns],
                           shuffled_dataset["target_nitrogen_oxides"],
                           "target_nitrogen_oxides")


# ### Ridge Model Training & Validating

# #### K-Fold Validation

# In[ ]:


k_fold_cross_validation(ModelName.Ridge,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_carbon_monoxide"],
                        "target_carbon_monoxide")

k_fold_cross_validation(ModelName.Ridge,
                           shuffled_dataset[feature_columns],
                           shuffled_dataset["target_benzene"],
                           "target_benzene")

k_fold_cross_validation(ModelName.Ridge,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_nitrogen_oxides"],
                        "target_nitrogen_oxides")


# ### DecisionTreeRegressor Model Training & Validating

# In[ ]:


print("*****Starting to train DecisionTreeRegressor models*****")

decision_tree_regressors = {}

decision_tree_regressors['carbon_monoxide_predictor'], _ = get_trained_model(DecisionTreeRegressor(random_state=RANDOM_STATE), 
                                                                          X_train, 
                                                                          y_train['target_carbon_monoxide'],
                                                                          'carbon_monoxide')
decision_tree_regressors['benzene_predictor'], _ = get_trained_model(DecisionTreeRegressor(random_state=RANDOM_STATE), 
                                                                  X_train,
                                                                  y_train['target_benzene'],
                                                                  'benzene')
decision_tree_regressors['nitrogen_oxides_predictor'], _ = get_trained_model(DecisionTreeRegressor(random_state=RANDOM_STATE),
                                                                          X_train,
                                                                          y_train['target_nitrogen_oxides'],
                                                                          'nitrogen_oxides')


# In[ ]:


print("*****Starting to validate DecisionTreeRegressor models.*****")

_ = validate_trained_model(decision_tree_regressors['carbon_monoxide_predictor'],
                          X_test,
                          y_test['target_carbon_monoxide'],
                          'carbon_monoxide')

_ = validate_trained_model(decision_tree_regressors['benzene_predictor'],
                          X_test,
                          y_test['target_benzene'],
                          'benzene')

_ = validate_trained_model(decision_tree_regressors['nitrogen_oxides_predictor'],
                          X_test,
                          y_test['target_nitrogen_oxides'],
                          'nitrogen_oxides')


# #### Permutation Importance

# In[ ]:


perm = PermutationImportance(decision_tree_regressors['carbon_monoxide_predictor'],
                             random_state=RANDOM_STATE).fit(X_test, y_test['target_carbon_monoxide'])
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[ ]:


perm = PermutationImportance(decision_tree_regressors['benzene_predictor'],
                             random_state=RANDOM_STATE).fit(X_test, y_test['target_benzene'])
eli5.show_weights(perm, feature_names=X_test.columns.tolist())


# In[ ]:


perm = PermutationImportance(decision_tree_regressors['nitrogen_oxides_predictor'],
                             random_state=RANDOM_STATE).fit(X_test, y_test['target_nitrogen_oxides'])
eli5.show_weights(perm, feature_names=X_test.columns.tolist())


# #### K-Fold Cross Validation

# In[ ]:


k_fold_cross_validation(ModelName.DecisionTreeRegressor,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_carbon_monoxide"],
                        "target_carbon_monoxide")

k_fold_cross_validation(ModelName.DecisionTreeRegressor,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_benzene"],
                        "target_benzene")

k_fold_cross_validation(ModelName.DecisionTreeRegressor,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_nitrogen_oxides"],
                        "target_nitrogen_oxides")


# ## RandomForestRegressor Model Training & Validating

# In[ ]:


print("*****Starting to train RandomForestRegressor models*****")

random_forest_regressors = {}
MAX_SAMPLES = None
MAX_DEPTH = 12

random_forest_regressors['carbon_monoxide_predictor'], _ = get_trained_model(RandomForestRegressor(max_samples=MAX_SAMPLES, 
                                                                                                max_depth=MAX_DEPTH,
                                                                                                random_state=RANDOM_STATE), 
                                                                          X_train, 
                                                                          y_train['target_carbon_monoxide'],
                                                                          'carbon_monoxide')
random_forest_regressors['benzene_predictor'], _ = get_trained_model(RandomForestRegressor(max_samples=MAX_SAMPLES,
                                                                                        max_depth=MAX_DEPTH,
                                                                                        random_state=RANDOM_STATE), 
                                                                  X_train,
                                                                  y_train['target_benzene'],
                                                                  'benzene')
random_forest_regressors['nitrogen_oxides_predictor'], _ = get_trained_model(RandomForestRegressor(max_samples=MAX_SAMPLES,
                                                                                                max_depth=MAX_DEPTH,
                                                                                                random_state=RANDOM_STATE),
                                                                          X_train,
                                                                          y_train['target_nitrogen_oxides'],
                                                                          'nitrogen_oxides')


# In[ ]:


print("*****Starting to validate RandomForestRegressor models.*****")

_ = validate_trained_model(random_forest_regressors['carbon_monoxide_predictor'],
                          X_test,
                          y_test['target_carbon_monoxide'],
                          'carbon_monoxide')

_ = validate_trained_model(random_forest_regressors['benzene_predictor'],
                          X_test,
                          y_test['target_benzene'],
                          'benzene')

_ = validate_trained_model(random_forest_regressors['nitrogen_oxides_predictor'],
                          X_test,
                          y_test['target_nitrogen_oxides'],
                          'nitrogen_oxides')


# #### Permutation Importance

# In[ ]:


perm = PermutationImportance(random_forest_regressors['carbon_monoxide_predictor'],
                             random_state=RANDOM_STATE).fit(X_test, y_test['target_carbon_monoxide'])
eli5.show_weights(perm, feature_names=X_test.columns.tolist())


# In[ ]:


perm = PermutationImportance(random_forest_regressors['benzene_predictor'],
                             random_state=RANDOM_STATE).fit(X_test, y_test['target_benzene'])
eli5.show_weights(perm, feature_names=X_test.columns.tolist())


# In[ ]:


perm = PermutationImportance(random_forest_regressors['nitrogen_oxides_predictor'],
                             random_state=RANDOM_STATE).fit(X_test, y_test['target_nitrogen_oxides'])
eli5.show_weights(perm, feature_names=X_test.columns.tolist())


# #### K-Fold Cross Validation

# In[ ]:


k_fold_cross_validation(ModelName.RandomForestRegressor,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_carbon_monoxide"],
                        "target_carbon_monoxide")

k_fold_cross_validation(ModelName.RandomForestRegressor,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_benzene"],
                        "target_benzene")

k_fold_cross_validation(ModelName.RandomForestRegressor,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_nitrogen_oxides"],
                        "target_nitrogen_oxides")


# ## XGBRegressor Model Training & Validating

# In[ ]:


print("*****Starting to train XGBRegressor models*****")

xgb_regressors = {}

xgb_regressors['carbon_monoxide_predictor'], _ = get_trained_model(XGBRegressor(random_state=RANDOM_STATE,
                                                                               eval_metric="rmsle"), 
                                                                          X_train, 
                                                                          y_train['target_carbon_monoxide'],
                                                                          'carbon_monoxide')
xgb_regressors['benzene_predictor'], _ = get_trained_model(XGBRegressor(random_state=RANDOM_STATE,
                                                                       eval_metric="rmsle"), 
                                                                  X_train,
                                                                  y_train['target_benzene'],
                                                                  'benzene')
xgb_regressors['nitrogen_oxides_predictor'], _ = get_trained_model(XGBRegressor(random_state=RANDOM_STATE,
                                                                               eval_metric="rmsle"),
                                                                          X_train,
                                                                          y_train['target_nitrogen_oxides'],
                                                                          'nitrogen_oxides')


# In[ ]:


print("*****Starting to validate XGBRegressor models.*****")

_ = validate_trained_model(xgb_regressors['carbon_monoxide_predictor'],
                          X_test,
                          y_test['target_carbon_monoxide'],
                          'carbon_monoxide')

_ = validate_trained_model(xgb_regressors['benzene_predictor'],
                          X_test,
                          y_test['target_benzene'],
                          'benzene')

_ = validate_trained_model(xgb_regressors['nitrogen_oxides_predictor'],
                          X_test,
                          y_test['target_nitrogen_oxides'],
                          'nitrogen_oxides')


# #### Permutation Importance

# In[ ]:


perm = PermutationImportance(xgb_regressors['carbon_monoxide_predictor'],
                             random_state=RANDOM_STATE).fit(X_test, y_test['target_carbon_monoxide'])
eli5.show_weights(perm, feature_names=X_test.columns.tolist())


# In[ ]:


perm = PermutationImportance(xgb_regressors['benzene_predictor'],
                             random_state=RANDOM_STATE).fit(X_test, y_test['target_benzene'])
eli5.show_weights(perm, feature_names=X_test.columns.tolist())


# In[ ]:


perm = PermutationImportance(xgb_regressors['nitrogen_oxides_predictor'],
                             random_state=RANDOM_STATE).fit(X_test, y_test['target_nitrogen_oxides'])
eli5.show_weights(perm, feature_names=X_test.columns.tolist())


# #### K-Fold Cross Validation

# In[ ]:


k_fold_cross_validation(ModelName.XGBRegressor,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_carbon_monoxide"],
                        "target_carbon_monoxide")

k_fold_cross_validation(ModelName.XGBRegressor,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_benzene"],
                        "target_benzene")

k_fold_cross_validation(ModelName.XGBRegressor,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_nitrogen_oxides"],
                        "target_nitrogen_oxides")


# ## XGBRFRegressor Model Training & Validating

# In[ ]:


print("*****Starting to train XGBRFRegressor models*****")

xgbrf_regressors = {}
MAX_SAMPLES = None
MAX_DEPTH = 24
LEARNING_RATE = 1.0

xgbrf_regressors['carbon_monoxide_predictor'], _ = get_trained_model(XGBRFRegressor(learning_rate=LEARNING_RATE,
                                                                          max_samples=MAX_SAMPLES, 
                                                                          max_depth=MAX_DEPTH,
                                                                          random_state=RANDOM_STATE,
                                                                          eval_metric="rmsle"), 
                                                                          X_train, 
                                                                          y_train['target_carbon_monoxide'],
                                                                          'carbon_monoxide')
xgbrf_regressors['benzene_predictor'], _ = get_trained_model(XGBRFRegressor(learning_rate=LEARNING_RATE,
                                                                  max_samples=MAX_SAMPLES, 
                                                                  max_depth=MAX_DEPTH,
                                                                  random_state=RANDOM_STATE,
                                                                  eval_metric="rmsle"), 
                                                                  X_train,
                                                                  y_train['target_benzene'],
                                                                  'benzene')
xgbrf_regressors['nitrogen_oxides_predictor'], _ = get_trained_model(XGBRFRegressor(learning_rate=LEARNING_RATE,
                                                                          max_samples=MAX_SAMPLES,                                                                          
                                                                          max_depth=MAX_DEPTH,
                                                                          random_state=RANDOM_STATE,
                                                                          eval_metric="rmsle"),
                                                                          X_train,
                                                                          y_train['target_nitrogen_oxides'],
                                                                          'nitrogen_oxides')


# In[ ]:


print("*****Starting to validate XGBRFRegressor models.*****")

validate_trained_model(xgbrf_regressors['carbon_monoxide_predictor'],
                      X_test,
                      y_test['target_carbon_monoxide'],
                      'carbon_monoxide')

validate_trained_model(xgbrf_regressors['benzene_predictor'],
                      X_test,
                      y_test['target_benzene'],
                      'benzene')

validate_trained_model(xgbrf_regressors['nitrogen_oxides_predictor'],
                      X_test,
                      y_test['target_nitrogen_oxides'],
                      'nitrogen_oxides')


# #### K-Fold Validation

# In[ ]:


k_fold_cross_validation(ModelName.XGBRFRegressor,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_carbon_monoxide"],
                        "target_carbon_monoxide")

k_fold_cross_validation(ModelName.XGBRFRegressor,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_benzene"],
                        "target_benzene")

k_fold_cross_validation(ModelName.XGBRFRegressor,
                        shuffled_dataset[feature_columns],
                        shuffled_dataset["target_nitrogen_oxides"],
                        "target_nitrogen_oxides")


# ## Train Best Model on Full Dataset
# 
# > This is a manul task. After going through the results of different models, I am picking one and re-training the model with full train dataset.

# In[ ]:


xgbrf_regressors['carbon_monoxide_predictor'], _ = get_trained_model(XGBRFRegressor(learning_rate=LEARNING_RATE,
                                                                                    max_samples=MAX_SAMPLES, 
                                                                                    max_depth=MAX_DEPTH,
                                                                                    random_state=RANDOM_STATE,
                                                                                    eval_metric="rmsle"), 
                                                                          dataset_copy[feature_columns], 
                                                                          dataset_copy['target_carbon_monoxide'],
                                                                          'carbon_monoxide')

xgbrf_regressors['benzene_predictor'], _ = get_trained_model(XGBRFRegressor(learning_rate=LEARNING_RATE,
                                                                            max_samples=MAX_SAMPLES, 
                                                                            max_depth=MAX_DEPTH,
                                                                            random_state=RANDOM_STATE,
                                                                            eval_metric="rmsle"), 
                                                                  dataset_copy[feature_columns],
                                                                  dataset_copy['target_benzene'],
                                                                  'benzene')

xgbrf_regressors['nitrogen_oxides_predictor'], _ = get_trained_model(XGBRFRegressor(learning_rate=LEARNING_RATE,
                                                                                    max_samples=MAX_SAMPLES,                                                                          
                                                                                    max_depth=MAX_DEPTH,
                                                                                    random_state=RANDOM_STATE,
                                                                                    eval_metric="rmsle"),
                                                                      dataset_copy[feature_columns], 
                                                                      dataset_copy['target_nitrogen_oxides'],
                                                                      'nitrogen_oxides')


# ## Preparing Submission File

# In[ ]:


test_dataset = pd.read_csv(TEST_CSV)
test_dataset.head()


# In[ ]:


test_dataset = get_processed_dataset(test_dataset)
test_dataset.head()


# In[ ]:


carbon_monoxide_pred = xgbrf_regressors['carbon_monoxide_predictor'].predict(test_dataset[feature_columns])
benzene_pred = xgbrf_regressors['benzene_predictor'].predict(test_dataset[feature_columns])
nitrogen_oxides_pred = xgbrf_regressors['nitrogen_oxides_predictor'].predict(test_dataset[feature_columns])


# In[ ]:


submission_df = pd.DataFrame({"date_time": test_dataset['date_time'],
                             "target_carbon_monoxide": carbon_monoxide_pred,
                             "target_benzene": benzene_pred,
                             "target_nitrogen_oxides": nitrogen_oxides_pred})
submission_df.head()


# In[ ]:


submission_df.to_csv("submission.csv",index=False)


# ## Resources:
# 
# - [Understanding Random Forest
# ](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)
# - [Using XGBoost in Python
# ](https://www.datacamp.com/community/tutorials/xgboost-in-python)
# - [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)
# - [Permutation Importance](https://www.kaggle.com/dansbecker/permutation-importance)
