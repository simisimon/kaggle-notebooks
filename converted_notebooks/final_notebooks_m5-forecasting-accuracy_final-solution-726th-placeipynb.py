#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from typing import Union

import numpy as np
import pandas as pd
pd.set_option("max_columns", 50)
from tqdm.auto import tqdm as tqdm
import pickle
from random import sample
import random
import lightgbm as lgbm
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
import gc


from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


from sklearn.preprocessing import LabelEncoder

random.seed(820)

#Evaluator, with it the final models for the ensamble will be picked.
# the evaluator takes predictions and scores them for the last month in the train.
class WRMSSEEvaluator(object):
    
    group_ids = ( 'all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id',
        ['state_id', 'cat_id'],  ['state_id', 'dept_id'], ['store_id', 'cat_id'],
        ['store_id', 'dept_id'], ['item_id', 'state_id'], ['item_id', 'store_id'])

    def __init__(self, 
                 train_df: pd.DataFrame, 
                 valid_df: pd.DataFrame, 
                 calendar: pd.DataFrame, 
                 prices: pd.DataFrame):
        '''
        intialize and calculate weights
        '''
        self.calendar = calendar
        self.prices = prices
        self.train_df = train_df
        self.valid_df = valid_df
        self.train_target_columns = [i for i in self.train_df.columns if i.startswith('d_')]
        self.weight_columns = self.train_df.iloc[:, -28:].columns.tolist()

        self.train_df['all_id'] = "all"

        self.id_columns = [i for i in self.train_df.columns if not i.startswith('d_')]
        self.valid_target_columns = [i for i in self.valid_df.columns if i.startswith('d_')]

        if not all([c in self.valid_df.columns for c in self.id_columns]):
            self.valid_df = pd.concat([self.train_df[self.id_columns], self.valid_df],
                                      axis=1, 
                                      sort=False)
        self.train_series = self.trans_30490_to_42840(self.train_df, 
                                                      self.train_target_columns, 
                                                      self.group_ids)
        self.valid_series = self.trans_30490_to_42840(self.valid_df, 
                                                      self.valid_target_columns, 
                                                      self.group_ids)
        self.weights = self.get_weight_df()
        self.scale = self.get_scale()
        self.train_series = None
        self.train_df = None
        self.prices = None
        self.calendar = None

    def get_scale(self):
        '''
        scaling factor for each series ignoring starting zeros
        '''
        scales = []
        for i in tqdm(range(len(self.train_series))):
            series = self.train_series.iloc[i].values
            series = series[np.argmax(series!=0):]
            scale = ((series[1:] - series[:-1]) ** 2).mean()
            scales.append(scale)
        return np.array(scales)
    
    def get_name(self, i):
        '''
        convert a str or list of strings to unique string 
        used for naming each of 42840 series
        '''
        if type(i) == str or type(i) == int:
            return str(i)
        else:
            return "--".join(i)
    
    def get_weight_df(self) -> pd.DataFrame:
        """
        returns weights for each of 42840 series in a dataFrame
        """
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()
        weight_df = self.train_df[["item_id", "store_id"] + self.weight_columns].set_index(
            ["item_id", "store_id"]
        )
        weight_df = (
            weight_df.stack().reset_index().rename(columns={"level_2": "d", 0: "value"})
        )
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)
        weight_df = weight_df.merge(
            self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"]
        )
        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
        weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level=2)[
            "value"
        ]
        weight_df = weight_df.loc[
            zip(self.train_df.item_id, self.train_df.store_id), :
        ].reset_index(drop=True)
        weight_df = pd.concat(
            [self.train_df[self.id_columns], weight_df], axis=1, sort=False
        )
        weights_map = {}
        for i, group_id in enumerate(tqdm(self.group_ids, leave=False)):
            lv_weight = weight_df.groupby(group_id)[self.weight_columns].sum().sum(axis=1)
            lv_weight = lv_weight / lv_weight.sum()
            for i in range(len(lv_weight)):
                weights_map[self.get_name(lv_weight.index[i])] = np.array(
                    [lv_weight.iloc[i]]
                )
        weights = pd.DataFrame(weights_map).T / len(self.group_ids)

        return weights

    def trans_30490_to_42840(self, df, cols, group_ids, dis=False):
        '''
        transform 30490 sries to all 42840 series
        '''
        series_map = {}
        for i, group_id in enumerate(tqdm(self.group_ids, leave=False, disable=dis)):
            tr = df.groupby(group_id)[cols].sum()
            for i in range(len(tr)):
                series_map[self.get_name(tr.index[i])] = tr.iloc[i].values
        return pd.DataFrame(series_map).T
    
    def get_rmsse(self, valid_preds) -> pd.Series:
        '''
        returns rmsse scores for all 42840 series
        '''
        score = ((self.valid_series - valid_preds) ** 2).mean(axis=1)
        rmsse = (score / self.scale).map(np.sqrt)
        return rmsse

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds],
                                axis=1, 
                                sort=False)
        valid_preds = self.trans_30490_to_42840(valid_preds, 
                                                self.valid_target_columns, 
                                                self.group_ids, 
                                                True)
        self.rmsse = self.get_rmsse(valid_preds)
        self.contributors = pd.concat([self.weights, self.rmsse], 
                                      axis=1, 
                                      sort=False).prod(axis=1)
        return np.sum(self.contributors)


# In[ ]:


# Loading data
train_df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")
calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")

#Split train-test for evaluator
train_fold_df = train_df.iloc[:, :-28]
valid_fold_df = train_df.iloc[:, -28:].copy()

# Evaluator created
e = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)
del train_fold_df, train_df, calendar, prices


# In[ ]:


#This cell loads train-reday data sets and pre-trained label encoders.

gc.collect()
AllTrain = pickle.load(open('../input/train-ready/train.pkl', "rb"))

AllTrain["d"] = pickle.load(open("../input/train-ready/init_train_d.pkl", "rb"))
AllTrain.dropna(inplace=True) # remove oldest year.

le_dept =  pickle.load(open('../input/train-ready/le_dept.pkl', "rb"))
le_state = pickle.load(open('../input/train-ready/le_state.pkl', "rb"))

le_item =  LabelEncoder()
le_store =  LabelEncoder()

AllTrain["item_id"] = le_item.fit_transform(AllTrain["item_id"])
AllTrain["store_id"] = le_store.fit_transform(AllTrain["store_id"])


# In[ ]:


# Set validation.
Firs27days_mar2016 = ((AllTrain.year == 2016) & (AllTrain.month == 3)& (AllTrain.mday <= 27))
FebJan2016 = ((AllTrain.year == 2016) & (AllTrain.month < 3))
YearSmallerThan2016 = (AllTrain.year < 2016)

MainTrain = AllTrain.loc[Firs27days_mar2016 | FebJan2016 | YearSmallerThan2016].copy()
Validation = AllTrain.loc[~AllTrain.index.isin(MainTrain.index)].copy()

# Down sample main only.
MainTrain_index = list(MainTrain.index)
MainTrain_ds_index = sample(MainTrain_index, (len(MainTrain_index)//100) * 32) # 32% of data
MainTrain_ds_index = sorted(MainTrain_ds_index)
MainTrain = MainTrain.loc[MainTrain_ds_index]
MainTrain = MainTrain.reset_index().drop("index", axis = 1)

MainTrain = MainTrain.loc[MainTrain.index.isin(MainTrain_ds_index)]


# In[ ]:


# Memory save
del YearSmallerThan2016, FebJan2016, Firs27days_mar2016, AllTrain


# In[ ]:


# Features selected
MainTrain = MainTrain[["sales", "id", "year","dept_id","wday", "item_id", "store_id", "mean_1_months_ago","mean_2_months_ago","avg_last_year", "d"]]
Validation = Validation[["sales", "id", "year","dept_id","wday", "item_id", "store_id", "mean_1_months_ago","mean_2_months_ago","avg_last_year", "d"]]


# In[ ]:


# 'd' is a feature representing the day sequential number.
first_day =  Validation.head(1).d.values[0]
Validation['d'] = Validation['d'] - (first_day - 1)


# In[ ]:


sample_submission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv") # actual file used for submission
sample_submission_ = sample_submission.loc[sample_submission["id"].str.contains("validation")].copy() # valudation building
sample_submission_copy = sample_submission_.copy() # test building


# In[ ]:


# Extract product_id
def get_product(row):
    split_row = row.split("_")
    return split_row[0] + "_" + split_row[1] + "_" + split_row[2] + "_" + split_row[3] + "_" + split_row[4]

sample_submission['product_id'] = sample_submission["id"].apply(get_product)


# In[ ]:


# Creating ground_truth DataFrames to compare with prediction on the last month in the train set.
ground_truth = sample_submission.loc[sample_submission['id'].str.contains("validation")].copy()
ground_truth = ground_truth[["product_id"]].copy()

for day in range(1,29):
    only_day_sales = Validation.loc[Validation["d"].astype("int") == day].copy()
    only_day_sales["product_id"] = only_day_sales.id.str[:-len("validation") - 1]
    ground_truth["F{}".format(day)] = ground_truth.merge(only_day_sales[["product_id", "sales"]], on="product_id")["sales"]
    


# In[ ]:


# Split X - features. y- target (sales)
X = MainTrain.drop(["id", "sales", 'd'], axis =1)
X = X.loc[:,~X.columns.duplicated()] # rem dup cols

y = MainTrain["sales"]


# In[ ]:


# This function trains diefferent models for the ensemble
def train_N_models(X,y, help1):
    predictors = {}

    print("trainig")
    if "rf" not in help1:
        # RF
        print("rf")
        regr = RandomForestRegressor(max_depth=5, random_state=0,
                                  n_estimators=70, n_jobs = -1)
        regr.fit(X, y) 
        predictors["rf"] = regr
        help1["rf"] = regr


    if "catboost" not in help1:
        #     catboost
        print("catboost")
        catb = CatBoostRegressor(iterations=100,
                                  learning_rate=0.078,
                                  depth=10,
                                random_seed = 32,
                                logging_level = "Silent",
                                thread_count = -1)
        catb.fit(X, y)
        predictors["catboost"] = catb
        help1["catboost"] = catb


    if "XGBoost" not in help1:
        # XGBoost
        print("XGBoost")
        model = xgb.XGBRegressor(colsample_bytree=0.4,
                         gamma=0,                 
                         learning_rate=0.17,
                         max_depth=3,
                         min_child_weight=1.5,
                         n_estimators=70,                                                                    
                         reg_alpha=0.75,
                         reg_lambda=0.45,
                         subsample=0.6,
                         seed=42) 

        model.fit(X,y)
        predictors["xgboost"] = model
        help1["XGBoost"] = model

    if "lightgbm" not in help1:
        # lightgbm
        print("lightgbm")
        parms = {"boosting_type" : 'dart',
                 "num_leaves" : 3,
                 "max_depth" :2, 
                 'learning_rate':0.25,
                 "n_estimators" : 80,
                 "objective" : "regression", 
                 "min_split_gain" : 0,
                 "min_child_weight" : 0.001,
                 "min_child_samples" : 20,
                 "reg_alpha" : 0,
                 "reg_lambda" : 0, 
                 "random_state" : 1406,
                 "n_jobs" : -1,
                 "silent" : False}

        model = lgbm.LGBMRegressor(boosting_type = parms["boosting_type"],
                                  num_leaves = parms["num_leaves"],max_depth = parms["max_depth"],
                                  learning_rate = parms["learning_rate"],n_estimators = parms["n_estimators"],
                                  objective = parms["objective"],min_split_gain = parms["min_split_gain"],
                                  min_child_weight = parms["min_child_weight"],
                                  min_child_samples = parms["min_child_samples"],reg_alpha = parms["reg_alpha"],
                                  reg_lambda = parms["reg_lambda"],random_state = parms["random_state"],
                                  n_jobs = parms["n_jobs"],silent = parms["silent"])

        model.fit(X, y)
        predictors["lightgbm"] = model
        help1["lightgbm"] = model


    if "lr" not in help1:
        print("lr")
        # liniar regression 
        lm = linear_model.LinearRegression(n_jobs = -1)
        model = lm.fit(X,y)
        predictors["lr"] = model
        help1["lr"] = model


    if "NN" not in help1:
        print("NN")
        # NN
        NN_model = Sequential()

        # The Input Layer :
        NN_model.add(Dense(30, kernel_initializer="glorot_normal",input_dim =X.shape[1], activation="selu"))

        # The Hidden Layers :
        NN_model.add(Dense(15, kernel_initializer="glorot_normal",activation="elu"))
        NN_model.add(Dense(15, kernel_initializer="glorot_normal",activation="elu"))

        # The Output Layer :
        NN_model.add(Dense(1, kernel_initializer='glorot_normal',activation="elu"))

        # Compile the network :
        NN_model.compile(loss="mean_squared_error", optimizer="Adadelta", metrics=["acc"])

        NN_model.fit(X, y, epochs=1, batch_size=400, validation_split = 0.2, verbose = True) 
        predictors["NN"] = NN_model
        help1["NN"] = NN_model


    return predictors


# In[ ]:


help1 = {}


# In[ ]:


# Train without the last month, for validation
if "d" in X:
    predictors_valid = train_N_models(X.drop("d", axis = 1) ,y, help1)
else:
    predictors_valid = train_N_models(X ,y, help1)


# In[ ]:


# For every day there is a test DataFrame containing the features of that day.
# In this case we only use the validation rows
test_days = []
test_columns = list(MainTrain.columns)
test_columns.remove("sales")
test_columns.remove("d")

for day in range(0, 28):
    test_days.append(pickle.load(open("../input/tests-28/test_{}.pkl".format(day), "rb")))
    test_days[day] = test_days[day].loc[test_days[day]["id"].str.contains("validation")].copy()

    test_days[-1]["dept_id"] = le_dept.transform(test_days[-1]["dept_id"])
    test_days[-1]["state_id"] = le_state.transform(test_days[-1]["state_id"])
    test_days[-1]["item_id"] = le_item.transform(test_days[-1]["item_id"])
    test_days[-1]["store_id"] = le_store.transform(test_days[-1]["store_id"])
    test_days[-1] = test_days[-1][test_columns]


# In[ ]:


# Make validation days - DataFrame for every day in the validation set.
validation_days = []
test_days_cols = test_days[0].columns
for day in range(1,29):
    valid_day = Validation.loc[Validation.d == day][test_days_cols]
    valid_day = valid_day.loc[:,~valid_day.columns.duplicated()] # rem dup cols

    valid_day = valid_day.set_index('item_id')
    valid_day = valid_day.reindex(index=test_days[0]['item_id'])
    validation_days.append(valid_day.reset_index())
    


# In[ ]:


# This function makes prediction given a validation day DataFrame
def predict_28_days_validation(model, X):
    temp = pd.DataFrame()
    for day, validation in enumerate(validation_days):
        print(day, end = "  ")
        col = "F{}".format(day + 1)
        sample_submission_copy[col] = model.predict(validation_days[day].drop(["id"], axis = 1)[X.columns])
        sample_submission_copy[col] =  sample_submission_copy[col].round(1)
        temp[col] = sample_submission_copy[col].to_numpy().flatten()    
    print()
    return temp.to_numpy()
# column is every product * 28 days all the way down


# In[ ]:


# Predictions for ensemble
preds_validation = pd.DataFrame()
for predictor_name, predictor in predictors_valid.items():
    print(predictor_name)
    preds_validation[predictor_name] = predict_28_days_validation(predictor, X).flatten()


# In[ ]:


preds_validation["ground_truth"] = ground_truth.drop("product_id", axis =1).to_numpy().flatten()
preds_validation = preds_validation.clip(0, 99)


# In[ ]:


# Initial ensemble 
X_ens = preds_validation.drop("ground_truth", axis = 1)
y_ens = preds_validation["ground_truth"]


regr = xgb.XGBRegressor (eta =0.1,  
                        nthread = -1, 
                        n_estimators= 30, 
                        max_depth= 2, 
                        max_delta_step= 16,
                         colsample_bytree= 0.4,
                         scale_pos_weight= 0.9,
                         base_score= 0.9,
                         eval_metric= 'rmse')
regr.fit(X_ens, y_ens) 
pickle.dump(regr, open("regr_ens.pkl", "wb"))


# In[ ]:


Validation = Validation.drop("d", axis = 1)


# In[ ]:


gc.collect()


# In[ ]:


# Train on all the months in the data
help1 = {}

MainTrain = MainTrain.loc[:,~MainTrain.columns.duplicated()] # rem dup cols
Validation = Validation.loc[:,~Validation.columns.duplicated()] # rem dup cols


X = MainTrain.append(Validation)
y = X["sales"]

X = X.drop(["item_id", "id", "sales", "d"], axis =1)
X = X.loc[:,~X.columns.duplicated()] # rem dup cols

predictors_test = train_N_models(X, y, help1)


# In[ ]:





# In[ ]:


# Predict for every day in the test set
preds_test = pd.DataFrame()
for predictor_name, predictor in predictors_test.items():
    print(predictor_name)
    preds_test[predictor_name] = predict_28_days_validation(predictor, X).flatten()       
preds_test = preds_test.clip(0,99)


# In[ ]:





# In[ ]:


# re-run
whos = get_ipython().run_line_magic('who_ls', '')
if "Save_preds_test" in whos:
    preds_test = Save_preds_test
else:
    Save_preds_test =  preds_test


# In[ ]:


# Final models selection.
X_ens = preds_validation.drop("ground_truth", axis = 1)
y_ens = preds_validation["ground_truth"]

regr = xgb.XGBRegressor (eta =0.3,  
                        nthread = -1, 
                        n_estimators= 50, 
                        max_depth= 3, 
                         max_delta_step= 6,
                        colsample_bytree= 0.4,
#                          scale_pos_weight= 0.9,
                         base_score= 0.9,
                         random_state=0,
                         eval_metric= 'rmse')

best_grade = 999
save_x_ens = X_ens.copy()
df = pd.DataFrame()
for i, model in enumerate(X_ens):
    model_name = list(save_x_ens.columns)[i]
    print(model_name)
    df[model_name] = X_ens[model_name]
    
    regr.fit(df, y_ens)
    
    Ens_preds = regr.predict(preds_test[df.columns])
    Ens_preds = Ens_preds.reshape(sample_submission_.shape[0], 28)
    Ens_preds_c = Ens_preds
    grade = e.score(Ens_preds_c)
    print(grade)
    
    if grade < best_grade:  # improve
        print("adding {} helped! grade now: {} prev grade: {}".format(model_name, grade, best_grade))
        best_grade = grade
        print(df.columns)
        columns_best_regr = df.columns
    else:
        print("adding {} did not help. get out.".format(model_name))
        df = df.drop(model_name, axis = 1)


# In[ ]:


for col in preds_test.columns:
    if col not in columns_best_regr:
        preds_test = preds_test.drop(col, axis = 1)


# In[ ]:


# Train final ensemble.
regr.fit(df[columns_best_regr], y_ens)

Ens_preds = regr.predict(preds_test[columns_best_regr])
Ens_preds = Ens_preds.reshape(sample_submission_.shape[0], 28)
e.score(Ens_preds)


# In[ ]:





# In[ ]:


# For every day there is a test DataFrame containing the features of that day.
# In this case we only use all the rows - validation and test
test_days = []
for day in range(0, 28):
    test_days.append(pickle.load(open("../input/tests-28/test_{}.pkl".format(day), "rb")))
    test_days[-1]["dept_id"] = le_dept.transform(test_days[-1]["dept_id"])
    test_days[-1]["state_id"] = le_state.transform(test_days[-1]["state_id"])
    test_days[-1]["item_id"] = le_item.transform(test_days[-1]["item_id"])
    test_days[-1]["store_id"] = le_store.transform(test_days[-1]["store_id"])


# In[ ]:


def predict_28_days_test(model, X):
    temp = pd.DataFrame()
    for day, test in enumerate(test_days):
        print(day, end = "  ")
        col = "F{}".format(day + 1)
        sample_submission[col] = model.predict(test.drop(["id"], axis = 1)[X.columns])
        sample_submission[col] =  sample_submission[col].round(1)
        temp[col] = sample_submission[col].to_numpy().flatten()    
    print()
    return temp.to_numpy()


# In[ ]:


# Memory cleanup
whos = get_ipython().run_line_magic('who_ls', '')
if "Ens_preds" in whos:
    del Ens_preds
    
if "X_ens" in whos:
    del X_ens

if "Validation" in whos:
    del Validation
    
gc.collect()


# In[ ]:


for predictor_name in columns_best_regr:
    print(predictor_name)


# In[ ]:


# Final predictions.
print("predicting!")

preds = pd.DataFrame()   
    

for predictor_name in columns_best_regr:
    print(predictor_name)
    preds[predictor_name] = predict_28_days_test(predictors_test[predictor_name], X).flatten()

if "lr" in preds:
    preds["lr"] = preds["lr"].clip(lower = 0.1)
    
Ens_preds_test = regr.predict(preds)
Ens_preds_test = Ens_preds_test.reshape(sample_submission.shape[0], 28)
    
F_cols = ["F{}".format(i) for i in range(1,29)]
sample_submission[F_cols] = Ens_preds_test
sample_submission = sample_submission.round(1)
sample_submission[F_cols] = sample_submission[F_cols].clip(lower = 0.01, axis=0) # drop negatives

if "product_id"  in sample_submission:
    sample_submission =  sample_submission.drop("product_id", axis = 1)
sample_submission.to_csv("submission.csv", index= False)

