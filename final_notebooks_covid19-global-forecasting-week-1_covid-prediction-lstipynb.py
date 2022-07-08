#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import math
data_dir = "/kaggle/input/covid19-global-forecasting-week-1/"


# In[ ]:


train_df_raw = pd.read_csv(data_dir+"train.csv")
test_df_raw = pd.read_csv(data_dir+"test.csv")
all_df = train_df_raw
test_df = test_df_raw
maxlen = 30
hidden_number = 32
input_number = 5
output_number = 2
batch_size = 32
epochs = 50
lr = 0.001


# In[ ]:


def map_datetime(date):
    return (date - datetime.datetime.strptime('2020-01-22', "%Y-%m-%d")).days

def to_datetime(date):
    return datetime.datetime.strptime(date, "%Y-%m-%d")


# In[ ]:


test_df = test_df.fillna({"Province/State": "NAN"})
all_df = all_df.fillna({"Province/State": "NAN"})

all_df["ConfirmedCases"] = (all_df["ConfirmedCases"] + 1).map(math.log10)
cases_max = all_df["ConfirmedCases"].max()
fatal_max = all_df["Fatalities"].max()

all_df["Lat"] = all_df["Lat"]/180.
all_df["Long"] = all_df["Long"]/180.
all_df["ConfirmedCases"] = all_df["ConfirmedCases"] / cases_max
all_df["Fatalities"] = all_df["Fatalities"] / fatal_max

all_df["Date"] = all_df["Date"].map(to_datetime)
all_df["Date_num"] = all_df["Date"].map(map_datetime)
date_max = all_df["Date_num"].max()
all_df["Date_num"] = all_df["Date_num"] / date_max

date_unit = all_df.iloc[1]["Date_num"] - all_df.iloc[0]["Date_num"]

val_df = all_df[all_df["Date"] > (all_df["Date"].max() - datetime.timedelta(days=(maxlen+1)))]
train_df = all_df.drop(all_df[all_df["Date_num"] == all_df["Date_num"].max()].index)


# In[ ]:


test_df["Lat"] = test_df["Lat"]/180.
test_df["Long"] = test_df["Long"]/180.
test_df["Date"] = test_df["Date"].map(to_datetime)
test_df["Date_num"] = test_df["Date"].map(map_datetime)
test_df["Date_num"] = test_df["Date_num"] / date_max


# In[ ]:


def make_sequences(train_df):
    inputs = []
    targets = []
    for i in range(len(train_df) - maxlen - 1):
        if train_df.iloc[i]["Lat"] == train_df.iloc[i+maxlen]["Lat"] and \
           train_df.iloc[i]["Long"] == train_df.iloc[i+maxlen]["Long"]:
            inputs.append(np.array(train_df.iloc[i:i+maxlen][["Date_num", "Lat", "Long", "ConfirmedCases", "Fatalities"]]).tolist())
            targets.append(np.array(train_df.iloc[i+maxlen][["ConfirmedCases", "Fatalities"]]).tolist())
    return inputs, targets

train_inputs, train_targets = make_sequences(train_df)
val_inputs, val_targets = make_sequences(val_df)


# In[ ]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(hidden_number, batch_input_shape=[None, maxlen, input_number], return_sequences=True))
model.add(tf.keras.layers.LSTM(hidden_number))
model.add(tf.keras.layers.Dense(output_number, activation="sigmoid"))

optimizer = tf.keras.optimizers.Adam(lr=lr)
model.compile(loss="mean_squared_error", optimizer=optimizer)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=5)
history = model.fit(train_inputs, train_targets,
                    batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(val_inputs, val_targets),
                      callbacks = [early_stopping]
                      )


# In[ ]:


results = []
for idx in test_df.groupby(["Province/State", "Country/Region"]).count().index:
    test_df_on_idx = test_df[(test_df["Province/State"] == idx[0]) &
                             (test_df["Country/Region"] == idx[1])]
    train_df_on_idx = train_df[(all_df["Country/Region"] == idx[1]) &
                               (all_df["Province/State"] == idx[0])]
    inputs = np.array(train_df_on_idx[["Date_num", "Lat", "Long", "ConfirmedCases", "Fatalities"]])[-maxlen:]
    inputs = inputs.reshape(maxlen, input_number)
    for day in range(43):
        if int(1000000000*(datetime.timedelta(days=day) + test_df_on_idx["Date"].min()).timestamp()) in train_df_on_idx["Date"].values.tolist():
            result = np.array(train_df_on_idx[train_df_on_idx["Date"] == (datetime.timedelta(days=day) + test_df_on_idx["Date"].min())][["Date", "Lat", "Long", "ConfirmedCases", "Fatalities"]])[0, 3:]
        else:
            result = model.predict(np.array(inputs).reshape(1, maxlen, input_number)).reshape(-1)
            inputs = np.concatenate((inputs[1:], np.append(inputs[-1, :3], result).reshape(1, input_number)), axis=0)
        results.append([10**(result[0]*cases_max), result[1]*fatal_max])


# In[ ]:


submit_df = pd.read_csv(data_dir+"submission.csv", index_col=0)
submit_df


# In[ ]:


cases = []
fatals = []
for i in range(len(results)):
    n = results[i][0] 
    f = results[i][1]
    try:
        cases.append(int(n))
    except:
        cases.append(0)
    
    try:
        fatals.append(int(f))
    except:
        fatals.append(0)
submit_df["ConfirmedCases"] = cases
submit_df["Fatalities"] = fatals


# In[ ]:


submit_df.to_csv("submission.csv")

