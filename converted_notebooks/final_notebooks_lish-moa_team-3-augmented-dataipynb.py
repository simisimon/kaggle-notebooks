#!/usr/bin/env python
# coding: utf-8

# # Team 3
# 
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#TF stuff
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

#XGBoost
from xgboost import XGBClassifier

#Scikit
from sklearn.metrics import log_loss
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


# ## **Read in the data**
# 
# Two key files here: train_feature.csv (these is roughly your 'X'), and train_targets.csv (which is your 'y'). More information on these can be found on the original competetion site: https://www.kaggle.com/c/lish-moa/data 

# 

# In[ ]:


train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
train_targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
print(train_features.head())
print(train_targets.head())
print(train_features.shape)
print(train_targets.shape)


# **Some simple encoding:** Two of our features 'cp_type' and 'cp_dose' are categorical. We will encode these numerically with a simple one-hot encoding. The pandas get_dummies function is a quick way of doing this. 

# In[ ]:


train_features_enc = pd.get_dummies(train_features, columns=['cp_type', 'cp_dose'], drop_first=True)
print(train_features_enc.head())


# We're now all good to go with respect to defining our 'X' and 'y'; all we drop below is the sig_id column which is simply there to align the features and targets dataframes.

# In[ ]:


X = train_features_enc.iloc[:,1:].to_numpy()
y = train_targets.iloc[:,1:].to_numpy() 


# # **Neural Network Models**
# 
# We now transition to building NN mnodels. Such models have shown a great deal of promise w/ tabular data recently, and we will walk through a squence of increasingly complex models here. We start with the simplest possible NN -- logistic regression! Its worth noting that in the model below we regularize the parameters of the network (something we will drop later on). Another intersting point worth noting is that the model below optimizes the loss jointly over all the 206 components of $y$ as opposed to optimizing each individually. 
# 
# Another point of note is that we find these sorts of models far easier to customize and optimize over that the pre-packaged scikit models. 

# In[ ]:


# A convenient plotting function as we train models
def plot_hist(hist, last = None):
    if last == None:
        last = len(hist.history["loss"])
    plt.plot(hist.history["loss"][-last:])
    plt.plot(hist.history["val_loss"][-last:])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


# **Residual Connections**
# 
# Our final effort changes the architecture from being a simple feed forward network to adding resdiula connections. This can be viewed as a 'rollout' of the optimization that a boosting method might use. Specifically, consider builing a 'simple' model with a single hidden layer and then fitting another single model to the residual, and then another to the residual of the cobined models, and so forth.. Below, we do this three times. 
# 
# This model dramatically improves on XGBoost. Our loss stands at 0.0156! 
# 
# 

# In[ ]:


#0.0156
def l3_res_model(input_shape, no_classes, lr):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    b_1 = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(b_1)
    x = layers.BatchNormalization()(x)
    b_2 = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(b_2)
    x = layers.BatchNormalization()(x)
    b_3 = layers.Dropout(0.2)(x)
    tot_op = tf.keras.layers.add([b_1, b_2, b_3])
    outputs = layers.Dense(no_classes, activation='sigmoid')(tot_op)
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = lr), metrics=['binary_crossentropy'])
    return model


tf.random.set_seed(1010)
np.random.seed(1010)

X_train = X
y_train = y

control_vehicle_mask = X_train[:,-2] == 0
X_train = X_train[~control_vehicle_mask,:]
y_train = y_train[~control_vehicle_mask]

nnclf = l3_res_model((875,),206,0.0005)
hist = nnclf.fit(X_train, y_train, batch_size=512, epochs=45, verbose=1)


# In[ ]:


test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
submission_data = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

X_test_final = pd.get_dummies(test_features, columns=['cp_type', 'cp_dose'], drop_first=True).iloc[:,1:].to_numpy()

predictions = pd.DataFrame(nnclf.predict(X_test_final), columns = submission_data.columns[1:])

predictions.insert(0, submission_data.columns[0], test_features[submission_data.columns[0]])

# Save final output
predictions.to_csv('submission.csv', index=False)


# In[ ]:




