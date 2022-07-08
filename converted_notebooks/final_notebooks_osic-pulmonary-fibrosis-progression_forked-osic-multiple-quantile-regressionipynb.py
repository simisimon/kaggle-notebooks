#!/usr/bin/env python
# coding: utf-8

# ![](http://)Thank you @ulrich07 for this lovely notebook, don't forget upvode it -->
# https://www.kaggle.com/ulrich07/osic-multiple-quantile-regression-starter

# In[ ]:


import numpy as np
import pandas as pd
import pydicom
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
from statistics import mean, pstdev


# In[ ]:


import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M


# In[ ]:


def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything(42)


# In[ ]:


ROOT = "../input/osic-pulmonary-fibrosis-progression"
BATCH_SIZE=128


# In[ ]:


tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
chunk = pd.read_csv(f"{ROOT}/test.csv")

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")


# In[ ]:


tr['WHERE'] = 'train'
chunk['WHERE'] = 'val'
sub['WHERE'] = 'test'
data = tr.append([chunk, sub])


# In[ ]:


print(tr.shape, chunk.shape, sub.shape, data.shape)
print(tr.Patient.nunique(), chunk.Patient.nunique(), sub.Patient.nunique(), 
      data.Patient.nunique())
#


# In[ ]:


data['min_week'] = data['Weeks']
data.loc[data.WHERE=='test','min_week'] = np.nan
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')


# In[ ]:


base = data.loc[data.Weeks == data.min_week]
base = base[['Patient','FVC']].copy()
base.columns = ['Patient','min_FVC']
base['nb'] = 1
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
base = base[base.nb==1]
base.drop('nb', axis=1, inplace=True)


# In[ ]:


data = data.merge(base, on='Patient', how='left')
data['base_week'] = data['Weeks'] - data['min_week']
del base


# In[ ]:


COLS = ['Sex','SmokingStatus'] #,'Age'
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)
#=================


# In[ ]:


#
data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )
data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )
data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )
data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )
FE += ['age','percent','week','BASE']


# In[ ]:


tr = data.loc[data.WHERE=='train']
chunk = data.loc[data.WHERE=='val']
sub = data.loc[data.WHERE=='test']
del data


# In[ ]:


tr.shape, chunk.shape, sub.shape


# In[ ]:





# ### BASELINE NN 

# In[ ]:


C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
#=============================#
def score(y_true, y_pred):
    #K.print_tensor(y_true)
    #K.print_tensor(y_pred)
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 5] - y_pred[:, 1]
    fvc_pred = y_pred[:, 3]
    
    #sigma_clip = sigma + C1
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    #K.print_tensor(K.mean(metric))
    return K.mean(metric)
#============================#
def qloss(y_true, y_pred):
    #print(K.int_shape(y_true))
    #print(K.int_shape(y_pred))
    # Pinball loss for multiple quantiles
    #K.print_tensor(y_true[:1])
    #K.print_tensor(y_pred[:1])
    qs = [0.09, 0.159, 0.34, 0.50, 0.66, 0.841, 0.91]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    #K.print_tensor(e[:1])
    #K.print_tensor((q*e)[:1])
    #K.print_tensor(((q-1)*e)[:1])
    v = tf.maximum(q*e, (q-1)*e)
    #K.print_tensor(v[:1])
    #K.print_tensor(K.mean(v))
    return K.mean(v)
#=============================#
def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
        #return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*tf.metrics.mean_absolute_error(y_true[:,0], y_pred[:,3])
    return loss
#=================
def make_model(nh):
    z = L.Input((nh,), name="Patient")
    x = L.Dense(100, activation="relu", name="d1")(z)
    x = L.Dense(100, activation="relu", name="d2")(x)
    #x = L.Dense(100, activation="relu", name="d3")(x)
    p1 = L.Dense(7, activation="linear", name="p1")(x)
    p2 = L.Dense(7, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])
    #preds = L.Lambda(lambda x: x[0], name="preds")([p1, p2])
    #preds = L.Lambda(lambda x: tf.cumsum(x[1], axis=1), name="preds")([p1, p2]) 
                     
    
    model = M.Model(z, preds, name="CNN")
    #model.compile(loss=qloss, optimizer="adam", metrics=[score])
    model.compile(loss=mloss(0.1), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])
    return model



# In[ ]:


y = tr['FVC'].values
z = tr[FE].values
ze = sub[FE].values
nh = z.shape[1]
pe = np.zeros((ze.shape[0], 7))
pred = np.zeros((z.shape[0], 7))


# In[ ]:


net = make_model(nh)
print(net.summary())
print(net.count_params())


# In[ ]:


NFOLD = 5
kf = KFold(n_splits=NFOLD)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'cnt = 0\nEPOCHS = 400\ntr_scores = []\nval_scores = []\nfor tr_idx, val_idx in kf.split(z):\n    cnt += 1\n    print(f"FOLD {cnt}")\n    net = make_model(nh)\n    net.fit(z[tr_idx], y[tr_idx], batch_size=BATCH_SIZE, epochs=EPOCHS, \n            validation_data=(z[val_idx], y[val_idx]), verbose=0) #\n    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))\n    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))\n    print("predict val...")\n    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)\n    print("predict test...")\n    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD\n    tr_scores.append(net.evaluate(z[tr_idx], y[tr_idx],verbose=0, batch_size=BATCH_SIZE)[1])\n    val_scores.append(net.evaluate(z[val_idx], y[val_idx],verbose=0, batch_size=BATCH_SIZE)[1])\n#==============\nprint("")\nprint("Mean training score: ", mean(tr_scores))\nprint("Mean validation score: ", mean(val_scores))\nprint("")\nprint("STD training score: ", pstdev(tr_scores))\nprint("STD validation score: ", pstdev(val_scores))\nprint("")\nprint(\'Overfitting metric - mean(val) - mean(tr):\', mean(val_scores) - mean(tr_scores))\nprint(\'Overfitting metric - STD(val) - STD(tr):\', pstdev(val_scores) - pstdev(tr_scores))\nprint("")\n')


# In[ ]:


pe[:10]


# In[ ]:


sigma_opt_mae = mean_absolute_error(y, pred[:, 3])
sigma_opt_mse = sqrt(mean_squared_error(y, pred[:, 3]))
unc = pred[:,5] - pred[:, 1]
unc_real = abs(y - pred[:,3])
sigma_mean = np.mean(unc)
print("MAE, SIGmean: ")
print(sigma_opt_mae, sigma_mean)
print("sqrt(MSE), SIGmean: ")
print(sigma_opt_mse, sigma_mean)


# In[ ]:


idxs = np.random.randint(0, y.shape[0], 100)
plt.plot(y[idxs], label="ground truth")
#plt.figure(figsize=(100,60))
plt.plot(pred[idxs, 0], label="q09")
plt.plot(pred[idxs, 1], label="q18")
plt.plot(pred[idxs, 2], label="q34")
plt.plot(pred[idxs, 3], label="q50")
plt.plot(pred[idxs, 4], label="q66")
plt.plot(pred[idxs, 5], label="q82")
plt.plot(pred[idxs, 6], label="q91")
plt.legend(loc="best")


# In[ ]:


print(unc.min(), unc.mean(), unc.max(), (unc>=0).mean())
print(unc_real.min(), unc_real.mean(), unc_real.max(), (unc_real>=0).mean())


# In[ ]:


plt.hist(unc, bins=100, label='guess')
#plt.hist(unc_real, bins=100, label='real')
plt.title("uncertainty in prediction")
plt.show()


# In[ ]:





# ### PREDICTION

# In[ ]:


sub.head()


# In[ ]:


sub['FVC1'] = 0.996*pe[:, 3]
sub['Confidence1'] = pe[:, 5] - pe[:, 1]


# In[ ]:


subm = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()


# In[ ]:


subm.loc[~subm.FVC1.isnull()].head(10)


# In[ ]:


subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
if sigma_mean<70:
    subm['Confidence'] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']


# In[ ]:


subm.head()


# In[ ]:


subm.describe().T


# In[ ]:


otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
for i in range(len(otest)):
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1


# In[ ]:


subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)

