#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###### import numpy as np
import tensorflow as tf
import random as rn
import numpy as np

import os

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)

from keras import backend as K

#tf.set_random_seed(1234)
tf.random.set_seed(1234)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

K.set_session(sess)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.layers import Input, Dense,LSTM,RepeatVector,GRU,Dropout,Reshape
from keras.layers import*
from keras.models import Model
from keras.models import Sequential
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
import matplotlib.pyplot as plt
from math import sqrt
import random
import warnings
warnings.simplefilter("ignore", DeprecationWarning)


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import random_projection
from sklearn import cluster
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import MiniBatchSparsePCA
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


def parser(x):
	return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[:,0][-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = []
	for x in X:
   		 new_row = new_row+[i for i in x]
            
	new_row.append(value) 
	new_row_2 = np.array(new_row)
	new_row_2 = new_row_2.reshape((1,new_row_2.shape[0]))
	inverted = scaler.inverse_transform(new_row_2)
	return inverted[0, -1]


def create_dataset(dataset,features, look_back=1):
	dataset = np.insert(dataset,[0]*look_back,0)    
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	dataY= np.array(dataY)        
	dataY = np.reshape(dataY,(dataY.shape[0],features))
	dataset = np.concatenate((dataX,dataY),axis=1)  
	return dataset



# convert series to supervised learning
def series_to_supervised(data,features, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	x = np.zeros(features, dtype=np.int)
	for i in range(n_in):
		data = np.insert(data,x,0)
	data = data.reshape(int(data.shape[0]/features),features) 
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(batch_size, X.shape[0], X.shape[1])
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]




# In[ ]:


space = {'seq_len':hp.choice('seq_len',[5,10,15,20,25,30]),
         'epochs_pre':hp.choice('epochs_pre',[i for i in range(50,1000)]),
         'epochs_finetune':hp.choice('epochs_finetune',[i for i in range(50,500)]),
         'units':hp.choice('units',[i for i in range(1,50)]),
         'dropout':hp.choice('dropout',[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
         'batch_size':hp.choice('batch_size',[73, 146, 219])}


# In[ ]:


params = {'seq_len': 10,
          'epochs_pre': 100,
          'epochs_finetune': 50,
          'units': 20,
          'dropout': 0.2,
          'batch_size': 73}


# In[ ]:


hidden_layers = [445,359,4]
batch_size = 219
dropout = 0.2
seq_len = 25
epochs_pre  = [625,115,933]
epochs_finetune = 197
window_size = 0
features = 8


# In[ ]:


from glob import glob
import pandas as pd
import numpy as np
data = glob('../input/dataarif54k/*/*.csv')
fn = (np.array([pd.read_csv(f)for f in data])).reshape(-1,10)
fn.shape 
fs = pd.DataFrame(fn).drop([0],axis=1)
fs


# In[ ]:


'''series = read_csv('../input/saecontoh/pollution.csv', header=0, index_col=0)
raw_values = series.values

# integer encode wind direction
encoder = LabelEncoder()
raw_values[:,4] = encoder.fit_transform(raw_values[:,4])

# transform data to be stationary
diff = difference(raw_values, 1)


dataset = diff.values
dataset = create_dataset(dataset,features,window_size)'''


# In[ ]:


fs.drop([9],axis=1).values


# In[ ]:


#pd.DataFrame(dataset) 


# In[ ]:


def rms (a):
    out = np.sqrt(np.mean(np.square(a), axis = 1))
  
    return out


# In[ ]:


def filters (chanel,pole,low,high):
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    dn = signal.filtfilt(b_notch, a_notch, chanel)
    b, a = scipy.signal.butter(pole, [low, high], 'band')
    df = scipy.signal.lfilter(b, a, dn)
    dframe = pd.DataFrame(df)
    
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(chanel, label = 'data')
    ax[0].legend()
    ax[1].plot(df,label ='filter' )
    ax[1].legend()
    return dframe


# In[ ]:


def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		end_ix = i + n_steps
		if end_ix > len(sequences):
			break
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X),np.array(y)


# In[ ]:


import scipy.signal
from scipy import signal
import matplotlib.pyplot as plt
def prep(x,y)  :
    
    ch1  = filters (x.ch1,pole,low,high)
    ch2  = filters (x.ch2,pole,low,high)
    ch3  = filters (x.ch3,pole,low,high)
    ch4  = filters (x.ch4,pole,low,high)
    ch5  = filters (x.ch5,pole,low,high)
    ch6  = filters (x.ch6,pole,low,high)
    ch7  = filters (x.ch7,pole,low,high)
    ch8  = filters (x.ch8,pole,low,high)
    
    

    datafil = pd.concat([ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,y],axis=1)
    datafil.columns= ['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8','label']
    
    h1  = datafil['ch1'].values
    h2  = datafil['ch2'].values
    h3  = datafil['ch3'].values
    h4  = datafil['ch4'].values
    h5  = datafil['ch5'].values
    h6  = datafil['ch6'].values
    h7  = datafil['ch7'].values
    h8  = datafil['ch8'].values
    lab = datafil['label'].values

    c1 = h1.reshape(len(h1),1)
    c2 = h2.reshape(len(h2),1)
    c3 = h3.reshape(len(h3),1)
    c4 = h4.reshape(len(h4),1)
    c5 = h5.reshape(len(h5),1)
    c6 = h6.reshape(len(h6),1)
    c7 = h7.reshape(len(h7),1)
    c8 = h8.reshape(len(h8),1)
    lb = lab.reshape(len(lab),1)
    
    seg =  np.hstack((c1,c2,c3,c4,c5,c6,c7,c8, lb))
    x, y = split_sequences(seg, n_steps)
    
    print('data windowing',x.shape, y.shape)
    xr = x.reshape(-1,n_chanel*n_steps)
    print('data reshape',xr.shape)
      
    out= []
    for a in  x :
                
                df = pd.DataFrame(a)
                df.columns = ['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
                
                rms1 = (np.mean(df['ch1'].values ** 2))
                rms2 = (np.mean(df['ch2'].values ** 2)) 
                rms3 = (np.mean(df['ch3'].values ** 2)) 
                rms4 = (np.mean(df['ch4'].values ** 2)) 
                rms5 = (np.mean(df['ch5'].values ** 2)) 
                rms6 = (np.mean(df['ch6'].values ** 2)) 
                rms7 = (np.mean(df['ch7'].values ** 2)) 
                rms8 = (np.mean(df['ch8'].values ** 2))
                rms_total = np.hstack((rms1,rms2,rms3,rms4,rms5,rms6,rms7,rms8))
    
                out.append(rms_total)
    print('data rms',np.array(out).shape)
    fig,ax         = plt.subplots(1,1,figsize=(10,5))
    ax.plot(out);
    data_rms       = np.array(out)

   
    
    return data_rms,x,y


# In[ ]:


std = StandardScaler()

dp = fs
dp.columns= [	'ch1',	'ch2',	'ch3',	'ch4',	'ch5',	'ch6',	'ch7',	'ch8','label']
Xp= dp.drop(['label'],axis=1)
Xp= std.fit_transform(Xp)
Xp = pd.DataFrame(Xp)
Xp.columns= [	'ch1',	'ch2',	'ch3',	'ch4',	'ch5',	'ch6',	'ch7',	'ch8']
Yp =dp.label
Yp


# In[ ]:


dp


# In[ ]:


Xp


# In[ ]:


## filtering
sf= 250

low = 7/sf
high = 13/sf
pole = 3
samp_freq = sf/ 2
notch_freq =60.0  
quality_factor = 1.58

#winddwing
n_steps  = 25
n_chanel = 8

X_rms,X_win,y= prep(Xp,Yp)


# In[ ]:


X_win.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_win.reshape(-1,25*8), y, test_size=0.2,shuffle =True)
X_train.shape, X_test.shape, y_train.shape, y_test.shape   


# In[ ]:


X_train[:50000].shape


# In[ ]:


y_train[:50000].shape


# In[ ]:


model = RandomForestClassifier()
model.fit(X_train[:50000],y_train[:50000])  
model.score(X_train[:50000],y_train[:50000]),model.score(X_test[:10000],y_test[:10000])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_win, y, test_size=0.2,shuffle =True)
X_train.shape, X_test.shape, y_train.shape, y_test.shape  


# In[ ]:


x_train = X_train[:35040]
x_train.shape


# In[ ]:


print('\nstart pretraining')
print('===============')

timesteps = x_train.shape[1]
input_dim = x_train.shape[2]
trained_encoder = []
x_train_temp = x_train


# In[ ]:


print('>> 1 LAYERS')
hidden = hidden_layers[0]
epochs = epochs_pre[0]


# In[ ]:


inputs = Input(batch_shape=(5, timesteps, x_train_temp.shape[2]))
encoded1 = CuDNNLSTM(
    hidden, # 35
    batch_input_shape=(5, timesteps, x_train_temp.shape[2]),
    stateful=False)(inputs)
encoded2 = CuDNNLSTM(
    hidden, # 35
    batch_input_shape=(5, timesteps, x_train_temp.shape[2]),
    stateful=True)(inputs)


# In[ ]:


ae1 = Model(inputs, encoded1)
ae2 = Model(inputs, encoded2)


# In[ ]:


ae1.predict(x_train_temp[0:5])


# In[ ]:


ae2.predict(x_train_temp[0:5])


# In[ ]:





# In[ ]:


print(f'pretrain Autoencoder: {0} ----> Encoder: {hidden} ----> Epochs: {epochs}')
print(x_train_temp.shape)
print('=============================================================')

inputs = Input(
    # (219, 25, 8)
    batch_shape=(batch_size, timesteps, x_train_temp.shape[2]))
encoded = CuDNNLSTM(
    hidden, # 35
    batch_input_shape=(batch_size, timesteps, x_train_temp.shape[2]),
    stateful = False)(inputs)
decoded = RepeatVector(timesteps)(encoded) 
decoded = CuDNNLSTM(input_dim, stateful=False, return_sequences=True)(decoded)

AE = Model(inputs, decoded)

encoder = Model(inputs,encoded)

AE.compile(loss='mean_squared_error', optimizer='Adam')

encoder.compile(loss='mean_squared_error', optimizer='Adam')

AE.summary()


# In[ ]:


AE.fit(
    x_train_temp, 
    x_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    verbose=1
)


# In[ ]:


# store trained encoder and its weights
trained_encoder.append(
    (AE.layers[1], AE.layers[1].get_weights())
)

# update training data
x_train_temp = encoder.predict(x_train_temp, batch_size=batch_size)


# In[ ]:


x_train_temp.shape


# In[ ]:


# reshape encoded input to 3D
inputs = Input(shape = (x_train_temp.shape[1],)) 
reshape = RepeatVector(timesteps)(inputs)
Repeat = Model(inputs,reshape)

x_train_temp = Repeat.predict(x_train_temp, batch_size=batch_size)


# In[ ]:


x_train_temp.shape


# In[ ]:


print('>> 2 LAYERS')
hidden = hidden_layers[1]
epochs = epochs_pre[1]


# In[ ]:


print(f'pretrain Autoencoder: {1} ----> Encoder: {hidden} ----> Epochs: {epochs}')
print(x_train_temp.shape)
print('=============================================================')

inputs = Input(
    # (219, 25, 35)
    batch_shape=(batch_size, timesteps, x_train_temp.shape[2]))
encoded = CuDNNLSTM(
    hidden, # 49
    batch_input_shape=(batch_size, timesteps, x_train_temp.shape[2]),
    stateful = False)(inputs)
decoded = RepeatVector(timesteps)(encoded) 
decoded = CuDNNLSTM(input_dim, stateful=False, return_sequences=True)(decoded)

AE = Model(inputs, decoded)

encoder = Model(inputs,encoded)

AE.compile(loss='mean_squared_error', optimizer='Adam')

encoder.compile(loss='mean_squared_error', optimizer='Adam')

AE.summary()


# In[ ]:


AE.fit(
    x_train_temp, 
    x_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    verbose=1
)


# In[ ]:


# store trained encoder and its weights
trained_encoder.append(
    (AE.layers[1], AE.layers[1].get_weights())
)

# update training data
x_train_temp = encoder.predict(x_train_temp, batch_size=batch_size)


# In[ ]:


x_train_temp.shape


# In[ ]:


# reshape encoded input to 3D
inputs = Input(shape = (x_train_temp.shape[1],)) 
reshape = RepeatVector(timesteps)(inputs)
Repeat = Model(inputs,reshape)

x_train_temp = Repeat.predict(x_train_temp, batch_size=batch_size)


# In[ ]:


x_train_temp.shape


# In[ ]:


Y1 = pd.DataFrame(y_train)
Y1


# In[ ]:


labelencoder = LabelEncoder()
Y1 = pd.get_dummies(Y1,columns=[0])
Y1.columns= ['enc1','enc2','enc3']
Y1


# In[ ]:


# Fine-turning
print('\nFine-turning')
print('============')

l = len(trained_encoder)
#build finetuning model
model = Sequential()
for i,encod in enumerate(trained_encoder):
    model.add(encod[0])
    model.layers[-1].set_weights(encod[1])
    model.add(Dropout(dropout))
    if(i+1 != l): model.add(RepeatVector(timesteps))

model.add(Dense(3))

model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])

model.fit(
    x_train, 
    Y1[:35040], 
    #epochs=epochs_finetune, 
    epochs= 2000,
    batch_size=batch_size, 
    verbose=1,
    shuffle=True
)


# In[ ]:


# save trained model
model.save('3layer_25.h5')

# redefine the model in order to test with one sample at a time (batch_size = 1)
new_model = Sequential()
new_model.add(
    CuDNNLSTM(
        hidden_layers[0],
        batch_input_shape=(1, timesteps, input_dim),
        stateful=False)
)
for layer in model.layers[1:]:
    new_model.add(layer)

# copy weights
old_weights = model.get_weights()
new_model.set_weights(old_weights)


# In[ ]:


new_model.summary()


# In[ ]:


from tensorflow import keras

mod = keras.models.load_model('./3layer_25.h5')
mod.summary()


# In[ ]:


plt.subplot(121)
plt.plot(np.argmax(mod.predict(x_train),axis=1)[:50])
plt.subplot(122)
plt.plot(y_train[:50])

