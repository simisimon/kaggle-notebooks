#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -U segmentation-models')


# # Initial Configs

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import keras.backend as K


# In[ ]:


LIMITED_RUN = False
without_sat = False
without_wind = False
dataset_path = '../input/oversample-normalized-data-1-hour-ahead/dataset.hp'


# In[ ]:


get_ipython().system('rm -r ./dataset.hp')


# In[ ]:


import shutil
shutil.copy(dataset_path, './')


# In[ ]:


dataset_dict = h5py.File('./dataset.hp', 'r+')


# In[ ]:


print(dataset_dict['X'].shape)
print(dataset_dict['Y'].shape)


# # Common Function Definitions

# In[ ]:


get_ipython().system('pip install tensorflow_addons')

import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import *
import tensorflow_addons as tfa
import segmentation_models as sm
import cv2

sm.set_framework('tf.keras')
sm.framework()

def pre_process(batch):
    for idx in range(len(batch)):
        batch[idx,:,:,:] = cv2.medianBlur(np.multiply(batch[idx,:,:,:], 255.0), 3)
    return np.divide(batch, 255.0)


def oversample_for_rain(X_TRAIN,Y_TRAIN):
    gamma = 0.7
    good_sample_count = 0
    good_sample=[]
    count_ = 0
    for Y in Y_TRAIN:
        Temp_y = np.count_nonzero(Y == 1)
        if Temp_y >= 819:
            good_sample_count += 1
            good_sample.append(count_)
        count_ = count_ + 1
    if ((good_sample_count/float(count_) < gamma) and (good_sample_count > 0)):
        print('good_sample Amount:',good_sample_count)
        print('total sample Amount:',count_)
        rep_amount = np.round(gamma*(count_)/((1-gamma)*good_sample_count))
        if rep_amount > 20:
            print('Rep Amount Above 20, Actual:', rep_amount)
            rep_amount = 20
        print('Considered Rep Amount:',rep_amount)
        for i in range(int(rep_amount)):
            X_TRAIN = np.concatenate((X_TRAIN, X_TRAIN[good_sample,:]), axis=0)
            Y_TRAIN = np.concatenate((Y_TRAIN, Y_TRAIN[good_sample,:]), axis=0)
    print('shapes', X_TRAIN.shape, Y_TRAIN.shape)    
    return(X_TRAIN,Y_TRAIN)


# In[ ]:


import math
class DataGenerator(keras.utils.Sequence):

    def __init__(self, keys, dataset, batch_multiply=False, suffix=''):
        """Constructor can be expanded,
           with batch size, dimentation etc.
        """
        self.shuffle = True

        self.dataset = dataset
        self.keys = keys
        self.batch_size = 10
        self.batch_multiply = batch_multiply
        self.on_epoch_end()
        self.x_name = 'X'+suffix
        self.y_name = 'Y'+suffix
        

    def __len__(self):
      'Take all batches in each iteration'
      return  math.ceil(int(len(self.keys)) / self.batch_size)

    def __getitem__(self, index):
      'Get next batch'
      # Generate indexes of the batch
      #indexes = self.indexes[index:(index+1)]
      # single file
      indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
      batch_file_keys = [self.keys[k] for k in indexes]

      # Set of X_train and y_train
      X, Y = self.__data_generation(batch_file_keys)
      #print("XY Shapes", np.array(X).shape, np.array(Y).shape)
      return list(X), list(Y)

    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.keys))
      if self.shuffle == True:
        np.random.shuffle(self.indexes)

    def __data_generation(self, batch_file_keys):
        'Generates data containing batch_size samples'
        # x_data_loc = self.x_files_loc
        # y_data_loc = self.y_files_loc
        # Generate data
        X=[]
        Y=[]
        for key_ in batch_file_keys:
            X.append(self.dataset[self.x_name][key_].astype('float32'))
            Y.append(self.dataset[self.y_name][key_].astype('float32'))
            # Store sample
        X = pre_process(np.array(X))
#         if self.batch_multiply:
#             X, Y = oversample_for_rain(np.array(X), np.array(Y))
        return np.array(X)[np.newaxis,:], np.array(Y)[np.newaxis,:]


# In[ ]:


import keras.backend as K
from sklearn.metrics import confusion_matrix
def get_f1(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.round(K.flatten(y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())
#     f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return f1

def get_csi(y_true, y_pred):
    y_true = K.round((K.flatten(y_true)))
    y_pred = K.round(K.flatten(y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    csi = tp/(tp + fp + fn + K.epsilon())
    
    return K.mean(csi)


# def get_f1_loss(y_true, y_pred):
#     print('A--')
#     y_true = K.flatten(y_true)
#     y_pred = K.flatten(y_pred)
#     print('AA')
#     xx = K.flatten(tf.math.confusion_matrix(y_true, y_pred))
#     tn = K.cast(xx[0], 'float32')
#     fp = K.cast(xx[1], 'float32')
#     fn = K.cast(xx[2], 'float32')
#     tp = K.cast(xx[3], 'float32')
#     print('CC', xx)
#     p = tp / (tp + fp)
#     r = tp / (tp + fn)
#     print('DD',p)
#     f1 = 2*p*r / (p+r)
#     print('EE')
# #     f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
#     return 1 - K.cast(K.mean(f1), 'float32')
def get_f1_loss(y_true, y_pred):
  y_true = K.flatten(y_true)
  y_pred = (K.flatten(y_pred))
  tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
  tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
  fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
  fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

  p = tp / (tp + fp + K.epsilon())
  r = tp / (tp + fn + K.epsilon())
  f1 = 2*p*r / (p+r+K.epsilon())
  f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
  
  return 1 - K.mean(f1)

# def get_csi_loss(y_true, y_pred):
#     y_true = (K.flatten(y_true))
#     y_pred = (K.flatten(y_pred))
#     tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
#     tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
#     fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
#     fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
#     csi = tp/(tp + fp + fn + K.epsilon())
#     csi = tf.where(tf.math.is_nan(csi), tf.zeros_like(csi), csi)
    
#     return 1 - K.mean(csi)

# def weighted_cross_entropy(beta):
#   def loss(y_true, y_pred):
#     weight_a = beta * tf.cast(y_true, tf.float32)
#     weight_b = 1 - tf.cast(y_true, tf.float32)
    
#     o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b 
#     return tf.reduce_mean(o)

#   return loss

# def FocalLoss(y_true, y_pred):    
#     alpha = 0.95
#     gamma = 2
#     inputs = K.flatten(y_true)
#     targets = K.flatten(y_pred)
    
#     BCE = K.binary_crossentropy(targets, inputs)
#     BCE_EXP = K.exp(-BCE)
#     focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
#     return focal_loss

# def DiceBCELoss(y_true, y_pred):
#     smooth=1e-6
#     #flatten label and prediction tensors
# #     print(K.shape(y_true), K.shape(y_pred))
#     inputs = K.flatten(y_true)
#     targets = K.flatten(y_pred)
# #     print(K.shape(inputs), K.shape(targets))
#     BCE =  K.binary_crossentropy(targets, inputs)
# #     print('SS')
#     intersection = K.sum(targets * inputs)
# #     print('JJ')
#     dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
#     Dice_BCE = BCE + dice_loss
#     return (Dice_BCE)

def weighted_binary_crossentropy(zero_weight, one_weight):
    def weighted_binary_crossentropy(y_true, y_pred):
        b_ce = K.binary_crossentropy(y_true, y_pred)
        # weighted calc
        weight_vector = y_true * one_weight + (1 - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy


# In[ ]:


n_classes = 2
activation = 'sigmoid'
LR = 0.0001
optim = keras.optimizers.Adam(LR)

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + focal_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


# In[ ]:


from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def train_and_test(backbone, loss, learning_rate,  dataset_dict, seq_length, train_gen, valid_gen, test_gen):
    model = None
    
    model = sm.Unet(backbone,input_shape=(128,128,seq_length), encoder_weights=None, classes=1, activation=activation)
    model.compile(optim, loss, metrics=metrics)
#     model.save_weights('model_' + backbone + '_' + str(learning_rate) + '_' + loss + '_0.h5')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    
    best_model_name = 'model_' + backbone + '_' + 'seq_' + str(seq_length) + '_best_model.tf'
    
    mc = ModelCheckpoint(best_model_name, monitor='val_f1-score', mode='max', verbose=1, save_best_only=True)
    
    print('--------------------------Model Training START---------------------------------')
    history=model.fit(x=train_gen,  validation_data=valid_gen, epochs=50, callbacks=[es, mc])
    print('--------------------------Model Training END---------------------------------')
    
    print('--------------------------VAL F1 Score---------------------------------')
    fig, ax = plt.subplots()
    ax.plot(history.history['val_f1-score'],'o-',label="val_f1_score")
    print('--------------------------VAL LOSS---------------------------------')
    fig, ax = plt.subplots()
    ax.plot(history.history['val_loss'],'o-',label="val_loss")
    
    model.load_weights(best_model_name)
    
    print('--------------------------EVALUATE BEST MODEL---------------------------------')
    model.evaluate(x=test_gen)
    
    #(x_files_loc, x_files_test, y_files_loc, y_files_test)
    test_batches = [23,69,11,155]
    print('--------------------------SAMPLE OUTPUTS---------------------------------')
    for test_batch in test_batches:
        print('Sample '+str(test_batch) + '---------------------------------')
        out_class = 0
        im_in = dataset_dict['X'][test_idx[test_batch]]
        im_out = dataset_dict['Y'][test_idx[test_batch]]

        Y_predict = model.predict(pre_process(im_in[np.newaxis,:]))
        # print('F1 Score for batch:', get_f1(im_out,Y_predict))
        # print('CSI Score for batch:', get_csi(im_out,Y_predict))
        # print('F1 Score for selected image class:', get_f1(im_out[batch_image_for_viz,:,:,out_class],Y_predict[batch_image_for_viz,:,:,out_class]))

        plt.figure()
        plt.pcolormesh(im_out,vmin=0,vmax=1)
        plt.title('test_idx'+ str(test_idx[test_batch]))
        plt.colorbar()
        

        plt.figure()
        plt.pcolormesh(Y_predict[0,:,:,0])
        plt.colorbar()

        plt.figure()
        Y_p = Y_predict[0,:,:,0].copy()
        Y_p_ = Y_p.copy()
        Y_p_[Y_p>=0.5] = 1
        Y_p_[Y_p<0.5] = 0
        plt.pcolormesh(Y_p_)
    return model


# # Training Common

# In[ ]:


for i in range(dataset_dict['X'].shape[3]):
    plt.figure()
    plt.pcolormesh(dataset_dict['X'][997,:,:,i])
    plt.colorbar()
plt.figure()
plt.pcolormesh(dataset_dict['Y'][997,:,:])
plt.colorbar()


# In[ ]:


im = pre_process(dataset_dict['X'][0:5,:,:,:])
for i in range(dataset_dict['X'].shape[3]):
    plt.figure()
    plt.pcolormesh(im[0,:,:,i])
    plt.colorbar()
plt.figure()
plt.pcolormesh(dataset_dict['Y'][0,:,:])
plt.colorbar()  


# In[ ]:


train_idx = list(range(dataset_dict['Y'].shape[0]))
valid_idx = list(range(dataset_dict['Y_valid'].shape[0]))
test_idx = list(range(dataset_dict['Y_test'].shape[0]))


# # Rain Only Training

# In[ ]:


train_rain = True


# In[ ]:


print('X Shape', dataset_dict['X'].shape)
print('Y Shape', dataset_dict['Y'].shape)


# In[ ]:


if train_rain:
    data_x = dataset_dict['X']
    data_x_rain =  data_x[:,:,:,:5]
    dataset_dict['X'].resize(data_x_rain.shape)
    data_x[...] = data_x_rain


# In[ ]:


if train_rain:
    data_test_x = dataset_dict['X_test']
    data_test_x_rain =  data_test_x[:,:,:,:5]
    dataset_dict['X_test'].resize(data_test_x_rain.shape)
    data_test_x[...] = data_test_x_rain


# In[ ]:


if train_rain:
    data_valid_x = dataset_dict['X_valid']
    data_valid_x_rain =  data_valid_x[:,:,:,:5]
    dataset_dict['X_valid'].resize(data_valid_x_rain.shape)
    data_valid_x[...] = data_valid_x_rain


# In[ ]:


if train_rain:
    del data_x
    del data_x_rain


# In[ ]:


if train_rain:
    print('X Shape', dataset_dict['X'].shape)
    print('Y Shape', dataset_dict['Y'].shape)


# In[ ]:


if train_rain:
    im = pre_process(dataset_dict['X'][0:5,:,:,:])
    for i in range(dataset_dict['X'].shape[3]):
        plt.figure()
        plt.pcolormesh(im[0,:,:,i])
        plt.colorbar()
    plt.figure()
    plt.pcolormesh(dataset_dict['Y'][0,:,:])
    plt.colorbar()


# In[ ]:


if train_rain:
    train_gen = DataGenerator(train_idx, dataset_dict, batch_multiply=False)
    valid_gen = DataGenerator(valid_idx, dataset_dict, batch_multiply=False, suffix='_valid')
    test_gen = DataGenerator(test_idx, dataset_dict, batch_multiply=False, suffix='_test')


# In[ ]:


# model_res = train_and_test('resnet18',  weighted_binary_crossentropy(0.3,0.7), 0.0001,  dataset_dict, 5, train_gen, valid_gen, test_gen)


# In[ ]:


if train_rain:
    train_and_test('vgg16',  'binary_crossentropy', 0.001,  dataset_dict, 5, train_gen, valid_gen, test_gen)


# In[ ]:


if train_rain:
    train_and_test('resnet50',  'binary_crossentropy', 0.001,  dataset_dict, 5, train_gen, valid_gen, test_gen)


# # Rain + Sat

# In[ ]:


rain_sat_train =False


# In[ ]:


print('X Shape', dataset_dict['X'].shape)
print('Y Shape', dataset_dict['Y'].shape)


# In[ ]:


if rain_sat_train:
    data_x = dataset_dict['X']
    data_x_rain =  data_x[:,:,:,[0,1,2,3,4,9,10]]
    dataset_dict['X'].resize(data_x_rain.shape)
    data_x[...] = data_x_rain


# In[ ]:


if rain_sat_train:
    data_test_x = dataset_dict['X_test']
    data_test_x_rain =  data_test_x[:,:,:,[0,1,2,3,4,9,10]]
    dataset_dict['X_test'].resize(data_test_x_rain.shape)
    data_test_x[...] = data_test_x_rain


# In[ ]:


if rain_sat_train:
    data_valid_x = dataset_dict['X_valid']
    data_valid_x_rain =  data_valid_x[:,:,:,[0,1,2,3,4,9,10]]
    dataset_dict['X_valid'].resize(data_valid_x_rain.shape)
    data_valid_x[...] = data_valid_x_rain


# In[ ]:


if rain_sat_train:
    print('X Shape', dataset_dict['X'].shape)
    print('Y Shape', dataset_dict['Y'].shape)


# In[ ]:


if rain_sat_train:
    im = pre_process(dataset_dict['X'][0:5,:,:,:])
    for i in range(dataset_dict['X'].shape[3]):
        plt.figure()
        plt.pcolormesh(im[0,:,:,i])
        plt.colorbar()
    plt.figure()
    plt.pcolormesh(dataset_dict['Y'][0,:,:])
    plt.colorbar()


# In[ ]:


if rain_sat_train:
    train_gen = DataGenerator(train_idx, dataset_dict, batch_multiply=False)
    valid_gen = DataGenerator(valid_idx, dataset_dict, batch_multiply=False)
    test_gen = DataGenerator(test_idx, dataset_dict, batch_multiply=False)


# In[ ]:


if rain_sat_train:
    train_and_test('vgg16', 'binary_crossentropy', 0.001,  dataset_dict, 7, train_gen, valid_gen, test_gen)


# In[ ]:


if rain_sat_train:
    train_and_test('resnet50', 'binary_crossentropy', 0.001,  dataset_dict, 7, train_gen, valid_gen, test_gen)


# # Rain + Sat + Wind

# In[ ]:


rain_sat_wind_train = False


# In[ ]:


if rain_sat_wind_train:
    print('X Shape', dataset_dict['X'].shape)
    print('Y Shape', dataset_dict['Y'].shape)


# In[ ]:


if rain_sat_wind_train:
    im = pre_process(dataset_dict['X'][0:5,:,:,:])
    for i in range(dataset_dict['X'].shape[3]):
        plt.figure()
        plt.pcolormesh(im[0,:,:,i])
        plt.colorbar()
    plt.figure()
    plt.pcolormesh(dataset_dict['Y'][0,:,:])
    plt.colorbar()


# In[ ]:


if rain_sat_wind_train:
    train_gen = DataGenerator(train_idx, dataset_dict, batch_multiply=False)
    valid_gen = DataGenerator(valid_idx, dataset_dict, batch_multiply=False, suffix='_valid')
    test_gen = DataGenerator(test_idx, dataset_dict, batch_multiply=False, suffix='_test')


# In[ ]:


if rain_sat_wind_train:
    train_and_test('vgg16',  'binary_crossentropy', 0.001,  dataset_dict, 11, train_gen, valid_gen, test_gen)


# In[ ]:


if rain_sat_wind_train:
    train_and_test('resnet50', 'binary_crossentropy', 0.001,  dataset_dict, 11, train_gen, valid_gen, test_gen)


# # Rain + Wind

# In[ ]:


rain_wind_train = False


# In[ ]:


if rain_wind_train:
    print('X Shape', dataset_dict['X'].shape)
    print('Y Shape', dataset_dict['Y'].shape)


# In[ ]:


if rain_wind_train:
    data_x = dataset_dict['X']
    data_x_rain =  data_x[:,:,:,[0,1,2,3,4,5,6,7,8]]
    dataset_dict['X'].resize(data_x_rain.shape)
    data_x[...] = data_x_rain


# In[ ]:


if rain_wind_train:
    data_test_x = dataset_dict['X_test']
    data_test_x_rain =  data_test_x[:,:,:,[0,1,2,3,4,5,6,7,8]]
    dataset_dict['X_test'].resize(data_test_x_rain.shape)
    data_test_x[...] = data_test_x_rain


# In[ ]:


if rain_wind_train:
    data_valid_x = dataset_dict['X_valid']
    data_valid_x_rain =  data_valid_x[:,:,:,[0,1,2,3,4,5,6,7,8]]
    dataset_dict['X_valid'].resize(data_valid_x_rain.shape)
    data_valid_x[...] = data_valid_x_rain


# In[ ]:


if rain_wind_train:
    print('X Shape', dataset_dict['X'].shape)
    print('Y Shape', dataset_dict['Y'].shape)


# In[ ]:


if rain_wind_train:
    im = pre_process(dataset_dict['X'][0:5,:,:,:])
    for i in range(dataset_dict['X'].shape[3]):
        plt.figure()
        plt.pcolormesh(im[0,:,:,i])
        plt.colorbar()
    plt.figure()
    plt.pcolormesh(dataset_dict['Y'][0,:,:])
    plt.colorbar()


# In[ ]:


if rain_wind_train:
    train_gen = DataGenerator(train_idx, dataset_dict, batch_multiply=False)
    valid_gen = DataGenerator(valid_idx, dataset_dict, batch_multiply=False, suffix='_valid')
    test_gen = DataGenerator(test_idx, dataset_dict, batch_multiply=False, suffix='_test')


# In[ ]:


if rain_wind_train:
    train_and_test('vgg16',  'binary_crossentropy', 0.001,  dataset_dict, 9, train_gen, valid_gen, test_gen)


# In[ ]:


if rain_wind_train:
    train_and_test('resnet50',  'binary_crossentropy', 0.001,  dataset_dict, 9, train_gen, valid_gen, test_gen)

