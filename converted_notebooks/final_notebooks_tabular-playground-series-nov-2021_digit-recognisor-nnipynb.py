#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Current Version 
# Score : 0.97139 
# Characterstics : 
    # Simple ANN model 
    # using simple 

## Use the CNN based approach 
#0.98807
## Tried with highe batch size to check the effect - faster training 
# Used GPU - faster training 

# V13 
# done regularisation 

# TODO for the next version 
## Hyper Parameter Optimization


# This Notebooks uses simple NN to solve the digit recognition problem. This is one of the many approaches that can be used to solve the problem. Can be a good starting point to be begin. 

# In[ ]:


import tensorflow as tf
from sklearn.model_selection import train_test_split


# In[ ]:


def lib_version(lib):
    print("{0} version is : {1}".format(lib.__name__ , lib.__version__))

for lib in [np , pd , tf] :
    lib_version(lib)


# In[ ]:


train_file = '/kaggle/input/digit-recognizer/train.csv'
test_file = '/kaggle/input/digit-recognizer/test.csv'


# In[ ]:


trainDF = pd.read_csv(train_file)


# Performing the normalization and data splitting.

# In[ ]:


y = trainDF['label']
X = trainDF.drop(['label'] , axis = 1)

# This step is used to convert the 1-D array to 3-D array where 28 , 28 , 1 represent witdth , height and color channel  
X = X.to_numpy(dtype='float32').reshape((-1, 28 , 28 , 1 ))

X = ( X * 1.0 )/ 255

X_train , X_Test , y_train , y_test = train_test_split( X , y , test_size = .2)


# Tried with batch size 32 though can be tried with Higher Batch Size. 

# In[ ]:


BATCH_SIZE=512
INPUT_SHAPE = ( X_train[1].shape ) 


# In[ ]:


def df_to_ds ( X  , y  , batch_size , shuffle):
    labels = y
    ds = tf.data.Dataset.from_tensor_slices((X , labels))
    if  shuffle :
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size)
    return ds
        


# In[ ]:


trainDS = df_to_ds ( X_train , y_train , BATCH_SIZE , True  )
valDS = df_to_ds ( X_Test, y_test , BATCH_SIZE , False  )


# In[ ]:


def get_scal(feature):
    def minmax(x):
        mini = trainDF[feature].min()
        maxi = trainDF[feature].max()
        return (x - mini)/(maxi-mini)
    return(minmax)

def transform ( inputs , cols ) : 
    ldf = inputs.copy()

    feature_columns =[]
    num_feature_columns = {colname:
            tf.feature_column.numeric_column(colname )
            for colname in cols
        }
    for num_key,num_col in num_feature_columns.items() :
        feature_columns.append(num_col)
      
    return ldf, feature_columns 


# In[ ]:


def build_simple_nn(optimizer , loss , metrics ):
    
    inputs = {
          colname: tf.keras.layers.Input(name=colname, shape=(), dtype='float32')
          for colname in X_train.columns
      }
    
    transformed , fc = transform (inputs ,  X_train.columns)
    dnn_inputs  = tf.keras.layers.DenseFeatures(fc)
    h1 = tf.keras.layers.Dense(784 , activation = 'relu' , name = 'hidden_1')
    b1 = tf.keras.layers.BatchNormalization()
    h2 = tf.keras.layers.Dense(256 , activation = 'relu' , name = 'hidden_2')
    b2 = tf.keras.layers.BatchNormalization()
    h3 = tf.keras.layers.Dense(64 , activation = 'relu' , name = 'hidden_3')
    b3 = tf.keras.layers.BatchNormalization()
    h4 = tf.keras.layers.Dense(16 , activation = 'relu' , name = 'hidden_4')
    output = tf.keras.layers.Dense(10 , activation = 'softmax' , name = 'output')
    x = dnn_inputs (transformed )
    x = h1 (x)
    x = b1 (x)
    x = h2 (x)
    x = b2 (x)
    x = h3 (x)
    x = b3 (x)
    x = h4 (x)
    outputs = output(x)
    
    model = tf.keras.models.Model(inputs , outputs)
    
    model.compile(optimizer , loss , metrics)
    #model.summary()
    return model


# In[ ]:


def build_covnet(optimizer , loss , metrics):
    inputs = tf.keras.layers.Input(shape = INPUT_SHAPE)
    
    conv1_a = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3) , padding ='same', activation = 'relu')
    conv1_b = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3) , padding ='same', activation = 'relu')
    b1 = tf.keras.layers.BatchNormalization()
    max_1_a = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2, 2))

    conv2_a = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3) , padding ='same', activation = 'relu')
    conv2_b = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3) , padding ='same', activation = 'relu')
    b2 = tf.keras.layers.BatchNormalization()
    max_2_a = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2, 2))

    conv3_a = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3) , padding ='same', activation = 'relu')
    conv3_b = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3) , padding ='same', activation = 'relu')
    b3 = tf.keras.layers.BatchNormalization()
    max_3_a = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2, 2))

    conv4_a = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3) , padding ='same', activation = 'relu')
    conv4_b = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3) , padding ='same', activation = 'relu')
    b4 = tf.keras.layers.BatchNormalization()
    max_4_a = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2, 2))
    
    dropout = tf.keras.layers.Dropout(.3)
    flatten = tf.keras.layers.Flatten()
    
    d1 = tf.keras.layers.Dense(units=128 , kernel_regularizer=tf.keras.regularizers.L2(1e-4) , activation = 'relu')
    b5 = tf.keras.layers.BatchNormalization()
    d2 = tf.keras.layers.Dense(units=64 ,kernel_regularizer=tf.keras.regularizers.L2(1e-4), activation = 'relu')
    b6 = tf.keras.layers.BatchNormalization()
    final = tf.keras.layers.Dense(units=10 , activation = 'softmax')

    x = conv1_a (inputs)
    x = conv1_b (x)
    x = b1(x)
    x = max_1_a (x)
    x = dropout(x)
    
    x = conv2_a (x)
    x = conv2_b (x)
    x = b2(x)
    x = max_2_a (x)
    x = dropout(x)
    
    x = conv3_a (x)
    x = conv3_b (x)
    x = b3(x)
    x = max_3_a (x)
    x = dropout(x)
    
    x = conv4_a (x)
    x = conv4_b (x)
    x = b4(x)
    x = max_4_a (x)
    x = dropout(x)
    
    x = flatten (x)
    
    x = d1(x)
    x = b5(x)
    x = d2(x)
    x = b6(x)
    outputs = final(x)

    model = tf.keras.models.Model(inputs , outputs)
    model.compile(optimizer , loss , metrics)
    #model.summary()
    return model
    
    


# In[ ]:


optimizer = tf.keras.optimizers.RMSprop(learning_rate=.01) #tf.keras.optimizers.Adam(learning_rate=.1 )
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']
nn_model = build_covnet ( optimizer , loss , metrics )


# In[ ]:


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience = 5 , restore_best_weights = True) , 
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy' , patience= 3 , factor=0.5)
]

results = nn_model.fit(  trainDS , validation_data = valDS , epochs = 100 , callbacks = callbacks) 


# In[ ]:


hist = pd.DataFrame(results.history )
hist[['accuracy' , 'val_accuracy']].plot()


# In[ ]:


hist[['loss' , 'val_loss' , 'lr']].plot()


# In[ ]:


testDF = pd.read_csv(test_file)


# In[ ]:


# This step is used to convert the 1-D array to 3-D array where 28 , 28 , 1 represent witdth , height and color channel  
testDF = testDF.to_numpy(dtype='float32').reshape((-1, 28 , 28 , 1 ))

testDF = ( testDF * 1.0 )/ 255


# In[ ]:


def sub_df_to_ds ( X  , batch_size , shuffle):
    
    ds = tf.data.Dataset.from_tensor_slices((X ))
    if  shuffle :
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size)
    return ds


# In[ ]:


testDS =  sub_df_to_ds ( testDF , BATCH_SIZE , False  )


# In[ ]:


sub_arr = nn_model.predict(testDS)


# In[ ]:


y_classes = sub_arr.argmax(axis=-1)
y_classes


# In[ ]:


np.arange(1 , 28001 )


# In[ ]:


iarr = np.array(np.arange(1 , 28001 ), dtype=np.int64)
ser1 = pd.Series(iarr, dtype='int64')
ser2 = pd.Series(y_classes.reshape(28000 ,), dtype=np.int64)
frame = { 'ImageId': ser1, 'Label': ser2 }
submission=pd.DataFrame(frame)


# In[ ]:


submission


# In[ ]:


submission.to_csv('submission.csv',index=False)

