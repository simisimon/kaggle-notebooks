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


# # Import Library

# In[ ]:


import matplotlib.pyplot as plt
import cv2
import random
from sklearn.model_selection import train_test_split


# # Load Image Data

# In[ ]:


import os
from tqdm import tqdm

DIR = '/kaggle/input/surface-crack-detection'
label = ['Negative', 'Positive']
def load_image(data_dir):
    img_data = []
    
    for word in label:
        path = os.path.join(data_dir, word)
        y_label = label.index(word)
        
        for img_name in tqdm(os.listdir(path)):
            img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            resizing = cv2.resize(img, (120,120))
            
            img_data.append([resizing, y_label])
            
    return np.array(img_data)

dataset = load_image(DIR)


# 이미지 데이터를 불러옴

# In[ ]:


plt.figure(figsize = (12,12))
plt.imshow(dataset[20003][0], cmap = 'gray')


# # Image Processing

# ### Gaussian Blur & Thresholding
# > 모델 학습에서 성능을 높이기 위해 이미지 프로세싱 기법을 적용
# > * 가우시안 블러를 통해 불순물을 제거
# > * 스레시홀딩 기법을 통해 바이너리 이미지로 변환
# >> 바이너리 이미지 : 검은색과 흰색으로만 표현한 이미지

# In[ ]:


#가우시간 블러 + 스레시홀딩
def threshold_gaussian(data_list):
    thr_gau = []
    for data in tqdm(data_list):
        #tmp = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
        tmp = cv2.GaussianBlur(data[0], (9,9), 0)

        tmp = cv2.adaptiveThreshold(
            tmp,
            maxValue = 255.0,
            adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType = cv2.THRESH_BINARY_INV,
            blockSize = 19,
            C = 9
        )
        thr_gau.append([tmp, data[1]])
    del tmp
    return np.array(thr_gau)
        
#스레시홀딩만 적용
def threshold(data_list):
    thr_only = []
    for data in tqdm(data_list):
        #tmp = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
        
        tmp = cv2.adaptiveThreshold(
        data[0],
        maxValue = 255.0,
        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType = cv2.THRESH_BINARY_INV,
        blockSize = 19,
        C = 9
        )
        thr_only.append([tmp, data[1]])
    del tmp
    return np.array(thr_only)

dataset_t = threshold(dataset)
dataset_tg = threshold_gaussian(dataset)


# > 비교를 위해 스레시홀딩만 적용한 이미지와 가우시안블러와 스레시홀딩을 둘 다 적용한 이미지를 출력

# In[ ]:


rand_po = random.randrange(20001,40001)

plt.figure(figsize = (12,12))

plt.subplot(1,3,1)
plt.title('Original')
plt.imshow(dataset[rand_po][0], cmap = 'gray')

plt.subplot(1,3,2)
plt.title('Threshold Only')
plt.imshow(dataset_t[rand_po][0], cmap = 'gray')

plt.subplot(1,3,3)
plt.title("GaussianBLUR and Threshold")
plt.imshow(dataset_tg[rand_po][0], cmap = 'gray')


# In[ ]:


rand_ne = random.randrange(1,20001)

plt.figure(figsize = (12,12))

plt.subplot(1,3,1)
plt.title('Original')
plt.imshow(dataset[rand_ne][0], cmap = 'gray')

plt.subplot(1,3,2)
plt.title('Threshold Only')
plt.imshow(dataset_t[rand_ne][0], cmap = 'gray')

plt.subplot(1,3,3)
plt.title("BLUR and Threshold")
plt.imshow(dataset_tg[rand_ne][0], cmap = 'gray')


# > 가우시안블러와 스레시홀딩을 둘 다 적용한 이미지가 깔끔하게 처리된 것을 볼 수 있음

# In[ ]:


del(dataset_t)


# # Data Split

# > 모델 학습에 입력할 train 데이터와 모델 평가에 쓰일 test 데이터를 분리

# In[ ]:


X = list(zip(*dataset_tg))[0]
y = list(zip(*dataset_tg))[1]

X = np.array(X).reshape(-1, 120, 120, 1)
X = X / 255
y = np.array(y)


# > * X는 이미지 데이터
# >> 0 ~ 255 사이의 값을 255로 나눠줌으로써 scaling 적용
# > * y는 0 또는 1의 label 데이터

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify  = y, random_state = 5, shuffle = True)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


'''X_or = list(zip(*dataset))[0]
y_or = list(zip(*dataset))[1]

X_or = np.array(X_or).reshape(-1, 120, 120, 1)
X_or = X_or / 255
y_or = np.array(y_or)

Xor_train, Xor_test, yor_train, yor_test = train_test_split(X_or, y_or, test_size = 0.3, stratify  = y_or, random_state = 5, shuffle = True)'''


# # Create CNN Model & Model Training

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D,MaxPool2D , Dropout,Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping


model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape = X.shape[1:]))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(128, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))

model.add(BatchNormalization())
model.add(Dense(1, activation="sigmoid"))


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

Dir = './model/'
if not os.path.exists(Dir):
  os.mkdir(Dir)
modelpath = "./model/{epoch:02d} - {val_loss:.4f}.hdf5"

checkpointer =ModelCheckpoint(filepath = modelpath, monital = 'val_loss', verbose = 1, save_best_only= True)
early = EarlyStopping(monitor = 'val_loss', patience = 5)


history = model.fit(X_train, y_train, validation_data = (X_test, y_test),epochs = 12, batch_size = 100, verbose = 0, callbacks = [early, checkpointer])
print('\nAccuracy : {:.4f}'.format(model.evaluate(X_test, y_test)[1]))


# In[ ]:


'''history = model.fit(Xor_train, yor_train, validation_data = (Xor_test, yor_test),epochs = 12, batch_size = 100, verbose = 0, callbacks = [early, checkpointer])
print('\nAccuracy : {:.4f}'.format(model.evaluate(Xor_test, yor_test)[1]))
'''


# # Model Accuracy

# In[ ]:


y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, c = 'red', markersize = 3,label = 'Testset_loss')
plt.plot(x_len, y_loss, c = 'blue', markersize = 3,label = 'Trainset_loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


y_vacc = history.history['val_accuracy']
y_acc = history.history['accuracy']

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vacc, c = 'red', markersize = 3,label = 'Testset_accuracy')
plt.plot(x_len, y_acc, c = 'blue', markersize = 3,label = 'Trainset_accuracy')
plt.legend(loc = 'lower right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

