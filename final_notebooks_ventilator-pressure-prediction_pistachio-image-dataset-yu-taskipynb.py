#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#첨부한 이미지 데이터셋을 사용하여 2종류의 피스타치오를 분류하는 모델을 학습하고 결과를 보고서로 제출하세요
#1. Kirmizi_Pistachio와 Siirt_Pistachio를 구분하는 binary classification입니다.
#2. 기존의 CNN 모델을 사용하여 transfer learning이나 fine tuning을 해도 좋고 직접 CNN 모델을 만들어서 학습해도 좋습니다.
#3. training data 70%, validation data 20%, test data 10%로 데이터 셋을 만들어서 학습과 성능 검증을 수행합니다. 세 가지 데이터셋에서 두 종류의 피스타치오 비율은 동일(거의!)하도록 합니다. scikitlearn train_test_split()을 사용하면 됩니다.
#4. 과제에는 아래의 내용이 포함되어야 합니다.
# - 사용한 모델과 학습 하이퍼파라미터, 
# - 커스텀 모델일 경우 model summary 표시
# - 모델 성능: 세 데이터셋의 accuracy와 loss,  validation과 test dataset의 accuracy,  precision, recall, f1 score
# 작성자 김건욱

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#라이브러리 설치

get_ipython().system('pip install split-folders')


# In[ ]:


#필요한 라이브러리 불러오기

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
import random
import os
import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

#코드분석 경고안함

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


# In[ ]:


#데이터 경로 설정

data_dir = "../input/pistachio-image-dataset/Pistachio_Image_Dataset/Pistachio_Image_Dataset/"


# In[ ]:


os.listdir(data_dir)


# In[ ]:


#훈련용 70%, 검증용 20%, 테스트용 10% 데이터 분리

splitfolders.ratio(data_dir, output="output", seed=101, ratio=(.7, .2, .1))


# In[ ]:


#훈련, 검증, 테스트 경로 설정

train_path = './output/train'
val_path = './output/val'
test_path = './output/test'


# In[ ]:


#이미지 사이즈 512, 배치사이즈 64 설정

img_size = 512
batch = 64


# In[ ]:


#라벨의 종류를 도출

labels = []
for i in os.listdir(train_path):
    labels+=[i]


# In[ ]:


#2가지 'Kirmizi_Pistachio', 'Siirt_Pistachio'를 분류하는 라벨로 구성됨

labels


# In[ ]:


#이미지 살펴보기, 함수만들기

def load_random_imgs_from_folder(folder,label):
  plt.figure(figsize=(15,15))
  for i in range(3):
    file = random.choice(os.listdir(folder))
    image_path = os.path.join(folder, file)
    img=mpimg.imread(image_path)
    ax=plt.subplot(1,3,i+1)
    ax.title.set_text(label)
    plt.xlabel(f'Name: {file}')
    plt.imshow(img)


# In[ ]:


# 함수실행

for label in labels:
    load_random_imgs_from_folder(f"{data_dir}/{label}",label)


# In[ ]:


# 이미지 증식

train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
                                   rotation_range = 0.5,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.1,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest'
                                  )

test_val_datagen = ImageDataGenerator(rescale = 1.0/255.0)


# In[ ]:


#이미지 증식 옵션, 디렉토리 설정

train_generator = train_datagen.flow_from_directory(directory = train_path,
                                                    batch_size = batch,
                                                    class_mode = "categorical",
                                                    target_size = (img_size,img_size)
                                                    )

val_generator = test_val_datagen.flow_from_directory(directory = val_path,
                                                    batch_size = batch,
                                                    class_mode = "categorical",
                                                    target_size = (img_size,img_size)
                                                    )

test_generator = test_val_datagen.flow_from_directory(directory = test_path,
                                                    batch_size = batch,
                                                    class_mode = "categorical",
                                                    target_size = (img_size,img_size)
                                                    )


# In[ ]:


#데이터 셋팅

data_train = image_dataset_from_directory(
    train_path,
    label_mode='categorical',
    seed=0,
    color_mode="rgb",
    image_size=(img_size,img_size),
    batch_size=64,
)

data_val = image_dataset_from_directory(
    val_path,
    label_mode='categorical',
    seed=0,
    color_mode="rgb",
    image_size=(img_size,img_size),
    batch_size=64,
)

data_test = image_dataset_from_directory(
    test_path,
    label_mode='categorical',
    seed=0,
    color_mode="rgb",
    image_size=(img_size,img_size),
    batch_size=64,
)


# In[ ]:


data_train


# In[ ]:


#전이학습, VGG16 활용

base_model = VGG16(weights='imagenet', include_top=False,
                            input_shape=(img_size, img_size,3))

# 기본 레이어
base_model.trainable = False

# 뒷단에 추가 레이처 작업
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(4096,activation="relu")(x)
x = Dense(4096,activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(2096,activation="relu")(x)
predictions = Dense(2, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers:
    if layer.trainable==True:
        print(layer)


# In[ ]:


# 모델 요약 확인
model.summary()


# In[ ]:


#콜백 옵션 설정, LOSS값을 보고, 5번 이상 개선되지 않으면 stop, 모델 체크포인트도 필수!

callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1),
                ModelCheckpoint('model.hdf5',save_best_only=True)]


# In[ ]:


#학습 파라메터 설정

opt = Adam(learning_rate=0.01)
model.compile(
  loss='categorical_crossentropy',
  optimizer=opt,
  metrics=['accuracy'])


# In[ ]:


#학습은 5번만 하기로 함

history=model.fit(data_train,
                  epochs=5,
                  validation_data=data_val,
                  validation_steps=int(0.1 * len(data_val)),
                  verbose=1,
                  callbacks=callbacks)


# In[ ]:


# 과제 요구사항(리마인드)

# - 사용한 모델과 학습 하이퍼파라미터 - model.summary() 참조(완료)
# - 커스텀 모델일 경우 model summary 표시 - model.summary() 참조(완료)
# - 모델 성능: 세 데이터셋의 accuracy와 loss,  
# - validation과 test dataset의 accuracy,  precision, recall, f1 score


# In[ ]:


#1. 훈련용 데이터의 정확도와 loss는 다음과 같음

history.history


# In[ ]:


#val의 loss와 정확도는 다음과 같음

result_val = model.evaluate(data_val, batch_size=64)
print("val loss, val acc:", result_val)


# In[ ]:


#test의 loss와 정확도는 다음과 같음

result_test = model.evaluate(data_test, batch_size=64)
print("test loss, test acc:", result_test)


# In[ ]:


# 본 과제에서는 validation과 test dataset의 accuracy,  precision, recall, f1 score 수치가 필요함
# keras에서는 acc, loss 평가지표만 제공하기 때문에 별도의 함수식을 만들어야함
# 아래와 같이 keras backend 라이브러리를 불러와서 별도의 metrics을 만듬

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:


#학습 파라메터 설정(다시 설정)

opt = Adam(learning_rate=0.01)
model.compile(
  loss='categorical_crossentropy',
  optimizer=opt,
  metrics=['acc',f1_m,precision_m, recall_m]) # 이부분이 앞서 만든 함수가 추가됨


# In[ ]:


#학습은 5번만 하기로 함(다시 학습)

history=model.fit(data_train,
                  epochs=5,
                  validation_data=data_val,
                  validation_steps=int(0.1 * len(data_val)),
                  verbose=1,
                  callbacks=callbacks)


# In[ ]:


# 훈련용 데이터 평가(f1 score, precison, reacall 추가)

loss, accuracy, f1_score, precision, recall = model.evaluate(data_train, verbose=0)
print("train loss, train acc, train f1, train pre, train recall:", loss, accuracy, f1_score, precision, recall)


# In[ ]:


# 검증용 데이터 평가(f1 score, precison, reacall 추가)

loss, accuracy, f1_score, precision, recall = model.evaluate(data_val, verbose=0)
print("val loss, val acc, val f1, val pre, val recall:", loss, accuracy, f1_score, precision, recall)


# In[ ]:


# 테스트용 데이터 평가(f1 score, precison, reacall 추가)

loss, accuracy, f1_score, precision, recall = model.evaluate(data_test, verbose=0)
print("test loss, test acc, test f1, test pre, test recall:", loss, accuracy, f1_score, precision, recall)


# In[ ]:


# 따라서 훈련용, 검증용, 테스트 데이터의 loss, accuracy, f1_score, precision, recall 모두 평가지표가 산출됨
# 작성자 김건욱

