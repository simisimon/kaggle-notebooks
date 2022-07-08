#!/usr/bin/env python
# coding: utf-8

# # Import labraries

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.layers import BatchNormalization
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf

import cv2
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')


# # Load Dataset

# In[ ]:


labels = ['Negative', 'Positive'] #negative:without crack, positive: with crack
img_size = 120
def read_images(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label) #os.path.join(): 2개의 문자열을 결합 -> 1개의 경로  
        print(path)
        class_num = labels.index(label)
        print(class_num)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) #이미지 파일을 Grayscale로 읽음
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) #cv2.resize: x축과 y축 방향으로의 스케일 비율 지정
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)
    

Dataset = read_images('/kaggle/input/surface-crack-detection')


# # Visualizing the Dataset

# In[ ]:


Im = []
for i in Dataset:
    if(i[1] == 0):
        Im.append("Negative")
    elif(i[1] == 1):
        Im.append("Positive")

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
sns.set_style('darkgrid')
axl = sns.countplot(Im)
axl.set_title("Number of Images")


# > negative, positive 각각 20000개 씩 있음

# # Normalization of image data

# In[ ]:


x = []
y = []

for feature, label in Dataset:
    x.append(feature) # x: 이미지 데이터
    y.append(label) # y: label(0,1) 데이터
    
#normalization(정규화)    
x = np.array(x).reshape(-1, img_size, img_size, 1)
x = x / 255

y = np.array(y)


# In[ ]:


plt.subplot(1, 2, 1)
plt.imshow(x[1000].reshape(img_size, img_size), cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(x[30000].reshape(img_size, img_size), cmap='gray')
plt.axis('off')


# # CNN Model
# 
# * **Convolution layer**는 컨볼루션 신경망에서 다차원 프로세스에서 컨볼루션 프로세스를 수행하는데 사용된다. 이 레이어는 feature map이라는 입력으로 정의되는 이미지 매트릭스에서 뉴런의 조정과 속성 학습을 가능하게 한다.
# 
# * **ReLu**는 컨볼루션 프로세스 후에 나타나는 기능맵을 평면화하는 작업을 수행한다. 음수 값을 0으로 변환하여 0과 양의 무한값 사이의 출력을 생성한다.
# 
# * **Pooling**는 합성곱에 의해 얻어진 Feature map으로부터 값을 샘플링해서 정보를 압축하는 과정이다. 맥스풀링 (Max-pooling)은 특정 영역에서 가장 큰 값을 샘플링하는 풀링 방식
# 
# CNN의 훈련 단계에서 정규화. 데이터 증대는 가중치의 정규화 및 배치 정규화를 위한 중요한 요소입니다(Srivastava et al., 2014). 이러한 이유로 Dropout이라는 방법을 사용합니다. 주요 목적은 *과적합*을 방지하는 것입니다.
# 

# In[ ]:


#합성곱 신경망 구성
model = Sequential()
model.add(Conv2D(64,3,padding="same", activation="relu", input_shape = x.shape[1:]))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(128, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

#Dense 층
model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(2, activation="softmax"))

#신경망에 대한 정보를 출력
model.summary()


# <a id = "5"></a><br>
# # **Model Training**
# 
# * epoch 횟수를 늘려 정확도를 높일 수 있음
# * 학습률을 다르게 시도해보는 것도 정확도를 높일 수 있음 
# 
# 낮은 학습률을 선택한 경우에는 학습 속도가 느리다. 하지만 높은 학습률을 선택하면 훈련 속도는 빨라지지만 정확도는 떨어질 수 있다.

# In[ ]:


#모델 컴파일
opt = Adam(lr=1e-5)

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) 

#모델 훈련하기
history = model.fit(x, y, epochs = 15, batch_size = 128, validation_split = 0.25, verbose=1)


# In[ ]:


print(history.history.keys())


# <a id = "6"></a><br>
# # **Graphs**

# In[ ]:


plt.figure(figsize=(12, 12))
plt.style.use('ggplot')
plt.subplot(2,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy of the Model')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(['train accuracy', 'validation accuracy'], loc='lower right', prop={'size': 12})

plt.subplot(2,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss of the Model')
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(['train loss', 'validation loss'], loc='best', prop={'size': 12})


# <a id = "7"></a><br>
# # Classification Report
# 
# * The classification_report function builds a text report showing the main classification metrics. 
# 
# * **Precision** for each class, it is defined as the ratio of true positives to the sum of true and false positives.
# 
# * **Recall** for each class, it is defined as the ratio of true positives to the sum of true positives and false negatives.
# * **F1 scores** are lower than accuracy measures as they embed precision and recall into their computation.

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix

predictions = model.predict_classes(x)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y, predictions, target_names = ['Negative','Positive']))

