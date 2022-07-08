#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
import os


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


train = ImageDataGenerator(rescale = 1./255)
validation = ImageDataGenerator(rescale=1/255)


# In[ ]:


train_dataset = train.flow_from_directory('/content/gdrive/MyDrive/Money/Train',target_size=(150,150),batch_size=3,class_mode='categorical')
validation_dataset = train.flow_from_directory('/content/gdrive/MyDrive/Money/Validation',target_size=(150,150),batch_size=3,class_mode='categorical')


# In[ ]:


from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform',padding ='same',input_shape=(150,150,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform',padding ='same',input_shape=(150,150,3)))
model.add(MaxPooling2D((2,2)))



model.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform',padding ='same'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform',padding ='same'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform',padding ='same'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform',padding ='same'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(64,activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(128, activation = 'relu', kernel_initializer= 'he_uniform'))
model.add(Dropout(0.2))
model.add(Dense(11,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_dataset,epochs=50,batch_size=64
                    ,validation_data=validation_dataset,verbose=1)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

path = '/content/gdrive/MyDrive/Money/Test'
for i in range (11):
    img = load_img(path+'//'+str(i+1)+'.jpg',target_size=(150,150))
    plt.imshow(img)
    plt.show()

    img = img_to_array(img)
    img=np.reshape(img,(1,150,150,3))
    img = img.astype('float32')
    img = img/255
    predict =  np.argmax(model.predict(img))
    if predict==0:
      print("1 000")
    elif predict==1: 
      print("10 000")
    elif predict==2: 
      print("100 000")
    elif predict==3: 
      print("200 ")
    elif predict==4: 
      print("2 000 ")
    elif predict==5: 
      print("20 000 ")
    elif predict==6: 
      print("200 000 ")
    elif predict==7: 
      print("500 ")
    elif predict==8: 
      print("5 000 ")
    elif predict==9: 
      print("50 000 ")
    elif predict==10: 
      print("500 000 ")

img.shape


# In[ ]:


train_dataset.class_indices


# In[ ]:


import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,6)
plt.show()


# In[ ]:




