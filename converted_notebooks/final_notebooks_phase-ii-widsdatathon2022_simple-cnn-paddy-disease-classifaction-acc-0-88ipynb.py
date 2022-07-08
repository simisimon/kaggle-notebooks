#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds


# In[ ]:


batch_size = 32
img_height = 180
img_width = 180


# In[ ]:


import pathlib
data_dir = pathlib.Path('/kaggle/input/paddy-disease-classification/train_images/')


# In[ ]:


image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# In[ ]:


image_sample = list(data_dir.glob('hispa/*'))
PIL.Image.open(str(image_sample[0]))


# In[ ]:


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[ ]:


val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[ ]:


class_names = train_ds.class_names
print(class_names)


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# In[ ]:


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.Normalization(),
        tf.keras.layers.Resizing(img_height, img_width),
        tf.keras.layers.Rescaling(1./255),  
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(factor=0.02),
        tf.keras.layers.RandomZoom(
            height_factor=0.4, width_factor=0.4
        ),
    ],
    name="data_augmentation",
)


# In[ ]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[ ]:


early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)


# In[ ]:


num_classes = len(class_names)

model = tf.keras.Sequential([
  data_augmentation,
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])


# In[ ]:


model.build((None,img_height,img_width,3))
model.summary()


# In[ ]:


model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


# In[ ]:


model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=100,
  callbacks=[early_stopping_cb],
)


# In[ ]:


test_dir = pathlib.Path('/kaggle/input/paddy-disease-classification/test_images/')


# In[ ]:


test_dir


# In[ ]:


test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode=None,
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size)


# In[ ]:


y_pred =  model.predict(test_ds)


# In[ ]:


y_pred_classes = y_pred.argmax(axis=1)


# In[ ]:


y_pred_classes


# In[ ]:


submission_classes = [class_names[x] for x in y_pred_classes]


# In[ ]:


import pandas as pd


# In[ ]:


submit = pd.read_csv('/kaggle/input/paddy-disease-classification/sample_submission.csv')


# In[ ]:


submit['label'] = submission_classes


# In[ ]:


submit.to_csv('submission.csv', index=False)


# In[ ]:




