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

import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import sklearn
from sklearn import metrics
import seaborn as sns


# In[ ]:


import zipfile

#unzip dataset will be found in "../working"
def unzip_dataset(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
        
unzip_dataset(path_to_zip_file = "../input/dogs-vs-cats-redux-kernels-edition/train.zip", directory_to_extract_to = "./")
unzip_dataset(path_to_zip_file = "../input/dogs-vs-cats-redux-kernels-edition/test.zip", directory_to_extract_to = "./")

print(os.listdir("../working"))


# In[ ]:


TRAIN_DIR = '../working/train'
TEST_DIR = '../working/test'
IMG_SIZE = 50
LR = 1e-3

def label_img(img):
    word_label = img.split('.')
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label[0] == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label[0] == 'dog': return [0,1]
    
def create_train_data():
    training_data = []
    training_label = []
    i = 0
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))[:,:, ::-1]
        training_data.append(np.array(img))
        training_label.append(np.array(label))
        # test code
#         i+=1
#         if i == 10:
#             break
        # ---
    return np.array(training_data), np.array(training_label)

def process_test_data():
    testing_data = []
    testing_ids = []
    i = 0
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        test_id = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append(np.array(img))
        testing_ids.append(np.array(test_id))
        # test code
#         if i == 10:
#             break
#         i += 1
        # ---
    return np.array(testing_data), np.array(testing_ids)


# In[ ]:


data, labels = create_train_data()
data = data / 255 # normalization


# In[ ]:


test_data, test_ids = process_test_data()
test_data = test_data / 255 # normalization


# In[ ]:


plt.imshow(data[0])


# # 2. Data Pre-processing

# In[ ]:


X_train, X_test, X_valid = tf.split(  
            data,
            num_or_size_splits=[int(0.7 * data.shape[0]), 0, int(0.3 * data.shape[0])],
            axis=0
        )
y_train, y_test, y_valid = tf.split(
            labels,
            [int(0.7 * data.shape[0]), 0, int(0.3 * data.shape[0])],
            axis=0
        )


# # 3. Model Building and Training

# In[ ]:


from tensorflow.keras import layers, models 
(_, data_length, data_width, _) = data.shape
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (data_length, data_width, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(2, activation = 'softmax'))


# In[ ]:


model.summary()


# In[ ]:


METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]


# In[ ]:


model.compile(optimizer = 'adam',
              loss = tf.keras.losses.binary_crossentropy,
              metrics = METRICS)


# In[ ]:


history = model.fit(X_train, y_train, epochs = 23, 
                    batch_size = 1000, shuffle = True,
                    validation_data = (X_valid, y_valid)
                   )


# # Model Evaluation

# In[ ]:


print(model.predict(tf.expand_dims(X_train[0], 0)))
print(model.predict_classes(tf.expand_dims(X_train[0], 0)))
# print(history.history)


# In[ ]:


model.evaluate(X_valid, y_valid)


# ## 4.1 Loss, AUC, Precision, and Recall Curve

# In[ ]:


train_predictions = model.predict_classes(X_train)
valid_predictions = model.predict_classes(X_valid)


# In[ ]:


def plot_metrics(history):
    metrics =  ['loss', 'auc', 'precision', 'recall']
    plt.figure(figsize=(15, 15))
    for n, metric in enumerate(metrics):
        name = metric
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color= 'b', label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color= 'r', linestyle="--", label='Val')
        plt.title(name)
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()
        
plot_metrics(history)


# ## 4.2 ROC Curve

# In[ ]:


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label = name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
#     plt.xlim([-0.5,20])
#     plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
plt.figure(figsize=(7, 7))
plot_roc("Train Baseline", y_train[:, 0], train_predictions, color ='b')
plot_roc("Test Baseline", y_valid[:, 0], valid_predictions, color = 'r', linestyle = '--')
plt.legend(loc = 'lower right')


# ## 4.3 Confusion Matrix

# In[ ]:


sns.set()
plt.figure(figsize = (7, 7))
def plot_cm(labels, predictions, p=0.5):
    cm = metrics.confusion_matrix(labels, predictions > p)
#     plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap = 'YlGnBu')
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
plot_cm(y_train[:, 0], train_predictions)


# # 4. Model Testing

# In[ ]:


# model.save('saved_model')
results = model.predict(test_data)
results = results[:, 1] / np.sum(results, axis = 1)
submission = pd.concat([pd.Series(test_ids), pd.Series(results)], axis = 1)


# In[ ]:


submission = submission.rename(columns = {0: 'id', 1: 'label'})
submission.head()


# In[ ]:


# submission.to_csv('submission.csv', index = False)

