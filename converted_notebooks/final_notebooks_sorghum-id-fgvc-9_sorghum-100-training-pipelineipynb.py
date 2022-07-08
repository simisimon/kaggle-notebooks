#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split


# In[ ]:


get_ipython().system('ls')


# In[ ]:


print(os.getcwd())


# In[ ]:


test_images = glob.glob(r'../input/sorghum-id-fgvc-9/test/*')
train_images = glob.glob(r'../input/sorghum-id-fgvc-9/train_images/*')

print(f'Total Training Images : {len(train_images)}')
print(f'Total Test Images : {len(test_images)}')


# In[ ]:


train_mapping_csv = pd.read_csv(r'../input/sorghum-id-fgvc-9/train_cultivar_mapping.csv')
sample_submission_csv = pd.read_csv(r'../input/sorghum-id-fgvc-9/sample_submission.csv')


# In[ ]:


train_mapping_csv.head()


# In[ ]:


image_names = train_mapping_csv['image'].to_list()
train_labels = train_mapping_csv['cultivar'].to_list()


# In[ ]:


label_names = np.unique(train_labels).tolist()[:-1]


# In[ ]:


class_index = {k : v for k,v in zip(label_names,range(len(label_names)))}
index_class = {v : k for k,v in zip(label_names,range(len(label_names)))}


# In[ ]:


train_filepath = []
train_filepath_labels = []
train_filepath_labels_index = []
filenames = []

for i in train_images:
    im_name = os.path.basename(i)
    im_label = train_labels[image_names.index(im_name)]
    train_filepath.append(i)
    train_filepath_labels.append(im_label)
    train_filepath_labels_index.append(str(class_index.get(im_label)))
    filenames.append(im_name)

filepath_df = pd.DataFrame.from_dict({'Filepaths':train_filepath, 'Labels' : train_filepath_labels, 'Filename' : filenames,'Label_Index': train_filepath_labels_index})


# In[ ]:


filepath_df.head()


# In[ ]:


def visualize_train_dataset(df,select_label=None):

    if select_label:
        print(f'Selected Label : {select_label}')
        df = df[df['Labels'] == select_label]
        
    r6_df = df.sample(16)
    
    filepaths = r6_df['Filepaths'].to_list()
    labels = r6_df['Labels'].to_list()
    
    plt.figure(figsize=(16,16))
    for indx,i in enumerate(filepaths):
        lb = labels[indx]
        im = cv2.imread(i)
        plt.subplot(4,4,indx+1)
        plt.title(lb)
        plt.imshow(im[:,:,::-1])
        plt.axis('off')
    plt.show()


# In[ ]:


visualize_train_dataset(filepath_df,select_label = label_names[52])


# In[ ]:


visualize_train_dataset(filepath_df,select_label = label_names[28])


# In[ ]:


from tensorflow.keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import tensorflow
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input


# In[ ]:


train_df, valid_df = train_test_split(filepath_df, test_size=0.2, random_state=42,stratify=filepath_df['Label_Index'], shuffle=True)
train_df.to_csv('TrainDF.csv')
valid_df.to_csv('ValidDF.csv')


# In[ ]:


train_datagen = ImageDataGenerator(
                             horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=30,
                             brightness_range=(0.2, 0.8),
                             zoom_range=[0.5,1.0],
                             preprocessing_function = preprocess_input
                            )

val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

img_height, img_width = 224,224
batch_size = 256

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df, directory='../input/sorghum-id-fgvc-9/train_images',
                                              x_col="Filename", 
                                              y_col="Label_Index", 
                                              batch_size=batch_size, 
                                              target_size=(img_height, img_width),
                                              shuffle=True)

valid_generator = val_datagen.flow_from_dataframe(dataframe=valid_df, directory='../input/sorghum-id-fgvc-9/train_images',
                                              x_col="Filename", 
                                              y_col="Label_Index", 
                                              batch_size=batch_size,  
                                              target_size=(img_height, img_width),
                                              shuffle=False)

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size


# In[ ]:


# https://colab.research.google.com/github/mschuessler/two4two/blob/trainKerasExample/examples/two4two_leNet.ipynb#scrollTo=SNaYUXbBFXJG


# In[ ]:


def create_model():
    tensorflow.keras.backend.clear_session()
    base_model = tensorflow.keras.applications.Xception(weights='imagenet',  
                                                        input_shape=(img_height, img_width, 3),
                                                        include_top=False)
    base_model.trainable = False
    inputs = tensorflow.keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)
    x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
    x = tensorflow.keras.layers.Dense(256,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tensorflow.keras.layers.Dense(128,activation='relu')(x)
    outputs = tensorflow.keras.layers.Dense(100)(x)
    model = tensorflow.keras.Model(inputs, outputs)
    return model


# In[ ]:


#model_x = create_model()


# In[ ]:


#model_x.summary()


# In[ ]:


model_x = tf.keras.models.load_model(r'../input/model-1/base_model_1/09-0.191683-0.183594.hdf5')


# In[ ]:


model_x.compile(optimizer=tensorflow.keras.optimizers.Adam(),
              loss=tensorflow.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_filepath = "base_model_2/{epoch:02d}-{accuracy:02f}-{val_accuracy:02f}.hdf5"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model_x.fit(train_generator,
            validation_data=valid_generator,
            validation_steps=STEP_SIZE_VALID,
            steps_per_epoch=STEP_SIZE_TRAIN,
            epochs=10,
            callbacks = [model_checkpoint_callback])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#model_test = tf.keras.models.load_model(r'../input/model-1/base_model_1/09-0.191683-0.183594.hdf5')


# In[ ]:


# import cv2
# from tqdm.autonotebook import tqdm
# def generate_predictions():
#     layer = tf.keras.layers.Softmax()
#     test_image_folder = r'../input/sorghum-id-fgvc-9/test'
#     sample_submission_csv = pd.read_csv(r'../input/sorghum-id-fgvc-9/sample_submission.csv')
#     test_filenames = sample_submission_csv['filename'].to_list()
#     _filenames = []
#     _preds = []
#     for f in tqdm(test_filenames,total=len(test_filenames)):
        
#         f = test_image_folder + '/' + f
#         im = cv2.imread(f)
#         im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#         im = cv2.resize(im, (224, 224))
#         im = preprocess_input(im)
#         #print(im.shape)
#         #im = im/255.0
#         im = np.expand_dims(im,0)
#         y = model_test.predict(im)
#         y = layer(y).numpy()
#         y_pred = np.argmax(y)
#         y_pred_name = index_class.get(y_pred)
#         _filenames.append(os.path.basename(f))
#         _preds.append(y_pred_name)
    
#     df = pd.DataFrame.from_dict({'filename':_filenames,'cultivar': _preds})
    
#     return df


# In[ ]:


#test_df = generate_predictions()


# In[ ]:


#test_df.head()


# In[ ]:


#test_df.to_csv('submission_1.csv',index=False)


# In[ ]:


# sample_submission_csv = pd.read_csv(r'../input/sorghum-id-fgvc-9/sample_submission.csv')
# sample_submission_csv.head()


# In[ ]:




