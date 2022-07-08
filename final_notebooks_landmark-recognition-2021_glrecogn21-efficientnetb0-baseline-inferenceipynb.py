#!/usr/bin/env python
# coding: utf-8

# ## About
# 
# In this notebook, I'll create a submission with the models of [GLRet21: EfficientNetB0 Baseline Training](https://www.kaggle.com/hidehisaarai1213/glret21-efficientnetb0-baseline-training).
# 
# This notebook is based on [GLRet21: EfficientNetB0 Baseline Inference](https://www.kaggle.com/hidehisaarai1213/glret21-efficientnetb0-baseline-inference) and
# 
# [Recognition Kernel](https://www.kaggle.com/camaskew/host-baseline-example?scriptVersionId=40191321)

# # If you copy the notbook please upvote it

# In[ ]:


get_ipython().system('pip install ../input/kerasapplications/ > /dev/null')
get_ipython().system('pip install ../input/efficientnet-keras-source-code/ > /dev/null')


# In[ ]:


import gc
import os
import math
import random
import re
import warnings
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple
import csv
import operator

import efficientnet.tfkeras as efn
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import spatial
from sklearn.preprocessing import normalize
from tqdm import tqdm


# In[ ]:


tf.__version__


# In[ ]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# ## Settings

# In[ ]:


DATADIR = Path("../input/landmark-recognition-2021")
TEST_IMAGE_DIR = DATADIR / "test"
TRAIN_IMAGE_DIR = DATADIR / "train"
TRAIN_LABELMAP_PATH = '../input/landmark-recognition-2021/train.csv'

NUM_TO_RERANK = 6
#TOP_K = 5  

N_CLASSES = 81313
THRESHOLD = 0.5

TEST = False


# In[ ]:


df_train = pd.read_csv('../input/landmark-recognition-2021/train.csv')

df_test = pd.read_csv('../input/landmark-recognition-2021/sample_submission.csv')


# ## Utilities

# In[ ]:


import time

from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    print(f"[{name}]")
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


# In[ ]:


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(1213)


# In[ ]:


def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy


# In[ ]:


strategy = auto_select_accelerator()
REPLICAS = strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE


# ## Model

# In[ ]:


class GeM(tf.keras.layers.Layer):
    def __init__(self, pool_size, init_norm=3.0, normalize=False, **kwargs):
        self.pool_size = pool_size
        self.init_norm = init_norm
        self.normalize = normalize

        super(GeM, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pool_size': self.pool_size,
            'init_norm': self.init_norm,
            'normalize': self.normalize,
        })
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.p = self.add_weight(name='norms', shape=(feature_size,),
                                 initializer=tf.keras.initializers.constant(self.init_norm),
                                 trainable=True)
        super(GeM, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        x = tf.math.maximum(x, 1e-6)
        x = tf.pow(x, self.p)

        x = tf.nn.avg_pool(x, self.pool_size, self.pool_size, 'VALID')
        x = tf.pow(x, 1.0 / self.p)

        if self.normalize:
            x = tf.nn.l2_normalize(x, 1)
        return x

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[-1]])


# In[ ]:


class ArcMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    '''
    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# In[ ]:


def build_model(size=256, efficientnet_size=0, weights="imagenet", count=0):
    inp = tf.keras.layers.Input(shape=(size, size, 3), name="inp1")
    label = tf.keras.layers.Input(shape=(), name="inp2")
    x = getattr(efn, f"EfficientNetB{efficientnet_size}")(
        weights=weights, include_top=False, input_shape=(size, size, 3))(inp)
    x = GeM(8)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, name="dense_before_arcface", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = ArcMarginProduct(
        n_classes=N_CLASSES,
        s=30,
        m=0.5,
        name="head/arc_margin",
        dtype="float32"
    )([x, label])
    output = tf.keras.layers.Softmax(dtype="float32")(x)
    model = tf.keras.Model(inputs=[inp, label], outputs=[output])
    opt = tf.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=opt,
        loss=[tf.keras.losses.SparseCategoricalCrossentropy()],
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model


# In[ ]:


def create_model_for_inference(weights_path: str):
    with strategy.scope():
        base_model = build_model(
            size=256,
            efficientnet_size=0,
            weights=None,
            count=0)
        base_model.load_weights(weights_path)
        model = tf.keras.Model(inputs=base_model.get_layer("inp1").input,
                               outputs=base_model.get_layer("dense_before_arcface").output)
        return model


# ## Feature Extraction

# In[ ]:


def to_hex(image_id) -> str:
    return '{0:0{1}x}'.format(image_id, 16)


def get_image_path(subset, image_id):
    name = to_hex(image_id)
    return os.path.join(DATASET_DIR, subset, name[0], name[1], name[2], '{}.jpg'.format(name))


def load_image_tensor(image_path):
    tensor = tf.convert_to_tensor(np.array(Image.open(image_path).convert("RGB")))
    tensor = tf.image.resize(tensor, size=(256, 256))
    tensor = tf.expand_dims(tensor, axis=0)
    return tf.cast(tensor, tf.float32) / 255.0


def create_batch(files):
    images = []
    for f in files:
        images.append(load_image_tensor(f))
    return tf.concat(images, axis=0)


# In[ ]:


def extract_global_features(image_root_dir, n_models=4,DEBUG='TEST'):
    image_paths = []
    for root, dirs, files in os.walk(image_root_dir):
        for file in files:
            if file.endswith('.jpg'):
                 image_paths.append(os.path.join(root, file))
                    
    if DEBUG == 'TRAIN':
        
        #image_paths = random.shuffle(image_paths)
        
        if TEST:
            
            image_paths = image_paths[:100000]
        
        else:
        
            image_paths = image_paths
    
    elif DEBUG == 'TEST':
        
        #image_paths = random.shuffle(image_paths)
        
        if TEST:
            
            image_paths = image_paths#[:1000]
        else:
        
            image_paths = image_paths
        
    
    
    
    num_embeddings = len(image_paths)

    ids = num_embeddings * [None]
    ids = []
    for path in image_paths:
        ids.append(path.split('/')[-1][:-4])
    
    embeddings = np.zeros((num_embeddings, 512))
    image_paths = np.array(image_paths)
    chunk_size = 512
    
    n_chunks = len(image_paths) // chunk_size
    if len(image_paths) % chunk_size != 0:
        n_chunks += 1

    for n in range(n_models):
        print(f"Getting Embedding for fold{n} model.")
        model = create_model_for_inference(f"../input/glret21-efficientnetb0-baseline-training/fold{n}.h5")
        for i in tqdm(range(n_chunks)):
            files = image_paths[i * chunk_size:(i + 1) * chunk_size]
            batch = create_batch(files)
            embedding_tensor = model.predict(batch)
            embeddings[i * chunk_size:(i + 1) * chunk_size] += embedding_tensor / n_models
        del model
        gc.collect()
        tf.keras.backend.clear_session()

    embeddings = normalize(embeddings, axis=1)

    return ids, embeddings

def load_labelmap():
    
    
    with open(TRAIN_LABELMAP_PATH, mode='r') as csv_file:
        
        csv_reader = csv.DictReader(csv_file)
        labelmap = {row['id']: row['landmark_id'] for row in csv_reader}

    return labelmap

def get_aggregate_score(dict_map):
    
    aggregate_scores = {}

    for ids, label, score in dict_map:


        if label not in aggregate_scores:

            aggregate_scores[label] = score

            #aggregate_scores[label] = label_map[ids]

        else:

            aggregate_scores[label] += score
            
    return aggregate_scores
    
    
def fill_prediction(ID):
    
    if ID in prediction_dict:
        
        if prediction_dict[ID][1] <= THRESHOLD:
            
            return ''
        
        else:
        
            return str(prediction_dict[ID][0]) + ' ' + str(prediction_dict[ID][1])
    
    else: 
        return ''
    


# ## Main

# In[ ]:


def get_predictions():
    with timer("Getting Test Embeddings"):
        test_ids, test_embeddings = extract_global_features(str(TEST_IMAGE_DIR))

    with timer("Getting Train Embeddings"):
        train_ids, train_embeddings = extract_global_features(str(TRAIN_IMAGE_DIR),DEBUG='TRAIN')

    PredictionString_list = []
    labelmap = load_labelmap()
    train_ids_labels_and_scores = {}
    test_ids_labels_and_scores = {}
    with timer("Matching..."):
        for test_index in range(test_embeddings.shape[0]):
            distances = spatial.distance.cdist(test_embeddings[np.newaxis, test_index, :], train_embeddings, 'cosine')[0]
            partition = np.argpartition(distances, NUM_TO_RERANK)[:NUM_TO_RERANK]
            nearest = sorted([(train_ids[p], distances[p]) for p in partition], key=lambda x: x[1])
            df = pd.DataFrame(columns=['ids', 'distance', 'label'])
            with timer(test_index):
                
                train_ids_labels_and_scores[test_ids[test_index]] = [
                    (train_id, labelmap[train_id],1 - cosine_distance)
                                    for train_id, cosine_distance in nearest
                                ]
    
        for ids in train_ids_labels_and_scores:
            
            a = get_aggregate_score(train_ids_labels_and_scores[ids])
            
            test_ids_labels_and_scores[ids] =  max(a.items(), key=operator.itemgetter(1))
            #test_ids_labels_and_scores[ids] = a
            
            

    return test_ids_labels_and_scores #test_ids, PredictionString_list




# In[ ]:


#prediction_dict['777f9efff0fc6b81']


# In[ ]:


#prediction_dict = get_predictions()


# In[ ]:





# In[ ]:


if TEST:
    num_pict = 1
else:
    num_pict = 10345
    


# In[ ]:


if len(df_test) == num_pict:
    df_test[['id', 'landmarks']].to_csv('submission.csv', index = False)

else:
    prediction_dict = get_predictions()
    df_test['landmarks'] = df_test['id'].apply(fill_prediction)
    df_test[['id', 'landmarks']].to_csv('submission.csv', index = False)

