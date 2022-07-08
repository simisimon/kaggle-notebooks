#!/usr/bin/env python
# coding: utf-8

# ### Work in progress/incomplete
# 
# #### Feel free to give feedback!

# In[ ]:


import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import math
import pandas as pd
from sklearn import model_selection
import glob
import os
from zipfile import ZipFile
import shutil
import tqdm.notebook as tqdm

import logging
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(num_gpus, "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
#     policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
#     tf.keras.mixed_precision.experimental.set_policy(policy)
#     print('Compute dtype: %s' % policy.compute_dtype)
#     print('Variable dtype: %s' % policy.variable_dtype)

    
if num_gpus == 0:
    strategy = tf.distribute.OneDeviceStrategy(device='CPU')
    print("Setting strategy to OneDeviceStrategy(device='CPU')")
elif num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='GPU')
    print("Setting strategy to OneDeviceStrategy(device='GPU')")
else:
    strategy = tf.distribute.MirroredStrategy()
    print("Setting strategy to MirroredStrategy()")


# In[ ]:


config = {
    'learning_rate': 1e-3,
    'momentum': 0.9,
    'K': 4,
    'margin': 0.3,
    'clip_grad': 10.0,
    'n_epochs': 4,
    'batch_size': 16,
    'input_size': (256, 256, 3),
    'dense_units': 1024,
    'dropout_rate': 0.0,
}


# In[ ]:


def read_df(input_path):
    files_paths = glob.glob(input_path + 'train/*/*/*/*')
    mapping = {}
    for path in files_paths:
        mapping[path.split('/')[-1].split('.')[0]] = path
    df = pd.read_csv(input_path + 'train.csv')
    df['path'] = df['id'].map(mapping)
    
    df = (
        df
        .groupby('landmark_id')['path']
        .agg(lambda x: ','.join(x))
        .reset_index()
    )
    return df

df = read_df('../input/landmark-retrieval-2020/')
df.head(10)


# In[ ]:


def _get_transform_matrix(rotation, shear, hzoom, wzoom, hshift, wshift):

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    # convert degrees to radians
    rotation = math.pi * rotation / 360.
    shear    = math.pi * shear    / 360.

    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')

    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    rot_mat = get_3x3_mat([c1,    s1,   zero ,
                           -s1,   c1,   zero ,
                           zero,  zero, one ])

    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_mat = get_3x3_mat([one,  s2,   zero ,
                             zero, c2,   zero ,
                             zero, zero, one ])

    zoom_mat = get_3x3_mat([one/hzoom, zero,      zero,
                            zero,      one/wzoom, zero,
                            zero,      zero,      one])

    shift_mat = get_3x3_mat([one,  zero, hshift,
                             zero, one,  wshift,
                             zero, zero, one   ])

    return tf.matmul(
        tf.matmul(rot_mat, shear_mat),
        tf.matmul(zoom_mat, shift_mat)
    )

def _spatial_transform(image,
                       rotation=3.0,
                       shear=2.0,
                       hzoom=8.0,
                       wzoom=8.0,
                       hshift=8.0,
                       wshift=8.0):

    ydim = tf.gather(tf.shape(image), 0)
    xdim = tf.gather(tf.shape(image), 1)
    xxdim = xdim % 2
    yxdim = ydim % 2

    # random rotation, shear, zoom and shift
    rotation = rotation * tf.random.normal([1], dtype='float32')
    shear = shear * tf.random.normal([1], dtype='float32')
    hzoom = 1.0 + tf.random.normal([1], dtype='float32') / hzoom
    wzoom = 1.0 + tf.random.normal([1], dtype='float32') / wzoom
    hshift = hshift * tf.random.normal([1], dtype='float32')
    wshift = wshift * tf.random.normal([1], dtype='float32')

    m = _get_transform_matrix(
        rotation, shear, hzoom, wzoom, hshift, wshift)

    # origin pixels
    y = tf.repeat(tf.range(ydim//2, -ydim//2,-1), xdim)
    x = tf.tile(tf.range(-xdim//2, xdim//2), [ydim])
    z = tf.ones([ydim*xdim], dtype='int32')
    idx = tf.stack([y, x, z])

    # destination pixels
    idx2 = tf.matmul(m, tf.cast(idx, dtype='float32'))
    idx2 = tf.cast(idx2, dtype='int32')
    # clip to origin pixels range
    idx2y = tf.clip_by_value(idx2[0,], -ydim//2+yxdim+1, ydim//2)
    idx2x = tf.clip_by_value(idx2[1,], -xdim//2+xxdim+1, xdim//2)
    idx2 = tf.stack([idx2y, idx2x, idx2[2,]])

    # apply destinations pixels to image
    idx3 = tf.stack([ydim//2-idx2[0,], xdim//2-1+idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    image = tf.reshape(d, [ydim, xdim, 3])
    return image

def _pixel_transform(image,
                     saturation_delta=0.3,
                     contrast_delta=0.1,
                     brightness_delta=0.2):
    image = tf.image.random_saturation(
        image, 1-saturation_delta, 1+saturation_delta)
    image = tf.image.random_contrast(
        image, 1-contrast_delta, 1+contrast_delta)
    image = tf.image.random_brightness(
        image, brightness_delta)
    return image

def _random_fliplr(image, p=0.25):
    r = tf.random.uniform(())
    mirror_cond = tf.math.less(r, p)
    image = tf.cond(
        mirror_cond,
        lambda: tf.reverse(image, [1]),
        lambda: image
    )
    return image

def preprocess_input(image, target_size, augment=False):
    
    image = tf.image.resize(
        image, target_size, method='bilinear')

    image = tf.cast(image, tf.uint8)
    if augment:
        image = _spatial_transform(image)
        image = _random_fliplr(image)
        image = _pixel_transform(image)
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image

def create_triplet_dataset(df, training, batch_size, input_size, K):
    
    def sample_input(image_paths, label, K):
        image_paths = tf.strings.split(image_paths, sep=',')
        labels = tf.tile([label], [K,])
        if K-len(image_paths) > 0:
            image_paths = tf.random.shuffle(image_paths)
            for i in tf.range(K-len(image_paths)):
                image_paths = tf.concat(
                    [image_paths, [tf.gather(image_paths, i)]], axis=0)
            return image_paths, labels
        idx = tf.argsort(tf.random.uniform(tf.shape(image_paths)))
        idx = tf.gather(idx, tf.range(K))
        image_paths = tf.gather(image_paths, idx)
        return image_paths, labels

    def read_image(image_path):
        image = tf.io.read_file(image_path)
        return tf.image.decode_jpeg(image, channels=3)

    def reshape(x, y):
        x = tf.reshape(x, (-1, *input_size))
        y = tf.reshape(y, (-1,))
        return x, y
    
    @tf.autograph.experimental.do_not_convert # to silence warning
    def nested(x, y):
        return (tf.data.Dataset.from_tensor_slices((x, y))
                .map(lambda x, y: (read_image(x), y),
                    tf.data.experimental.AUTOTUNE)
                .map(lambda x, y: (preprocess_input(
                        x, input_size[:2], True), y),
                     tf.data.experimental.AUTOTUNE)
                .batch(K))

    image_paths, labels = df.path, df.index
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if training:
        dataset = dataset.shuffle(10_000)
    dataset = dataset.map(
        lambda x, y: sample_input(x, y, K), tf.data.experimental.AUTOTUNE)
    dataset = dataset.flat_map(nested)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda x, y: reshape(x, y), tf.data.experimental.AUTOTUNE)
    return dataset


# In[ ]:


def create_model(input_shape, dense_units=512, dropout_rate=0.0):

    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=input_shape,
        weights=('../input/imagenet-weights/' +
                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    )

    pooling = tf.keras.layers.GlobalAveragePooling2D(name='head/pooling')
    dropout = tf.keras.layers.Dropout(dropout_rate, name='head/dropout')
    dense = tf.keras.layers.Dense(dense_units, name='head/dense', dtype='float32')

    image = tf.keras.layers.Input(input_shape, name='input/image')
    
    x = backbone(image)
    x = pooling(x)
    x = dropout(x)
    x = dense(x)
    return tf.keras.Model(
        inputs=image, outputs=x)


class DistributedModel:

    def __init__(self,
                 input_size,
                 batch_size,
                 finetuned_weights,
                 dense_units,
                 dropout_rate,
                 margin,
                 optimizer,
                 strategy,
                 mixed_precision,
                 clip_grad):

        self.model = create_model(
            input_shape=input_size,
            dense_units=dense_units,
            dropout_rate=dropout_rate)

        self.input_size = input_size

        if finetuned_weights:
            self.model.load_weights(finetuned_weights)

        self.mixed_precision = mixed_precision
        self.optimizer = optimizer
        self.strategy = strategy
        self.clip_grad = clip_grad

        # loss function
        self.loss_object = tfa.losses.TripletSemiHardLoss(
            margin=margin,
            distance_metric='L2',
        )
        # metrics
        self.mean_loss_train = tf.keras.metrics.Mean()

        if self.optimizer and self.mixed_precision:
            self.optimizer = \
                tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer, loss_scale='dynamic')

    def _compute_loss(self, labels, embedding):
        per_example_loss = self.loss_object(labels, embedding)
        return per_example_loss / strategy.num_replicas_in_sync

    def _backprop_loss(self, tape, loss, weights):
        gradients = tape.gradient(loss, weights)
        if self.mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=self.clip_grad)
        self.optimizer.apply_gradients(zip(clipped, weights))

    def _train_step(self, inputs):
        with tf.GradientTape() as tape:
            embedding = self.model(inputs[0], training=True)
            loss = self._compute_loss(inputs[1], embedding)
            self.mean_loss_train.update_state(loss)
            if self.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)
        self._backprop_loss(tape, loss, self.model.trainable_weights)

    @tf.function
    def _distributed_train_step(self, dist_inputs):
        per_replica_loss = self.strategy.run(self._train_step, args=(dist_inputs,))
        return self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    def train(self, train_ds, epochs, save_path):
        for epoch in range(epochs):
            dist_train_ds = self.strategy.experimental_distribute_dataset(train_ds)
            dist_train_ds = tqdm.tqdm(dist_train_ds)
            for i, inputs in enumerate(dist_train_ds):
                loss = self._distributed_train_step(inputs)
                dist_train_ds.set_description(
                    "Loss {:.5f}".format(self.mean_loss_train.result().numpy())
                )
            if save_path:
                self.model.save_weights(save_path)

            self.mean_loss_train.reset_states()


# In[ ]:


train_ds = create_triplet_dataset(
    df=df,
    training=True,
    batch_size=config['batch_size'],
    input_size=config['input_size'],
    K=config['K']
)

with strategy.scope():

    optimizer = tf.keras.optimizers.SGD(
        config['learning_rate'], momentum=config['momentum'])

    dist_model = DistributedModel(
        input_size=config['input_size'],
        batch_size=config['batch_size'],
        finetuned_weights=None,
        dense_units=config['dense_units'],
        dropout_rate=config['dropout_rate'],
        margin=config['margin'],
        optimizer=optimizer,
        strategy=strategy,
        mixed_precision=False,
        clip_grad=config['clip_grad'])

    dist_model.train(
        train_ds=train_ds, 
        epochs=config['n_epochs'], 
        save_path='model.h5')


# In[ ]:


newmodel = tf.keras.Model(
    inputs=dist_model.model.get_layer('input/image').input,
    outputs=dist_model.model.get_layer('head/dense').output)


@tf.function(input_signature=[
        tf.TensorSpec(
            shape=[None, None, 3],
            dtype=tf.uint8,
            name='input_image')
    ])
def serving(input_image):
    input_image = preprocess_input(
        input_image, target_size=config['input_size'][:2])
    outputs = newmodel(input_image[tf.newaxis])
    features = tf.math.l2_normalize(outputs[0])
    return {
        'global_descriptor': tf.identity(features, name='global_descriptor')
    }


tf.saved_model.save(
    obj=newmodel,
    export_dir='model',
    signatures={'serving_default': serving})


filepaths = []
for dirpath, _, filepath in os.walk('model'):
    for fp in filepath:
        filepaths.append(os.path.join(dirpath, fp))

with ZipFile('submission.zip', 'w') as zip:
    for fp in filepaths:
        print(fp, '/'.join(fp.split('/')[1:]))
        zip.write(fp, arcname='/'.join(fp.split('/')[1:]))


# In[ ]:




