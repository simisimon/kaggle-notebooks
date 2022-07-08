#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q efficientnet')
import re
import os
import numpy as np
import pandas as pd
from scipy.signal import get_window
from typing import Optional, Tuple
import warnings
import random
import math
import tensorflow as tf
import efficientnet.tfkeras as efn
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
import tensorflow_addons as tfa
from kaggle_datasets import KaggleDatasets


# In[ ]:


# Function to get hardware strategy
def get_hardware_strategy():
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        policy = mixed_precision.Policy('mixed_bfloat16')
        mixed_precision.set_global_policy(policy)
        tf.config.optimizer.set_jit(True)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)
    return tpu, strategy

tpu, strategy = get_hardware_strategy()


# In[ ]:


# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Data access (Train tf records)
GCS_PATH1 = KaggleDatasets().get_gcs_path('g2net-tf-records-tr-bp-filter-1')
GCS_PATH2 = KaggleDatasets().get_gcs_path('g2net-tf-records-tr-bp-filter-2')
GCS_PATH3 = KaggleDatasets().get_gcs_path('g2net-tf-records-tr-bp-filter-3')
# Data access (Test tf records)
GCS_PATH4 = KaggleDatasets().get_gcs_path('g2net-tf-records-ts-bp-filter-1')
GCS_PATH5 = KaggleDatasets().get_gcs_path('g2net-tf-records-ts-bp-filter-2')

# Configuration
EPOCHS = 15
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
IMAGE_SIZE = [512, 512]
# Seed
SEED = 42
# Learning rate
LR = 0.0001
# Verbosity
VERBOSE = 2

# Training filenames directory
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH1 + '/train*.tfrec') + tf.io.gfile.glob(GCS_PATH2 + '/train*.tfrec') + tf.io.gfile.glob(GCS_PATH3 + '/train*.tfrec')
# Testing filenames directory
TESTING_FILENAMES = tf.io.gfile.glob(GCS_PATH4 + '/test*.tfrec') + tf.io.gfile.glob(GCS_PATH5 + '/test*.tfrec')


# In[ ]:


# Function to create cqt kernel
def create_cqt_kernels(
    q: float,
    fs: float,
    fmin: float,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    norm: float = 1,
    window: str = "hann",
    fmax: Optional[float] = None,
    topbin_check: bool = True
) -> Tuple[np.ndarray, int, np.ndarray, float]:
    fft_len = 2 ** _nextpow2(np.ceil(q * fs / fmin))
    
    if (fmax is not None) and (n_bins is None):
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
    elif (fmax is None) and (n_bins is not None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
    else:
        warnings.warn("If nmax is given, n_bins will be ignored", SyntaxWarning)
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        
    if np.max(freqs) > fs / 2 and topbin_check:
        raise ValueError(f"The top bin {np.max(freqs)} Hz has exceeded the Nyquist frequency, \
                           please reduce the `n_bins`")
    
    kernel = np.zeros((int(n_bins), int(fft_len)), dtype=np.complex64)
    
    length = np.ceil(q * fs / freqs)
    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = np.ceil(q * fs / freq)
        
        if l % 2 == 1:
            start = int(np.ceil(fft_len / 2.0 - l / 2.0)) - 1
        else:
            start = int(np.ceil(fft_len / 2.0 - l / 2.0))

        sig = get_window(window, int(l), fftbins=True) * np.exp(
            np.r_[-l // 2:l // 2] * 1j * 2 * np.pi * freq / fs) / l
        
        if norm:
            kernel[k, start:start + int(l)] = sig / np.linalg.norm(sig, norm)
        else:
            kernel[k, start:start + int(l)] = sig
    return kernel, fft_len, length, freqs


def _nextpow2(a: float) -> int:
    return int(np.ceil(np.log2(a)))

# Function to prepare cqt kernel
def prepare_cqt_kernel(
    sr=22050,
    hop_length=512,
    fmin=32.70,
    fmax=None,
    n_bins=84,
    bins_per_octave=12,
    norm=1,
    filter_scale=1,
    window="hann"
):
    q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
    print(q)
    return create_cqt_kernels(q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)

# Function to create cqt image
def create_cqt_image(wave, hop_length=16):
    CQTs = []
    for i in range(3):
        x = wave[i]
        x = tf.expand_dims(tf.expand_dims(x, 0), 2)
        x = tf.pad(x, PADDING, "REFLECT")

        CQT_real = tf.nn.conv1d(x, CQT_KERNELS_REAL, stride=hop_length, padding="VALID")
        CQT_imag = -tf.nn.conv1d(x, CQT_KERNELS_IMAG, stride=hop_length, padding="VALID")
        CQT_real *= tf.math.sqrt(LENGTHS)
        CQT_imag *= tf.math.sqrt(LENGTHS)

        CQT = tf.math.sqrt(tf.pow(CQT_real, 2) + tf.pow(CQT_imag, 2))
        CQTs.append(CQT[0])
    return tf.stack(CQTs, axis=2)

HOP_LENGTH = 6
cqt_kernels, KERNEL_WIDTH, lengths, _ = prepare_cqt_kernel(
    sr=2048,
    hop_length=HOP_LENGTH,
    fmin=20,
    fmax=1024,
    bins_per_octave=9)
LENGTHS = tf.constant(lengths, dtype=tf.float32)
CQT_KERNELS_REAL = tf.constant(np.swapaxes(cqt_kernels.real[:, np.newaxis, :], 0, 2))
CQT_KERNELS_IMAG = tf.constant(np.swapaxes(cqt_kernels.imag[:, np.newaxis, :], 0, 2))
PADDING = tf.constant([[0, 0],
                        [KERNEL_WIDTH // 2, KERNEL_WIDTH // 2],
                        [0, 0]])


# In[ ]:


# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

# Function to prepare image
def prepare_image(wave):
    # Decode raw
    wave = tf.reshape(tf.io.decode_raw(wave, tf.float64), (3, 4096))
    normalized_waves = []
    # Normalize
    for i in range(3):
        normalized_wave = wave[i] / tf.math.reduce_max(wave[i])
        normalized_waves.append(normalized_wave)
    # Stack and cast
    wave = tf.stack(normalized_waves)
    wave = tf.cast(wave, tf.float32)
    # Create image
    image = create_cqt_image(wave, HOP_LENGTH)
    # Resize image
    image = tf.image.resize(image, [*IMAGE_SIZE])
    # Reshape
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

# This function parse our images and also get the target variable
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        'wave': tf.io.FixedLenFeature([], tf.string),
        'wave_id': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = prepare_image(example['wave'])
    image_id = example['wave_id']
    target = tf.cast(example['target'], tf.float32)
    return image, image_id, target

# This function parse our images and also get the target variable
def read_unlabeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        'wave': tf.io.FixedLenFeature([], tf.string),
        'wave_id': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = prepare_image(example['wave'])
    image_id = example['wave_id']
    return image, image_id

# This function loads TF Records and parse them into tensors
def load_dataset(filenames, ordered = False, labeled = True):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False 
        
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls = AUTO) 
    return dataset

# This function is to get our training dataset
def get_training_dataset(filenames, ordered = False, labeled = True):
    dataset = load_dataset(filenames, ordered = ordered, labeled = labeled)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

# This function is to get our validation and test dataset
def get_val_test_dataset(filenames, ordered = True, labeled = True):
    dataset = load_dataset(filenames, ordered = ordered, labeled = labeled)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) 
    return dataset

# Function to count how many photos we have in
def count_data_items(filenames):
    # The number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_TESTING_IMAGES = count_data_items(TESTING_FILENAMES)
print(f'Dataset: {NUM_TRAINING_IMAGES} training images')
print(f'Dataset: {NUM_TESTING_IMAGES} testing images')


# In[ ]:


# Learning rate callback function
def get_lr_callback():
    lr_start   = 0.0001
    lr_max     = 0.000015 * BATCH_SIZE
    lr_min     = 0.0000001
    lr_ramp_ep = 3
    lr_sus_ep  = 0
    lr_decay   = 0.7
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start   
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max    
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min    
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = VERBOSE)
    return lr_callback

# Function to create our EfficientNetB7 model
def get_model():
    with strategy.scope():
        inp = tf.keras.layers.Input(shape = (*IMAGE_SIZE, 3))
        x = efn.EfficientNetB3(include_top = False, weights = 'imagenet')(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        output = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
        model = tf.keras.models.Model(inputs = [inp], outputs = [output])
        opt = tf.keras.optimizers.Adam(learning_rate = LR)
        opt = tfa.optimizers.SWA(opt)
        model.compile(
            optimizer = opt,
            loss = [tf.keras.losses.BinaryCrossentropy()],
            metrics = [tf.keras.metrics.AUC()]
        )
        return model
    
# Function to train a model with 100% of the data
def train_and_evaluate(SEED=42):
    print('\n')
    print('-'*50)
    print(f'Training EFFB7 with 100% of the data with seed {SEED} for {EPOCHS} epochs')
    if tpu:
        tf.tpu.experimental.initialize_tpu_system(tpu)
    train_dataset = get_training_dataset(TRAINING_FILENAMES, ordered = False, labeled = True)
    train_dataset = train_dataset.map(lambda image, image_id, target: (image, target))
    STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // (BATCH_SIZE * 4)
    K.clear_session()
    # Seed everything
    seed_everything(SEED)
    model = get_model()
    history = model.fit(train_dataset,
                        steps_per_epoch = STEPS_PER_EPOCH,
                        epochs = EPOCHS,
                        callbacks = [get_lr_callback()], 
                        verbose = VERBOSE)
        
    print('\n')
    print('-'*50)
    print('Test inference...')
    # Predict the test set 
    dataset = get_val_test_dataset(TESTING_FILENAMES, ordered = True, labeled = False)
    image = dataset.map(lambda image, image_id: image)
    test_predictions = model.predict(image).astype(np.float32).reshape(-1)
    # Get the test set image_id
    image_id = dataset.map(lambda image, image_id: image_id).unbatch()
    image_id = next(iter(image_id.batch(NUM_TESTING_IMAGES))).numpy().astype('U')
    # Create dataframe output
    test_df = pd.DataFrame({'id': image_id, 'target': test_predictions})
    # Save test dataframe to disk
    test_df.to_csv(f'TEST_EfficientNetB7_{IMAGE_SIZE[0]}_{SEED}.csv', index = False)
    


# In[ ]:


train_and_evaluate(SEED=2020)


# In[ ]:


train_and_evaluate(SEED=1991)

