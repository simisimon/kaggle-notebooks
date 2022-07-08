#!/usr/bin/env python
# coding: utf-8

# This notebook trains a model running on TPU.  Summaries are as follows:
# 
# * Model: FPN in [Segmentation Models](https://github.com/qubvel/segmentation_models)
# * Backbone: EfficientNet B1
# * Image Size: 224
# * Learning Rate: maximum 1e-3, cosine decay with warmup
# * Epochs: 30 (2 for warmup and the rests are cosine decay)
# * Batch Size: 128
# * Folds: 5 (trains 1 fold only)
# * Loss: 0.5 * BCE + 0.5 * Dice
# * Data Augmentations: [Albumentations](https://albumentations.ai/) like
# 
# The performance of the trained model is not good, ~0.79 on the LB score.
# 
# # Reference
# 
# Thanks a lot to the authors for sharing the valuable information.
# 
# * [UWMGI: UNet Keras [Train] with EDA](https://www.kaggle.com/code/ammarnassanalhajali/uwmgi-unet-keras-train-with-eda)
# * [UWMGI: Unet [Train] [PyTorch]](https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch)
# * [UWMGI Image Segmentation EDA](https://www.kaggle.com/code/tt195361/uwmgi-image-segmentation-eda)
# * [UWMGI Image Segmentation Make TFRecords](https://www.kaggle.com/code/tt195361/uwmgi-image-segmentation-make-tfrecords)
# 
# # Preparation

# In[ ]:


get_ipython().system('git clone https://github.com/tt195361/TfDataAugmentation.git')

import sys
sys.path.append('TfDataAugmentation')

import TfDataAugmentation as Tfda


# In[ ]:


get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
get_ipython().system('pip install ../input/segmentation-models-keras/Keras_Applications-1.0.8-py3-none-any.whl --quiet')
get_ipython().system('pip install ../input/segmentation-models-keras/image_classifiers-1.0.0-py3-none-any.whl --quiet')
get_ipython().system('pip install ../input/segmentation-models-keras/efficientnet-1.0.0-py3-none-any.whl --quiet')
get_ipython().system('pip install ../input/segmentation-models-keras/segmentation_models-1.0.1-py3-none-any.whl --quiet')

print("Segmentation Models installed.")


# In[ ]:


DEBUG = False


# In[ ]:


import numpy as np
import pandas as pd
import segmentation_models as sm
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from kaggle_datasets import KaggleDatasets
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
import joblib

print(tf.__version__)


# In[ ]:


SEG_MODEL = sm.FPN
BACKBONE = 'efficientnetb1'
IMAGE_SIZE = 224
BATCH_SIZE = 128
INIT_LR = 1e-4
WARMUP_EPO = 2 if not DEBUG else 1
COSINE_EPO = 28 if not DEBUG else 2
N_EPOCHS = WARMUP_EPO + COSINE_EPO
N_FOLDS = 5

VID = 'V24'
FOLD_I_LIST = [0]
FOLD_I_LIST = FOLD_I_LIST[:2] if DEBUG else FOLD_I_LIST

print("N_EPOCHS:   ", N_EPOCHS)
print("FOLD_I_LIST:", FOLD_I_LIST)


# In[ ]:


DATA_SRC = 'uwmgi-image-segmentation-tfrecords'
AUTOTUNE = tf.data.experimental.AUTOTUNE


# # TPU

# In[ ]:


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() 
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # otherwise detect GPUs
    strategy = tf.distribute.MirroredStrategy() # single-GPU or multi-GPU
    
REPLICAS = strategy.num_replicas_in_sync

print(f"Running on {REPLICAS} replicas")


# In[ ]:


GCS_DS_PATH = KaggleDatasets().get_gcs_path(DATA_SRC)

GCS_DS_PATH


# # Dataset

# In[ ]:


def decode_image(image_bytes):
    image_png = tf.image.decode_png(image_bytes)
    image_png = tf.reshape(image_png, [IMAGE_SIZE, IMAGE_SIZE, 1])
    image_png = tf.cast(image_png, dtype=tf.float32) / 255.0
    return image_png

def decode_mask(mask_bytes, height, width):
    mask_png = tf.image.decode_png(mask_bytes)
    # loaded image's shape is [width, height, channel]
    mask_png = tf.reshape(mask_png, [width, height, 3])
    mask_float = tf.cast(mask_png, dtype=tf.float32)
    return mask_float

def resize_image(image):
    resized_image = tf.image.resize(
        image, [IMAGE_SIZE, IMAGE_SIZE],
        method=tf.image.ResizeMethod.BILINEAR)
    return resized_image


# In[ ]:


def read_tfrecord(example):
    TFREC_FORMAT = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'case_no': tf.io.FixedLenFeature([], tf.int64),
        'day_no': tf.io.FixedLenFeature([], tf.int64),
        'slice_no': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'fold': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'space_h': tf.io.FixedLenFeature([], tf.float32),
        'space_w': tf.io.FixedLenFeature([], tf.float32),
    }
    
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    sample_id = example['id']
    height = example['height']
    width = example['width']
    image = decode_image(example['image'])
    mask = decode_mask(example['mask'], height, width)
    fold = example['fold']
    
    resized_mask = resize_image(mask)
    return image, resized_mask, (sample_id, height, width), fold

def make_raw_ds(filenames):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    return dataset


# In[ ]:


tfrec_file_pattern = os.path.join(GCS_DS_PATH, '*.tfrec')
tfrec_file_names = tf.io.gfile.glob(tfrec_file_pattern)
raw_ds = make_raw_ds(tfrec_file_names)

print(raw_ds)


# In[ ]:


train_data_file_path = os.path.join(GCS_DS_PATH, 'train_data.csv')
train_data_df = pd.read_csv(train_data_file_path)
fold_count_dict = \
    train_data_df['fold'] \
        .value_counts() \
        .sort_index() \
        .to_dict()

fold_count_dict


# In[ ]:


def get_train_count(fold_i):
    counts = [ 
        count for fold, count in fold_count_dict.items() \
        if fold != fold_i ]
    return sum(counts)

def get_val_count(fold_i):
    return fold_count_dict[fold_i]


# In[ ]:


def pick_image_mask(image, mask, info, fold):
    return image, mask

def pick_image_mask_info(image, mask, info, fold):
    return image, mask, info

def select_train(ds, fold_i):
    ds = ds.filter(lambda image, mask, info, fold: fold != fold_i)
    return ds
    
def select_val(ds, fold_i):
    ds = ds.filter(lambda image, mask, info, fold: fold == fold_i)
    return ds


# In[ ]:


cut_size = IMAGE_SIZE // 20

transforms = Tfda.Compose([
    Tfda.HorizontalFlip(p=0.5),
    Tfda.ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.05, rotate_limit=10,
        interpolation='bilinear', border_mode='constant', p=0.5),
    Tfda.OneOf([
        Tfda.GridDistortion(
            num_steps=5, distort_limit=0.05,
            interpolation='bilinear', border_mode='constant', p=0.5),        
        Tfda.OpticalDistortion(
            distort_limit=0.5, shift_limit=0.05,
            interpolation='bilinear', border_mode='constant', p=0.5),
        ], p=0.25),
    Tfda.Cutout(
        num_holes=8, max_h_size=cut_size, max_w_size=cut_size, p=0.5),
])

def data_augment(image, mask):
    result = transforms(image=image, mask=mask)
    aug_image = result["image"]
    aug_mask = result["mask"]
    return aug_image, aug_mask


# In[ ]:


def make_datasets(fold_i):
    train_ds = select_train(raw_ds, fold_i) \
        .map(pick_image_mask, num_parallel_calls=AUTOTUNE) \
        .cache() \
        .repeat() \
        .shuffle(1024) \
        .map(data_augment, num_parallel_calls=AUTOTUNE) \
        .batch(BATCH_SIZE) \
        .prefetch(AUTOTUNE)
        
    val_ds = select_val(raw_ds, fold_i) \
        .map(pick_image_mask, num_parallel_calls=AUTOTUNE) \
        .batch(BATCH_SIZE) \
        .cache() \
        .prefetch(AUTOTUNE)
    
    train_steps = get_train_count(fold_i) // BATCH_SIZE
    val_steps = get_val_count(fold_i) // BATCH_SIZE

    return train_ds, val_ds, train_steps, val_steps


# In[ ]:


def make_pred_dataset(fold_i):
    pred_ds = select_val(raw_ds, fold_i) \
        .map(pick_image_mask_info, num_parallel_calls=AUTOTUNE) \
        .batch(BATCH_SIZE) \
        .prefetch(AUTOTUNE)
    return pred_ds


# # Visualization

# In[ ]:


def draw_images_masks(ds):
    rows = 6
    cols = 5
    n_imgs = cols * rows
    plt.figure(figsize=(12, 2.5 * rows))
    for i, (image, mask) in enumerate(ds.take(n_imgs)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap="gray")
        plt.imshow(mask, alpha=0.5)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# In[ ]:


train_ds, val_ds, _, _ = make_datasets(0)

print("train_ds")
draw_images_masks(train_ds.unbatch().skip(95))

print("val_ds")
draw_images_masks(val_ds.unbatch().skip(80))


# # Model

# In[ ]:


dice_loss_fun = sm.losses.DiceLoss()
bce_loss_fun = sm.losses.BinaryCELoss()

def bce_dice_loss(y_true, y_pred):
    dice_loss = dice_loss_fun(y_true, y_pred)
    bce_loss = bce_loss_fun(y_true, y_pred)
    return 0.5 * dice_loss + 0.5 * bce_loss


# In[ ]:


# https://www.kaggle.com/code/ammarnassanalhajali/uwmgi-unet-keras-train-with-eda
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + smooth) \
        / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return coef


# In[ ]:


def make_input_3_channels(x):
    x_3_channels = tf.concat([x, x, x], axis=-1)
    return x_3_channels

def make_model():
    seg_model = SEG_MODEL(
        BACKBONE, encoder_weights='imagenet', 
        classes=3, activation='sigmoid')
    
    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name="inputs")
    x = L.Lambda(
        make_input_3_channels, name="3_chan")(inputs)
    outputs = seg_model(x)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="seg_model")

    # 'steps_per_execution' instructs to send multiple batches to TPU
    # at once. Each core in TPU should receive 128 elements.
    steps_per_execution = 128 // (BATCH_SIZE // REPLICAS)
    print("steps_per_execution: ", steps_per_execution)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=bce_dice_loss,
        metrics=[dice_coef],
        steps_per_execution=steps_per_execution)
    return model


# In[ ]:


with strategy.scope():
    model = make_model()
    
initial_weights = model.get_weights()
model.summary()


# In[ ]:


LR_START = INIT_LR
LR_MAX = 1e-3
LR_MIN = 1e-5
LR_RAMPUP_EPOCHS = WARMUP_EPO
LR_SUSTAIN_EPOCHS = 0
EPOCHS = N_EPOCHS

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        decay_total_epochs = EPOCHS - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
        decay_epoch_index = epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
    return lr

rng = [i for i in range(EPOCHS)]
lr_y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, lr_y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}". \
      format(lr_y[0], max(lr_y), lr_y[-1]))


# In[ ]:


def make_callbacks(best_model_file_name):
    cb_monitor = 'val_loss'
    cb_mode = 'min'
    cb_verbose = 1

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_model_file_name, save_best_only=True,
        save_weights_only=False, monitor=cb_monitor, mode=cb_mode,
        verbose=cb_verbose)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    
    return [checkpoint, lr_callback]


# In[ ]:


def fit_one_fold(fold_i, best_model_file_name):
    train_dataset, val_dataset, train_steps, val_steps = make_datasets(fold_i)
    callbacks = make_callbacks(best_model_file_name)

    history = model.fit(
        train_dataset, 
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps)
    return history


# In[ ]:


def plot_history(history, title, labels, subplot):
    plt.subplot(*subplot)
    plt.title(title)
    for label in labels:
        plt.plot(history.history[label], label=label)
    plt.legend()
    
def plot_fit_result(history):
    plt.figure(figsize=(12, 4))
    plot_history(history, "Loss", ['loss', 'val_loss'], (1, 2, 1))
    plot_history(history, "dice_coef", ['dice_coef', 'val_dice_coef'], (1, 2, 2))
    plt.show()


# In[ ]:


def resize_image_to(image, height, width):
    resized_image = tf.image.resize(
        image, [width, height],
        method=tf.image.ResizeMethod.BILINEAR)
    return resized_image


# In[ ]:


def encode_pred(one_pred):
    curr_pred = one_pred.flatten()
    
    prev_pred = np.empty_like(curr_pred)
    prev_pred[1:] = curr_pred[:-1]
    prev_pred[0] = 0
    
    next_pred = np.empty_like(curr_pred)
    next_pred[:-1] = curr_pred[1:]
    next_pred[-1] = 0
    
    pixel_no = np.arange(len(curr_pred))
    start_pixels = pixel_no[(prev_pred == 0) & (curr_pred == 1)]
    end_pixels = pixel_no[(curr_pred == 1) & (next_pred == 0)]
    
    encode_list = []
    for start_pixel, end_pixel in zip(start_pixels, end_pixels):
        encode_list.append(str(start_pixel))
        encode_list.append(str(end_pixel - start_pixel + 1))
    
    encoded_pred = ' '.join(encode_list)
    return encoded_pred


# In[ ]:


def make_predictions(raw_pred, height, width):
    resized_image = resize_image_to(raw_pred, height, width)
    bin_pred = np.where(resized_image >= 0.5, 1, 0)
    large_bowel_pred = encode_pred(bin_pred[:, :, 0])
    small_bowel_pred = encode_pred(bin_pred[:, :, 1])
    stomach_pred = encode_pred(bin_pred[:, :, 2])
    return large_bowel_pred, small_bowel_pred, stomach_pred


# In[ ]:


def predict_one_fold(fold_i):
    pred_ds = make_pred_dataset(fold_i)
    pred_batch_list = []
    mask_batch_list = []
    pred_list = []
    for i, (image_batch, mask_batch, info_batch) in enumerate(pred_ds):
        if DEBUG and 3 <= i:
            break
        print('.', end='', flush=True)

        pred_batch = model(image_batch, training=False)
        pred_batch_list.append(pred_batch)
        mask_batch_list.append(mask_batch.numpy())
        
        sample_id_batch, height_batch, width_batch = info_batch
        for pred, sample_id, height, width in \
                zip(pred_batch, sample_id_batch, height_batch, width_batch):
            sample_id = sample_id.numpy().decode('utf-8')
            height = height.numpy()
            width = width.numpy()
        
            large_bowel_pred, small_bowel_pred, stomach_pred = \
                make_predictions(pred, height, width)
            pred_list.append([
                sample_id, large_bowel_pred, small_bowel_pred, 
                stomach_pred, fold_i])
    print()
    
    preds = np.concatenate(pred_batch_list, axis=0)
    masks = np.concatenate(mask_batch_list, axis=0)
    pred_df = pd.DataFrame(
        pred_list, 
        columns=['id', 'large_bowel', 'small_bowel', 'stomach', 'fold'])
    return preds, masks, pred_df


# In[ ]:


def save_binary(name, bin_file, file_name_format):
    file_name = file_name_format.format(VID, fold_i)
    joblib.dump(bin_file, file_name)
    print("{0} are saved to {1}.".format(name, file_name))
    
def save_df(name, df, file_name_format):
    file_name = file_name_format.format(VID, fold_i)
    df.to_csv(file_name, index=False)
    print("{0} is saved to {1}.".format(name, file_name))


# In[ ]:


for fold_i in FOLD_I_LIST:
    print("####################")
    print("# Fold {0}".format(fold_i))
    model.set_weights(initial_weights)
    best_model_file_name = "seg_model_{0}_{1}.hdf5".format(VID, fold_i)
    history = fit_one_fold(fold_i, best_model_file_name)
    plot_fit_result(history)
    
    model.load_weights(best_model_file_name)
    preds, masks, pred_df = predict_one_fold(fold_i)
    
    save_binary("preds", preds, "preds_{0}_{1}.joblib")
    save_binary("masks", masks, "masks_{0}_{1}.joblib")
    save_df("pred_df", pred_df, "pred_{0}_{1}.csv")
    print()

