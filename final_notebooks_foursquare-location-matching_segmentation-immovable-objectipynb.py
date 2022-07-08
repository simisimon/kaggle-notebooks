#!/usr/bin/env python
# coding: utf-8

# credit :[this notebook by ammarali32](https://www.kaggle.com/code/ammarali32/imc-2022-kornia-loftr-from-0-533-to-0-721) and followers...
# 
# 
# ### This notebook partly use above notebook's code! Thanks!

# ## create masks by cityscape trained deeplabv3 model(from [tensorflow model_zoo](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/cityscapes.md))
# ### Any comments and advice welcome!

# In[ ]:


import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
import cv2
import csv
from glob import glob

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[ ]:


# reffered from https://github.com/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb
class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 854
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    
    with tf.io.gfile.GFile(path, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

  def close(self):
        self.sess.close()


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray(["movable", "immovable"])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


# ## Use xception backbone(very good segmentation result, but too slow to submit😢)

# In[ ]:


MODEL = DeepLabModel("../input/deeplab-cityscapes-xception71-trainvalfine/deeplab_cityscapes_xception71_trainvalfine_2018_09_08/trainval_fine/frozen_inference_graph.pb")


# In[ ]:


src = '/kaggle/input/image-matching-challenge-2022/'

test_samples = []
with open(f'{src}/test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]


resized_ims = []
masks = []
immovable = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) # 'road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain'
movable = (10, 11, 12, 13, 14, 15, 16, 17, 18) #'sky','person','rider','car','truck','bus','train','motorcycle','bicycle'
masks = {}

# for i in ims:
for i, row in enumerate(test_samples):
  sample_id, batch_id, image_1_id, image_2_id = row
  
  image_1 = Image.open(f'{src}/test_images/{batch_id}/{image_1_id}.png')
  image_2 = Image.open(f'{src}/test_images/{batch_id}/{image_2_id}.png')
    
  resized_im_1, seg_map_1 = MODEL.run(image_1)  
  resized_im_2, seg_map_2 = MODEL.run(image_2)
    
  mask_1 = np.zeros(seg_map_1.shape)
  mask_2 = np.zeros(seg_map_2.shape)

  for j in immovable:
    mask_1[seg_map_1 == j] = 1
    mask_2[seg_map_2 == j] = 1

  if i<3:
    vis_segmentation(resized_im_1, mask_1.astype(np.uint8))
    vis_segmentation(resized_im_2, mask_2.astype(np.uint8))

  masks[batch_id] = (mask_1.astype(np.uint8), mask_2.astype(np.uint8))


# ## Use mobilenetv3 backbone(fast, but not good segmentation😢)

# In[ ]:


MODEL = DeepLabModel("../input/deeplab-mnv3-large-cityscapes-trainfine-2019-11-15/deeplab_mnv3_large_cityscapes_trainfine/frozen_inference_graph.pb")


# In[ ]:


src = '/kaggle/input/image-matching-challenge-2022/'

test_samples = []
with open(f'{src}/test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]


resized_ims = []
masks = []
immovable = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) # 'road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain'
movable = (10, 11, 12, 13, 14, 15, 16, 17, 18) #'sky','person','rider','car','truck','bus','train','motorcycle','bicycle'
masks = {}

# for i in ims:
for i, row in enumerate(test_samples):
  sample_id, batch_id, image_1_id, image_2_id = row
  
  image_1 = Image.open(f'{src}/test_images/{batch_id}/{image_1_id}.png')
  image_2 = Image.open(f'{src}/test_images/{batch_id}/{image_2_id}.png')
    
  resized_im_1, seg_map_1 = MODEL.run(image_1)  
  resized_im_2, seg_map_2 = MODEL.run(image_2)
    
  mask_1 = np.zeros(seg_map_1.shape)
  mask_2 = np.zeros(seg_map_2.shape)

  for j in immovable:
    mask_1[seg_map_1 == j] = 1
    mask_2[seg_map_2 == j] = 1

  if i<3:
    vis_segmentation(resized_im_1, mask_1.astype(np.uint8))
    vis_segmentation(resized_im_2, mask_2.astype(np.uint8))

  masks[batch_id] = (mask_1.astype(np.uint8), mask_2.astype(np.uint8))


# In[ ]:


# trick for release vram used by tf1 model
from numba import cuda 
device = cuda.get_current_device()
device.reset()

