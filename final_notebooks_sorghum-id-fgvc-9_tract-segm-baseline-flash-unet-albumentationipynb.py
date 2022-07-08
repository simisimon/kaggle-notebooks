#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis 🔎

# In[ ]:


get_ipython().system('pip uninstall -y torchtext')
get_ipython().system('mkdir -p frozen_packages/')
get_ipython().system('cp ../input/demo-flash-semantic-segmentation/frozen_packages/* frozen_packages/')
get_ipython().system('cp ../input/tract-segm-eda-3d-interactive-viewer/frozen_packages/* frozen_packages/')
# !pip install -q --upgrade torch torchvision
get_ipython().system('pip install -q "lightning-flash[image]" "torchmetrics<0.8" --pre --no-index --find-links frozen_packages/')
get_ipython().system('pip install -q -U timm segmentation-models-pytorch --no-index --find-links frozen_packages/')
# !pip install -q "https://github.com/PyTorchLightning/lightning-flash/archive/refs/heads/segm/multi-label.zip"
get_ipython().system("pip install -q 'kaggle-image-segmentation' --no-index --find-links frozen_packages/")

get_ipython().system(' pip list | grep -e torch -e lightning')
get_ipython().system(' nvidia-smi -L')


# In[ ]:


import os, glob
import pandas as pd
import matplotlib.pyplot as plt

DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
df_train = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))
display(df_train.head())

df_pred = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
WITH_SUBMISSION = not df_pred.empty


# In[ ]:


all_imgs = glob.glob(os.path.join(DATASET_FOLDER, "train", "case*", "case*_day*", "scans", "*.png"))
all_imgs = [p.replace(DATASET_FOLDER, "") for p in all_imgs]

print(f"images: {len(all_imgs)}")
print(f"annotated: {len(df_train['id'].unique())}")


# In[ ]:


from pprint import pprint
from kaggle_imsegm.data_io import extract_tract_details

pprint(extract_tract_details(df_train['id'].iloc[0], DATASET_FOLDER))

df_train[['Case','Day','Slice', 'image', 'image_path', 'height', 'width']] = df_train['id'].apply(
    lambda x: pd.Series(extract_tract_details(x, DATASET_FOLDER))
)
display(df_train.head())


# # Prepare custom dataset 💽

# In[ ]:


import os.path
from typing import Callable, Tuple, Sequence

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from kaggle_imsegm.dataset import TractDataset2D

ds = TractDataset2D(df_train, DATASET_FOLDER)
print(len(ds))


# In[ ]:


spl = ds[255]
img, seg = spl["input"], spl["target"]
print(img.shape)
fig, axarr = plt.subplots(ncols=4, figsize=(12, 3))
axarr[0].imshow(np.rollaxis(img.numpy(), 0, 3), cmap="gray")
print(np.argmax(seg, axis=0).shape)
for i in range(seg.shape[0]):
    axarr[i + 1].imshow(seg[i, ...])


# In[ ]:


from typing import Any, Callable, Dict, Tuple, Type, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import albumentations as alb

from kaggle_imsegm.transform import FlashAlbumentationsAdapter
from kaggle_imsegm.dataset import TractData

COLOR_MEAN: float = 0.349977
COLOR_STD: float = 0.215829
DEFAULT_TRANSFORM = FlashAlbumentationsAdapter(
    [alb.Resize(224, 224), alb.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255)]
)
    
dm = TractData(df_train, DATASET_FOLDER, dataloader_kwargs=dict(batch_size=12, num_workers=3))
dm.setup()
print(len(dm.train_dataloader()))
print(len(dm.val_dataloader()))


# In[ ]:


from kaggle_imsegm.visual import show_tract_datamodule_samples_2d

_= show_tract_datamodule_samples_2d(dm.val_dataloader(), nb=3)


# # Lightning⚡Flash & UNet++ & albumentations
# 
# lets follow the Semantinc segmentation example: https://lightning-flash.readthedocs.io/en/stable/reference/semantic_segmentation.html

# In[ ]:


import torch

import flash
from flash.core.data.utils import download_data
from flash.image import SemanticSegmentation, SemanticSegmentationData

print(flash.__version__)


# ### 1. Create the DataModule

# In[ ]:


from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union
import albumentations as alb
from flash.core.data.io.input_transform import InputTransform
from flash.image.segmentation.input_transform import prepare_target, remove_extra_dimensions
# from kaggle_imsegm.augment import FlashAlbumentationsAdapter

IMAGE_SIZE = (320, 320)
TRAIN_TRANSFORM = FlashAlbumentationsAdapter([
    alb.Resize(*IMAGE_SIZE),
    alb.VerticalFlip(p=0.5),
    alb.HorizontalFlip(p=0.5),
    alb.RandomRotate90(p=0.5),
    alb.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.08, rotate_limit=10, p=1.),
    alb.GaussNoise(var_limit=(0.001, 0.02), mean=0, per_channel=False, p=1.0),
    # alb.OneOf([
    #     alb.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
    #     alb.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
    # ], p=0.25),
    alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    alb.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255),
])
VAL_TRANSFORM = FlashAlbumentationsAdapter([
    alb.Resize(*IMAGE_SIZE), alb.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255)
])


# In[ ]:


sample_imgs = glob.glob(os.path.join(DATASET_FOLDER, "test", "**", "*.png"), recursive=True)
if not sample_imgs:
    sample_imgs = glob.glob(os.path.join(DATASET_FOLDER, "train", "case123", "**", "*.png"), recursive=True)
print(f"images: {len(sample_imgs)}")
sample_imgs = [p.replace(DATASET_FOLDER + os.path.sep, "") for p in sample_imgs[70:75]]
tab_preds = pd.DataFrame({"image_path": sample_imgs})


# In[ ]:


datamodule = TractData(
    df_train,
    dataset_dir=DATASET_FOLDER,
    df_predict=tab_preds,
    train_transform=TRAIN_TRANSFORM,
    input_transform=VAL_TRANSFORM,
    dataloader_kwargs=dict(batch_size=18, num_workers=3),
    val_split=0.01 if WITH_SUBMISSION else 0.1, 
)
datamodule.setup()
LABELS = datamodule.labels
assert len(LABELS) == 3


# In[ ]:


_= show_tract_datamodule_samples_2d(datamodule.train_dataloader(), nb=5, skip_empty=True)


# ### 2. Build the task

# In[ ]:


# import segmentation_models_pytorch as smp
from kaggle_imsegm.model import MixedLoss
from kaggle_imsegm.transform import SemanticSegmentationOutputTransform

model = SemanticSegmentation(
    backbone="efficientnet-b3",
    head="unetplusplus",
    pretrained=False,
    optimizer="AdamW",
    learning_rate=7e-3,
    loss_fn=MixedLoss("dice", smooth=0.01),
    lr_scheduler=("cosineannealinglr", {"T_max": 500, "eta_min": 1e-6}),
    num_classes=3,
    multi_label=True,
    output_transform=SemanticSegmentationOutputTransform(),
)


# ### 3. Create the trainer and finetune the model

# In[ ]:


import pytorch_lightning as pl

GPUs = torch.cuda.device_count()

trainer = flash.Trainer(
    max_epochs=15 if WITH_SUBMISSION else 5,
    logger=pl.loggers.CSVLogger(save_dir='logs/'),
    gpus=GPUs,
    precision=16 if GPUs else 32,
    accumulate_grad_batches=24,
    gradient_clip_val=0.01,
    limit_train_batches=1.0 if WITH_SUBMISSION else 0.1,
    limit_val_batches=1.0 if WITH_SUBMISSION else 0.2,
)


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Train the model
trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")

# Save the model!
trainer.save_checkpoint("semantic_segmentation_model.pt")


# In[ ]:


import seaborn as sn

metrics = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')
del metrics["step"]
metrics.set_index("epoch", inplace=True)
# display(metrics.dropna(axis=1, how="all").head())
g = sn.relplot(data=metrics, kind="line")
plt.gcf().set_size_inches(12, 4)
plt.grid()


# ### 4. Segment a few images!

# In[ ]:


from itertools import chain

preds = trainer.predict(model, datamodule=datamodule)  #, output="preds"
preds = list(chain(*preds))


# In[ ]:


fig, axarr = plt.subplots(ncols=4, nrows=len(sample_imgs), figsize=(12, 3 * len(sample_imgs)))
for i, pred in enumerate(preds):
    print(pred.keys())
    img = pred['input']
    print(img.shape, img.min(), img.max())
    axarr[i, 0].imshow(img)
    for j, seg in enumerate(pred['preds']):
        print(seg.shape, seg.min(), seg.max())
        im = axarr[i, j + 1].imshow(seg, vmin=-10, vmax=10)
        plt.colorbar(im, ax=axarr[i, j + 1])


# # Inference 🔥

# In[ ]:


model = SemanticSegmentation.load_from_checkpoint(
    "semantic_segmentation_model.pt"
)


# In[ ]:


sfolder = "test" if WITH_SUBMISSION else "train"
ls_images = glob.glob(os.path.join(DATASET_FOLDER, sfolder, "**", "*.png"), recursive=True)
ls_images = [p.replace(DATASET_FOLDER + os.path.sep, "") for p in ls_images]
case_day = [os.path.dirname(p).split(os.path.sep)[-2] for p in ls_images]
df_pred = pd.DataFrame({'Case_Day': case_day, 'image_path': ls_images})

if not WITH_SUBMISSION:
    df_pred = df_pred[df_pred["Case_Day"].str.startswith("case123_day")]
display(df_pred.head())


# ## Predictions for test scans

# In[ ]:


import numpy as np
from itertools import chain
from scipy.ndimage import binary_opening
from skimage.morphology import disk
from kaggle_imsegm.mask import rle_encode

preds = []
for case_day, tab_preds in tqdm(df_pred.groupby("Case_Day")):
    dm = TractData(
        df_train[df_train["id"].str.startswith("case123_day")],  # FAKE
        dataset_dir=DATASET_FOLDER,
        df_predict=tab_preds,
        train_transform=TRAIN_TRANSFORM,
        input_transform=VAL_TRANSFORM,
        dataloader_kwargs=dict(batch_size=10, num_workers=3),
    )
    # dm.setup()
    results = trainer.predict(model, datamodule=dm)
    results = list(chain(*results))
    assert len(tab_preds["image_path"]) == len(results)
    for img_path, spl in zip(tab_preds["image_path"], results):
        name, _ = os.path.splitext(os.path.basename(img_path))
        id_ = f"{case_day}_" + "_".join(name.split("_")[:2])
        # print(spl.keys())
        for i, mask in enumerate(spl["preds"]):
            mask = (mask >= 0).astype(np.uint8)
            mask = binary_opening(mask, structure=disk(4)).astype(np.uint8)
            # print(seg.shape)
            rle = rle_encode(mask)[1] if np.sum(mask) > 1 else ""
            preds.append({"id": id_, "class": LABELS[i], "predicted": rle})

assert len(preds) == 3 * len(df_pred)
df_pred = pd.DataFrame(preds)


# In[ ]:


display(df_pred[df_pred["predicted"] != ""].head())


# ## Finalize submissions

# In[ ]:


df_ssub = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
del df_ssub['predicted']
if WITH_SUBMISSION:
    assert len(df_ssub) == len(df_pred)
df_pred = df_ssub.merge(df_pred, on=['id','class'])

df_pred[['id', 'class', 'predicted']].to_csv("submission.csv", index=False)

get_ipython().system('head submission.csv')

