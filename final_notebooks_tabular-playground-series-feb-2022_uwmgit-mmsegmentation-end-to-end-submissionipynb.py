#!/usr/bin/env python
# coding: utf-8

# # 1. Install packages

# In[ ]:


import sys
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
sys.path.append("../input/segmentation-models-pytorch/segmentation_models.pytorch-0.2.1")
sys.path.append("../input/pretrainedmodels/pretrainedmodels-0.7.4")
sys.path.append("../input/efficientnet-pytorch/EfficientNet-PyTorch-master")

get_ipython().system('pip install ../input/mmdetection/addict-2.4.0-py3-none-any.whl > /dev/null')
get_ipython().system('pip install ../input/mmdetection/yapf-0.31.0-py2.py3-none-any.whl > /dev/null')
get_ipython().system('pip install ../input/mmdetection/terminaltables-3.1.0-py3-none-any.whl > /dev/null')
get_ipython().system('pip install ../input/mmdetection/einops* > /dev/null')
get_ipython().system('pip install ../input/mmdetection/mmcv_full-1.3.17-cp37-cp37m-linux_x86_64.whl > /dev/null')


# # 2. Install mmsegmentation 
# 
# This is from my own [mmseg github repo](https://github.com/CarnoZhao/Kaggle-UWMGIT) (leave a star if you like it!)
# 
# I have integrated `segmentation_models_pytorch` in this version of `mmsegmentation`. Although `segmentation_models_pytorch`'s simple Unet performs better than some models of `mmsegmentation`, anyway, `mmsegmentation` is still a good library for segmentation task when you want to compare various models in a unified training pipeline.
# 
# I only hard-coded `smp.Unet` in `./mmseg/models/segmentors/smp_models.py`. You can add more `smp` models in it!

# In[ ]:


get_ipython().system('git clone https://github.com/CarnoZhao/Kaggle-UWMGIT && cd ../input/kaggleuwmgit && pip install -e .')


# # 3. Prepare data

# In[ ]:


import os
import glob
import numpy as np
import pandas as pd

import cv2
from PIL import Image
from tqdm.auto import tqdm


# ## 3.1 Read csv and extract meta info

# In[ ]:


df_train = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
df_train = df_train.sort_values(["id", "class"]).reset_index(drop = True)
df_train["patient"] = df_train.id.apply(lambda x: x.split("_")[0])
df_train["days"] = df_train.id.apply(lambda x: "_".join(x.split("_")[:2]))

all_image_files = sorted(glob.glob("../input/uw-madison-gi-tract-image-segmentation/train/*/*/scans/*.png"), key = lambda x: x.split("/")[5] + "_" + x.split("/")[7])
size_x = [int(os.path.basename(_)[:-4].split("_")[-4]) for _ in all_image_files]
size_y = [int(os.path.basename(_)[:-4].split("_")[-3]) for _ in all_image_files]
spacing_x = [float(os.path.basename(_)[:-4].split("_")[-2]) for _ in all_image_files]
spacing_y = [float(os.path.basename(_)[:-4].split("_")[-1]) for _ in all_image_files]
df_train["image_files"] = np.repeat(all_image_files, 3)
df_train["spacing_x"] = np.repeat(spacing_x, 3)
df_train["spacing_y"] = np.repeat(spacing_y, 3)
df_train["size_x"] = np.repeat(size_x, 3)
df_train["size_y"] = np.repeat(size_y, 3)
df_train["slice"] = np.repeat([int(os.path.basename(_)[:-4].split("_")[-5]) for _ in all_image_files], 3)
df_train


# ## 3.2 Make mmseg-format data (2.5D by default)
# 
# 
# Here, I used 2.5d data with stride=2. Thanks this good trick from [https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-stride-2-data](https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-stride-2-data)

# In[ ]:


def rle_decode(mask_rle, shape):
    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    ends = starts + lengths
    h, w = shape
    img = np.zeros((h * w,), dtype = np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    return img.reshape(shape)

get_ipython().system('mkdir -p ./mmseg_train/{images,labels,splits}')
for day, group in tqdm(df_train.groupby("days")):
    patient = group.patient.iloc[0]
    imgs = []
    msks = []
    file_names = []
    for file_name in group.image_files.unique():
        img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
        segms = group.loc[group.image_files == file_name]
        masks = {}
        for segm, label in zip(segms.segmentation, segms["class"]):
            if not pd.isna(segm):
                mask = rle_decode(segm, img.shape[:2])
                masks[label] = mask
            else:
                masks[label] = np.zeros(img.shape[:2], dtype = np.uint8)
        masks = np.stack([masks[k] for k in sorted(masks)], -1)
        imgs.append(img)
        msks.append(masks)
        
    imgs = np.stack(imgs, 0)
    msks = np.stack(msks, 0)
    for i in range(msks.shape[0]):
        img = imgs[[max(0, i - 2), i, min(imgs.shape[0] - 1, i + 2)]].transpose(1,2,0) # 2.5d data
        msk = msks[i]
        new_file_name = f"{day}_{i}.png"
        cv2.imwrite(f"./mmseg_train/images/{new_file_name}", img)
        cv2.imwrite(f"./mmseg_train/labels/{new_file_name}", msk)


# ## 3.3 Make fold splits

# In[ ]:


all_image_files = glob.glob("./mmseg_train/images/*")
patients = [os.path.basename(_).split("_")[0] for _ in all_image_files]


from sklearn.model_selection import GroupKFold

split = list(GroupKFold(5).split(patients, groups = patients))

for fold, (train_idx, valid_idx) in enumerate(split):
    with open(f"./mmseg_train/splits/fold_{fold}.txt", "w") as f:
        for idx in train_idx:
            f.write(os.path.basename(all_image_files[idx])[:-4] + "\n")
    with open(f"./mmseg_train/splits/holdout_{fold}.txt", "w") as f:
        for idx in valid_idx:
            f.write(os.path.basename(all_image_files[idx])[:-4] + "\n")


# # 4. Training
# 
# ## 4.1 Make config
# 
# This is only **a simple baseline**, you can change anything in it
# 
# From my own experiment, when using larger backbone, larger image size and more augs, the public score will be easily exceed 0.865.
# 
# Here, I only train for 1k iters. **More iters are required to get a valid score**.
# 
# I have made a single model submission scored 0.878 using this training pipeline!

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\ncat <<EOT >> ./Kaggle-UWMGIT/config.py\nnum_classes = 3\n\n# model settings\nnorm_cfg = dict(type=\'SyncBN\', requires_grad=True)\nloss = [\n    dict(type=\'CrossEntropyLoss\', use_sigmoid=True, loss_weight=1.0),\n]\nmodel = dict(\n    type=\'SMPUnet\',\n    backbone=dict(\n        type=\'timm-efficientnet-b0\',\n        pretrained="imagenet"\n    ),\n    decode_head=dict(\n        num_classes=num_classes,\n        align_corners=False,\n        loss_decode=loss\n    ),\n    # model training and testing settings\n    train_cfg=dict(),\n    test_cfg=dict(mode="whole", multi_label=True))\n\n# dataset settings\ndataset_type = \'CustomDataset\'\ndata_root = \'../mmseg_train/\'\nclasses = [\'large_bowel\', \'small_bowel\', \'stomach\']\npalette = [[0,0,0], [128,128,128], [255,255,255]]\nimg_norm_cfg = dict(mean=[0,0,0], std=[1,1,1], to_rgb=True)\nsize = 256\nalbu_train_transforms = [\n    dict(type=\'RandomBrightnessContrast\', p=0.5),\n]\ntrain_pipeline = [\n    dict(type=\'LoadImageFromFile\', to_float32=True, color_type=\'unchanged\', max_value=\'max\'),\n    dict(type=\'LoadAnnotations\'),\n    dict(type=\'Resize\', img_scale=(size, size), keep_ratio=True),\n    dict(type=\'RandomFlip\', prob=0.5, direction=\'horizontal\'),\n    dict(type=\'Albu\', transforms=albu_train_transforms),\n    dict(type=\'Normalize\', **img_norm_cfg),\n    dict(type=\'Pad\', size=(size, size), pad_val=0, seg_pad_val=255),\n    dict(type=\'DefaultFormatBundle\'),\n    dict(type=\'Collect\', keys=[\'img\', \'gt_semantic_seg\']),\n]\ntest_pipeline = [\n    dict(type=\'LoadImageFromFile\', to_float32=True, color_type=\'unchanged\', max_value=\'max\'),\n    dict(\n        type=\'MultiScaleFlipAug\',\n        img_scale=(size, size),\n        flip=False,\n        transforms=[\n            dict(type=\'Resize\', keep_ratio=True),\n            dict(type=\'RandomFlip\'),\n            dict(type=\'Normalize\', **img_norm_cfg),\n            dict(type=\'Pad\', size=(size, size), pad_val=0, seg_pad_val=255),\n            dict(type=\'ImageToTensor\', keys=[\'img\']),\n            dict(type=\'Collect\', keys=[\'img\']),\n        ])\n]\ndata = dict(\n    samples_per_gpu=32,\n    workers_per_gpu=4,\n    train=dict(\n        type=dataset_type,\n        multi_label=True,\n        data_root=data_root,\n        img_dir=\'images\',\n        ann_dir=\'labels\',\n        img_suffix=".png",\n        seg_map_suffix=\'.png\',\n        split="splits/fold_0.txt",\n        classes=classes,\n        palette=palette,\n        pipeline=train_pipeline),\n    val=dict(\n        type=dataset_type,\n        multi_label=True,\n        data_root=data_root,\n        img_dir=\'images\',\n        ann_dir=\'labels\',\n        img_suffix=".png",\n        seg_map_suffix=\'.png\',\n        split="splits/holdout_0.txt",\n        classes=classes,\n        palette=palette,\n        pipeline=test_pipeline),\n    test=dict(\n        type=dataset_type,\n        multi_label=True,\n        data_root=data_root,\n        test_mode=True,\n        img_dir=\'test/images\',\n        ann_dir=\'test/labels\',\n        img_suffix=".jpg",\n        seg_map_suffix=\'.png\',\n        classes=classes,\n        palette=palette,\n        pipeline=test_pipeline))\n\n# yapf:disable\nlog_config = dict(\n    interval=50,\n    hooks=[\n        dict(type=\'CustomizedTextLoggerHook\', by_epoch=False),\n    ])\n# yapf:enable\ndist_params = dict(backend=\'nccl\')\nlog_level = \'INFO\'\nload_from = None\nresume_from = None\nworkflow = [(\'train\', 1)]\ncudnn_benchmark = True\n\ntotal_iters = 1\n# optimizer\noptimizer = dict(type=\'AdamW\', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05)\noptimizer_config = dict(type=\'Fp16OptimizerHook\', loss_scale=\'dynamic\')\n# learning policy\nlr_config = dict(policy=\'poly\',\n                 warmup=\'linear\',\n                 warmup_iters=500,\n                 warmup_ratio=1e-6,\n                 power=1.0, min_lr=0.0, by_epoch=False)\n# runtime settings\nfind_unused_parameters=True\nrunner = dict(type=\'IterBasedRunner\', max_iters=int(total_iters * 1000))\ncheckpoint_config = dict(by_epoch=False, interval=int(total_iters * 1000), save_optimizer=False)\nevaluation = dict(by_epoch=False, interval=min(5000, int(total_iters * 1000)), metric=[\'imDice\', \'mDice\'], pre_eval=True)\nfp16 = dict()\n\nwork_dir = f\'./work_dirs/tract/baseline\'\nEOT\n')


# # 4.2 Let's start training

# In[ ]:


# reinstall for inner bash usage
get_ipython().system('cp -r ../input/segmentation-models-pytorch/segmentation_models.pytorch-0.2.1 ./ && cd segmentation_models.pytorch-0.2.1  && pip install -e .')
get_ipython().system('cp -r ../input/timm-pytorch-image-models/pytorch-image-models-master ./ && cd pytorch-image-models-master  && pip install -e .')


# In[ ]:


get_ipython().system('cd Kaggle-UWMGIT && python ./tools/train.py ./config.py --gpu-ids 0')


# # 5. Inferencing
# 
# ## 5.1 Load trained models

# In[ ]:


sys.path.append('./Kaggle-UWMGIT')
from mmseg.apis import init_segmentor, inference_segmentor
from mmcv.utils import config

cfgs = [
    "./Kaggle-UWMGIT/work_dirs/tract/baseline/config.py",
]

ckpts = [
    "./Kaggle-UWMGIT/work_dirs/tract/baseline/latest.pth",
]

models = []
for cfg, ckpt in zip(cfgs, ckpts):
    cfg = config.Config.fromfile(cfg)
    cfg.model.backbone.pretrained = None
    cfg.model.test_cfg.logits = True
    cfg.data.test.pipeline[1].transforms.insert(2, dict(type="Normalize", mean=[0,0,0], std=[1,1,1], to_rgb=False))

    model = init_segmentor(cfg, ckpt)
    models.append(model)


# ## 5.2 Make test submission csv

# In[ ]:


import os
import cv2
import numpy as np
import pandas as pd
import glob
from tqdm.auto import tqdm
from scipy.ndimage import binary_closing, binary_opening, measurements

def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

classes = ['large_bowel', 'small_bowel', 'stomach']
data_dir = "../input/uw-madison-gi-tract-image-segmentation/"
test_dir = os.path.join(data_dir, "test")
sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
test_images = glob.glob(os.path.join(test_dir, "**", "*.png"), recursive = True)

if len(test_images) == 0:
    test_dir = os.path.join(data_dir, "train")
    sub = pd.read_csv(os.path.join(data_dir, "train.csv"))[["id", "class"]].iloc[:100 * 3]
    sub["predicted"] = ""
    test_images = glob.glob(os.path.join(test_dir, "**", "*.png"), recursive = True)
    
id2img = {_.rsplit("/", 4)[2] + "_" + "_".join(_.rsplit("/", 4)[4].split("_")[:2]): _ for _ in test_images}
sub["file_name"] = sub.id.map(id2img)
sub["days"] = sub.id.apply(lambda x: "_".join(x.split("_")[:2]))
fname2index = {f + c: i for f, c, i in zip(sub.file_name, sub["class"], sub.index)}
sub


# ## 5.3 Start Inferencing

# In[ ]:


subs = []
for day, group in tqdm(sub.groupby("days")):
    imgs = []
    for file_name in group.file_name.unique():
        img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
        old_size = img.shape[:2]
        s = int(os.path.basename(file_name).split("_")[1])
        file_names = [file_name.replace(f"slice_{s:04d}", f"slice_{s + i:04d}") for i in range(-2, 3)]
        file_names = [_ for _ in file_names if os.path.exists(_)]
        imgs = [cv2.imread(file_names[0], cv2.IMREAD_ANYDEPTH)] + [img] + [cv2.imread(file_names[-1], cv2.IMREAD_ANYDEPTH)]
        
        new_img = np.stack(imgs, -1)
        new_img = new_img.astype(np.float32) / new_img.max()

        res = [inference_segmentor(model, new_img)[0] for model in models]
        res = (sum(res) / len(res)).round().astype(np.uint8)
        res = cv2.resize(res, old_size[::-1], interpolation = cv2.INTER_NEAREST)
        for j in range(3):
            rle = rle_encode(res[...,j])
            index = fname2index[file_name + classes[j]]
            sub.loc[index, "predicted"] = rle


# ## 5.4 Format submission

# In[ ]:


sub = sub[["id", "class", "predicted"]]
sub.to_csv("submission.csv", index = False)
sub

