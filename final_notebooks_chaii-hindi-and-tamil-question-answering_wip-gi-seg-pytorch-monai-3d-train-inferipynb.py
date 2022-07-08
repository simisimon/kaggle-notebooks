#!/usr/bin/env python
# coding: utf-8

# # GI-Seg PyTorch ⚡ MONAI 3D Train & Infer
# - Downloaded weight and requirements come from [GI-Seg Downloads](https://www.kaggle.com/clemchris/gi-seg-download)
# - Dataset: [UW-Madison GI Tract Image Segmentation Masks](https://www.kaggle.com/datasets/clemchris/uw-madison-gi-tract-image-segmentation-masks)
# 
# 
# ## Sources --> please upvote them if you find this notebook useful
# - Awsaf's [UWMGI: Unet [Train] [PyTorch]](https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch)
# - Awsaf's [UWMGI: 2.5D stride=2 Data](https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-stride-2-data)
# - Awsaf's [UWMGI: Unet [Infer] [PyTorch]](https://www.kaggle.com/code/awsaf49/uwmgi-unet-infer-pytorch)
# 
# ## Scores
# - V01: 0.XXX

# # Installs

# In[ ]:


get_ipython().system('cd ../input/gi-seg-downloads &&  pip install -q monai-0.8.1-202202162213-py3-none-any.whl torchmetrics-0.8.2-py3-none-any.whl')


# # Imports

# In[ ]:


from pathlib import Path
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import cupy as cp
import cv2
import matplotlib.pyplot as plt
import monai
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from joblib import delayed
from joblib import Parallel
from monai.data import CSVDataset
from monai.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Metric
from torchmetrics import MetricCollection
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm


# # Paths & Settings

# In[ ]:


KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"
OUTPUT_DIR = KAGGLE_DIR / "working"

INPUT_DATA_DIR = INPUT_DIR / "uw-madison-gi-tract-image-segmentation"
INPUT_DATA_NPY_DIR = INPUT_DIR / "uw-madison-gi-tract-image-segmentation-masks"

SPATIAL_SIZE = (192, 192, 128)
N_SPLITS = 5
RANDOM_SEED = 2022
VAL_FOLD = 0
BATCH_SIZE = 1
NUM_WORKERS = 2
OPTIMIZER = "Adam"
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-6
SCHEDULER = None
MIN_LR = 1e-6

FAST_DEV_RUN = False # Debug training
GPUS = 1
MAX_EPOCHS = 30
PRECISION = 32

DEVICE = "cuda"
THR = 0.45

DEBUG = False # Debug complete pipeline


# # Prepare 3D Data

# In[ ]:


def add_3d_paths(df, stage):
    df["image_3d"] = df["image_path"].str.split("/scans").str[0] + "_image_3d.npy"
    df["image_3d"] = df["image_3d"].str.replace("input", "working")
    
    if stage == "train":
        df["mask_3d"] = df["image_3d"].str.replace("_image_", "_mask_")
        
    return df


# In[ ]:


train_df = pd.read_csv(INPUT_DATA_NPY_DIR / "train_preprocessed.csv")

if DEBUG:
    train_df = train_df.head(1_000)

train_df = add_3d_paths(train_df, stage="train")


# In[ ]:


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED) # uint16
    return image


def load_mask(row):
    shape = (row.height, row.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)

    rles = eval(row.segmentation.replace("nan", "''"))
    for i, rle in enumerate(rles):
        if rle:
            mask[..., i] = rle_decode(rle, shape[:2])

    return mask * 255


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1

    return mask.reshape(shape)  # Needed to align to RLE direction


def create_3d_image_mask(group_df, stage):
    image_3d, mask_3d = [], []
    for row in group_df.itertuples():
        image_3d.append(load_image(row.image_path))  # uint16
        
        if stage == "train":
            mask_3d.append(load_mask(row))  # uint8

    image_3d = np.stack(image_3d, axis=-1)

    dir_3d = Path(row.image_3d).parent
    dir_3d.mkdir(parents=True, exist_ok=True)
    np.save(row.image_3d, image_3d)

    if stage == "train":
        mask_3d = np.stack(mask_3d, axis=-1)
        np.save(row.mask_3d, mask_3d)

    return group_df.id.to_list()


def create_3d_npy_data(df, stage):
    grouped = df.groupby(["case", "day"])
    ids = Parallel(n_jobs=NUM_WORKERS)(
        delayed(create_3d_image_mask)(group_df, stage)
        for _, group_df in tqdm(grouped, total=len(grouped), desc="Iterating over case-day groups")
    )

    columns_to_drop = ["id", "slice", "image_path"]
    if stage == "train":
        columns_to_drop += ["classes", "segmentation", "rle_len", "empty", "mask_path", "image_paths"]

    df = df.drop(columns=columns_to_drop)
    df = df.drop_duplicates().reset_index(drop=True)
    df["ids"] = ids

    return df


# In[ ]:


train_df = create_3d_npy_data(train_df, stage="train")

if DEBUG:
    print(len(train_df))
    display(train_df.head())


# In[ ]:


train_df.to_csv(f"train_preprocessed_3d.csv")


# # LitDataModule

# In[ ]:


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_path: str,
        test_csv_path: Optional[str],
        val_fold: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_df = pd.read_csv(train_csv_path)

        if test_csv_path is not None:
            self.test_df = pd.read_csv(test_csv_path)
        else:
            self.test_df = None

        self.train_transforms, self.val_transforms, self.test_transforms = self._init_transforms()

    def _init_transforms(self):
        spatial_size = SPATIAL_SIZE

        transforms = [
            monai.transforms.LoadImaged(keys=["image_3d", "mask_3d"]),
            monai.transforms.AddChanneld(keys=["image_3d"]),
            monai.transforms.AsChannelFirstd(keys=["mask_3d"], channel_dim=2),
            monai.transforms.ScaleIntensityd(keys=["image_3d", "mask_3d"]),
            #monai.transforms.ResizeWithPadOrCrop(keys=["image_3d", "mask_3d"], spatial_size=spatial_size),
            monai.transforms.Resized(keys=["image_3d", "mask_3d"], spatial_size=spatial_size, mode="nearest"),
        ]

        test_transforms = [
            monai.transforms.LoadImaged(keys=["image_3d"]),
            monai.transforms.AddChanneld(keys=["image_3d"]),
            monai.transforms.ScaleIntensityd(keys=["image_3d"]),
            #monai.transforms.ResizeWithPadOrCrop(keys=["image_3d"], spatial_size=spatial_size),
            monai.transforms.Resized(keys=["image_3d"], spatial_size=spatial_size, mode="nearest"),
        ]

        train_transforms = monai.transforms.Compose(transforms)
        val_transforms = monai.transforms.Compose(transforms)
        test_transforms = monai.transforms.Compose(test_transforms)

        return train_transforms, val_transforms, test_transforms

    def setup(self, stage: Optional[str] = None):
        train_df = self.train_df[self.train_df.fold != self.hparams.val_fold].reset_index(drop=True)
        val_df = self.train_df[self.train_df.fold == self.hparams.val_fold].reset_index(drop=True)

        if stage == "fit" or stage is None:
            self.train_dataset = self._dataset(train_df, transforms=self.train_transforms)
            self.val_dataset = self._dataset(val_df, transforms=self.val_transforms)

        if stage == "test" or stage is None:
            if self.test_df is not None:
                self.test_dataset = self._dataset(self.test_df, transforms=self.test_transforms)
            else:
                self.test_dataset = self._dataset(val_df, transforms=self.val_transforms)

    def _dataset(self, df: pd.DataFrame, transforms: Callable) -> CSVDataset:
        return CSVDataset(src=df, transform=transforms)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: CSVDataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )


# # Visualize Cases

# In[ ]:


data_module = LitDataModule(
    train_csv_path="train_preprocessed_3d.csv",
    test_csv_path=None,
    val_fold=VAL_FOLD,
    batch_size=4,
    num_workers=NUM_WORKERS,
)
data_module.setup()

train_dataloader = data_module.train_dataloader()
batch = next(iter(train_dataloader))


# In[ ]:


for batch_idx, _ in enumerate(batch["image_3d"]):
    image_3d = batch["image_3d"][batch_idx]
    mask_3d = batch["mask_3d"][batch_idx]

    fig, ax = plt.subplots()
    _, images_grid = monai.visualize.utils.matshow3d(volume=image_3d, every_n=10, frame_dim=-1, fig=fig)
    _, masks_grid = monai.visualize.utils.matshow3d(volume=mask_3d, every_n=10, frame_dim=-1, channel_dim=0, fig=fig)
    plt.title(f"Case {batch['case'][batch_idx]}, Day {batch['day'][batch_idx]}")
    plt.imshow(images_grid, cmap="bone")
    plt.imshow(masks_grid, alpha=0.5)
    plt.axis("off")
    plt.tight_layout()


# # Metrics

# In[ ]:


class DiceMetric(Metric):
    def __init__(self):
        super().__init__()

        self.post_processing = monai.transforms.Compose(
            [
                monai.transforms.Activations(sigmoid=True),
                monai.transforms.AsDiscrete(threshold=0.5),
            ]
        )
        self.add_state("dice", default=[])

    def update(self, y_pred, y_true):
        y_pred = self.post_processing(y_pred)
        self.dice.append(monai.metrics.compute_meandice(y_pred, y_true))

    def compute(self):
        return torch.mean(torch.stack(self.dice))


# # LitModule

# In[ ]:


class LitModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        scheduler: Optional[str],
        T_max: int,
        T_0: int,
        min_lr: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = self._init_model()

        self.loss_fn = self._init_loss_fn()

        self.metrics = self._init_metrics()

    def _init_model(self):
        return monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def _init_loss_fn(self):
        return monai.losses.DiceLoss(sigmoid=True)

    def _init_metrics(self):
        val_metrics = MetricCollection({"val_dice": DiceMetric()})
        test_metrics = MetricCollection({"test_dice": DiceMetric()})
        
        return torch.nn.ModuleDict(
            {
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler is not None:
            if self.hparams.scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.hparams.T_max, eta_min=self.hparams.min_lr
                )
            else:
                raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")

            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        else:
            return {"optimizer": optimizer}

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    def shared_step(self, batch, stage, log=True):
        images, masks = batch["image_3d"], batch["mask_3d"]
        y_pred = self(images)

        loss = self.loss_fn(y_pred, masks)

        if stage != "train":
            metrics = self.metrics[f"{stage}_metrics"](y_pred, masks)
        else:
            metrics = None

        if log:
            batch_size = images.shape[0]
            self._log(loss, batch_size, metrics, stage)

        return loss

    def _log(self, loss, batch_size, metrics, stage):
        on_step = True if stage == "train" else False

        self.log(f"{stage}_loss", loss, on_step=on_step, on_epoch=True, prog_bar=True, batch_size=batch_size)

        if metrics is not None:
            self.log_dict(metrics, on_step=on_step, on_epoch=True, batch_size=batch_size)

    @classmethod
    def load_eval_checkpoint(cls, checkpoint_path, device):
        module = cls.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)
        module.eval()

        return module


# # Train

# In[ ]:


def train(
    random_seed: int = RANDOM_SEED,
    train_csv_path: str = "train_preprocessed_3d.csv",
    val_fold: str = VAL_FOLD,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    scheduler: Optional[str] = SCHEDULER,
    min_lr: float = MIN_LR,
    gpus: int = GPUS,
    fast_dev_run: bool = FAST_DEV_RUN,
    max_epochs: int = MAX_EPOCHS,
    precision: int = PRECISION,
    debug: bool = DEBUG,
):
    pl.seed_everything(random_seed)

    if debug:
        max_epochs = 2

    data_module = LitDataModule(
        train_csv_path=train_csv_path,
        test_csv_path=None,
        val_fold=val_fold,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    module = LitModule(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler=scheduler,
        T_max=int(30_000 / batch_size * max_epochs) + 50,
        T_0=25,
        min_lr=min_lr,
    )

    trainer = pl.Trainer(
        fast_dev_run=fast_dev_run,
        gpus=gpus,
        log_every_n_steps=10,
        logger=pl.loggers.CSVLogger(save_dir='logs/'),
        max_epochs=max_epochs,
        precision=precision,
    )

    trainer.fit(module, datamodule=data_module)

    if not fast_dev_run:
        trainer.test(module, datamodule=data_module)
        
    return trainer


# In[ ]:


trainer = train()


# In[ ]:


# From https://www.kaggle.com/code/jirkaborovec?scriptVersionId=93358967&cellId=22
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")[["epoch", "train_loss_epoch", "val_loss"]]
metrics.set_index("epoch", inplace=True)

sns.relplot(data=metrics, kind="line", height=5, aspect=1.5)
plt.grid()


# # Infer

# ### Load Test Data

# In[ ]:


def extract_metadata_from_id(df):
    df[["case", "day", "slice"]] = df["id"].str.split("_", n=2, expand=True)

    df["case"] = df["case"].str.replace("case", "").astype(int)
    df["day"] = df["day"].str.replace("day", "").astype(int)
    df["slice"] = df["slice"].str.replace("slice_", "").astype(int)

    return df


def extract_metadata_from_path(path_df):
    path_df[["parent", "case_day", "scans", "file_name"]] = path_df["image_path"].str.rsplit("/", n=3, expand=True)

    path_df[["case", "day"]] = path_df["case_day"].str.split("_", expand=True)
    path_df["case"] = path_df["case"].str.replace("case", "")
    path_df["day"] = path_df["day"].str.replace("day", "")

    path_df[["slice", "width", "height", "spacing", "spacing_"]] = (
        path_df["file_name"].str.replace("slice_", "").str.replace(".png", "").str.split("_", expand=True)
    )
    path_df = path_df.drop(columns=["parent", "case_day", "scans", "file_name", "spacing_"])

    numeric_cols = ["case", "day", "slice", "width", "height", "spacing"]
    path_df[numeric_cols] = path_df[numeric_cols].apply(pd.to_numeric)

    return path_df


# In[ ]:


sub_df = pd.read_csv(INPUT_DATA_DIR / "sample_submission.csv")
test_set_hidden = not bool(len(sub_df))

if test_set_hidden:
    test_df = pd.read_csv(INPUT_DATA_DIR / "train.csv")[: 1000 * 3]
    test_df = test_df.drop(columns=["class", "segmentation"]).drop_duplicates()
    image_paths = [str(path) for path in (INPUT_DATA_DIR / "train").rglob("*.png")]
else:
    test_df = sub_df.drop(columns=["class", "predicted"]).drop_duplicates()
    image_paths = [str(path) for path in (INPUT_DATA_DIR / "test").rglob("*.png")]

test_df = extract_metadata_from_id(test_df)

path_df = pd.DataFrame(image_paths, columns=["image_path"])
path_df = extract_metadata_from_path(path_df)

test_df = test_df.merge(path_df, on=["case", "day", "slice"], how="left")
test_df = add_3d_paths(test_df, stage="test")

print(len(test_df))
test_df.head()


# In[ ]:


test_df = create_3d_npy_data(test_df, stage="test")


# ### Save Test DataFrame

# In[ ]:


test_df.to_csv("test_preprocessed_3d.csv", index=False)


# ## Run inference

# In[ ]:


def mask2rle(mask):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    mask = cp.array(mask)
    pixels = mask.flatten()
    pad = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def masks2rles(masks, ids, height, width):
    pred_strings = []
    pred_ids = []
    pred_classes = []

    for idx in tqdm(range(masks.shape[0])):
        mask = masks[idx]
        
        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(mask[..., midx])

        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * len(rle))
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])

    return pred_strings, pred_ids, pred_classes


@torch.no_grad()
def infer(model_paths, device, thr):
    data_module = LitDataModule(
        train_csv_path="train_preprocessed_3d.csv",
        test_csv_path="test_preprocessed_3d.csv",
        val_fold=0,
        batch_size=1,
        num_workers=0,
    )
    
    data_module.setup(stage="test")
    test_dataloader = data_module.test_dataloader()
    
    pred_strings = []
    pred_ids = []
    pred_classes = []

    for batch in tqdm(test_dataloader):
        images_3d = batch["image_3d"].to(device, dtype=torch.float)
        ids, height, width = batch["ids"][0], batch["height"][0], batch["width"][0]
        
        ids = eval(ids)
        height = int(height)
        width = int(width)

        size_3d = images_3d.size()
        masks_3d = torch.zeros((size_3d[0], 3, size_3d[2], size_3d[3], size_3d[4]), device=device, dtype=torch.float32)

        for path in model_paths:
            model = LitModule.load_eval_checkpoint(path, device=device)
            out_3d = model(images_3d)
            out_3d = torch.nn.Sigmoid()(out_3d)
            masks_3d += out_3d / len(model_paths)

        # Remove batch dim
        masks_3d = torch.squeeze(masks_3d) 
        
        # Resize to original shape
        spatial_size = (width, height, len(ids))
        resize_transform = monai.transforms.Resize(spatial_size=spatial_size, mode="nearest")
        masks_3d = resize_transform(masks_3d)
        
        # Use depth as batch dim
        masks = masks_3d.permute((3, 0, 1, 2))
            
        masks = (masks.permute((0, 2, 3, 1)) > thr).to(torch.uint8).cpu().detach().numpy()  # shape: (n, h, w, c)
        result = masks2rles(masks, ids, height, width)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])
        
    pred_df = pd.DataFrame({"id": pred_ids, "class": pred_classes, "predicted": pred_strings})

    return pred_df


# In[ ]:


model_paths = list((Path(trainer.logger.log_dir) / "checkpoints").glob("*.ckpt"))
model_paths


# In[ ]:


pred_df = infer(model_paths, DEVICE, THR)


# ## Submit

# In[ ]:


if not test_set_hidden:
    sub_df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv")
    del sub_df["predicted"]
else:
    sub_df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")[: 1000 * 3]
    del sub_df["segmentation"]

sub_df = sub_df.merge(pred_df, on=["id", "class"])
sub_df.to_csv("submission.csv", index=False)
display(sub_df.head(5))


# ## 
