#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pytorch_lightning efficientnet_pytorch colorednoise torchlibrosa audiomentations')


# In[ ]:


import sys
import random
import os
import math
import time

import numpy as np
import pandas as pd
import librosa
import librosa.display
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import label_ranking_average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from audiomentations import Compose, AddGaussianNoise, Gain, TimeStretch, PitchShift


# In[ ]:


def set_random_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[ ]:


BASE_INPUT_DIR = '/kaggle/input/'
TRAIN_INPUT_DIR = os.path.join(BASE_INPUT_DIR, 'rfcx-species-audio-detection/train')
TEST_INPUT_DIR = os.path.join(BASE_INPUT_DIR, 'rfcx-species-audio-detection/test')
OUTPUT_DIR = './output'

train_tp = pd.read_csv(os.path.join(BASE_INPUT_DIR, 'rfcx-species-audio-detection/train_tp.csv'))
submission = pd.read_csv(os.path.join(BASE_INPUT_DIR, 'rfcx-species-audio-detection/sample_submission.csv'))


# In[ ]:


sys.path.append(os.path.join(BASE_INPUT_DIR, 'panns-sed'))

from models import Cnn14_DecisionLevelAtt, AttBlock, init_layer
from pytorch_utils import move_data_to_device, do_mixup


# In[ ]:


PROJECT_ID = 'kaggle-moa-296003'
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)


# In[ ]:


bucket = storage_client.get_bucket('tn-kaggle-data')
blob = bucket.blob('panns_weights.zip')
blob.download_to_filename('./panns_weights.zip')
get_ipython().system('unzip ./panns_weights.zip')
get_ipython().system('rm panns_weights.zip')


# In[ ]:


# LRAP. Instance-level average
# Assume float preds [BxC], labels [BxC] of 0 or 1
def LRAP(preds, labels):
    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
    pos_matrix = torch.tensor(np.array([i+1 for i in range(labels.size(-1))])).unsqueeze(0)
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = (scores.sum(-1) / labels.sum(-1)).mean()
    return score.item()


# label-level average
# Assume float preds [BxC], labels [BxC] of 0 or 1
def LWLRAP(labels, preds):
    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
    # Number of GT labels per instance
    num_labels = labels.sum(-1)
    pos_matrix = torch.tensor(np.array([i+1 for i in range(labels.size(-1))]), device=DEVICE).unsqueeze(0)
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = scores.sum() / labels.sum()
    return score.item()


class ImprovedPANNsLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.normal_loss = nn.BCELoss()
        self.bce = nn.BCELoss()

    def forward(self, output_dict, target):
        return self.bce(output_dict['clipwise_output'], target)


# # Constants & Config

# In[ ]:


DEVICE= "cuda:0"

NUM_SPECIES = 24

IMG_SIZE = (224, 512, 3)
IMG_HEIGHT = IMG_SIZE[0]
IMG_WIDTH = IMG_SIZE[1]

FMIN = 40.0
FMAX = 24000.0

SR = 48000
FRAME_SIZE = 1024
HOP_LENGTH = 320
N_MELS = 64

CLIP_DURATION = 60
SEGMENT_DURATION = 10


# In[ ]:


cfg = {
    'preprocess': {
        'frame_size': FRAME_SIZE,
        'hop_length': HOP_LENGTH,
        'sub_segment_duration': 6,

        'do_mixup': True,
        'mixup_alpha': 0.2
    },
    'training': {
        'n_folds': 5,
        'batch_size': 24,
        'epochs': 100,
        'max_lr': 1e-4,
    },
    'model': {
        'base_model': 'cnn14_att',
    },
    'inference': {
        'segment_stride': 6
    }
}


# # Preprocessing

# In[ ]:


class RFCXDataset(torch.utils.data.Dataset):

    def __init__(self, samples, configs, is_train):
        self.samples = samples
        self.cfg = configs
        self.is_train = is_train
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5),
        ])
        
    def _load_audio(self, recording_id):
        filepath = os.path.join(TRAIN_INPUT_DIR, recording_id + '.flac')
        data, _ = librosa.load(filepath, sr=SR)
        return data
    
    def _cut_wav(self, audio_data, sample):
        tmin = sample['t_min']
        tmax = sample['t_max']
        sub_segment_duration = cfg['preprocess']['sub_segment_duration']

        if self.is_train:
            if tmax - tmin < sub_segment_duration:
                min_left = max(0.0, tmax - sub_segment_duration)
                max_left = min(tmin, CLIP_DURATION - sub_segment_duration)
            else:
                shrinkage = (tmax - tmin) - sub_segment_duration
                min_left = tmin
                max_left = tmin + shrinkage
            left_cut = np.random.uniform(low=min_left, high=max_left)
        else:
            if tmax - tmin < sub_segment_duration:
                extension = max(0.0, sub_segment_duration - (tmax - tmin))/2
                left_extend = extension
                if tmax + extension > CLIP_DURATION:
                    left_extend += tmax + extension - CLIP_DURATION
                left_cut = max(0.0, tmin - left_extend)
            else:
                shrinkage = (tmax - tmin) - sub_segment_duration
                left_cut = tmin + shrinkage/2

        left_cut_sample = int(np.floor(left_cut * SR))
        right_cut_sample = left_cut_sample + sub_segment_duration*SR
        cut = audio_data[left_cut_sample:right_cut_sample]
        assert len(cut) == sub_segment_duration*SR
        return cut
    
    def _one_hot(self, idx):
        target = np.zeros(NUM_SPECIES, dtype=np.float32)
        sparse_label = self.samples.species_id.iloc[idx]
        target[sparse_label] = 1
        return target

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx: int):
        sample = self.samples.iloc[idx, :]
        data = self._load_audio(sample.recording_id)
        wav, target = self._cut_wav(data, sample), self._one_hot(idx)
        if self.is_train:
            wav = self.augment(samples=wav, sample_rate=SR)
        return wav, target


# In[ ]:


class Mixup(object):

    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)


# # Model

# In[ ]:


def create_model(load_pretrained=True):
    with_mixup = cfg['preprocess']['do_mixup']
    args = {
        'sample_rate': SR,
        'window_size': FRAME_SIZE,
        'hop_size': HOP_LENGTH,
        'mel_bins': N_MELS,
        'fmin': FMIN, 
        'fmax': FMAX,
        'classes_num': 527,
    }
    base_model = Cnn14_DecisionLevelAtt(**args)
    if load_pretrained:
        checkpoint = torch.load(os.path.join(BASE_INPUT_DIR, 'panns-sed/Cnn14_DecisionLevelAtt_mAP=0.425.pth'))
        base_model.load_state_dict(checkpoint['model'])

    base_model.att_block = AttBlock(2048, NUM_SPECIES, activation="sigmoid")
    base_model.att_block.init_weights()
    init_layer(base_model.fc1)
    return base_model


# In[ ]:


class LightModel(LightningModule):

    def __init__(self, model, train_samples, val_samples):
        super().__init__()
        self.model = model
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.loss_fn = ImprovedPANNsLoss()
        self.mixup_augmenter = Mixup(cfg['preprocess']['mixup_alpha'])

    def forward(self, batch, mixup_lambda=None):
        return self.model(batch, mixup_lambda=mixup_lambda)
    
    def train_dataloader(self):
        batch_size = cfg['training']['batch_size']
        if cfg['preprocess']['do_mixup']:
            batch_size *= 2
        return DataLoader(
            RFCXDataset(self.train_samples, cfg, True),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            RFCXDataset(self.val_samples, cfg, False), 
            batch_size=cfg['training']['batch_size'],
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )

    def training_step(self, batch, batch_idx):
        mixup_lambda = None
        if cfg['preprocess']['do_mixup']:
            mixup_lambda = self.mixup_augmenter.get_lambda(batch_size=len(batch))
            mixup_lambda = move_data_to_device(mixup_lambda, DEVICE)
        y, target = [x.to(DEVICE) for x in batch] 
        target = do_mixup(target, mixup_lambda)
        output = model(y, mixup_lambda=mixup_lambda)
        bceLoss = self.loss_fn(output, target)
        loss = bceLoss
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y, target = [x.to(self.device) for x in batch]
        output = model(y)
        bceLoss = self.loss_fn(output, target)
        loss = bceLoss
        self.log('val_loss', loss, on_epoch=True)
        lwap = LWLRAP(target, output['clipwise_output'])
        self.log("LwAP", lwap, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['max_lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        lr_scheduler = {"scheduler": scheduler }
        return [optimizer], [lr_scheduler]


# # Inference

# In[ ]:


class RFCXTestDataset(torch.utils.data.Dataset):

    def __init__(self, samples, segmented=False):
        self.samples = samples
        self.segmented = segmented
        
    def load_test_data(self, recording_id):
        filepath = os.path.join(TEST_INPUT_DIR, recording_id + '.flac')
        data, _ = librosa.load(filepath, sr=SR)
        all_segments = []
        if self.segmented:
            for i in range(0, 55, cfg['inference']['segment_stride']):
                sample_min = i * SR
                sample_max = sample_min + cfg['preprocess']['sub_segment_duration']*SR
                segment = data[sample_min:sample_max]
                assert len(segment) == cfg['preprocess']['sub_segment_duration']*SR
                all_segments.append(segment)
        else:
            all_segments.append(data)
        return torch.tensor(all_segments)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx: int):
        sample = self.samples.iloc[idx, :]
        return sample.recording_id, self.load_test_data(sample.recording_id)


# In[ ]:


test_dataset = RFCXTestDataset(submission)
data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)


# In[ ]:


models_path = './output'
model_files = os.listdir(models_path)
file_by_fold = []
for i in range(cfg['training']['n_folds']):
    for f in model_files:
        if f.startswith(f'weights-{i}'):
            file_by_fold.append(f)
            break


# In[ ]:


submission_folds = []

for fold in range(5):
    fold_start = time.time()
    checkpoint_path = os.path.join(models_path, file_by_fold[fold])
    model = LightModel(create_model(load_pretrained=False), train_tp, train_tp).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE)["state_dict"])
    model.eval()
    fold_submission = submission.copy()
    fold_preds = []
    print(f'Load Fold {fold}. Elapsed:', round(time.time() - fold_start, 2))

    with torch.no_grad():
        start = time.time()
        for i, batch in enumerate(data_loader):
            for recording_id, inputs in zip(batch[0], batch[1]):
                inputs = inputs.to(DEVICE)
                segment_preds = model(inputs)["clipwise_output"]
                rec_pred = torch.max(segment_preds, dim=0).values.detach().cpu().numpy()
                fold_preds.append(rec_pred)
                print(recording_id, 'elapsed:', round(time.time() - start, 2))

    fold_submission.iloc[:, 1:] = np.stack(fold_preds)
    fold_submission.to_csv(f'fold_{fold}.csv', index=False)
    submission_folds.append(fold_submission)
    print('Fold', fold, 'Elapsed:', round(time.time() - fold_start, 2))


# In[ ]:


from scipy.stats.mstats import gmean

probs = np.stack([sf.iloc[:, 1:].values for sf in submission_folds])
submission.iloc[:, 1:] = gmean(probs, axis=0)
submission.to_csv('./submission.csv', index=False)

