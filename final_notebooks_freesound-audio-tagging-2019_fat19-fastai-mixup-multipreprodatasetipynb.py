#!/usr/bin/env python
# coding: utf-8

# # FAT2019 fast.ai implements with MixUp, MultiPreprocessed
# 
# Based from https://www.kaggle.com/vinayaks/2d-cnn-high-score-fast-ai
# 
# History:
# * V38: Modify version history
# * V37: Downgrade kaggle docker-container
# * V36: Rename title, and add some comments
# * V35: [PREDICTION] Final submission commit
# * V34: [LEARNING] Final training transferred V27 weights
# * V21: [LEARNING] Training with curated and noisy dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import IPython
import IPython.display
import PIL
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F

import librosa

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system(' ls ../input/fat2019_prep_mels1')


# In[ ]:


get_ipython().system(' ls ../input/fat2019-multipreprocessed-package')


# In[ ]:


get_ipython().system(' ls ../input/fat19-fastai-weights-of-mixup-mp')


# ## utils

# In[ ]:


# Fix result

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 999
seed_everything(SEED)


# In[ ]:


def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


# ## Mode definisions

# In[ ]:


TRAIN_MODE = False
CONTINUOUS_TRAIN = True
MIXMATCH_SSL = 0 # >= 0 (0 is mixup mode)


# ## File/folder definitions
# 
# - `df_train` will handle training data.
# - `df_test` will handle test data.

# In[ ]:


DATA = Path('../input/freesound-audio-tagging-2019')
PREPROCESSED_N1K = Path('../input/fat2019_prep_mels1')
PREPROCESSED_MP = Path('../input/fat2019-multipreprocessed-package')
LAST_WEIGHTS = Path('../input/fat19-fastai-weights-of-mixup-mp')
WORK = Path('work')
Path(WORK).mkdir(exist_ok=True, parents=True)

CSV_TRAIN_CURATED = PREPROCESSED_MP/'train_curated_valid.csv'
CSV_TRAIN_NOISY = PREPROCESSED_MP/'train_noisy_valid.csv'
CSV_TRAIN_NOISY_BEST50S = PREPROCESSED_N1K/'trn_noisy_best50s.csv'
CSV_SUBMISSION = DATA/'sample_submission.csv'

SAVED_WEIGHT = LAST_WEIGHTS/'last'
DEPLOYED_MODEL = LAST_WEIGHTS/'export.pkl'

def add_prefix_path(df, prefix):
    fnames = df.fname.copy()
    df.fname = prefix + os.sep
    df.fname = df.fname.str.cat(fnames)
    return df
def filter_df(df):
    few_labels = list(map(lambda p: len(p[1]) <= 3, df.labels.str.split(',').iteritems()))
    return df[few_labels]

df_curated = add_prefix_path(pd.read_csv(CSV_TRAIN_CURATED), 'train_curated')
df_noisy = add_prefix_path(pd.read_csv(CSV_TRAIN_NOISY), 'train_noisy')
# df_noisy = filter_df(df_noisy) # filtering fewer labels than 3 or equal it
df_noisy50s = add_prefix_path(pd.read_csv(CSV_TRAIN_NOISY_BEST50S), 'train_noisy')
df_submission = pd.read_csv(CSV_SUBMISSION)
df_test = add_prefix_path(df_submission.copy(), 'test')

df_train = df_curated
# df_train = pd.concat([df_curated, df_noisy], ignore_index=True)
# df_train = pd.concat([df_curated, df_noisy50s], ignore_index=True, sort=True)
print('Training dataset size:', len(df_train))
df_train.head()


# ## Masking filters

# In[ ]:


# post augmentations
USE_MASK_FREQ       = True
MASK_FREQ_RANGE     = 8   #[mels]
MASK_FREQ_MAX_COUNT = 3

USE_MASK_TIME       = True
MASK_TIME_RANGE     = 8   #[frames]
MASK_TIME_MAX_COUNT = 3

def freq_mask(x, num=1, mask_size=10, mask_value=None, inplace=False):
    cloned = x.clone() if not inplace else x
    num_bins = cloned.shape[1]
    mask_value = cloned.mean() if mask_value is None else mask_value
    for i in range(num):
        f = random.randrange(0, mask_size + 1)
        if f == 0 or f >= num_bins: continue

        f_low = random.randrange(0, num_bins - f)
        f_high = f_low + f
        cloned[:, f_low:f_high] = mask_value
    return cloned

def time_mask(x, num=1, mask_size=10, mask_value=None, inplace=False):
    cloned = x.clone() if not inplace else x
    num_frames = cloned.shape[2]
    mask_value = cloned.mean() if mask_value is None else mask_value
    for i in range(num):
        t = random.randrange(0, mask_size + 1)
        if t == 0 or t >= num_frames: continue

        t_beg = random.randrange(0, num_frames - t)
        t_end = t_beg + t
        cloned[:, :, t_beg:t_end] = mask_value
    return cloned


# ## Preprocessing for PCM data

# In[ ]:


# Data processing configuration
class AugmentationConfig:
    padding_scale = 1.
    whitenoise = True
    whitenoise_level = 1e-3 # 0.~1.
    pitchshift = True
    pitchshift_steps = 2. # steps (12/oct)

class PreproConfig:
    sr = 44100
    duration = 2. # secs
    n_out = 128
    n_mels = 128
    n_fft = n_mels * 20
    
    hop_len = int(sr * duration // n_out)
    sample_size = int(sr * duration)
    padding_size = int(sample_size * AugmentationConfig.padding_scale)

# preprocessor
def load_audio(file, eps=1e-6):
    pcm, sr = librosa.load(file, sr=PreproConfig.sr)
    if len(pcm) <= 0:
        # raise 'No audio samples'
        pcm = np.zeros(PreproConfig.sample_size) + 1e-8
        sr = PreproConfig.sr

    pcm_max = np.max(np.abs(pcm))
    if pcm_max > eps:
        pcm = pcm / pcm_max

    min_size = PreproConfig.padding_size
    if len(pcm) > min_size: pcm, _ = librosa.effects.trim(pcm)
    pcm = pad_zeros(pcm, min_size)
    return pcm, pcm_max

def pad_zeros(x, size):
    if len(x) > size: # long enough
        return x
    else: # pad blank
        padding = size - len(x)
        offset = padding // 2
        return np.pad(x, (offset, size - len(x) - offset), 'constant')

def conv_pcm_to_magphs(x):
    fft = librosa.core.stft(x, n_fft=PreproConfig.n_fft, hop_length=PreproConfig.hop_len)
    return np.abs(fft), np.angle(fft)

def conv_magphs_to_mels(magphs):
    mel_basis = librosa.filters.mel(PreproConfig.sr, PreproConfig.n_fft, n_mels=PreproConfig.n_mels)
    return np.dot(mel_basis, magphs)

def conv_mag_to_melspec(mag):
    mels = conv_magphs_to_mels(mag ** 2.)
    mels = librosa.power_to_db(mels)
    m_max = mels.max()
    m_min = mels.min()
    return (mels - m_min) / (m_max - m_min)

def normalize_melspec(melspec, eps=1e-6, dtype=np.uint8):
    mean = melspec.mean()
    std = melspec.std()
    ms_std = (melspec - mean) / (std + eps)
    ms_min, ms_max = ms_std.min(), ms_std.max()
    if (ms_max - ms_min) > eps:
        ms_norm = 255. * (ms_std - ms_min) / (ms_max - ms_min)
        return ms_norm.astype(dtype)
    else:
        return np.zeros_like(ms_std, dtype=dtype)

def preprocess(pcm, debug=False, dtype=np.uint8):
    melspec, melphss = conv_pcm_to_magphs(pcm)
    melspec = conv_mag_to_melspec(melspec)
    melspec = normalize_melspec(melspec, dtype=dtype)
    return melspec


# Loader for MultiPreprocessed dataset

# In[ ]:


# multi-preprocessed data loader
class MPLoader:
    cache = dict()
    cache_vmem_percent = 45.0
    
    data_type = 'msp' # msp, mph, mfcc
    
    use_augmentation = True
    max_of_augid = 2
    
    def reset():
        MPLoader.cache = dict()
    
    def get(fname, augmentation=False, cache=False):
        if fname in MPLoader.cache:
            data = MPLoader.cache[fname]
        else:
            data = MPLoader.load(fname, cache)

        if augmentation and MPLoader.use_augmentation:
            return random.choice(data)
        else:
            return data[0]

    def load(fname, cache=False):
        data_raw = np.load((PREPROCESSED_MP/fname).with_suffix('.wav.npz'))
        data = list()

        if MPLoader.use_augmentation:
            for i in range(1, MPLoader.max_of_augid+1):
                tid = f'{MPLoader.data_type}{i}'
                if tid in data_raw:
                    data.append(MPLoader._mp2tensor(data_raw[tid]))
                else:
                    break
        else:
            tid = f'{MPLoader.data_type}1'
            data.append(MPLoader._mp2tensor(data_raw[tid]))

        if cache: MPLoader.cache[fname] = data
        return data

    def load_and_process(fname, cache=False):
        pcm, _ = load_audio(DATA/fname)
        mels = preprocess(pcm)
        data = [MPLoader._mp2tensor(mels)]
        if cache: MPLoader.cache[fname] = data
        return data
    
    def _mp2tensor(mp):
        mp = torch.FloatTensor(mp).div_(255)
        return mp
    
    def full_load(df, use_preprocess=True):
        vmem_limit = MPLoader.cache_vmem_percent
        virt = psutil.virtual_memory()
        if virt.percent >= vmem_limit:
            vmem_limit = virt.percent + 0.1
        
        for i, row in tqdm_notebook(df.iterrows(), total=len(df)):
            if use_preprocess:
                MPLoader.load(row.fname, cache=True)
            else:
                MPLoader.load_and_process(row.fname, cache=True)
            if i % 10 == 0:
                virt = psutil.virtual_memory()
                if virt.percent >= vmem_limit:
                    mb = 1024 * 1024
                    print(f'Stopped loading preprocessed data as cache, '
                          f'cause memory usage reached {vmem_limit}%.')
                    print('Memory usage:', virt.available>>20, '/', virt.total>>20, 'MB')
                    break
                elif i % 1000 == 0:
                    print(virt.percent, '%', virt.available>>20, 'MB')
        print('Completed loading preprocessed data.')

if TRAIN_MODE:
    MPLoader.full_load(df_train)
    if MIXMATCH_SSL > 0:
        MPLoader.full_load(df_noisy)


# In[ ]:


plt.figure(figsize=(8,4))
plt.imshow(MPLoader.get(df_train.fname[0]).numpy(), origin='lower')
plt.title(df_train.labels[0])
plt.show()


# ## Custom `open_image` for fast.ai library to load data from memory
# 
# - Important note: Random cropping 1 sec, this is working like augmentation.

# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.vision.data import *
from fastai.callbacks import *
import random

TIME_DIM = 128

def open_fat2019_image(fn, convert_mode, after_open)->Image:
    # open
    fname = '/'.join(fn.split('/')[-2:])
    x = MPLoader.get(fname, augmentation=True)
    # crop 2sec
    base_dim, time_dim = x.shape
    if time_dim < TIME_DIM:
        x2 = torch.zeros((base_dim,TIME_DIM), dtype=x.dtype)
        crop = random.randint(0, TIME_DIM - time_dim)
        x2[:, crop:crop+time_dim] = x
        x = x2
    else:
        crop = random.randint(0, time_dim - TIME_DIM)
        x = x[:, crop:crop+TIME_DIM]
    x = torch.stack((x,x,x), dim=0)
    # masking
    if USE_MASK_FREQ: freq_mask(
        x,
        num=random.randrange(MASK_FREQ_MAX_COUNT),
        mask_size=MASK_FREQ_RANGE,
        mask_value=0,
        inplace=True,
    )
    if USE_MASK_TIME: time_mask(
        x,
        num=random.randrange(MASK_TIME_MAX_COUNT),
        mask_size=MASK_TIME_RANGE,
        mask_value=0,
        inplace=True,
    )
    # standardize
    return Image(x)

vision.data.open_image = open_fat2019_image


# ## Follow multi-label classification
# 
# - Almost following fast.ai course: https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-planet.ipynb
# - But `pretrained=False`

# In[ ]:


BATCH_SIZE = 48 # used 3-times CUDA memory

tfms = get_transforms(do_flip=True, max_rotate=0, max_lighting=0.1, max_zoom=0, max_warp=0.)
# src = (ImageList.from_csv(WORK, Path('..')/CSV_TRAIN_CURATED, folder='train_curated')
src = (ImageList.from_df(df_train, WORK, folder='')
#        .split_by_rand_pct(0.1, seed=SEED) # for Validation
       .split_none()
       .label_from_df(label_delim=',')
      )
data = (src.transform(tfms, size=128)
        .databunch(bs=BATCH_SIZE).normalize(imagenet_stats)
       )

if MIXMATCH_SSL > 0:
    noisy_src = (ImageList.from_df(df_noisy, WORK, folder='')
                 .split_none()
                 .label_from_df(label_delim=',')
                )
    noisy_data = (noisy_src.transform(tfms, size=128)
                  .databunch(bs=BATCH_SIZE).normalize(imagenet_stats)
                 )


# In[ ]:


if TRAIN_MODE: data.show_batch(3)


# In[ ]:


def lwlrap(y_pred,y_true):
    score, weight = calculate_per_class_lwlrap(y_true.cpu().numpy(), y_pred.cpu().numpy())
    lwlrap = (score * weight).sum()
    return torch.from_numpy(np.array(lwlrap))


# MixMatch is **INCOMPLETE**! This is useless in FAT2019.

# In[ ]:


# Customized MixMatch for multi-label
# referenced:
#  - https://github.com/perrying/realistic-ssl-evaluation-pytorch
#  - https://github.com/YU1ut/MixMatch-pytorch
class MixMatchCallback(LearnerCallback):
    def __init__(self, learn:Learner,
                 unlabeled_dl:DeviceDataLoader,
                 temperature:float=0.5, n_augment:int=2,
                 alpha:float=0.75, lambda_u:float=100, rampup:int=16):
        super().__init__(learn)
        self.unlabeled_dl = unlabeled_dl
        self.T = temperature
        self.K = n_augment
        self.beta_distirb = torch.distributions.beta.Beta(alpha, alpha)
        self.lambda_u = lambda_u
        self.rampup = rampup
        self.n_iterations = len(learn.data.train_dl)

    def on_train_begin(self, **kwargs):
        self.unlabeled_iter = iter(self.unlabeled_dl)
        self.learn.loss_func = MultiSemiLoss(self.learn.loss_func, lambda_u=self.lambda_u, rampup=self.rampup)

    def on_train_end(self, **kwargs):
        self.unlabeled_iter = None
        self.learn.loss_func = self.learn.loss_func.get_old()
    
    def unlabeled_next_batch(self):
        try:
            return next(self.unlabeled_iter)
        except:
            self.unlabeled_iter = iter(self.unlabeled_dl)
            return next(self.unlabeled_iter)

    def sharpen(self, y):
        y = y.pow(1 / self.T)
        return y / y.sum(dim=1, keepdim=True)

    def on_batch_begin(self, epoch, iteration, last_input, last_target, train, **kwargs):
        if not train: return
        with torch.no_grad():
            bs = len(last_input)
            
            # K augmentation and make prediction labels
            u_x_hat = [self.unlabeled_next_batch()[0] for _ in range(self.K)]
#             u_x_hat = [u_x for _ in range(self.K)]
            y_hat = sum([self.learn.model(u_x_hat[i]).sigmoid_() for i in range(self.K)]) / self.K
            y_hat = self.sharpen(y_hat)
            y_hat = y_hat.repeat(self.K, 1)
            
            # mixup
            u_x_hat = torch.cat([last_input] + u_x_hat, dim=0)
            y_hat = torch.cat((last_target, y_hat), dim=0)
            index = torch.randperm(u_x_hat.shape[0])
            
            lam = self.beta_distirb.sample().item()
            lam = max(lam, 1-lam)
            
            mixed_input = lam * u_x_hat + (1-lam) * u_x_hat[index]
            mixed_input = self.interleave(torch.split(mixed_input, bs), bs)
            mixed_target = lam * y_hat + (1-lam) * y_hat[index]
            
            loss_func = self.learn.loss_func
            loss_func.labeled_bs = bs
            loss_func.epoch = iteration / self.n_iterations
        return {'last_input': mixed_input, 'last_target': mixed_target}

    def on_loss_begin(self, last_input, last_output, iteration, **kwargs):
        bs = len(last_input)
        last_output = self.interleave(torch.split(last_output, bs), bs)
        return {'last_output': last_output}

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return torch.cat([torch.cat(v, dim=0) for v in xy], dim=0)

MM_LOG = pd.DataFrame(columns=['loss', 'loss_labeled', 'loss_unlabeled', 'weight'])

class MultiSemiLoss(nn.Module):
    "Adapt the loss function `crit` to go with mixmatch."
    
    def __init__(self, crit, lambda_u=100.0, rampup=16):
        super().__init__()
        self.crit = crit
        self.labeled_bs = 0
        self.lambda_u = lambda_u
        self.rampup = rampup
        self.epoch = 0
        
    def forward(self, output, target):
        if self.labeled_bs == 0: return self.crit(output, target)
        
        output_x, target_x = output[:self.labeled_bs], target[:self.labeled_bs]
        output_u, target_u = output[self.labeled_bs:], target[self.labeled_bs:]
        
        q = output_u.sigmoid()
        
        Lx = self.crit(output_x, target_x)
        Lu = torch.mean((q - target_u) ** 2)
        
        w = self.lambda_u * (np.clip(self.epoch / self.rampup, 0., 1.) if self.rampup > 0 else 1.)
        
        self.labeled_bs = 0
        d = Lx + w * Lu
        MM_LOG.loc[self.epoch] = [float(d), float(Lx), float(Lu), w]
        return d
    
    def get_old(self):
        return self.crit

def _mixmatch(learn:Learner, unlabeled_data:ImageDataBunch,
              temperature:float=0.5, n_augment:int=2,
              alpha:float=0.75, lambda_u:float=100, rampup:int=16) -> Learner:
    "Add mixup https://arxiv.org/abs/1905.02249 to `learn`."
    unlabeled_dl = unlabeled_data.dl(DatasetType.Train)
    learn.callback_fns.append(partial(MixMatchCallback,
                                      unlabeled_dl=unlabeled_dl,
                                      temperature=temperature, n_augment=n_augment,
                                      alpha=alpha, lambda_u=lambda_u, rampup=rampup))
    return learn
Learner.mixmatch = _mixmatch


# In[ ]:


class MixUpCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True):
        super().__init__(learn)
        self.alpha,self.stack_x,self.stack_y = alpha,stack_x,stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        if self.stack_x:
            new_input = [last_input, last_input[shuffle], lambd]
        else: 
            new_input = (last_input * lambd.view(lambd.size(0),1,1,1) + x1 * (1-lambd).view(lambd.size(0),1,1,1))
        if self.stack_y:
            new_target = torch.cat([last_target[:,None].float(), y1[:,None].float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + y1.float() * (1-lambd)
        return {'last_input': new_input, 'last_target': new_target}  
    
    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()

class MixUpLoss(nn.Module):
    "Adapt the loss function `crit` to go with mixup."
    
    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'): 
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        else: 
            self.crit = partial(crit, reduction='none')
            self.old_crit = crit
        self.reduction = reduction
        
    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = self.crit(output,target[:,0].long()), self.crit(output,target[:,1].long())
            d = (loss1 * target[:,2] + loss2 * (1-target[:,2])).mean()
        else:  d = self.crit(output, target)
        if self.reduction == 'mean': return d.mean()
        elif self.reduction == 'sum': return d.sum()
        return d
    
    def get_old(self):
        if hasattr(self, 'old_crit'):  return self.old_crit
        elif hasattr(self, 'old_red'): 
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit

def _mixup(learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True) -> Learner:
    "Add mixup https://arxiv.org/abs/1710.09412 to `learn`."
    learn.callback_fns.append(partial(MixUpCallback, alpha=alpha, stack_x=stack_x, stack_y=stack_y))
    return learn
Learner.mixup = _mixup


# In[ ]:


from fastai.core import *
from fastai.basic_data import *
from fastai.basic_train import *
from fastai.torch_core import *
def _tta_only(learn:Learner, ds_type:DatasetType=DatasetType.Valid, num_pred:int=5) -> Iterator[List[Tensor]]:
    "Computes the outputs for several augmented inputs for TTA"
    dl = learn.dl(ds_type)
    ds = dl.dataset
    old = ds.tfms
    aug_tfms = [o for o in learn.data.train_ds.tfms]
    try:
        pbar = master_bar(range(num_pred))
        for i in pbar:
            ds.tfms = aug_tfms
            yield get_preds(learn.model, dl, pbar=pbar)[0]
    finally: ds.tfms = old

Learner.tta_only = _tta_only

def _TTA(learn:Learner, beta:float=0, ds_type:DatasetType=DatasetType.Valid, num_pred:int=5, with_loss:bool=False) -> Tensors:
    "Applies TTA to predict on `ds_type` dataset."
    preds,y = learn.get_preds(ds_type)
    all_preds = list(learn.tta_only(ds_type=ds_type, num_pred=num_pred))
    avg_preds = torch.stack(all_preds).mean(0)
    if beta is None: return preds,avg_preds,y
    else:            
        final_preds = preds*beta + avg_preds*(1-beta)
        if with_loss:
            with NoneReduceOnCPU(learn.loss_func) as lf: loss = lf(final_preds, y)
            return final_preds, y, loss
        return final_preds, y

Learner.TTA = _TTA


# In[ ]:


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x
    
class Classifier(nn.Module):
    def __init__(self, num_classes=1000): # <======== modificaition to comply fast.ai
        super().__init__()
        
        self.conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=512),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 1)) # <======== modificaition to comply fast.ai
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*4, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        #x = torch.mean(x, dim=3)   # <======== modificaition to comply fast.ai
        #x, _ = torch.max(x, dim=2) # <======== modificaition to comply fast.ai
        x = self.avgpool(x)         # <======== modificaition to comply fast.ai
        x = self.fc(x)
        return x


# In[ ]:


labels = df_submission.columns[1:].tolist()
label_size = len(labels)

def calc_P_R_AP(y_true, y_pred):
    P = [None] * label_size
    R = [None] * label_size
    AP = np.zeros(label_size)
    for i in range(label_size):
        P[i], R[i], _ = precision_recall_curve(y_true[:,i], y_pred[:,i])
        AP[i] = average_precision_score(y_true[:,i], y_pred[:,i])

    df_ap = pd.DataFrame(data=AP, index=labels, columns=['AP'])
    return P, R, AP, df_ap


# In[ ]:


def normalize_predict(y):
    min_pred = y.min(axis=1).reshape(-1,1)
    max_pred = y.max(axis=1).reshape(-1,1)
    return (y - min_pred) / (max_pred - min_pred)


# In[ ]:


def borrowed_model(pretrained=False, **kwargs):
    return Classifier(**kwargs)

if TRAIN_MODE:
    f_score = partial(fbeta, thresh=0.2)
    learn = cnn_learner(
        data,
        borrowed_model, pretrained=False,
        metrics=[lwlrap],
        loss_func=nn.MultiLabelSoftMarginLoss()
    )
    if MIXMATCH_SSL > 0:
        if CONTINUOUS_TRAIN:
            learn.mixmatch(noisy_data, temperature=0.5, n_augment=MIXMATCH_SSL, alpha=0.4, lambda_u=50., rampup=0)
        else:
#             learn.mixmatch(noisy_data, temperature=0.5, n_augment=MIXMATCH_SSL, alpha=0.4, lambda_u=100., rampup=16)
            learn.mixmatch(noisy_data, temperature=0.5, n_augment=MIXMATCH_SSL, alpha=0.4, lambda_u=10., rampup=1000)
    else:
        learn.mixup(alpha=0.4, stack_y=False)
    if CONTINUOUS_TRAIN: learn.load(Path('../..')/SAVED_WEIGHT)
    learn.unfreeze()


# In[ ]:


# Weights
if TRAIN_MODE and CONTINUOUS_TRAIN:
    df_ap = pd.read_csv(LAST_WEIGHTS/'labels_ap.csv', index_col=0)
#     y_pred, y_true = learn.TTA(ds_type=DatasetType.Valid, num_pred=25)
#     y_pred = normalize_predict(y_pred.cpu().numpy())
#     _, _, _, df_ap = calc_P_R_AP(y_true.numpy(), y_pred)
#     fig = plt.figure(figsize=(8,11))
#     ax = fig.add_subplot(1,1,1)
#     df_ap.sort_values('AP').plot.barh(
#         title='Average Precision per Labels',
#         grid=True, legend=False, xlim=(0.,1.), ax=ax,
#     )
#     plt.tight_layout()
#     plt.show()
    
    loss_weights = torch.FloatTensor((1/df_ap.AP).values ** 4).cuda()
    print(loss_weights)
    learn.loss_func = nn.MultiLabelSoftMarginLoss(weight=loss_weights)
else:
    loss_weights = None


# In[ ]:


if TRAIN_MODE:
    learn.lr_find()
    learn.recorder.plot(suggestion=True)


# In[ ]:


if TRAIN_MODE:
    gc.collect() # for GPU memory releasing when interrupted in training
    callbacks = [
        SaveModelCallback(learn, every='improvement', monitor='lwlrap', name='best'),
    ]
    if CONTINUOUS_TRAIN:
        learn.fit_one_cycle(50, slice(1e-7,1e-4), callbacks=callbacks)
    else:
        learn.fit_one_cycle(300, 2e-2, callbacks=callbacks)


# In[ ]:


# if TRAIN_MODE:
# #     gc.collect()
#     MM_LOG.plot(logy=True)
#     plt.show()


# In[ ]:


USE_MASK_FREQ = USE_MASK_TIME = False
MPLoader.use_augmentation = False

if TRAIN_MODE:
    y_pred, y_true = learn.get_preds(ds_type=DatasetType.Train)
    y_pred = torch.from_numpy(normalize_predict(y_pred.cpu().numpy()))
    print('Local LwLRAP score (Train dataset):', float(lwlrap(y_pred, y_true).float()))


# In[ ]:


if TRAIN_MODE:
    learn.save('last')
    learn.export()


# ## Validate and Analyze model

# In[ ]:


if TRAIN_MODE:
    from sklearn.metrics import *
    
    y_true2 = y_true.numpy()
    y_pred2 = y_pred.numpy()
    
    labels = df_submission.columns[1:].tolist()
    label_size = len(labels)
    
    P = [None] * label_size
    R = [None] * label_size
    AP = np.zeros(label_size)
    for i in range(label_size):
        P[i], R[i], _ = precision_recall_curve(y_true2[:,i], y_pred2[:,i])
        AP[i] = average_precision_score(y_true2[:,i], y_pred2[:,i])
    df_ap = pd.DataFrame(data=AP, index=labels, columns=['AP'])
    df_ap.to_csv('labels_ap.csv')
    
    P_micro, R_micro, _ = precision_recall_curve(y_true2.ravel(), y_pred2.ravel())
    AP_micro = average_precision_score(y_true2, y_pred2, average='micro')

    fig = plt.figure(figsize=(8,11))
    ax = fig.add_subplot(1,1,1)
    df_ap.sort_values('AP').plot.barh(
        title='Average Precision per Labels (μAP=%.5f)' % AP_micro,
        grid=True, legend=False, xlim=(0.,1.), ax=ax,
    )
    plt.tight_layout()
    plt.show()


# In[ ]:


if TRAIN_MODE:
    from itertools import cycle
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(12, 9))
    f_scores = np.linspace(0.2, 0.9, num=7)

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    l, = plt.plot(R_micro, P_micro, color='gold', lw=2)

    for i, color in zip(range(label_size), colors):
        l, = plt.plot(R[i], P[i], color=color, lw=2)

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')

    plt.show()


# ## Test prediction and making submission file simple
# - Switch to test data.
# - Overwrite results to sample submission; simple way to prepare submission file.

# In[ ]:


MPLoader.reset()
MPLoader.full_load(df_test, use_preprocess=False)


# In[ ]:


USE_MASK_FREQ = USE_MASK_TIME = False

test = ImageList.from_df(df_test, WORK, folder='')
learn = load_learner(WORK, test=test) if TRAIN_MODE else load_learner('.', DEPLOYED_MODEL, test=test)
preds, _ = learn.TTA(ds_type=DatasetType.Test, num_pred=50)


# In[ ]:


df_submission[learn.data.classes] = preds
df_submission.to_csv('submission.csv', index=False)
df_submission.head()


# In[ ]:


test_val = normalize_predict(df_submission[df_submission.columns[1:]].head(240).values).T
plt.figure(figsize=(16,4))
plt.title('Prediction map of Test dataset')
plt.imshow(np.where(test_val > 1, test_val, test_val))
plt.xlabel('Samples')
plt.ylabel('Tags')
plt.colorbar()
plt.show()


# ## GradCAM

# In[ ]:


from fastai.callbacks.hooks import *

def visualize_cnn_by_cam(learn, data_index):
    x, _y = learn.data.train_ds[data_index]
    y = _y.data
    if not isinstance(y, (list, np.ndarray)): # single label -> one hot encoding
        y = np.eye(learn.data.train_ds.c)[y]

    m = learn.model.eval()
    xb,_ = learn.data.one_item(x)
    xb_im = Image(learn.data.denorm(xb)[0])
    xb = xb.cuda()

    def hooked_backward(cat):
        with hook_output(m[0]) as hook_a:
            with hook_output(m[0], grad=True) as hook_g:
                preds = m(xb)
                preds[0,int(cat)].backward()
        return hook_a,hook_g
    def show_heatmap(img, hm, label):
        img_ch, img_w, img_h = img.data.shape
        
        _,axs = plt.subplots(1, 2)
        axs = axs.flat
        axs[0].set_title(label, size=10)
        img.show(ax=axs[0])
        axs[1].set_title(f'CAM of {label}', size=10)
        
        # convert conv heatmap resolution to image
        hm = (hm - hm.min()) / (hm.max() - hm.min()) * 255
        hm = hm.numpy().astype(np.uint8)
        hm = PIL.Image.fromarray(hm)
        hm = hm.resize((img_w, img_h), PIL.Image.ANTIALIAS)
        hm = np.uint8(hm) / 255
        img.show(ax=axs[1])
        axs[1].imshow(hm, alpha=0.6, cmap='magma');
        plt.show()

    for y_i in np.where(y > 0)[0]:
        hook_a,hook_g = hooked_backward(cat=y_i)
        acts = hook_a.stored[0].cpu()
        grad = hook_g.stored[0][0].cpu()
        grad_chan = grad.mean(1).mean(1)
        mult = (acts*grad_chan[...,None,None]).mean(0)
        show_heatmap(img=xb_im, hm=mult, label=str(learn.data.train_ds.y[data_index]))

if TRAIN_MODE:
    learn = cnn_learner(data, borrowed_model, pretrained=False, metrics=[lwlrap]).mixup(stack_y=False)
    if TRAIN_MODE:
        learn.load('last')
    else:
        learn.load(Path('../..')/SAVED_WEIGHT)

    for idx in range(10):
        visualize_cnn_by_cam(learn, idx)


# ## Top Losses

# In[ ]:


def plot_multi_sound_top_losses(interp, df, samples:int=3, figsize:Tuple[int, int]=(8, 8)):
    losses, idxs = interp.top_losses()
    l_dim = len(losses.size())
    infolist, ordlosses_idxs, mismatches_idxs, mismatches, losses_mismatches, mismatchescontainer = [],[],[],[],[],[]
    truthlabels = np.asarray(interp.y_true, dtype=int)
    classes_ids = [k for k in enumerate(interp.data.classes)]
    predclass = np.asarray(interp.pred_class)
    for i,pred in enumerate(predclass):
        where_truth = np.nonzero((truthlabels[i]>0))[0]
        mismatch = np.all(pred!=where_truth)
        if mismatch:
            mismatches_idxs.append(i)
            if l_dim > 1 : losses_mismatches.append((losses[i][pred], i))
            else: losses_mismatches.append((losses[i], i))
        if l_dim > 1: infotup = (i, pred, where_truth, losses[i][pred], np.round(interp.probs[i], decimals=3)[pred], mismatch)
        else: infotup = (i, pred, where_truth, losses[i], np.round(interp.probs[i], decimals=3)[pred], mismatch)
        infolist.append(infotup)
    ds = interp.data.dl(interp.ds_type).dataset
    mismatches = ds[mismatches_idxs]
    ordlosses = sorted(losses_mismatches, key = lambda x: x[0], reverse=True)
    for w in ordlosses: ordlosses_idxs.append(w[1])
    mismatches_ordered_byloss = ds[ordlosses_idxs]
    print(f'{str(len(mismatches))} misclassified samples over {str(len(interp.data.valid_ds))} samples in the validation set.')
    for ima in range(len(mismatches_ordered_byloss)):
        mismatchescontainer.append(mismatches_ordered_byloss[ima][0])
    for sampleN in range(samples):
        actualclasses = ''
#         print(ordlosses[sampleN])
#         print(ordlosses_idxs[sampleN])
#         print(mismatches_ordered_byloss[sampleN][1])
#         print(df.iloc[ordlosses_idxs[sampleN]])
#         print(classes_ids[infolist[ordlosses_idxs[sampleN]][1]])
        pred_samples = df[df['labels'] == classes_ids[infolist[ordlosses_idxs[sampleN]][1]][1]]
        for clas in infolist[ordlosses_idxs[sampleN]][2]:
            actualclasses = f'{actualclasses} -- {str(classes_ids[clas][1])}'
        imag = mismatches_ordered_byloss[sampleN][0]
        plt.figure(figsize=(12,6))
        ax = plt.subplot(1,3,1)
        imag = show_image(imag, ax=ax, figsize=figsize)
        imag.set_title(f"""Predicted: {classes_ids[infolist[ordlosses_idxs[sampleN]][1]][1]} \nActual: {actualclasses}\nLoss: {infolist[ordlosses_idxs[sampleN]][3]}\nProbability: {infolist[ordlosses_idxs[sampleN]][4]}""",
                        loc='left')
        if len(pred_samples) > 0:
            ax = plt.subplot(1,3,2)
            ax.axis('off')
            ax.imshow(MPLoader.get(pred_samples.iloc[0].fname)[:,0:0+128], cmap='gray')
            plt.title(f"{classes_ids[infolist[ordlosses_idxs[sampleN]][1]][1]} #1")
            if len(pred_samples) > 1:
                ax = plt.subplot(1,3,3)
                ax.axis('off')
                ax.imshow(MPLoader.get(pred_samples.iloc[1].fname)[:,0:0+128], cmap='gray')
                plt.title(f"{classes_ids[infolist[ordlosses_idxs[sampleN]][1]][1]} #2")

        plt.tight_layout()
        plt.show()

if TRAIN_MODE:
    interp = learn.interpret(ds_type=DatasetType.Train)
    plot_multi_sound_top_losses(interp, df_curated, 10, figsize=(4,4))

