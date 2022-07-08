#!/usr/bin/env python
# coding: utf-8

# # About this notebook
# 
# This is fully based on the work below and adding torch_xla to it to trian on TPU's
# - [Luke](https://arxiv.org/pdf/2010.01057v1.pdf)-base starter notebook https://www.kaggle.com/yasufuminakama/jigsaw4-luke-base-starter-train from nakama
# - Approach References
#     - https://www.kaggle.com/tanlikesmath/xlm-roberta-pytorch-xla-tpu#Training
#     - Thanks for sharing nakama ,@debarshichanda and @nbroad 
#     
#  Also just ran this in **debug** mode presently just to do a quick test . 

# In[ ]:


# for TPU
get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev')


# # Directory settings

# In[ ]:


# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# # CFG

# In[ ]:


# ====================================================
# CFG
# ====================================================
class CFG:
    competition='Jigsaw4'
    _wandb_kernel='gauravbrills'
    debug=True
    apex=True
    print_freq=50
    num_workers=8
    model="studio-ousia/luke-base"
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=3
    encoder_lr=1e-5
    decoder_lr=1e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=64 #64
    fc_dropout=0.
    text="text"
    target="target"
    target_size=1
    head=32
    tail=32
    max_len=head+tail
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    margin=0.5
    seed=2021
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=True


# # Library

# In[ ]:


# ====================================================
# Library
# ====================================================
import os
import gc
import re
import sys
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

os.system('pip uninstall -q transformers -y')
os.system('pip uninstall -q tokenizers -y')
os.system('pip uninstall -q huggingface_hub -y')

os.system('mkdir -p /tmp/pip/cache-tokenizers/')
os.system('cp ../input/tokenizers-0103/tokenizers-0.10.3-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl /tmp/pip/cache-tokenizers/')
os.system('pip install -q --no-index --find-links /tmp/pip/cache-tokenizers/ tokenizers')

os.system('mkdir -p /tmp/pip/cache-huggingface-hub/')
os.system('cp ../input/huggingface-hub-008/huggingface_hub-0.0.8-py3-none-any.whl /tmp/pip/cache-huggingface-hub/')
os.system('pip install -q --no-index --find-links /tmp/pip/cache-huggingface-hub/ huggingface_hub')

os.system('mkdir -p /tmp/pip/cache-transformers/')
os.system('cp ../input/transformers-470/transformers-4.7.0-py3-none-any.whl /tmp/pip/cache-transformers/')
os.system('pip install -q --no-index --find-links /tmp/pip/cache-transformers/ transformers')

import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import LukeTokenizer, LukeModel, LukeConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# for TPU
os.environ['XLA_USE_BF16']="1"
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# for TPU

#torch.set_default_tensor_type('torch.FloatTensor')

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import tensorflow as tf

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
except ValueError:
    tpu = None
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()


# In[ ]:


import torch_xla.version as xv
print('PYTORCH:', xv.__torch_gitrev__)
print('XLA:', xv.__xla_gitrev__)


# # Utils

# In[ ]:


# ====================================================
# Utils
# ====================================================
def get_score(df):
    score = len(df[df['less_toxic_pred'] < df['more_toxic_pred']]) / len(df)
    return score


def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


# # Data Loading

# In[ ]:


# ====================================================
# Data Loading
# ====================================================
train = pd.read_csv('../input/jigsaw-toxic-severity-rating/validation_data.csv')
if CFG.debug:
    train = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)
test = pd.read_csv('../input/jigsaw-toxic-severity-rating/comments_to_score.csv')
submission = pd.read_csv('../input/jigsaw-toxic-severity-rating/sample_submission.csv')
print(train.shape)
print(test.shape, submission.shape)
display(train.head())
display(test.head())
display(submission.head())


# # CV split

# In[ ]:


# ====================================================
# CV split
# ====================================================
#Fold = GroupKFold(n_splits=CFG.n_fold)
Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

#for n, (trn_index, val_index) in enumerate(Fold.split(train, train, train['worker'])):
for n, ( trn_index, val_index) in enumerate(Fold.split(X=train, y=train.worker)):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)
display(train.groupby('fold').size())


# # tokenizer

# In[ ]:


# ====================================================
# tokenizer
# ====================================================
tokenizer = LukeTokenizer.from_pretrained(CFG.model, lowercase=True)
tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
CFG.tokenizer = tokenizer


# # Dataset

# In[ ]:


# ====================================================
# Dataset
# ====================================================
def prepare_input(text, cfg):
    if cfg.tail == 0:
        inputs = cfg.tokenizer.encode_plus(text, 
                                           return_tensors=None, 
                                           add_special_tokens=True, 
                                           max_length=cfg.max_len,
                                           pad_to_max_length=True,
                                           truncation=True)
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
    else:
        inputs = cfg.tokenizer.encode_plus(text,
                                           return_tensors=None, 
                                           add_special_tokens=True, 
                                           truncation=True)
        for k, v in inputs.items():
            v_length = len(v)
            if v_length > cfg.max_len:
                v = np.hstack([v[:cfg.head], v[-cfg.tail:]])
            if k == 'input_ids':
                new_v = np.ones(cfg.max_len) * cfg.tokenizer.pad_token_id
            else:
                new_v = np.zeros(cfg.max_len)
            new_v[:v_length] = v 
            inputs[k] = torch.tensor(new_v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.less_toxic = df['less_toxic'].fillna("none").values
        self.more_toxic = df['more_toxic'].fillna("none").values

    def __len__(self):
        return len(self.less_toxic)

    def __getitem__(self, item):
        less_toxic_inputs = prepare_input(str(self.less_toxic[item]), self.cfg)
        more_toxic_inputs = prepare_input(str(self.more_toxic[item]), self.cfg)
        label = torch.tensor(1, dtype=torch.float)
        return less_toxic_inputs, more_toxic_inputs, label


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.text = df[cfg.text].fillna("none").values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        inputs = prepare_input(text, self.cfg)
        return inputs


# # Model

# In[ ]:


# ====================================================
# Model
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = LukeConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = LukeModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = LukeModel(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, cfg.target_size)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = torch.mean(last_hidden_states, 1)
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output


# # Helper functions

# In[ ]:


# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    import torch_xla
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (less_toxic_inputs, more_toxic_inputs, labels) in enumerate(train_loader):
        for k, v in less_toxic_inputs.items():
            less_toxic_inputs[k] = v.to(device)
        for k, v in more_toxic_inputs.items():
            more_toxic_inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            less_toxic_y_preds = model(less_toxic_inputs)
            more_toxic_y_preds = model(more_toxic_inputs)
            loss = criterion(more_toxic_y_preds, less_toxic_y_preds, labels)
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
        #print({f"[fold{fold}] loss": losses.val,  f"[fold{fold}] lr": scheduler.get_lr()[0]})
    return losses.avg


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device, dtype=torch.long)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.extend(y_preds.sigmoid().cpu().detach().numpy().tolist())
    print("preds ",len(preds) )   
    #predictions = np.concatenate(preds)
    return preds


# In[ ]:


# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):
    device = xm.xla_device()
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    
    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    validation = folds.loc[val_idx].reset_index(drop=True)
    
    valid_folds = sorted(set(validation['less_toxic'].unique()) | set(validation['more_toxic'].unique()))
    valid_folds = pd.DataFrame({'text': valid_folds}).reset_index()
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TestDataset(CFG, valid_folds)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
              train_dataset,
              num_replicas=xm.xrt_world_size(),
              rank=xm.get_ordinal(),
              shuffle=True)
    train_loader = DataLoader(train_dataset,
                              sampler= train_sampler,
                              batch_size=CFG.batch_size,
                              #shuffle=True,
                              num_workers=0,#CFG.num_workers,
                              pin_memory=True, drop_last=True)
    gc.collect()
    xm.master_print('parallel loader created... training now')
    
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
      valid_dataset,
      num_replicas=xm.xrt_world_size(),
      rank=xm.get_ordinal(),
      shuffle=False)
    valid_loader = DataLoader(valid_dataset,
                              sampler= valid_sampler,
                              batch_size=CFG.batch_size,
                              #shuffle=False,
                              num_workers=0,#CFG.num_workers, 
                              pin_memory=True, drop_last=False)
    
    
    gc.collect()
    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    xm.save(model.config, OUTPUT_DIR+'config.pth') # torch.save
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler=='linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler=='cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.MarginRankingLoss(margin=CFG.margin)
    
    best_score = 0.

    for epoch in range(CFG.epochs):

        start_time = time.time()
        LOGGER.info("epoch ",epoch)
        para_loader = pl.ParallelLoader(train_loader, [device])

        # train
        avg_loss = train_fn(fold, para_loader.per_device_loader(device), model, 
                            criterion, optimizer, epoch, scheduler, device)
        
        del para_loader
        para_loader = pl.ParallelLoader(valid_loader, [device])
        
        # eval
        preds = inference_fn(para_loader.per_device_loader(device), model, device)
        del para_loader
        gc.collect()
        
        # scoring
        valid_folds['pred'] = preds
        if 'less_toxic_pred' in validation.columns:
            validation = validation.drop(columns='less_toxic_pred')
        if 'more_toxic_pred' in validation.columns:
            validation = validation.drop(columns='more_toxic_pred')
        rename_cols = {CFG.text: 'less_toxic', 'pred': 'less_toxic_pred'}
        validation = validation.merge(valid_folds[[CFG.text, 'pred']].rename(columns=rename_cols), 
                                      on='less_toxic', how='left')
        rename_cols = {CFG.text: 'more_toxic', 'pred': 'more_toxic_pred'}
        validation = validation.merge(valid_folds[[CFG.text, 'pred']].rename(columns=rename_cols), 
                                      on='more_toxic', how='left')
        score = get_score(validation)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}') 
        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {score:.4f} Model')
            xm.save({'model': model.state_dict(), #torch.save
                        'preds': preds},
                        OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

    preds = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth", #torch.load
                       map_location=torch.device('cpu'))['preds']
    valid_folds['pred'] = preds
    if 'less_toxic_pred' in validation.columns:
        validation = validation.drop(columns='less_toxic_pred')
    if 'more_toxic_pred' in validation.columns:
        validation = validation.drop(columns='more_toxic_pred')
    rename_cols = {CFG.text: 'less_toxic', 'pred': 'less_toxic_pred'}
    validation = validation.merge(valid_folds[[CFG.text, 'pred']].rename(columns=rename_cols), 
                                  on='less_toxic', how='left')
    rename_cols = {CFG.text: 'more_toxic', 'pred': 'more_toxic_pred'}
    validation = validation.merge(valid_folds[[CFG.text, 'pred']].rename(columns=rename_cols), 
                                  on='more_toxic', how='left')

    #torch.cuda.empty_cache()
    gc.collect()
    
    return validation


# In[ ]:


def _run():  
    def get_result(oof_df):
        score = get_score(oof_df)
        LOGGER.info(f'Score: {score:<.4f}')
    
    if CFG.train:
        # train 
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold) 
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)
    
    #wandb.finish()  


# In[ ]:


import time

# Start training processes
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    a = _run()

FLAGS={}
start_time = time.time()
if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=1, start_method='fork')


# In[ ]:


print('Time taken: ',time.time()-start_time)

