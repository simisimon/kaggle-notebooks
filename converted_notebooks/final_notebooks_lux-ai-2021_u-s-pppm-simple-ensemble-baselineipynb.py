#!/usr/bin/env python
# coding: utf-8

# Ensemble of the three notebooks.Please upvote the original notebooks:
# 
# **1. Deberta v3 large**
# [PPPM / Deberta-v3-large baseline [inference]](https://www.kaggle.com/code/yasufuminakama/pppm-deberta-v3-large-baseline-inference)
# 
# **2. Roberta-large**
# [PatentPhrase RoBERTa Inference](https://www.kaggle.com/code/santhoshkumarv/patentphrase-roberta-inference-lb-0-814)
# 
# And use ensemble strategy from:
# [Tips for ensambling](https://www.kaggle.com/code/jellyz9/tips-for-ensambling)

# ## 1.Deberta v3 large

# In[ ]:


# ====================================================
# Directory settings
# ====================================================
import os

INPUT_DIR = '../input/us-patent-phrase-to-phrase-matching/'
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# In[ ]:


# ====================================================
# CFG
# ====================================================
class CFG:
    num_workers=4
    path="../input/pppm-deberta-v3-large-baseline-w-w-b-train/"
    config_path=path+'config.pth'
    model="microsoft/deberta-v3-large"
    batch_size=32
    fc_dropout=0.2
    target_size=1
    max_len=133
    seed=42
    n_fold=4
    trn_fold=[0, 1, 2, 3]


# In[ ]:


import os
import gc
import math
import time
import random
import scipy as sp
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

os.system('pip uninstall -y transformers')
os.system('pip uninstall -y tokenizers')
os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels-dataset transformers')
os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels-dataset tokenizers')
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")

from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
get_ipython().run_line_magic('env', 'TOKENIZERS_PARALLELISM=true')


import warnings 
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
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


# In[ ]:


oof_df = pd.read_pickle(CFG.path+'oof_df.pkl')
labels = oof_df['score'].values
preds = oof_df['pred'].values
score = get_score(labels, preds)
LOGGER.info(f'CV Score: {score:<.4f}')


# In[ ]:


# ====================================================
# Data Loading
# ====================================================
test = pd.read_csv(INPUT_DIR+'test.csv')
submission = pd.read_csv(INPUT_DIR+'sample_submission.csv')
print(f"test.shape: {test.shape}")
print(f"submission.shape: {submission.shape}")
display(test.head())
display(submission.head())


# In[ ]:


# ====================================================
# CPC Data
# ====================================================
cpc_texts = torch.load(CFG.path+"cpc_texts.pth")
test['context_text'] = test['context'].map(cpc_texts)
display(test.head())


# In[ ]:


test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']
display(test.head())


# In[ ]:


# ====================================================
# tokenizer
# ====================================================
CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path+'tokenizer/')


# In[ ]:


# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text,
                           add_special_tokens=True,
                           max_length=cfg.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs


# In[ ]:


# ====================================================
# Model
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output


# In[ ]:


# ====================================================
# inference
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


# In[ ]:


test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(test_dataset,
                         batch_size=CFG.batch_size,
                         shuffle=False,
                         num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
predictions = []
for fold in CFG.trn_fold:
    model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
    state = torch.load(CFG.path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                       map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, device)
    predictions.append(prediction)
    del model, state, prediction; gc.collect()
    torch.cuda.empty_cache()
pred1 = np.mean(predictions, axis=0)


# # 2.Roberta-large

# In[ ]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(seed=2019)


# In[ ]:


@dataclass(frozen=True)
class CFG:
    num_workers: Optional[int] = 4
    config_path: Optional[str] = '../input/robertalarge'
    model_path: Optional[str] = '../input/phrase-matching-roberta-training-pytorch-wandb'
    model_name: Optional[str] = 'roberta-large'
    batch_size: Optional[int] = 32
    max_len: Optional[int] = 128
    seed: Optional[int] = 2019
    num_targets: Optional[int] = 1
    n_folds: Optional[int] = 5
    tokenizer = AutoTokenizer.from_pretrained('../input/robertalarge')


# In[ ]:


PATH = '../input/us-patent-phrase-to-phrase-matching'
test = pd.read_csv(os.path.join(PATH, 'test.csv'))
sub = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))


# In[ ]:


context_mapping = {
        "A": "Human Necessities",
        "B": "Operations and Transport",
        "C": "Chemistry and Metallurgy",
        "D": "Textiles",
        "E": "Fixed Constructions",
        "F": "Mechanical Engineering",
        "G": "Physics",
        "H": "Electricity",
        "Y": "Emerging Cross-Sectional Technologies",
}
    
test.context = test.context.apply(lambda x: context_mapping[x[0]])


# In[ ]:


class PhraseDataset:
    def __init__(self, anchor, target, context, tokenizer, max_len):
        self.anchor = anchor
        self.target = target
        self.context = context
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, item):
        anchor = self.anchor[item]
        context = self.context[item]
        target = self.target[item]

        encoded_text = CFG.tokenizer.encode_plus(
            context + " " + anchor,
            target,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
        )
        input_ids = encoded_text["input_ids"]
        attention_mask = encoded_text["attention_mask"]

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
        }


# In[ ]:


def inference_fn(model, test_loader):  
    model.eval()
    predictions = []
    tk0 = tqdm(test_loader, total=len(test_loader))
    for data in tk0:
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        
        with torch.no_grad():
            output = model(ids, mask)
        predictions.append(output.sigmoid().detach().cpu().numpy())
        
    return np.concatenate(predictions)


# In[ ]:


class PatentModel(torch.nn.Module):
    def __init__(self):
        super(PatentModel, self).__init__()
        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(CFG.config_path)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
            }
        )
        
        self.transformer = AutoModel.from_pretrained(CFG.config_path, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size, CFG.num_targets)
        
    def forward(self, ids, mask):
        transformer_out = self.transformer(input_ids=ids, attention_mask=mask)
        last_hidden_states = transformer_out[0]
        last_hidden_states = self.dropout(torch.mean(last_hidden_states, 1))
        logits1 = self.output(self.dropout1(last_hidden_states))
        logits2 = self.output(self.dropout2(last_hidden_states))
        logits3 = self.output(self.dropout3(last_hidden_states))
        logits4 = self.output(self.dropout4(last_hidden_states))
        logits5 = self.output(self.dropout5(last_hidden_states))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        
        return logits


# In[ ]:


def run_fold(test, fold, seed=42):    
    
    seed_everything(seed)
    
    test_dataset = PhraseDataset(
        test.anchor.values,
        test.target.values,
        test.context.values,
        CFG.tokenizer, 
        CFG.max_len
    ) 
    
    test_loader = DataLoader(test_dataset, 
                              batch_size=CFG.batch_size * 2, 
                              shuffle=False, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    model = PatentModel()
    
    model.load_state_dict(
        torch.load(f'{CFG.model_path}/{CFG.model_name.replace("-","_")}_patent_model_{fold}.pth',
        map_location=torch.device('cuda')
        )
    )
    
    model.to(device)

    preds = inference_fn(model, test_loader)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return preds


# In[ ]:


def inference_model(test, seed):
    
    predictions = []
    
    for f in range(CFG.n_folds):    
        preds = run_fold(test, f, seed) 
        predictions.append(preds)
        
    test_preds = np.column_stack(predictions)
        
    return test_preds


# In[ ]:


if __name__ == '__main__':
    pred2 =  np.mean(inference_model(test, CFG.seed),axis=1)


# # Ensemble

# In[ ]:


w1 = 0.66
w2 = 0.33


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

MMscaler = MinMaxScaler()

pred1_mm = MMscaler.fit_transform(pred1.reshape(-1,1)).reshape(-1)
pred2_mm = MMscaler.fit_transform(pred2.reshape(-1,1)).reshape(-1)

final_predictions =  pred1_mm * w1 + pred2_mm * w2


# # Submission

# In[ ]:


sub['score'] = final_predictions
sub.to_csv('submission.csv', index=False)


# In[ ]:


sub.head()


# In[ ]:




