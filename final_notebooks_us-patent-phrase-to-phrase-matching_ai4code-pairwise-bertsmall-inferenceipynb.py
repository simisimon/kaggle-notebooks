#!/usr/bin/env python
# coding: utf-8

# # Pairwise inference
# In this notebook, we demonstrate how to use pairwise model to predict in this competition. Please note that the inference time is much longer than pointwise method or using cosine similarity. 
# 
# 1. I used a bert-small model pretrained with pairwise-mlm.
# 2. Training with pairwise examples with negative samples randomly sampled.
# 3. **Inference and predict for all the pairs for test dataset.**
# 
# * [Pretrain](https://www.kaggle.com/code/yuanzhezhou/ai4code-pairwise-bertsmall-pretrain/notebook)
# * [Training](https://www.kaggle.com/yuanzhezhou/ai4code-pairwise-bertsmall-training)
# * [Inference](https://www.kaggle.com/yuanzhezhou/ai4code-pairwise-bertsmall-inference)

# In[ ]:


import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

pd.options.display.width = 180
pd.options.display.max_colwidth = 120

BERT_PATH = "../input/huggingface-bert-variants/distilbert-base-uncased/distilbert-base-uncased"

data_dir = Path('../input/AI4Code')


# In[ ]:


NUM_TRAIN = 200


def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )


paths_train = list((data_dir / 'train').glob('*.json'))[:NUM_TRAIN]
notebooks_train = [
    read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
]
df = (
    pd.concat(notebooks_train)
    .set_index('id', append=True)
    .swaplevel()
    .sort_index(level='id', sort_remaining=False)
)

df


# In[ ]:


# Get an example notebook
nb_id = df.index.unique('id')[6]
print('Notebook:', nb_id)

print("The disordered notebook:")
nb = df.loc[nb_id, :]
display(nb)
print()


# In[ ]:


df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()  # Split the string representation of cell_ids into a list

df_orders


# In[ ]:


len(df_orders.loc["002ba502bdac45"])


# In[ ]:


cell_order = df_orders.loc[nb_id]

print("The ordered notebook:")
nb.loc[cell_order, :]


# In[ ]:


def get_ranks(base, derived):
    return [base.index(d) for d in derived]

cell_ranks = get_ranks(cell_order, list(nb.index))
nb.insert(0, 'rank', cell_ranks)

nb


# In[ ]:


df_orders_ = df_orders.to_frame().join(
    df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
    how='right',
)

ranks = {}
for id_, cell_order, cell_id in df_orders_.itertuples():
    ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

df_ranks = (
    pd.DataFrame
    .from_dict(ranks, orient='index')
    .rename_axis('id')
    .apply(pd.Series.explode)
    .set_index('cell_id', append=True)
)

df_ranks


# In[ ]:


df_ancestors = pd.read_csv(data_dir / 'train_ancestors.csv', index_col='id')
df_ancestors


# In[ ]:


df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=["id"])
df


# In[ ]:


df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

df["pct_rank"].hist(bins=10)


# In[ ]:


dict_cellid_source = dict(zip(df['cell_id'].values, df['source'].values))


# In[ ]:


import numpy as np
import pandas as pd
import os
import re
# import fasttext
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import nltk
nltk.download('wordnet')

stemmer = WordNetLemmatizer()

def preprocess_text(document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()
        #return document

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    
def preprocess_df(df):
    """
    This function is for processing sorce of notebook
    returns preprocessed dataframe
    """
    return [preprocess_text(message) for message in df.source]

df.source = df.source.apply(preprocess_text)


# In[ ]:


from sklearn.model_selection import GroupShuffleSplit

NVALID = 0.1  # size of validation set

splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)

train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))

train_df = df.loc[train_ind].reset_index(drop=True)
val_df = df.loc[val_ind].reset_index(drop=True)


# In[ ]:


from tqdm.notebook import tqdm

def generate_triplet(df, mode='train'):
  triplets = []
  ids = df.id.unique()
  random_drop = np.random.random(size=10000)>0.9
  count = 0

  for id, df_tmp in tqdm(df.groupby('id')):
    df_tmp_markdown = df_tmp[df_tmp['cell_type']=='markdown']

    df_tmp_code = df_tmp[df_tmp['cell_type']=='code']
    df_tmp_code_rank = df_tmp_code['rank'].values
    df_tmp_code_cell_id = df_tmp_code['cell_id'].values

    for cell_id, rank in df_tmp_markdown[['cell_id', 'rank']].values:
      labels = np.array([(r==(rank+1)) for r in df_tmp_code_rank]).astype('int')

      for cid, label in zip(df_tmp_code_cell_id, labels):
        count += 1
        if label==1:
          triplets.append( [cell_id, cid, label] )
          # triplets.append( [cid, cell_id, label] )
        elif mode == 'test':
          triplets.append( [cell_id, cid, label] )
          # triplets.append( [cid, cell_id, label] )
        elif random_drop[count%10000]:
          triplets.append( [cell_id, cid, label] )
          # triplets.append( [cid, cell_id, label] )
    
  return triplets

triplets = generate_triplet(train_df)
val_triplets = generate_triplet(val_df, mode = 'test')


# In[ ]:


val_df.head()


# In[ ]:


from bisect import bisect


def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


# In[ ]:


from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel

MAX_LEN = 128

    
class MarkdownModel(nn.Module):
    def __init__(self):
        super(MarkdownModel, self).__init__()
        self.distill_bert = AutoModel.from_pretrained("../input/mymodelpairbertsmallpretrained/models/checkpoint-18000")
        self.top = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.2)
        
    def forward(self, ids, mask):
        x = self.distill_bert(ids, mask)[0]
        x = self.dropout(x)
        x = self.top(x[:, 0, :])
        x = torch.sigmoid(x) 
        return x


# In[ ]:


from torch.utils.data import DataLoader, Dataset



class MarkdownDataset(Dataset):
    
    def __init__(self, df, max_len, mode='train'):
        super().__init__()
        self.df = df
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained("../input/mymodelpairbertsmallpretrained/my_own_tokenizer", do_lower_case=True)
        self.mode=mode

    def __getitem__(self, index):
        row = self.df[index]

        label = row[-1]

        txt = dict_cellid_source[row[0]] + '[SEP]' + dict_cellid_source[row[1]]

        inputs = self.tokenizer.encode_plus(
            txt,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        return ids, mask, torch.FloatTensor([label])




    def __len__(self):
        return len(self.df)


# train_ds = MarkdownDataset(triplets, max_len=MAX_LEN, mode='test')
# val_ds = MarkdownDataset(val_triplets, max_len=MAX_LEN, mode='test')


# train_ds[1]


# In[ ]:


def adjust_lr(optimizer, epoch):
    if epoch < 1:
        lr = 5e-5
    elif epoch < 2:
        lr = 1e-3
    elif epoch < 5:
        lr = 1e-4
    else:
        lr = 1e-5

    for p in optimizer.param_groups:
        p['lr'] = lr
    return lr
    
def get_optimizer(net):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-4, betas=(0.9, 0.999),
                                 eps=1e-08)
    return optimizer

BS = 128
NW = 8

# train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=NW,
#                           pin_memory=False, drop_last=True)
# val_loader = DataLoader(val_ds, batch_size=BS * 8, shuffle=False, num_workers=NW,
#                           pin_memory=False, drop_last=False)


# In[ ]:


def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader, mode='train'):
    model.eval()
    
    tbar = tqdm(val_loader, file=sys.stdout)
    
    preds = np.zeros(len(val_loader.dataset), dtype='float32')
    labels = []
    count = 0

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            pred = model(inputs[0], inputs[1]).detach().cpu().numpy().ravel()

            preds[count:count+len(pred)] = pred
            count += len(pred)
            
            if mode=='test':
              labels.append(target.detach().cpu().numpy().ravel())
    if mode=='test':
      return preds
    else:
      return np.concatenate(labels), np.concatenate(preds)


# In[ ]:


paths_test = list((data_dir / 'test').glob('*.json'))
notebooks_test = [
    read_notebook(path) for path in tqdm(paths_test, desc='Test NBs')
]
test_df = (
    pd.concat(notebooks_test)
    .set_index('id', append=True)
    .swaplevel()
    .sort_index(level='id', sort_remaining=False)
).reset_index()


# In[ ]:


test_df.source = test_df.source.apply(preprocess_text)
dict_cellid_source = dict(zip(test_df['cell_id'].values, test_df['source'].values))


# In[ ]:


test_df["rank"] = test_df.groupby(["id", "cell_type"]).cumcount()
test_df["pred"] = test_df.groupby(["id", "cell_type"])["rank"].rank(pct=False)


# In[ ]:


test_triplets = generate_triplet(test_df, mode = 'test')


# In[ ]:


test_df["pct_rank"] = 0
test_ds = MarkdownDataset(test_triplets, max_len=MAX_LEN)
test_loader = DataLoader(test_ds, batch_size=BS * 4, shuffle=False, num_workers=NW,
                          pin_memory=False, drop_last=False)


import gc 
gc.collect()
len(test_ds), test_ds[0]


# In[ ]:


import sys 

model = MarkdownModel()
model = model.cuda()
model.load_state_dict(torch.load('../input/mymodelbertsmallpretrained120000/my_own_model.bin'))
y_test = validate(model, test_loader, mode='test')



# In[ ]:


preds_copy = y_test


# In[ ]:


pred_vals = []
count = 0
for id, df_tmp in tqdm(test_df.groupby('id')):
  df_tmp_mark = df_tmp[df_tmp['cell_type']=='markdown']
  df_tmp_code = df_tmp[df_tmp['cell_type']!='markdown']
  df_tmp_code_rank = df_tmp_code['rank'].rank().values
  N_code = len(df_tmp_code_rank)
  N_mark = len(df_tmp_mark)

  preds_tmp = preds_copy[count:count+N_mark * N_code]

  count += N_mark * N_code

  for i in range(N_mark):
    pred = preds_tmp[i*N_code:i*N_code+N_code] 

    softmax = np.exp((pred-np.mean(pred)) *20)/np.sum(np.exp((pred-np.mean(pred)) *20)) 

    rank = np.sum(softmax * df_tmp_code_rank)
    pred_vals.append(rank)

del model
del test_triplets[:]
del dict_cellid_source
gc.collect()


# In[ ]:


test_df.loc[test_df["cell_type"] == "markdown", "pred"] = pred_vals


# In[ ]:


sub_df = test_df.sort_values("pred").groupby("id")["cell_id"].apply(lambda x: " ".join(x)).reset_index()
sub_df.rename(columns={"cell_id": "cell_order"}, inplace=True)
sub_df.head()


# In[ ]:


sub_df.to_csv("submission.csv", index=False)


# In[ ]:





# # Please upvote if you find it helpful! :D
