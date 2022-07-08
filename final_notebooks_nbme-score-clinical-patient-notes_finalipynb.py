#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import re
from ast import literal_eval
from itertools import chain
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# In[ ]:


# Data Understanding
data_dir = "/kaggle/input/nbme-score-clinical-patient-notes"
# Training data files
train=pd.read_csv(data_dir+"/train.csv")
patient_notes=pd.read_csv(data_dir+"/patient_notes.csv")
features=pd.read_csv(data_dir+"/features.csv")

# Test data file/s
test=pd.read_csv(data_dir+"/test.csv")

# submission sample 
submission=pd.read_csv(data_dir+"/sample_submission.csv")


# # Train 
# ****Column Description :****
# 
# * id - Unique identifier for each patient note / feature pair.
# * case_num - The case to which this patient note belongs.
# * pn_num - The patient note annotated in this row.
# * feature_num - The feature annotated in this row.
# * annotation - The text(s) within a patient note indicating a feature. A feature may be indicated multiple times within a single note.
# * location - Character spans indicating the location of each annotation within the note. Multiple spans may be needed to represent an annotation, in which case the spans are delimited by a semicolon ;.

# In[ ]:


train.head()


# In[ ]:


print('Number of rows in train data: {}'.format(train.shape[0]))
print('Number of columns in train data: {}'.format(train.shape[1]))
print('Number of unique cases: {}'.format(train.case_num.nunique()))
print('Number of unique patients: {}'.format(train.pn_num.nunique()))


# # Features
# ****Column Description :****
# 
# * feature_num - A unique identifier for each feature.
# * case_num - The case to which this patient note belongs.
# * feature_text - A description of the feature.

# In[ ]:


features.head()


# In[ ]:


# Sample Feature Text
features["feature_text"].iloc[4], features["feature_text"].iloc[40], features["feature_text"].iloc[41]


# # Patient Notes
# **Column Description :**
# * pn_num - A unique identifier for each patient note.
# * case_num - A unique identifier for the clinical case a patient note represents.
# * pn_history - The text of the encounter as recorded by the test taker.

# In[ ]:


patient_notes.head()


# In[ ]:


# Sample Patient Note
print(patient_notes["pn_history"].iloc[8])


# In[ ]:


import ast
def create_labels_for_scoring(df):
    # example: ['0 1', '3 4'] -> ['0 1; 3 4']
    df['location_for_create_labels'] = [ast.literal_eval(f'[]')] * len(df)
    for i in range(len(df)):
        lst = df.loc[i, 'location']
        if lst:
            new_lst = ';'.join(lst)
            df.loc[i, 'location_for_create_labels'] = ast.literal_eval(f'[["{new_lst}"]]')
    # create labels
    truths = []
    for location_list in df['location_for_create_labels'].values:
        truth = []
        if len(location_list) > 0:
            location = location_list[0]
            for loc in [s.split() for s in location.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                truth.append([start, end])
        truths.append(truth)
    return truths


# In[ ]:


# Data Preprocess
def process_feature_text(text):
    return text.replace("-OR-", ";-").replace("-", " ").replace("I-year", "1-year")


def clean_spaces(txt):
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)
    return txt


# In[ ]:


train = pd.read_csv("/kaggle/input/nbme-score-clinical-patient-notes/train.csv")

# Merge Datasets to Prepare Training Data
merged_df = train.merge(features, how="left", on=["case_num", "feature_num"])
merged_df = merged_df.merge(patient_notes, how="left", on=['case_num', 'pn_num'])

# Preprocess
merged_df['pn_history'] = merged_df['pn_history'].apply(lambda x: x.strip())
merged_df['pn_history'] = merged_df['pn_history'].apply(clean_spaces)
merged_df['pn_history'] = merged_df['pn_history'].apply(lambda x: x.lower())
merged_df['feature_text'] = merged_df['feature_text'].apply(process_feature_text)
merged_df['feature_text'] = merged_df['feature_text'].apply(clean_spaces)
merged_df['feature_text'] = merged_df['feature_text'].apply(lambda x: x.lower())


merged_df['annotation_len'] = train['annotation'].apply(len)


# In[ ]:


merged_df['annotation'] = merged_df['annotation'].apply(ast.literal_eval)
merged_df['location'] = merged_df['location'].apply(ast.literal_eval)


# In[ ]:


truths = create_labels_for_scoring(merged_df)
merged_df


# In[ ]:


# Split data as train and test
test_size = int(len(merged_df)* (0.2))
train_df, test_df = train_test_split(merged_df, test_size=test_size, random_state=500)
print(len(train_df), len(test_df))


# In[ ]:


#   k fold ekle///

# Fold = GroupKFold(n_splits=CFG.n_fold)
# groups = train['pn_num'].values
# for n, (train_index, val_index) in enumerate(Fold.split(train, train['location'], groups)):
#     train.loc[val_index, 'fold'] = int(n)
# train['fold'] = train['fold'].astype(int)
# display(train.groupby('fold').size())


# In[ ]:


class CustomDataset(Dataset):
    def __init__(self, data, param, tokenizer):
        self.data = data
        self.param = param
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        inputs, label, offset_mapping, sequence_ids = prepare_dataset(self.data, self.param, self.tokenizer, item)
        return inputs, label, offset_mapping, sequence_ids


# In[ ]:


def prepare_dataset(data, param, tokenizer, item):
    pn_history = data['pn_history'].values[item]
    feature_text = data['feature_text'].values[item]
    locations = data['location'].values[item]
    annotation_length = data['annotation_len'].values[item]
    
    inputs = tokenizer(pn_history, feature_text, max_length = param['max_len'], padding = param['padding'], return_offsets_mapping = False)
      
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    
    encodings = tokenizer(pn_history, max_length = param['max_len'], padding = param['padding'], return_offsets_mapping=True)     
    
    offset_mapping = encodings['offset_mapping']
    sequence_ids = encodings.sequence_ids()
    offset_mapping = np.array(offset_mapping)
    sequence_ids = np.array(sequence_ids).astype("float16")
    ignore_idxes = np.where(np.array(sequence_ids) != 0)[0]
    
    label = np.zeros(len(offset_mapping))    
    label[ignore_idxes] = -1
    
    if annotation_length != 0:
        for location in locations:
            for loc in [s.split() for s in location.split(';')]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx
                if (start_idx != -1) & (end_idx != -1):
                    label[start_idx:end_idx] = 1
    label = torch.tensor(label, dtype=torch.float)
    return inputs, label, offset_mapping, sequence_ids


# In[ ]:


class CustomModel(nn.Module):
    def __init__(self,param):
        super().__init__()
        self.param = param       
        self.bert = AutoModel.from_pretrained(param['model_name'])  # BERT model
        self.config = AutoConfig.from_pretrained(param['model_name'], output_hidden_states=True)
        self.dropout = nn.Dropout(p=param['dropout'])
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.fc1(outputs[0])
        logits = self.fc2(self.dropout(logits)).squeeze(-1)
        return logits


# In[ ]:


params = {
    "max_len": 416,
    "padding": "max_length",
    "return_offsets_mapping": True,
    "truncation": "only_second",
    "model_name": "bert-base-uncased",
    "dropout": 0.2,
    "lr": 1e-5,
    "test_size": 0.2,
    "seed": 1268,
    "batch_size": 8
}

tokenizer = AutoTokenizer.from_pretrained(params['model_name'])

training_data = CustomDataset(train_df, params, tokenizer)
train_dataloader = DataLoader(training_data, batch_size=params['batch_size'], shuffle=True)

test_data = CustomDataset(test_df, params, tokenizer)
test_dataloader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)


# In[ ]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CustomModel(params).to(DEVICE)

criterion = torch.nn.BCEWithLogitsLoss(reduction = "none")
optimizer = optim.AdamW(model.parameters(), lr=params['lr'])


# In[ ]:


from sklearn.metrics import accuracy_score

def get_location_predictions(preds, offset_mapping, sequence_ids, test=False):
    all_predictions = []
    for pred, offsets, seq_ids in zip(preds, offset_mapping, sequence_ids):
        pred = 1 / (1 + np.exp(-pred))
        start_idx = None
        end_idx = None
        current_preds = []
        for pred, offset, seq_id in zip(pred, offsets, seq_ids):
            if seq_id is None or seq_id == 0:
                continue

            if pred > 0.5:
                if start_idx is None:
                    start_idx = offset[0]
                end_idx = offset[1]
            elif start_idx is not None:
                if test:
                    current_preds.append(f"{start_idx} {end_idx}")
                else:
                    current_preds.append((start_idx, end_idx))
                start_idx = None
        if test:
            all_predictions.append("; ".join(current_preds))
        else:
            all_predictions.append(current_preds)
            
    return all_predictions


def calculate_char_cv(predictions, offset_mapping, sequence_ids, labels):
    all_labels = []
    all_preds = []
    for preds, offsets, seq_ids, labels in zip(predictions, offset_mapping, sequence_ids, labels):

        num_chars = max(list(chain(*offsets)))
        char_labels = np.zeros(num_chars)

        for o, s_id, label in zip(offsets, seq_ids, labels):
            if s_id is None or s_id == 0:
                continue
            if int(label) == 1:
                char_labels[o[0]:o[1]] = 1

        char_preds = np.zeros(num_chars)

        for start_idx, end_idx in preds:
            char_preds[start_idx:end_idx] = 1

        all_labels.extend(char_labels)
        all_preds.extend(char_preds)

    results = precision_recall_fscore_support(all_labels, all_preds, average="binary", labels=np.unique(all_preds))
    accuracy = accuracy_score(all_labels, all_preds)
    

    return {
        "Accuracy": accuracy,
        "precision": results[0],
        "recall": results[1],
        "f1": results[2]
    }


# In[ ]:


def train_model(model, dataloader, optimizer, criterion):
    model.train()
    train_loss = []
    for step, (inputs, labels, a,b) in enumerate(dataloader):             
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)     
            
        labels = labels.to(DEVICE)            
        logits = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
        loss = criterion(logits, labels)
        size = inputs['input_ids'].size(0)
        loss = torch.masked_select(loss, labels > -1.0).mean()
        train_loss.append(loss.item() * size)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return sum(train_loss)/len(train_loss)


# In[ ]:


def eval_model(model, dataloader, criterion):
    model.eval()
    valid_loss = []
    preds = []
    offsets = []
    seq_ids = []
    valid_labels = []
    for step, (inputs, labels, offset_mapping, sequence_ids) in enumerate(dataloader):             
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)     

        labels = labels.to(DEVICE)            
        logits = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
        loss = criterion(logits, labels)
        loss = torch.masked_select(loss, labels > -1.0).mean()
        size = inputs['input_ids'].size(0)
        valid_loss.append(loss.item() * size)

        preds.append(logits.detach().cpu().numpy())
        offsets.append(offset_mapping.numpy())
        seq_ids.append(sequence_ids.numpy())
        valid_labels.append(labels.detach().cpu().numpy())
        

    preds = np.concatenate(preds, axis=0)
    offsets = np.concatenate(offsets, axis=0)
    seq_ids = np.concatenate(seq_ids, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    location_preds = get_location_predictions(preds, offsets, seq_ids, test=False)
    score = calculate_char_cv(location_preds, offsets, seq_ids, valid_labels)

    return sum(valid_loss)/len(valid_loss), score


# In[ ]:


import time

train_loss_data, valid_loss_data = [], []
score_data_list = []
valid_loss_min = np.Inf
since = time.time()
epochs = 3
best_loss = np.inf

for i in range(epochs):
    print("Epoch: {}/{}".format(i + 1, epochs))
    # first train model
    train_loss = train_model(model, train_dataloader, optimizer, criterion)
    train_loss_data.append(train_loss)
    print(f"Train loss: {train_loss}")
    # evaluate model
    valid_loss, score = eval_model(model, test_dataloader, criterion)
    valid_loss_data.append(valid_loss)
    score_data_list.append(score)
    print(f"Valid loss: {valid_loss}")
    print(f"Valid score: {score}")
    
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), "nbme_bert_v2.pth")

    
time_elapsed = time.time() - since
print('Training completed in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


# In[ ]:





# In[ ]:




