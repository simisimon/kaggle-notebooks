#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install ../input/xgboost-090/xgboost-0.90-py2.py3-none-manylinux1_x86_64.whl --no-dependencies')


# In[ ]:


# general
import pandas as pd
import numpy as np
import os
import copy
import pickle
import random
from joblib import Parallel, delayed
from multiprocessing import Manager
from tqdm.notebook import tqdm
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import KFold
from collections import Counter
from bisect import bisect_left
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.ensemble import GradientBoostingClassifier
from skopt.space import Real
from skopt import gp_minimize
import sys
import xgboost
import gc
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
# nlp
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


# In[ ]:


class Config:
    savename = "deberta-base-v2"
    n_folds = 5
    num_workers = 12
    fold = 0
    model = "microsoft/deberta-base"
    lr = 2.5e-5
    n_accum = 1
    max_grad_norm = 10
    output = "/content/model"
    input = "/content/data"
    max_len = 1600
    max_len_valid = 1600
    num_labels = 15
    batch_size = 4
    valid_batch_size = 4
    epochs = 6
    accumulation_steps = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    apex = True
    debug = False
    if debug:
        n_folds = 2
        epochs = 2


# In[ ]:


output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

labels_to_ids = {v:k for k,v in enumerate(output_labels)}
ids_to_labels = {k:v for k,v in enumerate(output_labels)}
disc_type_to_ids = {'Evidence':(11,12),'Claim':(5,6),'Lead':(1,2),'Position':(3,4),'Counterclaim':(7,8),'Rebuttal':(9,10),'Concluding Statement':(13,14)}


# In[ ]:


def get_texts(path):
    names, texts = [], []
    for f in list(os.listdir(path)):
        names.append(f.replace('.txt', ''))
        texts.append(open(path + f, 'r').read())
    texts = pd.DataFrame({'id': names, 'text': texts})
    return texts


# In[ ]:


def split_mapping(unsplit):
    splt = unsplit.split()
    offset_to_wordidx = np.full(len(unsplit),-1)
    txt_ptr = 0
    for split_index, full_word in enumerate(splt):
        while unsplit[txt_ptr:txt_ptr + len(full_word)] != full_word:
            txt_ptr += 1
        offset_to_wordidx[txt_ptr:txt_ptr + len(full_word)] = split_index
        txt_ptr += len(full_word)
    return offset_to_wordidx


# In[ ]:


class feedbackDataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len, get_wids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids # for validation

  def __getitem__(self, index):
        # GET TEXT AND WORD LABELS 
        text = self.data.text[index]        
        word_labels = self.data.entities[index] if not self.get_wids else None

        # TOKENIZE TEXT
        encoding = self.tokenizer(text,
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        
        word_ids = encoding.word_ids()  
        split_word_ids = np.full(len(word_ids),-1)
        offset_to_wordidx = split_mapping(text)
        offsets = encoding['offset_mapping']
        
        # CREATE TARGETS AND MAPPING OF TOKENS TO SPLIT() WORDS
        label_ids = []
        # Iterate in reverse to label whitespace tokens until a Begin token is encountered
        for token_idx, word_idx in reversed(list(enumerate(word_ids))):
            
            if word_idx is None:
                if not self.get_wids: label_ids.append(-100)
            else:
                if offsets[token_idx] != (0,0):
                    #Choose the split word that shares the most characters with the token if any
                    split_idxs = offset_to_wordidx[offsets[token_idx][0]:offsets[token_idx][1]]
                    split_index = stats.mode(split_idxs[split_idxs != -1]).mode[0] if len(np.unique(split_idxs)) > 1 else split_idxs[0]
                    
                    if split_index != -1: 
                        if not self.get_wids: label_ids.append( labels_to_ids[word_labels[split_index]] )
                        split_word_ids[token_idx] = split_index
                    else:
                        # Even if we don't find a word, continue labeling 'I' tokens until a 'B' token is found
                        if label_ids and label_ids[-1] != -100 and ids_to_labels[label_ids[-1]][0] == 'I':
                            split_word_ids[token_idx] = split_word_ids[token_idx + 1]
                            if not self.get_wids: label_ids.append(label_ids[-1])
                        else:
                            if not self.get_wids: label_ids.append(-100)
                else:
                    if not self.get_wids: label_ids.append(-100)
        
        encoding['labels'] = list(reversed(label_ids))

        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids: 
            item['wids'] = torch.as_tensor(split_word_ids)
        
        return item

  def __len__(self):
        return self.len


# In[ ]:


class FeedbackModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(model_name)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": self.num_labels,
            }
        )
        self.transformer = AutoModel.from_config(config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, ids, mask, token_type_ids=None):

        if token_type_ids:
            transformer_out = self.transformer(ids, mask, token_type_ids)
        else:
            transformer_out = self.transformer(ids, mask)
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        return logits, logits1, logits2, logits3, logits4, logits5


# In[ ]:


def get_predictions(all_labels, all_scores, df):    
    proba_thresh = {
        "Lead": 0.7,
        "Position": 0.55,
        "Evidence": 0.65,
        "Claim": 0.55,
        "Concluding Statement": 0.7,
        "Counterclaim": 0.5,
        "Rebuttal": 0.55,
    }
    final_preds = []
    
    for i in range(len(df)):
        idx = df.id.values[i]
        pred = all_labels[i]
        score = all_scores[i]
        preds = []
        j = 0
        
        while j < len(pred):
            cls = pred[j]
            if cls == 'O': 
                pass
            else: 
                cls = cls.replace('B','I')
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            # print(end - j)
            if cls != 'O' and cls != '' and end - j > 7:
                if np.mean(score[j:end]) > proba_thresh[cls.replace('I-','')]:
                    final_preds.append((idx, cls.replace('I-',''), 
                                        ' '.join(map(str, list(range(j, end))))))
            j = end
    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['id','class','predictionstring']
    return df_pred

def threshold(df):

    min_thresh = {
        "Lead": 9,
        "Position": 5,
        "Evidence": 14,
        "Claim": 3,
        "Concluding Statement": 11,
        "Counterclaim": 6,
        "Rebuttal": 4,
    }

    df = df.copy()
    for key, value in min_thresh.items():
        index = df.loc[df["class"] == key].query(f"len<{value}").index
        df.drop(index, inplace=True)
    return df


# In[ ]:


@torch.no_grad()
def inference(model, data_loader, weights):
    model.eval()

    ensemble_preds = np.zeros((len(data_loader.dataset), Config.max_len, len(labels_to_ids)), dtype=np.float32)
    wids = np.full((len(data_loader.dataset), Config.max_len), -100)

    for weight in weights:
        model.load_state_dict(torch.load(weight))
        infer_pbar = tqdm(enumerate(data_loader), total = len(data_loader))

        for step, data in infer_pbar:
            wids[step*Config.valid_batch_size:(step+1)*Config.valid_batch_size] = data['wids'].numpy()

            input_ids = data["input_ids"].to(Config.device)
            input_mask = data["attention_mask"].to(Config.device)

            batch_size = input_ids.shape[0]
            logits, logits1, logits2, logits3, logits4, logits5 = model(input_ids,
                                                                            input_mask)
            
            val_preds = logits.cpu().tolist()
            all_preds = torch.nn.functional.softmax(logits, dim=2).cpu().detach().numpy() 
            ensemble_preds[step*Config.valid_batch_size:(step+1)*Config.valid_batch_size] += all_preds / 4

    predictions = []
    # INTERATE THROUGH EACH TEXT AND GET PRED
    for text_i in range(ensemble_preds.shape[0]):
        token_preds = ensemble_preds[text_i]

        prediction = []
        previous_word_idx = -1
        prob_buffer = []
        word_ids = wids[text_i][wids[text_i] != -100]
        for idx,word_idx in enumerate(word_ids):                            
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:              
                prediction.append(token_preds[idx])
                previous_word_idx = word_idx
        predictions.append(prediction)
                
    
    return predictions  
    


# In[ ]:


# @torch.no_grad()
# def inference(model, data_loader, weights):
#     model.eval()

#     ensemble_preds = np.zeros((len(data_loader.dataset), Config.max_len, len(labels_to_ids)), dtype=np.float32)
#     wids = np.full((len(data_loader.dataset), Config.max_len), -100)

#     for weight in weights:
#         model.load_state_dict(torch.load(weight))
#         infer_pbar = tqdm(enumerate(data_loader), total = len(data_loader))

#         for step, data in infer_pbar:
#             wids[step*Config.valid_batch_size:(step+1)*Config.valid_batch_size] = data['wids'].numpy()

#             input_ids = data["input_ids"].to(Config.device)
#             input_mask = data["attention_mask"].to(Config.device)

#             batch_size = input_ids.shape[0]
#             logits, logits1, logits2, logits3, logits4, logits5 = model(input_ids,
#                                                                             input_mask)
            
#             val_preds = logits.cpu().tolist()
#             all_preds = torch.nn.functional.softmax(logits, dim=2).cpu().detach().numpy() 
#             ensemble_preds[step*Config.valid_batch_size:(step+1)*Config.valid_batch_size] += all_preds / 5

#     predictions = []
#     # INTERATE THROUGH EACH TEXT AND GET PRED
#     for text_i in range(ensemble_preds.shape[0]):
#         token_preds = ensemble_preds[text_i]
        
#         prediction = []
#         previous_word_idx = -1
#         prob_buffer = []
#         word_ids = wids[text_i][wids[text_i] != -100]
#         for idx,word_idx in enumerate(word_ids):                            
#             if word_idx == -1:
#                 pass
#             elif word_idx != previous_word_idx:              
#                 if prob_buffer:
#                     prediction.append(np.mean(prob_buffer, dtype=np.float32, axis=0))
#                     prob_buffer = []
#                 prob_buffer.append(token_preds[idx])
#                 previous_word_idx = word_idx
#             else: 
#                 prob_buffer.append(token_preds[idx])
#         prediction.append(np.mean(prob_buffer, dtype=np.float32, axis=0))
#         predictions.append(prediction)
                
    
#     return predictions  
    


# In[ ]:


train_df = pd.read_csv("../input/fb-corrected-train/corrected_train.csv")
MAX_SEQ_LEN = {}
train_df['len'] = train_df['predictionstring'].apply(lambda x:len(x.split()))
max_lens = train_df.groupby('discourse_type')['len'].quantile(.995)
for disc_type in disc_type_to_ids:
    MAX_SEQ_LEN[disc_type] = int(max_lens[disc_type])

#The minimum probability prediction for a 'B'egin class for which we will evaluate a word sequence
MIN_BEGIN_PROB = {
    'Claim': .35,
    'Concluding Statement': .15,
    'Counterclaim': .04,
    'Evidence': .1,
    'Lead': .32,
    'Position': .25,
    'Rebuttal': .01,
}


# In[ ]:


class SeqDataset(object):
    
    def __init__(self, features, labels, groups, wordRanges, truePos):
        
        self.features = np.array(features, dtype=np.float32)
        self.labels = np.array(labels)
        self.groups = np.array(groups, dtype=np.int16)
        self.wordRanges = np.array(wordRanges, dtype=np.int16)
        self.truePos = np.array(truePos)


# In[ ]:


def seq_dataset(disc_type, test_word_preds, pred_indices=None):
    word_preds = test_word_preds
    window = pred_indices if pred_indices else range(len(word_preds))
    X = np.empty((int(1e6),13), dtype=np.float32)
    X_ind = 0
    y = []
    truePos = []
    wordRanges = []
    groups = []
    for text_i in tqdm(window):
        text_preds = np.array(word_preds[text_i])
        num_words = len(text_preds)
        disc_begin, disc_inside = disc_type_to_ids[disc_type]
        
        # The probability that a word corresponds to either a 'B'-egin or 'I'-nside token for a class
        prob_or = lambda word_preds: (1-(1-word_preds[:,disc_begin]) * (1-word_preds[:,disc_inside]))
        
        # Iterate over every sub-sequence in the text
        quants = np.linspace(0,1,7)
        prob_begins = np.copy(text_preds[:,disc_begin])
        min_begin = MIN_BEGIN_PROB[disc_type]
        for pred_start in range(num_words):
            prob_begin = prob_begins[pred_start]
            if prob_begin > min_begin:
                begin_or_inside = []
                for pred_end in range(pred_start+1,min(num_words+1, pred_start+MAX_SEQ_LEN[disc_type]+1)):
                    
                    new_prob = prob_or(text_preds[pred_end-1:pred_end])
                    insert_i = bisect_left(begin_or_inside, new_prob)
                    begin_or_inside.insert(insert_i, new_prob[0])

                    # Generate features for a word sub-sequence

                    # The length and position of start/end of the sequence
                    features = [pred_end - pred_start, pred_start / float(num_words), pred_end / float(num_words)]
                    
                    # 7 evenly spaced quantiles of the distribution of relevant class probabilities for this sequence
                    features.extend(list(sorted_quantile(begin_or_inside, quants)))

                    # The probability that words on either edge of the current sub-sequence belong to the class of interest
                    features.append(prob_or(text_preds[pred_start-1:pred_start])[0] if pred_start > 0 else 0)
                    features.append(prob_or(text_preds[pred_end:pred_end+1])[0] if pred_end < num_words else 0)

                    # The probability that the first word corresponds to a 'B'-egin token
                    features.append(text_preds[pred_start,disc_begin])

                    exact_match = None

                    true_pos = None

                    # For efficiency, use a numpy array instead of a list that doubles in size when full to conserve constant "append" time complexity
                    if X_ind >= X.shape[0]:
                        new_X = np.empty((X.shape[0]*2,13), dtype=np.float32)
                        new_X[:X.shape[0]] = X
                        X = new_X
                    X[X_ind] = features
                    X_ind += 1
                    
                    y.append(exact_match)
                    truePos.append(true_pos)
                    wordRanges.append((np.int16(pred_start), np.int16(pred_end)))
                    groups.append(np.int16(text_i))

    return SeqDataset(X[:X_ind], y, groups, wordRanges, truePos)


# In[ ]:


def sorted_quantile(array, q):
    array = np.array(array)
    n = len(array)
    index = (n - 1) * q
    left = np.floor(index).astype(int)
    fraction = index - left
    right = left
    right = right + (fraction > 0).astype(int)
    i, j = array[left], array[right]
    return i + (j - i) * fraction


# In[ ]:


def predict_strings(disc_type, probThresh, test_groups, train_ind=None):
    string_preds = []
    submitSeqDs = submitSeqSets[disc_type]
    
    # Average the probability predictions of a set of classifiers
    get_tp_prob = lambda testDs, classifiers: np.mean([clf.predict_proba(testDs.features)[:,1] for clf in classifiers], axis=0) if testDs.features.shape[0] > 0 else np.array([])
    
    
    # Point to submission set values
    predict_df = test_texts
    text_df = test_texts
    groupIdx = np.isin(submitSeqDs.groups, test_groups)
    testDs = SeqDataset(submitSeqDs.features[groupIdx], submitSeqDs.labels[groupIdx], submitSeqDs.groups[groupIdx], submitSeqDs.wordRanges[groupIdx], submitSeqDs.truePos[groupIdx])

    # Classifiers are always loaded from disc during submission
    with open( f"../input/fb-deberta-3/FB_deberta-large-1600-2-itpt-ens-clf/clfs/{disc_type}_clf.p", "rb" ) as clfFile:
        classifiers = pickle.load( clfFile )  
    prob_tp = get_tp_prob(testDs, classifiers)
        
    text_to_seq = {}
    for text_idx in test_groups:
        # The probability of true positive and (start,end) of each sub-sequence in the curent text
        prob_tp_curr = prob_tp[testDs.groups == text_idx]
        word_ranges_curr = testDs.wordRanges[testDs.groups == text_idx]
        sorted_seqs = list(reversed(sorted(zip(prob_tp_curr, [tuple(wr) for wr in word_ranges_curr]))))
        text_to_seq[text_idx] = sorted_seqs
    
    for text_idx in test_groups:
        
        i = 1
        split_text = text_df.loc[text_df.id == predict_df.id.values[text_idx]].iloc[0].text.split()
        
        # Start and end word indices of sequence candidates kept in sorted order for efficiency
        starts = []
        ends = []
        
        # Include the sub-sequence predictions in order of predicted probability
        for prob, wordRange in text_to_seq[text_idx]:
            
            # Until the predicted probability is lower than the tuned threshold
            if prob < probThresh: break
                
            # Binary search already-placed word sequence intervals, and insert the new word sequence interval if it does not intersect an existing interval.
            insert = bisect_left(starts, wordRange[0])
            if (insert == 0 or ends[insert-1] <= wordRange[0]) and (insert == len(starts) or starts[insert] >= wordRange[1]):
                starts.insert(insert, wordRange[0])
                ends.insert(insert, wordRange[1])
                string_preds.append((predict_df.id.values[text_idx], disc_type, ' '.join(map(str, list(range(wordRange[0], wordRange[1]))))))
                i += 1     
    return string_preds

def sub_df(string_preds):
    return pd.DataFrame(string_preds, columns=['id','class','predictionstring'])
    
# Convert skopt's uniform distribution over the tuning threshold to a distribution that exponentially decays from 100% to 0%
def prob_thresh(x): 
    return .01*(100-np.exp(100*x))

# Convert back to the scalar supplied by skopt
def skopt_thresh(x): 
    return np.log((x/.01-100.)/-1.)/100.


# In[ ]:


def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end]])


def link_evidence(oof):
    thresh = 1
    idu = oof['id'].unique()
    idc = idu[1]
    eoof = oof[oof['class'] == "Evidence"]
    neoof = oof[oof['class'] != "Evidence"]
    for thresh2 in range(26,27, 1):
        retval = []
        for idv in idu:
            for c in  ['Lead', 'Position', 'Evidence', 'Claim', 'Concluding Statement',
                   'Counterclaim', 'Rebuttal']:
                q = eoof[(eoof['id'] == idv) & (eoof['class'] == c)]
                if len(q) == 0:
                    continue
                pst = []
                for i,r in q.iterrows():
                    pst = pst +[-1] + [int(x) for x in r['predictionstring'].split()]
                start = 1
                end = 1
                for i in range(2,len(pst)):
                    cur = pst[i]
                    end = i
                    #if pst[start] == 205:
                    #   print(cur, pst[start], cur - pst[start])
                    if (cur == -1 and c != 'Evidence') or ((cur == -1) and ((pst[i+1] > pst[end-1] + thresh) or (pst[i+1] - pst[start] > thresh2))):
                        retval.append((idv, c, jn(pst, start, end)))
                        start = i + 1
                v = (idv, c, jn(pst, start, end+1))
                #print(v)
                retval.append(v)
        roof = pd.DataFrame(retval, columns = ['id', 'class', 'predictionstring']) 
        roof = roof.merge(neoof, how='outer')
        return roof


# In[ ]:


model_dict = dict(
#         deberta_base_v2 = dict(
#             model_name = "../input/fb-deberta/deberta-base/deberta-base",
#             config = "../input/fb-deberta/deberta-base/deberta-base/config.json",
#             weights = [f"../input/fb-deberta-2/FB_deberta-base-PP-V2/FB_deberta-base-PP-V2/FB_deberta-base-v2/models/model_{fold}" for fold in range(5)]
#         ),
#     funnel_transformer_v2 = dict(
#             model_name = "../input/fb-funnel-transformer/funnel-transformer-intermediate/funnel-transformer-intermediate",
#             config = "../input/fb-funnel-transformer/funnel-transformer-intermediate/funnel-transformer-intermediate/config.json",
#             weights = [f"../input/fb-funnel-transformer/funnel-transformer-intermediate-PP-v2/funnel-transformer-intermediate-weights/models/model_{fold}" for fold in range(5)]
#         ),
    deberta_large_v2 = dict(
            model_name = "../input/fb-deberta/deberta-large/deberta-large",
            config = "../input/fb-deberta/deberta-large/deberta-large/config.json",
        weights = ["../input/fb-deberta-2/FB_deberta-large-ITPT-tuned-PP-v2/deberta-large-ITPT-tuned/models/model_1",
                  "../input/fb-deberta-2/FB_deberta_large_PP_v2/FB_deberta_large_PP_v2/deberta-large-v2/models/model_2",
                  "../input/fb-deberta-2/FB_deberta_large_PP_v2/FB_deberta_large_PP_v2/deberta-large-v2/models/model_3",
                  "../input/fb-deberta-2/FB_deberta_large_PP_v2/FB_deberta_large_PP_v2/deberta-large-v2/models/model_4",]
#             weights = [f"../input/fb-deberta-2/FB_deberta-large-ITPT-tuned-PP-v2/deberta-large-ITPT-tuned/models/model_{fold}" for fold in range(5)]
        )
)


# In[ ]:


if __name__ == "__main__":
    test_texts = get_texts("../input/feedback-prize-2021/test/")
    for key, item in model_dict.items():        
        manager = Manager()
        tokenizer = AutoTokenizer.from_pretrained(item["model_name"])
        test_dataset = feedbackDataset(test_texts, tokenizer, Config.max_len, True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size = Config.valid_batch_size,
                                                pin_memory = True,
                                                num_workers = Config.num_workers,  
                                                shuffle = False)
        model = FeedbackModel(item["config"],
                             Config.num_labels)
        model.to(Config.device)
        predictions = inference(model, test_loader, item["weights"])
        uniqueSubmitGroups = range(len(predictions))
        
        print('Making submit sequence datasets...')
        submitSeqSets = manager.dict()
        
        def sequenceDataset(disc_type, submit=False):
            print(f"Making {disc_type} dataset")
            submitSeqSets[disc_type] = seq_dataset(disc_type, predictions)
            
        Parallel(n_jobs=-1, backend='multiprocessing')(
                delayed(sequenceDataset)(disc_type, True) 
               for disc_type in disc_type_to_ids
            )
        print('Done.')
        seq_cache = {} 
        clfs = []
        thresholds = {}
        for disc_type in disc_type_to_ids:
            with open( f"../input/fb-deberta-3/FB_deberta-large-1600-2-itpt-ens-result/results/{disc_type}_res.p", "rb" ) as res_file:
                train_result = pickle.load( res_file )  
            thresholds[disc_type] = train_result['pred_thresh']
            print(disc_type, train_result)
            
        sub = pd.concat([sub_df(predict_strings(disc_type, thresholds[disc_type], uniqueSubmitGroups)) for disc_type in disc_type_to_ids ]).reset_index(drop=True)
        sub.to_csv("submission.csv", index = None)


# In[ ]:


sub.head()


# In[ ]:




