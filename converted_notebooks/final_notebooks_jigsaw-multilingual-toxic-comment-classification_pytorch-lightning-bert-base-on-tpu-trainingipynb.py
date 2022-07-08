#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --version 1.7 --apt-packages libomp5 libopenblas-dev')


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install pytorch-lightning==1.1.5\n!pip install transformers\n')


# In[ ]:


import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoModel, AutoTokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import  ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

RANDOM_SEED = 716

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

pl.seed_everything(RANDOM_SEED)


# In[ ]:


get_ipython().system('ls ../input/jigsaw-multilingual-toxic-comment-classification/')


# In[ ]:


# load data as pandas dataframe
train_df = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train_df.head()


# In[ ]:


train_df.shape


# In[ ]:


test_df = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/test.csv")


# In[ ]:


# split data to train and valid
train_df, val_df = train_test_split(train_df, test_size=0.1)
train_df.shape, val_df.shape, test_df.shape


# In[ ]:


# distribution of labels
LABEL_COLUMNS = train_df.columns.tolist()[2:]
train_df[LABEL_COLUMNS].sum().sort_values().plot(kind="barh");


# In[ ]:


# distribution of clean and toxic comments
train_toxic = train_df[train_df[LABEL_COLUMNS].sum(axis=1) > 0]
train_clean = train_df[train_df[LABEL_COLUMNS].sum(axis=1) == 0]

pd.DataFrame(dict(
  toxic=[len(train_toxic)], 
  clean=[len(train_clean)]
)).plot(kind='barh');


# In[ ]:


# sample clean comments and keep all the toxic comments
train_df = pd.concat([
  train_toxic,
  train_clean.sample(20000)
])

train_df.shape, val_df.shape


# In[ ]:


# define the model to use from huggingface
BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)


# In[ ]:


# check a sample visually

sample_row = train_df.iloc[456]
sample_comment = sample_row.comment_text
sample_labels = sample_row[LABEL_COLUMNS]

print(sample_comment)
print()
print(sample_labels.to_dict())


# In[ ]:


# encode the sample comment
encoding = tokenizer.encode_plus(
    sample_comment,
    add_special_tokens=True,
    max_length=128,
    return_token_type_ids=False,
    padding="max_length",
    return_attention_mask=True,
    return_tensors="pt"
)

encoding.keys()


# In[ ]:


# define max token count as 128
MAX_TOKEN_COUNT = 128


# In[ ]:


# define dataset
class ToxicDataset(Dataset):

    def __init__(self,
               data,
               tokenizer,
               max_token_len=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

  
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data_row = self.data.iloc[idx]

        comment_text = data_row.comment_text
        labels = data_row[LABEL_COLUMNS]

        encoding = self.tokenizer.encode_plus(
          comment_text,
          add_special_tokens=True,
          max_length=self.max_token_len,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

        return dict(
          comment_text=comment_text,
          input_ids=encoding["input_ids"].flatten(),
          attention_mask=encoding["attention_mask"].flatten(),
          labels=torch.FloatTensor(labels)
        )


# In[ ]:


# create dataset for train
train_dataset = ToxicDataset(
    train_df,
    tokenizer,
    max_token_len=MAX_TOKEN_COUNT
)

# pick out a sample item
sample_item = train_dataset[0]


# In[ ]:


# check the shape
sample_item['input_ids'].shape, sample_item['attention_mask'].shape


# In[ ]:


# load the model
model = AutoModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)


# In[ ]:


# try passing a sample batch through the model and check the output shape
sample_batch = next(iter(DataLoader(train_dataset, batch_size=8, num_workers=2)))
sample_batch['input_ids'].shape, sample_batch['attention_mask'].shape


# In[ ]:


# make outputs
output = model(sample_batch['input_ids'], sample_batch['attention_mask'])


# In[ ]:


# check shapes
output.last_hidden_state.shape, output.pooler_output.shape


# In[ ]:


"""
    Datamodule for Lightning
"""

class ToxicDataModule(pl.LightningDataModule):

    def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):

        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):

        self.train_dataset = ToxicDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.test_dataset = ToxicDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )


# In[ ]:


# set epochs and batch_size
N_EPOCHS = 10
BATCH_SIZE = 32

# create datamodule to be used in trainer later
data_module = ToxicDataModule(
    train_df,
    val_df,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_token_len=MAX_TOKEN_COUNT
)


# In[ ]:


from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from transformers import AdamW, get_linear_schedule_with_warmup

THRESHOLD = 0.5

"""
    Define the model class extending LightningModule
    
"""

class ToxicModel(pl.LightningModule):

    def __init__(self, n_classes=len(LABEL_COLUMNS), n_training_steps=None, n_warmup_steps=None):

        super().__init__()

        self.base = AutoModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.base.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss() # loss function

  
    def forward(self, input_ids, attention_masks, labels=None):

        output = self.base(input_ids, attention_mask=attention_masks)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)

        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)

        return loss, output

    def training_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_masks = batch["attention_mask"]
        labels = batch["labels"]

        loss, output = self(input_ids, attention_masks, labels)

        self.log("train_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": output, "labels": labels}

    def validation_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_masks = batch["attention_mask"]
        labels = batch["labels"]

        loss, output = self(input_ids, attention_masks, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)

        return {"val_loss": loss, "predictions": output, "labels": labels}

    def test_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_masks = batch["attention_mask"]
        labels = batch["labels"]

        loss, output = self(input_ids, attention_masks, labels)

        self.log("test_loss", loss, prog_bar=True, logger=True)

        return loss

  
    def training_epoch_end(self, training_outputs):

        labels = []
        outputs = []

        for output in training_outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_preds in output["predictions"].detach().cpu():
                outputs.append(out_preds)

        labels = torch.stack(labels).int()
        preds = torch.stack(outputs)

        for i, name in enumerate(LABEL_COLUMNS):
            class_roc_auc = auroc(preds[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc,
                                            self.current_epoch)
        
        preds = torch.where(preds >= THRESHOLD, 1., 0.)
        train_acc = accuracy(preds, labels)
        self.log("train_acc", train_acc, prog_bar=True, logger=True)
        
            
    def validation_epoch_end(self, val_outputs):

        labels = []
        outputs = []

        for output in val_outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_preds in output["predictions"].detach().cpu():
                outputs.append(out_preds)

        labels = torch.stack(labels).int()
        preds = torch.stack(outputs)

        preds = torch.where(preds >= THRESHOLD, 1., 0.)
        val_acc = accuracy(preds, labels)
        self.log("val_acc", val_acc, prog_bar=True, logger=True)
      

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=self.n_warmup_steps,
          num_training_steps=self.n_training_steps
        )

        return dict(
          optimizer=optimizer,
          lr_scheduler=dict(
            scheduler=scheduler,
            interval='step'
          )
        )


# In[ ]:


steps_per_epoch = len(train_df) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS

warmup_steps = total_training_steps // 5

warmup_steps, total_training_steps


# In[ ]:


model = ToxicModel(
    n_classes=len(LABEL_COLUMNS),
    n_warmup_steps=warmup_steps,
    n_training_steps=total_training_steps
)


# In[ ]:


_, predictions = model(sample_batch["input_ids"], sample_batch["attention_mask"])
predictions


# In[ ]:


accuracy(torch.where(predictions > 0.5, 1, 0), sample_batch["labels"])


# In[ ]:


get_ipython().system('rm -rf lightning_logs/')
get_ipython().system('rm -rf checkpoints/')


# In[ ]:


checkpoint_callback = ModelCheckpoint(
  dirpath="checkpoints",
  filename="best-checkpoint",
  save_top_k=1,
  verbose=True,
  monitor="val_loss",
  mode="min"
)

logger = TensorBoardLogger("lightning_logs", name="toxic-comments")

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)


# In[ ]:


trainer = pl.Trainer(
  logger=logger,
  callbacks=[early_stopping_callback, checkpoint_callback],
  max_epochs=N_EPOCHS,
  tpu_cores=8,
  progress_bar_refresh_rate=30
)


# In[ ]:


trainer.fit(model, data_module)


# In[ ]:


trainer.test()


# In[ ]:


trained_model = ToxicModel.load_from_checkpoint(
  trainer.checkpoint_callback.best_model_path,
  n_classes=len(LABEL_COLUMNS),
  n_training_steps=1,
  n_warmup_steps=1
)
trained_model.eval()
trained_model.freeze()


# In[ ]:


trained_model.device


# In[ ]:


test_comment = "Hi, I'm Meredith and I'm an alch... good at supplier relations"

encoding = tokenizer.encode_plus(
  test_comment,
  add_special_tokens=True,
  max_length=128,
  return_token_type_ids=False,
  padding="max_length",
  return_attention_mask=True,
  return_tensors='pt',
)

_, test_prediction = trained_model(encoding["input_ids"], encoding["attention_mask"])
test_prediction = test_prediction.flatten().numpy()

for label, prediction in zip(LABEL_COLUMNS, test_prediction):
    print(f"{label}: {prediction}")


# In[ ]:




