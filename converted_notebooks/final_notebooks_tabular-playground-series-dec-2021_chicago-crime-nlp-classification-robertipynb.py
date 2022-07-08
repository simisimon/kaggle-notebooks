#!/usr/bin/env python
# coding: utf-8

# # NLP Binary Classification

# ## Goal: Predict probability of arrest based on crime class + details + locations
# 
# ### Training: Chicago crime dataset 2018-2020
# ### Test: Chicago crime dataset 2021
# 
# The goal of binary text classification is to classify a text sequence into one of two classes. A transformer-based binary text classification model typically consists of a transformer model with a classification layer on top of it. The classification layer will have two output neurons, corresponding to each class.
# 
# https://simpletransformers.ai/docs/binary-classification/

# In[ ]:


get_ipython().system('pip install simpletransformers')


# In[ ]:


from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import torch
import glob
import os

# from google.colab import drive  # for Colab only
# drive.mount('/content/drive')

cuda_available = torch.cuda.is_available()
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# ## Data Preprocessing

# In[ ]:


# path = r'/content/drive/MyDrive/crime_taxi/'  # for Colab only
path = r'../input/chicago-crime-dataset-2018-to-2021/' 

all_files = glob.glob(os.path.join(path , "Crimes*.csv"))
train_files = [path + 'Crimes_-_2019.csv', path + 'Crimes_-_2020.csv', path + '/Crimes_-_2018.csv']
valid_files = [path + '/Crimes_-_2021.csv']
print("train_files:",train_files, "\nvaild_files:", valid_files)

train_df = pd.concat((pd.read_csv(f, usecols=['Primary Type','Description','Arrest','Location Description']) for f in train_files))
valid_df = pd.concat((pd.read_csv(f, usecols=['Primary Type','Description','Arrest','Location Description']) for f in valid_files))


# In[ ]:


train_df.head()


# In[ ]:


train_df = train_df.dropna()
valid_df = valid_df.dropna()

print(train_df.isnull().sum())
print(valid_df.isnull().sum())

train_df["text"] = train_df["Primary Type"] +": "+ train_df["Description"] +" AT "+ train_df["Location Description"] 
train_df["text"] = train_df["text"].str.lower()
train_df["labels"] = train_df["Arrest"].astype(int) # 0 for unarrested and 1 for arrested
train_df = train_df.drop(columns=["Primary Type", "Description", "Arrest","Location Description"])

valid_df["text"] = valid_df["Primary Type"] +": "+ valid_df["Description"] +" AT "+ valid_df["Location Description"] 
valid_df["text"] = valid_df["text"].str.lower()
valid_df["labels"] = valid_df["Arrest"].astype(int) # 0 for unarrested and 1 for arrested
valid_df = valid_df.drop(columns=["Primary Type", "Description", "Arrest","Location Description"])

# valid_df = train_df.sample(frac=0.1, random_state=66)
# train_df = train_df.drop(valid_df.index)


# # save to tsv for lazy_loading
# train_df.to_csv("/content/drive/MyDrive/crime_taxi/Crimes_-_2001_to_Present_train.tsv", sep="\t", index=False)
# valid_df.to_csv("/content/drive/MyDrive/crime_taxi/Crimes_-_2001_to_Present_valid.tsv", sep="\t", index=False)

print(train_df.head())
print(valid_df.head())

print("train_df.shape:", train_df.shape,"valid_df.shape:", valid_df.shape) #(741337, 2) (207087, 2)


# ## Optional model configuration

# In[ ]:


# model_args = ClassificationArgs(num_train_epochs=1)

model_args = {
    "num_train_epochs": 1,
    "train_batch_size": 8,
    "overwrite_output_dir": True,
    # "do_lower_case": True,
    "no_cache": True,
    "best_model_dir": "./",
    # "lazy_loading": True,
    # "lazy_labels_column": 1,
    # "lazy_text_column": 0,
}


# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "roberta-base", args=model_args
)

model.args.use_multiprocessing = False
model.args.use_multiprocessing_for_evaluation = False

# Train the model
model.train_model(train_df)

# # Train the model with lazy_loading
# model.train_model("/content/drive/MyDrive/crime_taxi/Crimes_-_2001_to_Present_train.tsv", eval_data="/content/drive/MyDrive/crime_taxi/Crimes_-_2001_to_Present_valid.tsv")


# ## Evaluate the model

# In[ ]:


result, model_outputs, wrong_predictions = model.eval_model(valid_df)


# ## Make predictions with the model

# In[ ]:


predictions, raw_outputs = model.predict(["motor vehicle theft: $500 and under on bridge",
                    "interference with public officer: kidnapping on street"])
predictions, raw_outputs

