#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# This is a starter notebook for [U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/overview) using JAX and Flax. In this competition, you will train your models on a novel semantic similarity dataset to extract relevant information by matching key phrases in patent documents.
# 
# ### About Dataset
# The dataset contains pairs of phrases (an anchor and a target phrase) and asked to rate how similar they are on a scale from 0 (not at all similar) to 1 (identical in meaning).
# 
# **Files**
# 
# - `train.csv` - the training set, containing phrases, contexts, and their similarity scores
# - `test.csv` - the test set set, identical in structure to the training set but without the score
# - `sample_submission.csv` - a sample submission file in the correct format
# 
# **Columns** 
# 
# - `id` - a unique identifier for a pair of phrases
# - `anchor` - the first phrase
# - `target` - the second phrase
# - `context` - the CPC classification (version 2021.05), which indicates the subject within which the similarity is to be scored
# - `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.
# 
# ### Intro to JAX
# [JAX](https://github.com/google/jax) is a framework which is used for high-performance numerical computing and machine learning research developed at [Google Research](https://research.google/) teams. It allows you to build Python applications with a NumPy-consistent API that specializes in differentiating, vectorizing, parallelizing, and compiling to GPU/TPU Just-In-Time. JAX was designed with performance and speed as a first priority, and is natively compatible with common machine learning accelerators such as [GPUs](https://www.kaggle.com/docs/efficient-gpu-usage) and [TPUs](https://www.kaggle.com/docs/tpu). Large ML models can take ages to train -- you might be interested in using JAX for applications where speed and performance are particularly important!
# ### When to use JAX vs TensorFlow?
# [TensorFlow](https://www.tensorflow.org/guide) is a fantastic product, with a rich and fully-featured ecosystem, capable of supporting most every use case a machine learning practitioner might have (e.g. [TFLite](https://www.tensorflow.org/lite) for on-device inference computing, [TFHub](https://tfhub.dev/) for sharing pre-trained models, and many additional specialized applications as well). This type of broad mandate both contrasts and compliments JAX's philosophy, which is more narrowly focused on speed and performance.  We recommend using JAX in situations where you do want to maximize speed and performance but you do not require any of the long tail of features and additional functionalities that only the [TensorFlow ecosystem](https://www.tensorflow.org/learn) can provide.
# ### Intro to the FLAX
# Just like [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) focuses on speed, other members of the JAX ecosystem are encouraged to specialize as well.  For example, [Flax](https://flax.readthedocs.io/en/latest/) focuses on neural networks and [jgraph](https://github.com/deepmind/jraph) focuses on graph networks.  
# 
# [Flax](https://flax.readthedocs.io/en/latest/) is a JAX-based neural network library that was initially developed by  Google Research's Brain Team (in close collaboration with the JAX team) but is now open source.  If you want to train machine learning models on GPUs and TPUs at an accelerated speed, or if you have an ML project that might benefit from bringing together both [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), consider using [Flax](https://flax.readthedocs.io/en/latest/) for your next project! [Flax](https://flax.readthedocs.io/en/latest/) is especially well-suited for projects that use large language models, and is a popular choice for cutting-edge [machine learning research](https://arxiv.org/search/?query=JAX&searchtype=all&abstracts=show&order=-announced_date_first&size=50).
# 

# ### **Imports**
# Importing the required libraries for this notebook

# In[ ]:


import os, re
import time
import jax
import flax
import optax
import datasets
import pandas as pd 
import numpy as np
from jax import jit
import jax.numpy as jnp
import tensorflow as tf
from flax.training import train_state
from itertools import chain
from tqdm.notebook import tqdm
from typing import Callable
from flax import traverse_util
from datasets import load_dataset, load_metric ,Dataset,list_metrics,load_from_disk, concatenate_datasets
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from transformers import FlaxAutoModelForSequenceClassification, AutoConfig, AutoTokenizer, BertTokenizer,AutoModelForSequenceClassification,RobertaTokenizer
import warnings
warnings.filterwarnings("ignore")
import pyarrow as pa
from scipy import stats
from scipy.stats import pearsonr
# to suppress warnings caused by cuda version
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ### **Loading the model checkpoint and tokenizer**
# We'll be using Huggingface's pre-trained [bert model](https://huggingface.co/bert-base-uncased) in this notebook.

# In[ ]:


seed=0
model_checkpoint = '../input/huggingface-bert-variants/bert-base-uncased/bert-base-uncased' 
tokenizer = BertTokenizer.from_pretrained(model_checkpoint,use_fast=True)


# ### **Loading and preprocess the data**
# We'll start with reading train data and looking at the first five rows.

# In[ ]:


train=pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv')
train.head()


# [`sep_token`](https://huggingface.co/docs/transformers/main_classes/tokenizer) is a special token separating two different sentences in the same input used by BERT

# In[ ]:


sep = tokenizer.sep_token
sep


# In[ ]:


# Combine context, anchor and target columns
train['inputs'] = train.context + sep + train.anchor + sep + train.target


# In[ ]:


train.head()


# ### Label Encoding
# We're using JAX to train a classifier model to predict score labels in this notebook. The training data score is separated into five intervals, each of which can be classified into one of five classes [0,1,2,3,4].

# In[ ]:


# https://www.kaggle.com/code/vbookshelf/hugging-face-trainer-a-classification-workflow
def create_label(x):
    
    if x == 0:
        return 0

    if x == 0.25:
        return 1
    
    if x == 0.5:
        return 2

    if x == 0.75:
        return 3

    if x == 1.0:
        return 4

train['labels'] = train['score'].apply(create_label)


# In[ ]:


train.head()


# Converting train dataframe to HuggingFace Dataset using Huggingface's [`from_pandas`](https://huggingface.co/docs/datasets/loading) function from Dataset class 

# In[ ]:


train_ds = Dataset.from_pandas(train)


# In[ ]:


train_ds


# Now, this function will preprocess the dataset by taking batch of data and returns the tokenized processed data

# In[ ]:


def preprocess_function(input_batch):
    '''
    INPUT - input batch from from original dataset
    RETURNS preprocessed data
    '''
    texts = (input_batch["inputs"],)
    processed = tokenizer(*texts, padding="max_length", max_length=128, truncation=True)
    processed["labels"] = input_batch["labels"]
    return processed


# In[ ]:


encoded_ds = train_ds.map(preprocess_function,batched=True, remove_columns= ['id', 'anchor', 'target', 'context', 'labels','score', 'inputs'])


# Spliting the dataset into train and eval sets

# In[ ]:


encoded_ds = encoded_ds.train_test_split(test_size=0.2)
encoded_ds


# In[ ]:


train_dataset = encoded_ds["train"]
validation_dataset = encoded_ds["test"]


# ### **Model Config**
# Defining all the model config parameters below

# In[ ]:


num_labels = 5
seed = 0
num_train_epochs = 5
learning_rate = 2e-5
per_device_batch_size = 128
weight_decay=1e-2


# In[ ]:


total_batch_size = per_device_batch_size * jax.local_device_count()
print("The overall batch size (both for training and eval) is", total_batch_size)


# Evaluating num of train steps and defining learning_rate_function using optax for jax.
# 
# Here I am using [cosine onecycle learning rate scheduler](https://optax.readthedocs.io/en/latest/api.html#optax.cosine_onecycle_schedule) from optax library

# In[ ]:


num_train_steps = len(train_dataset) // total_batch_size * num_train_epochs
learning_rate_function = optax.cosine_onecycle_schedule(transition_steps=num_train_steps, peak_value=learning_rate, pct_start=0.1)


# ### **Evaluation Metrics**
# For this notebook, we'll be using [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) to evaluate similarity between the predicted and actual scores.

# In[ ]:


def simple_corr(preds, labels):
    preds = preds.reshape(len(preds))
    corr,_ = pearsonr(preds,labels)
    return corr

class CORR(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="Calculates PearsonR metric.",
            citation="TODO: _CITATION",
            inputs_description="_KWARGS_DESCRIPTION",
            features=datasets.Features({
                'predictions': datasets.Value('float32'),
                'references': datasets.Value('float32'),
            }),
            codebase_urls=[],
            reference_urls=[],
            format='numpy'
        )

    def _compute(self, predictions, references):
        return {"PEARSONR": simple_corr(predictions, references)}
    
metric = CORR()


# In[ ]:


metric


# Below step is downloading the pretrained model,as we are doing sentence classification we can use FlaxAutoModelForSequenceClassification class. from_pretrained method will download and cache the model.
# 
# 

# In[ ]:


config = AutoConfig.from_pretrained(model_checkpoint, num_labels=num_labels)
model = FlaxAutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config, seed=seed)


# ### **Train state**

# Flax provides a class [flax.training.train_state.TrainState](https://flax.readthedocs.io/en/latest/flax.training.html#train-state), which stores the model parameters, the loss function, the optimizer, and exposes an apply_gradients function to update the model's weight parameters.

# In[ ]:


class TrainState(train_state.TrainState):
    '''
    Derived TrainState class that saves the forward pass of the model as an eval function and a loss function
    '''
    logits_function: Callable = flax.struct.field(pytree_node=False)
    loss_function: Callable = flax.struct.field(pytree_node=False)


# Here we are using Adam optimizer with weight decay, and again we are using [optax](https://optax.readthedocs.io/en/latest/) library.
# 
# Here is the interesting article on adam optimizer - https://www.fast.ai/2018/07/02/adam-weight-decay/

# In[ ]:


def decay_mask_fn(params):
    '''
    This function's task is to make sure that weight decay is not applies to any bias or Layernorm weights
    '''
    flat_params = traverse_util.flatten_dict(params)
    flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)


# In[ ]:


def adamw(weight_decay):
    return optax.adamw(learning_rate=learning_rate_function, b1=0.9, b2=0.999, eps=1e-6, weight_decay=weight_decay,mask=decay_mask_fn)
adamw = adamw(weight_decay)


# Now , computing the softmax cross entropy between sets of logits and labels using the [optax](https://optax.readthedocs.io/en/latest/) library.

# In[ ]:


## Defining the loss and the evaluation function
@jit
def loss_function(logits, labels):
    xentropy = optax.softmax_cross_entropy(logits, onehot(labels, num_classes=num_labels))
    return jnp.mean(xentropy)

@jit
def eval_function(logits):
    return logits.argmax(-1)


# In[ ]:


# Instantiate a TrainState.
state = TrainState.create(
    apply_fn=model.__call__,
    params=model.params,
    tx=adamw,
    logits_function=eval_function,
    loss_function=loss_function,
)


# ### **Training and evaluate functions**
# We'd pass state, batch, and dropout rng to the train function, which would return new state, metrics, and new dropout rng. We'll define a loss function that runs the forward pass in the train function. Then we'll use [`jax.value_and_grad`](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html) to differentiate this loss function. The mean gradient will then be computed across all devices. The gradients will then be applied to the weights. Finally, the train step function will be parallelized over all accessible machines.

# In[ ]:


def train_step(state, batch, dropout_rng):
    # take targets
    targets = batch.pop("labels")
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
    
    #define loss function which runs the forward pass 
    def loss_function(params):
        logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
        loss = state.loss_function(logits, targets)
        return loss
    
    grad_fn = jax.value_and_grad(loss_function) #differentiate the loss function
    loss, grad = grad_fn(state.params) 
    grad = jax.lax.pmean(grad, "batch") #compute the mean gradient over all devices
    new_state = state.apply_gradients(grads=grad) #applies the gradients to the weights.
    metrics = jax.lax.pmean({'loss': loss, 'learning_rate': learning_rate_function(state.step)}, axis_name='batch')
    
    return new_state, metrics, new_dropout_rng
parallel_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,)) # parallelized training over all TPU devices


# In[ ]:


# Define evaluation step
def eval_step(state, batch):
    logits = state.apply_fn(**batch, params=state.params, train=False)[0] #stack the model's forward pass with the logits function
    return state.logits_function(logits)
parallel_eval_step = jax.pmap(eval_step, axis_name="batch")


# ### **Data loader**
# 

# In[ ]:


# Returns batch model input
# 1. define random permutation 
# 2. randomized dataset is extracted and then it converted to a JAX array and sharded over all local TPU devices.
def train_data_loader(rng, dataset, batch_size):
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)
        yield batch
        
# similar to train data loader 
def eval_data_loader(dataset, batch_size): 
    for i in range(len(dataset) // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)
        yield batch


# In[ ]:


# replicate/copy the weight parameters on each device, to passthem to our pmapped functions.
state = flax.jax_utils.replicate(state)


# In[ ]:


# generating a seeded PRNGKey for the dropout layers and dataset shuffling.
rng = jax.random.PRNGKey(seed)
dropout_rngs = jax.random.split(rng, jax.local_device_count())


# ### **Training**
# Now, we'll define the training loop and train the pre-trained model. Each epoch includes a training phase for each batch. After an epoch has been finished, we can provide training metrics and conduct an evaluation. To begin, we'll perform the train step, which entails loading data batches using the `train_data_loader` defined above, then applying the `parallel_train_step` defined above, then taking data batches from the `eval_data_loader`, and then implementing the `parallel_eval_step` function defined above to obtain predictions, and evaluating metrics to check the model's performance.

# In[ ]:


start = time.time()
# Full training loop
for i, epoch in enumerate(tqdm(range(1, num_train_epochs + 1), desc=f"Epoch ...", position=0, leave=True)):
    rng, input_rng = jax.random.split(rng)

    # train
    with tqdm(total=len(train_dataset) // total_batch_size, desc="Training...", leave=False) as progress_bar_train:
        for batch in train_data_loader(input_rng, train_dataset, total_batch_size):
            state, train_metrics, dropout_rngs = parallel_train_step(state, batch, dropout_rngs)
            progress_bar_train.update(1)

    # evaluate
    with tqdm(total=len(validation_dataset) // total_batch_size, desc="Evaluating...", leave=False) as progress_bar_eval:
          for batch in eval_data_loader(validation_dataset, total_batch_size):
                labels = batch.pop("labels")
                predictions = parallel_eval_step(state, batch)
                metric.add_batch(predictions=chain(*predictions), references=chain(*labels))
                progress_bar_eval.update(1)
    eval_metric = metric.compute()
    loss = round(flax.jax_utils.unreplicate(train_metrics)['loss'].item(), 3)
    eval_score = round(list(eval_metric.values())[0],5)
    metric_name = list(eval_metric.keys())[0]

    print(f"{i+1}/{num_train_epochs} | Train loss: {loss} | Eval {metric_name}: {eval_score}")
    
print("Total time: ", time.time() - start, "seconds")


# ### **Test Generation**
# Now we'll load and preprocess the test data in preparation for competition submission.

# In[ ]:


test=pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv')
submission=pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv')


# In[ ]:


# Let's have a look at the test data
test.head()


# In[ ]:


# Combine context, anchor and target columns
test['inputs'] = test.context + sep + test.anchor + sep + test.target


# In[ ]:


test.head()


# Converting test dataframe to HuggingFace Dataset using Huggingface's [`from_pandas`](https://huggingface.co/docs/datasets/loading) function from Dataset class 

# In[ ]:


test_ds = Dataset.from_pandas(test)


# In[ ]:


test_ds


# In[ ]:


# preprocess test dataset
def preprocess_test(input_batch):
    '''
    INPUT - input batch from from original dataset
    RETURNS preprocessed data
    '''
    texts = (input_batch["inputs"],)
    processed = tokenizer(*texts, padding="max_length", max_length=128, truncation=True)
    return processed


# In[ ]:


encoded_test = test_ds.map(preprocess_test,batched=True, remove_columns= ['id', 'anchor', 'target', 'context', 'inputs'])


# In[ ]:


# similar to train dataloader, it takes dataset and batch_size
def test_data_loader(dataset,batch_size):
    if len(dataset)<batch_size:
        batch = dataset[:]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)
        yield batch
    else:
        for i in range(len(dataset) // batch_size):
            batch = dataset[i * batch_size : (i + 1) * batch_size]
            batch = {k: jnp.array(v) for k, v in batch.items()}
            batch = shard(batch)
            yield batch
        batch = dataset[(i+1) * batch_size:]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)
        yield batch


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Running on test dataset , storing predictions \npreds=[]\nfor batch in test_data_loader(encoded_test, total_batch_size):\n    predictions = parallel_eval_step(state, batch)\n    preds.append(predictions[0])\n')


# In[ ]:


# convert it into numpy array
predictions=[]
for pred in preds:
    predictions.extend(np.array(pred))


# In[ ]:


test['preds'] = predictions


# In[ ]:


test.head()


# In[ ]:


# Change the preds back to the corresponding float values
def change_preds(x):
    
    if x == 0:
        return 0

    if x == 1:
        return 0.25
    
    if x == 2:
        return 0.5

    if x == 3:
        return 0.75

    if x == 4:
        return 1.0
    
test['modified_preds'] = test['preds'].apply(change_preds)


# In[ ]:


test.head()


# In[ ]:


cols = ['id', 'modified_preds']
df = test[cols]
modified_preds = df['modified_preds']
submission['score'] = modified_preds
submission.to_csv('submission.csv',index=False)


# ### **Conclusion**
# In this notebook, we've used a novel semantic similarity dataset to train a model to extract meaningful information from patent documents by matching key terms. We used [JAX](https://github.com/google/jax) and [FLAX](https://flax.readthedocs.io/en/latest/) to train the pre-trained BERT neural network from huggingface for this dataset. In this notebook, we utilised the classifier BERT model to predict score labels. Each of the five intervals in the training data score can be categorised into one of five classes [0,1,2,3,4]. The anticipated classes will be changed back to float increments after inference. Using this approach, we were able to achieve a public score of 0.6937.
# 
# Save the version of your notebook that you want to submit before proceeding. When it's finished, go to the data section and look for the output there. Select the option for Output. When you click the submit button in output, you'll be able to see how your model performed. Click the Edit button to reopen the notebook and return to your model to improve it.
# 
# 
# To see more examples of how to use [JAX](https://github.com/google/jax) and [FLAX](https://flax.readthedocs.io/en/latest/) with different data formats, please see this [discussion post](https://www.kaggle.com/discussions/getting-started/315696).  
# 
# Now, it's your turn to  create some amazing notebooks using [JAX](https://github.com/google/jax) and [FLAX](https://flax.readthedocs.io/en/latest/) for [U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/c/us-patent-phrase-to-phrase-matching) competition. 
# 
# ### **Works Cited**
# - https://flax.readthedocs.io/en/latest/index.html
# - https://github.com/google/flax/tree/main/examples
# - https://www.kaggle.com/heyytanay/sentiment-clf-jax-flax-on-tpus-w-b/notebook
# - https://www.kaggle.com/asvskartheek/bert-tpus-jax-huggingface/notebook
# - https://huggingface.co/docs/datasets/package_reference/main_classes.html#dataset
# -  https://colab.sandbox.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb#scrollTo=Mn1GdGpipfWK
# - https://www.kaggle.com/code/vbookshelf/hugging-face-trainer-a-classification-workflow
# - https://www.kaggle.com/code/yashvi/predict-book-review-rating-using-jax-flax/notebook#Test-Generation
