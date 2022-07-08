#!/usr/bin/env python
# coding: utf-8

# **A better baseline (Code repository: https://github.com/sattree/gpr_pub)**
# ===
# 
# In my previous kernel, namely 'Reproducing GAP results', I had briefly mentioned a Confidence Model. While we can train our models on the GAP domain to get better results, the burning question remains: **Can we do better** with existing resources and leverage Domain Adaptation techniques to establish a good baseline, perhaps even a strong baseline - a baseline stemming from existing work done on same or similar problems?
# 
# To demonstrate the strength of existing coref models and establish a domain transfer baseline, we **must not access labels from the GAP dataset**. This kernel investigates and demonstrates just that, **without any training on the GAP dataset**.
# 
# ***
# 
# This is the concluding kernel in this tri-series of self-contained installments to introduce the GPR problem.
# 1. Coref visualization - https://www.kaggle.com/sattree/1-coref-visualization-jupyter-allenlp-stanford
# 1. Reproducing GAP results - lays the foundation for this work - https://www.kaggle.com/sattree/2-reproducing-gap-results
# 1. **A better baseline - without any training**
# 
# The notebook only presents results and demonstrates model usage to avoid flooding it with code statements. The model codes are available at https://github.com/sattree/gpr_pub. Hope you will find the code structure easily navigable.
# ***
# **Executive Summary**
# ---
# 
# The primary contribution of this kernel is the design of a Confidence Model allowing existing pre-trained solutions to be applied to this problem.
# 
# **Confidence Model by Bootstrapping**
# 
# Given a set of m classifiers and their predicted labels, we can estimate the probability of a sample belonging to a particular class as follows:
# 
# $$P(\frac{y=A}{C_1,C_2,....,C_m}) = \frac{P(\frac{C_1,C_2,....,C_m}{y=A}) P(y=A)}{\sum_{k={A, B, NEITHER}} P(\frac{C_1,C_2,...,C_m}{y=k}) P(y=k)}$$
# 
# $$P(\frac{C_1,C_2,....,C_m}{y=A}) = P(\frac{C_1}{y=A})P(\frac{C_2}{C_1,y=A}) ... P(\frac{C_{m}}{C_1,C_2,...,C_{m-1},y=A})   $$
# 
# Barring sparsity issues, the probabilities in RHS above can be easily estimated from labeled training data. However, since we do not have access to true labels, we will take a bootstrapping approach by identifying proxy labels. We will introduce a bias that when the majority of a rich set of models agree on a label, then it must be a good proxy for the true label with high probability. This will seed our bootstrapping solution, albeit a noisy one.
# 
# In what follows, the above problem is formulated as a density estimation problem and an estimation procedure is described.
# 
# 1. **Generate predictions from constituent models**: Drifting from our approach in the previous kernel, we will allow each statistical model to assign multiple labels to a sample (multi-label classfication), meaning that a model can assign a pronoun to both 'A' and 'B', even though we know apriori that a given pronoun in the text will only resolve to either 'A' or 'B' (if at all) but not both. This is important from the perspective of confidence modelling, since this indicates that there is evidence that the pronoun refers to one of the two mentions, but not enough to know which one. This is in contrast to the case where there is evidence that the pronoun does not refer to either. We would like our confidence model to capture this behavior in the domain.
# 
# 1. **Identify a diverse set of models** (noisyness is not important here): To keep things simple, we will pre-select Lee et al (given that it is state of the art, and leverages most modern developments in lingusitic modelling generalization, such as ELMo embeddings, etc.). The qualitative superiority of this model both in terms of precision as well as recall is also illustrated below. Thereafter, we will perform univariate testing wrt to every other classifier in the bag. Non-parametric mutual information measure is used to estimate the correlation. Given the limited size of data, only rejection of classifiers can be done conclusively - hence the models with a degree of parity with Lee et all and presumably less diversity are rejected.
# 
# 1. **Density estimation on labels** to model the conditional distribution of classes and models: Given the noise in labels, this step becomes tricky.
#     1. **Choose a simple model** like Logistic Regression to avoid tuning to noise in the labels. LR is also known to produce well calibrated probabilities.
#     1. **Impose L1 regularization** to encourage classifier selection relevant for each target class, since not every classifier performs equally well for 'A', 'B', and 'NEITHER'. This also helps in weeding out noise in the labels.
#     1. **Model selection by cross-validation** Since the labels are noisy, we must make sure that the model does not fit perfectly to the data. We select models that give around 95% cross-validation performance as opposed to 100%.
#     
# 1. **Backoff to non-smoothed labels**: Density estimation from multiple models has a smoothing effect (as can be seen from before and after figures). While it is good to avoid a large negative penalty for misclassification, it also hurts our strong predictions for easy samples to some extent. Hence we will restore the majority labels for such samples that were supposedly 'easy samples' to begin with.
# 
# A confidence model is built to convert labels of models into probabilties to be compatible with logloss. The underlying coref models include a combination of heuristic and coref models mentioned in the GAP paper and a few others.
# 
# **Heuristic Models**
# 
# * Random - not included for confidence modeling as it enforces a prediction on all samples
# * Token Distance - not included for confidence modeling as it enforces a prediction on all samples
# * Syntactical Distance
# * Parallelism
# * URL
# 
# **Off the shelf Coref Solvers**
# * Lee et al (cited in the GAP paper) - end to end coref system, one of the most recent state-of-the art
# * AllenNLP
# * Huggingface - neural coref - not included in this kernel due to memory limitations, but the results can be found in the github repository
# * Stanford (cited in the GAP paper)  - includes three variants
#     1.     Deterministic coref model
#     1.     Statistical coref model
#     1.     Neural coref model
# * Berkeley Coref System (BCS) - built using a diverse set of interesting linguistic features
# * Wiseman et al (cited in the GAP paper) - I wanted to include this to leverage its diversity in features but couldn't get it to work.
# 
# Moving on, below is a summary of the results and findings:
# 
# <table>
#     <tr>
#         <td> <img src="https://github.com/sattree/gpr_pub/blob/master/docs/baseline4.png?raw=true" alt="" style="width: 85%;"/>
#                 <div style="text-align:center; margin-top: 8px;">Diversity(parity) of heuristic and pre-trained coref models. Higher parity indicates higher correlation in the predictions and lower diversity.</div>
#         </td>
#          <td> <img src="https://github.com/sattree/gpr_pub/blob/master/docs/baseline5.png?raw=true" alt="" style="width: 85%;"/>
#                 <div style="text-align:center; margin-top: 8px;">Class-wise distribution of votes from the shortlisted set of classifiers. Class 'B' seems to be underrepresented.</div>
#         </td>
#     </tr>
#     <tr>
#         <td> <img src="https://github.com/sattree/gpr_pub/blob/master/docs/baseline2.png?raw=true" alt="" style="width: 85%;"/>
#                 <div style="text-align:center; margin-top: 8px;">Class-specific distribution of probabilties estimated by the confidence model. The probabilties seem to have been over-smoothed.</div>
#         </td>
#         <td> <img src="https://github.com/sattree/gpr_pub/blob/master/docs/baseline3.png?raw=true" alt="" style="width: 85%;"/>
#                 <div style="text-align:center; margin-top: 8px;">The same distribution after adjustment for highly confident (easy) samples.</div>
#         </td>
#     </tr>
# </table>
# 
# 
# **Further Improvements**
# 
# As you can see, a very strong baseline for the problem has been established.
# 
# Straightforward extensions:
# 1. Use the labels from the GAP dataset to improve the confidence model. This alone should give a good boost to the logloss score.
# 2. The current demonstration paves the way for building models by training on the GAP dataset and leveraging insights from existing general coref models. I expect a logloss score of 0.4 to be easily achievable.
# 
# ***

# Download and set up all the required data and models.

# In[ ]:


# Download and install all dependencies
# gpr_pub contains the heuristics models and supplementary code
get_ipython().system('git clone https://github.com/sattree/gpr_pub.git')

get_ipython().system('pip install allennlp --ignore-installed greenlet')

# !pip install ../input/neural-coref/en_coref_lg-3.0.0/en_coref_lg-3.0.0/
# Huggingface neuralcoref model has issues with spacy-2.0.18
# !conda install -y cymem==1.31.2 spacy==2.0.12

get_ipython().system('pip install attrdict pyhocon')

get_ipython().system('wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip')
get_ipython().system('unzip stanford-corenlp-full-2018-10-05.zip')
get_ipython().system('rm stanford-corenlp-full-2018-10-05.zip')

# setup berkeley coref system
get_ipython().system('git clone https://github.com/gregdurrett/berkeley-entity.git')
get_ipython().system('curl -s http://nlp.cs.berkeley.edu/downloads/berkeley-entity-models.tgz | tar xvz -C berkeley-entity')
get_ipython().system('mkdir berkeley-entity/data')
get_ipython().system('wget http://www.cs.utexas.edu/~gdurrett/data/gender.data.tgz')
get_ipython().system('tar -xvf gender.data.tgz')
get_ipython().system('mv gender.data berkeley-entity/data/')
get_ipython().system('rm gender.data.tgz')


# In[ ]:


from IPython.core.display import display, HTML
# Add css styles and js events to DOM, so that they are available to rendered html
display(HTML(open('gpr_pub/visualization/highlight.css').read()))
display(HTML(open('gpr_pub/visualization/highlight.js').read()))


# In[ ]:


# Required for Lee et al coref model
# details can be found here https://github.com/kentonl/e2e-coref

import tensorflow as tf
TF_CFLAGS = " ".join(tf.sysconfig.get_compile_flags())
TF_LFLAGS = " ".join(tf.sysconfig.get_link_flags())

# Linux (build from source)
get_ipython().system('g++ -std=c++11 -shared gpr_pub/modified_e2e_coref/coref_kernels.cc -o coref_kernels.so -fPIC $TF_CFLAGS $TF_LFLAGS -O2')


# In[ ]:


import os
import pyhocon
import sys
import logging

from attrdict import AttrDict
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.metrics import classification_report, log_loss
from sklearn.externals.joblib import Parallel, delayed
from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser
from tqdm import tqdm, tqdm_notebook

from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive

import spacy


# In[ ]:


import sys
sys.path.insert(0, 'gpr_pub/modified_e2e_coref/')
sys.path.insert(0, 'gpr_pub/')


# In[ ]:


from gpr_pub.models.coref import Coref
from gpr_pub.models.heuristics.stanford_base import StanfordModel
from gpr_pub.models.heuristics.spacy_base import SpacyModel

# Heuristics models implement coref resolution based on heuristics described in the paper
# Pronoun resolution is a simple wrapper to convert coref predictions into class-specific labels
from gpr_pub.models.heuristics.random_distance import RandomModel
from gpr_pub.models.heuristics.token_distance import TokenDistanceModel
from gpr_pub.models.heuristics.syntactic_distance import StanfordSyntacticDistanceModel
from gpr_pub.models.heuristics.parallelism import AllenNLPParallelismModel as ParallelismModel
from gpr_pub.models.heuristics.url_title import StanfordURLTitleModel as URLModel

from gpr_pub.models.pretrained.lee_et_al import LeeEtAl2017
from gpr_pub.models.pretrained.stanford import StanfordCorefModel
from gpr_pub.models.pretrained.allennlp import AllenNLPCorefModel
from gpr_pub.models.pretrained.huggingface import HuggingfaceCorefModel
from gpr_pub.models.pretrained.berkley_coref_system import BCS

from gpr_pub.models.pronoun_resolution import PronounResolutionModel, PronounResolutionModelV2


# In[ ]:


from gpr_pub import visualization
from gpr_pub.utils import CoreNLPServer


# In[ ]:


SPACY_MODEL = spacy.load('en_core_web_lg')

STANFORD_CORENLP_PATH = 'stanford-corenlp-full-2018-10-05/'
server = CoreNLPServer(classpath=STANFORD_CORENLP_PATH,
                        corenlp_options=AttrDict({'port': 9090, 
                                                  'timeout': '600000',
                                                  'thread': '2',
                                                  'quiet': 'true',
                                                  'preload': 'tokenize,ssplit,pos,lemma,parse,depparse,ner,coref'}))
server.start()
STANFORD_SERVER_URL = server.url
STANFORD_MODEL = CoreNLPParser(url=STANFORD_SERVER_URL)


# In[ ]:


model_url = 'https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz'
archive = load_archive(model_url, cuda_device=0)
ALLEN_COREF_MODEL = Predictor.from_archive(archive)

model_url = 'https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz'
archive = load_archive(model_url, cuda_device=0)
ALLEN_DEP_MODEL = Predictor.from_archive(archive)

# HUGGINGFACE_COREF_MODEL = en_coref_lg.load()


# **Load test data**
# ***

# In[ ]:


test = pd.read_csv('../input/gendered-pronoun-resolution/test_stage_1.tsv', sep='\t')

# normalizing column names
test.columns = map(lambda x: x.lower().replace('-', '_'), test.columns)
with pd.option_context('display.max_rows', 10, 'display.max_colwidth', 15):
    display(test)


# In[ ]:


lee_coref_model = LeeEtAl2017(SPACY_MODEL, config = {'name': 'final',
                                                     'log_root': '../input/e2e-coref-data/',
                                                    'model': 'gpr_pub/modified_e2e_coref/experiments.conf',
                                                    'context_embeddings_root': '../input/e2e-coref-data/',
                                                    'head_embeddings_root': '../input/e2e-coref-data/',
                                                    'char_vocab_root': '../input/e2e-coref-data/'
                                                    })


# In[ ]:


sample = test.loc[17]
data = ALLEN_COREF_MODEL.predict(sample.text)
print('{:-<100}'.format('Example where Coref resolves to both gold mentions: Pronoun={}, A={}, B={}'.format(sample.pronoun, sample.a, sample.b)))
visualization.render(data, allen=True, jupyter=True)

sample = test.loc[13]
data = ALLEN_COREF_MODEL.predict(sample.text)
print('{:-<100}'.format('Example where a single antecedent contains both gold mentions: Pronoun={}, A={}, B={}'.format(sample.pronoun, sample.a, sample.b)))
visualization.render(data, allen=True, jupyter=True)

sample = test.loc[13]
data = lee_coref_model.predict(**sample)
print('{:-<100}'.format('Example to compare the performance of Lee et al: Pronoun={}, A={}, B={}'.format(sample.pronoun, sample.a, sample.b)))
visualization.render({'document': data[0], 'clusters': data[1]}, allen=True, jupyter=True)


# **Setup models and generate predictions**
# ***

# Heuristic models

# In[ ]:


random_coref_model = RandomModel(SPACY_MODEL)
random_proref_model = PronounResolutionModel(random_coref_model)

token_distance_coref_model = TokenDistanceModel(SPACY_MODEL)
token_distance_proref_model = PronounResolutionModel(token_distance_coref_model)

syntactic_distance_coref_model = StanfordSyntacticDistanceModel(STANFORD_MODEL)
syntactic_distance_proref_model = PronounResolutionModel(syntactic_distance_coref_model, n_jobs=1)

parallelism_coref_model = ParallelismModel(ALLEN_DEP_MODEL, SPACY_MODEL)
parallelism_proref_model = PronounResolutionModel(parallelism_coref_model)

url_title_coref_model = URLModel(STANFORD_MODEL)
url_title_proref_model = PronounResolutionModel(url_title_coref_model, n_jobs=1)


# In[ ]:


# preds = random_proref_model.predict(test)
# test['random_a_coref'], test['random_b_coref'] = zip(*preds)

# preds = token_distance_proref_model.predict(test)
# test['token_distance_a_coref'], test['token_distance_b_coref'] = zip(*preds)

preds = syntactic_distance_proref_model.predict(test)
test['syntactic_distance_a_coref'], test['syntactic_distance_b_coref'] = zip(*preds)

preds = parallelism_proref_model.predict(test)
test['parallelism_a_coref'], test['parallelism_b_coref'] = zip(*preds)

preds = url_title_proref_model.predict(test)
test['parallelism_url_a_coref'], test['parallelism_url_b_coref'] = zip(*preds)


# Pre-trained coref model

# In[ ]:


stanford_coref_model = StanfordCorefModel(STANFORD_MODEL, algo='clustering')
deterministic_stanford_proref_model = PronounResolutionModelV2(stanford_coref_model, n_jobs=1, multilabel=True)

stanford_coref_model = StanfordCorefModel(STANFORD_MODEL, algo='statistical')
statistical_stanford_proref_model = PronounResolutionModelV2(stanford_coref_model, n_jobs=1, multilabel=True)

stanford_coref_model = StanfordCorefModel(STANFORD_MODEL, algo='neural', greedyness=0.5)
neural_stanford_proref_model = PronounResolutionModelV2(stanford_coref_model, n_jobs=1, multilabel=True)

allen_coref_model = AllenNLPCorefModel(ALLEN_COREF_MODEL, SPACY_MODEL)
allen_proref_model = PronounResolutionModelV2(allen_coref_model, n_jobs=2, multilabel=True)

# huggingface_coref_model = HuggingfaceCorefModel(HUGGINGFACE_COREF_MODEL)
# hugginface_proref_model = PronounResolutionModelV2(huggingface_coref_model, multilabel=True)

lee_coref_model = LeeEtAl2017(SPACY_MODEL, config = {'name': 'final',
                                                     'log_root': '../input/e2e-coref-data/',
                                                    'model': 'gpr_pub/modified_e2e_coref/experiments.conf',
                                                    'context_embeddings_root': '../input/e2e-coref-data/',
                                                    'head_embeddings_root': '../input/e2e-coref-data/',
                                                    'char_vocab_root': '../input/e2e-coref-data/'
                                                    })
lee_proref_model = PronounResolutionModelV2(lee_coref_model, multilabel=True)

bcs_coref_model = BCS(STANFORD_MODEL)
bcs_proref_model = PronounResolutionModelV2(bcs_coref_model, multilabel=True)


# In[ ]:


# preds = hugginface_proref_model.predict(test)
# test['huggingface_ml_a_coref'], test['huggingface_ml_b_coref'] = zip(*preds)

preds = allen_proref_model.predict(test)
test['allen_ml_a_coref'], test['allen_ml_b_coref'] = zip(*preds)

preds = deterministic_stanford_proref_model.predict(test)
test['stanford_ml_deterministic_a_coref'], test['stanford_ml_deterministic_b_coref'] = zip(*preds)

preds = statistical_stanford_proref_model.predict(test)
test['stanford_ml_statistical_a_coref'], test['stanford_ml_statistical_b_coref'] = zip(*preds)

preds = neural_stanford_proref_model.predict(test)
test['stanford_ml_neural_a_coref'], test['stanford_ml_neural_b_coref'] = zip(*preds)

preds = lee_proref_model.predict(test)
test['lee_a_coref'], test['lee_b_coref'] = zip(*preds)

preds = bcs_proref_model.predict(test, preprocessor=BCS.preprocess)
test['bcs_a_coref'], test['bcs_b_coref'] = zip(*preds)


# **Confidence Model (by bootstrapping)**
# ***

# In[ ]:


# Investigate diversity of the models
models = (
        'parallelism_url',
        'allen_ml', 
        # 'huggingface_ml', 
        'parallelism', 
        'stanford_ml_deterministic', 
        'syntactic_distance', 
        'stanford_ml_statistical',
        'stanford_ml_neural',
        'bcs',
        'lee',
       )

scores = []
for model in models[:-1]:
    score = mutual_info_score(test['{}_a_coref'.format(model)], test['lee_a_coref'])
    score2 = mutual_info_score(test['{}_b_coref'.format(model)], test['lee_b_coref'])
    scores.append((model, score, score2))
    
models = pd.DataFrame(scores, columns=['model', 'score_a', 'score_b']).set_index('model').sort_values('score_b')
models['parity(~diversity)'] = models.min(axis=1)
models


# In[ ]:


# shortlisted diverse set of models
models = (
        'parallelism_url',
        'allen_ml',
        'parallelism', 
        'syntactic_distance', 
        'stanford_ml_statistical',
        'lee',
        'bcs'
       )

models_a = [model+'_a_coref' for model in models]
models_b = [model+'_b_coref' for model in models]

test['votes_a'] = test[models_a].sum(axis=1)
test['votes_b'] = test[models_b].sum(axis=1)
test['votes_a_b'] = test[models_a+models_b].sum(axis=1)

plt.hist([test['votes_a'], test['votes_b']], label=['Class A', 'Class B'], bins=range(1, len(models)+2))
plt.legend()
plt.show()


# Now, let's invest some time in understanding this figure. Given that we have rejected highly correlated models, the number of votes should be directly reflective of confidence and ease in classification. We will use the data sampled based on voting stength to learn how different models interact with each other, and then transfer that behavior to the remaining samples where model predictions are sparse.
# 
# As can be seen from the figure above, the model predictions are biased towards samples belonging to class 'A'. Therefore, to balance the sampled set, we will set the threshold for proxy labels as 5 votes and above, and 4 votes and above for classes 'A' and 'B' respectively. We can also see that the category 'NEITHER' continues to elude us. The question of whether there is evidence in the data to support linking of any entity at all is rather an involved one.

# In[ ]:


# define proxy labels based on votes
mask_a = test['votes_a'] >=5
mask_b = test['votes_b'] >=4
mask_a_b = test['votes_a_b'] <= 1

true_proxy = test[mask_a | mask_b | mask_a_b]

true_proxy['label'] = 2
true_proxy.loc[mask_a, 'label'] = 0
true_proxy.loc[mask_b, 'label'] = 1


# In[ ]:


feats = models_a + models_b

X = true_proxy[feats]
y = true_proxy['label']

print(X.shape)

clf = LogisticRegression(multi_class='auto', solver='liblinear', penalty='l1', C=.05, max_iter=30)

scores = cross_val_score(clf, X, y, cv=StratifiedKFold(3, random_state=21))
scores


# In[ ]:


clf.fit(X, y)

X_tst = test[feats]
probabilties = clf.predict_proba(X_tst)

plt.hist(probabilties, range=(0,1), label=['A', 'B', 'NEITHER'])
plt.legend()
plt.title('Class-wise probability distributions from Confidence Model')
plt.show()


# Judging from the above figure, it appears that the confidence model has over-smoothed examples, specifically class A.
# In order to avoid over-smoothing really easy samples, we will reinstate the predictions for samples where the top 3 diverse models agree a 100%. Once again, to avoid logloss from going berserk on that 1 sample (there may be more in reality) where the above may not hold, we will set a small lower bound.

# In[ ]:


models_perfect = ('parallelism_url',
        'parallelism', 
        'lee')

models_a = [model+'_a_coref' for model in models_perfect]
models_b = [model+'_b_coref' for model in models_perfect]

mask_a_perfect = test[models_a].all(axis=1)
mask_b_perfect = test[models_b].all(axis=1)
print(test[mask_a_perfect].shape, test[mask_b_perfect].shape)

# set the lower bound, assuming 1% chance of failure
probabilties[mask_a_perfect] = [1,.02,.02]
probabilties[mask_b_perfect] = [.01,1,.01]

# Softmax of probabilities of joint model to convert them to labels for analysis
y_pred = np.zeros_like(probabilties)
y_pred[np.arange(len(probabilties)), probabilties.argmax(1)] = 1
y_pred = y_pred.astype(bool)

plt.hist(probabilties, range=(0, 1), label=['A', 'B', 'NEITHER'])
plt.title('Class-wise probability distributions from Confidence Model after adjustment')
plt.legend()
plt.show()


# In[ ]:


res = pd.concat([pd.DataFrame(y_pred, columns=['A', 'B', 'NEITHER']), 
                 pd.DataFrame(probabilties, columns=['prob_A', 'prob_B', 'prob_NEITHER'])], 
                axis=1)

plt.figure(figsize=(16,4))
plt.subplot(1,3,1)
plt.hist([res[res['A']]['prob_A'], res[~res['A']]['prob_A']], bins=10, rwidth=0.7, label=['True', 'False'])
plt.title('Distribution of probabilties over \nsamples predicted as class A')
plt.legend()

plt.subplot(1,3,2)
plt.hist([res[res['B']]['prob_B'], res[~res['B']]['prob_B']], bins=10, rwidth=0.7, label=['True', 'False'])
plt.title('Distribution of probabilties over \nsamples predicted as class B')
plt.legend()

plt.subplot(1,3,3)
plt.hist([res[res['NEITHER']]['prob_NEITHER'], res[~res['NEITHER']]['prob_NEITHER']], bins=10, rwidth=0.7, label=['True', 'False'])
plt.title('Distribution of probabilties over \nsamples predicted as class NEITHER')
plt.legend()

plt.show()


# Interesting!
# 
# - Class A can definitely benefit from more work.
# - Class B seems to be modeled well, we can see a nice exponential envelope.
# - Class NEITHER continues to elude us but the distribution looks reasonable.

# Generate predictions for submission

# In[ ]:


sub_df = pd.read_csv('../input/gendered-pronoun-resolution/sample_submission_stage_1.csv')
sub_df.loc[:, 'A'] = probabilties[:, 0]
sub_df.loc[:, 'B'] = probabilties[:, 1]
sub_df.loc[:, 'NEITHER'] = probabilties[:, 2]

sub_df.to_csv("submission.csv", index=False)

sub_df.head()


# In[ ]:


get_ipython().system('rm -r stanford-corenlp-full-2018-10-05/')
get_ipython().system('rm -r gpr_pub/')
get_ipython().system('rm -r berkeley-entity/')
get_ipython().system('rm -r tmp/')

