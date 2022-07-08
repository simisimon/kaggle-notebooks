#!/usr/bin/env python
# coding: utf-8

# # IEEE-CIS Fraud Detection &mdash; LightGBM Split Points
# 
# ## Adversarial Validation Version
# 
# This notebook shows some techniques to snoop on the gradient boosting process used by LightGBM - using its lesser-known APIs. This is the adversarial validation version, training LightGBM to predict whether rows are from the train or test set. The [original version of this kernel is here][5].
# 
# By counting the split points used in the decision trees, we can see the ways the algorithm divides the input space up. Given that the train and test sets are different time eras, this may lead to new insights about what areas in *feature space* are are specific to past or "future" data.
# 
# For more info on LightGBM see [pdf by Microsoft][3] or the [LightGBM github][4].
# 
# For another example of gradient boosting model analysis with XGBoost see the great [xgbfi][2] tool by [Faron][1].
# 
# ___
# 
# We start by building a model...
# 
#  [1]: https://www.kaggle.com/mmueller
#  [2]: https://github.com/Far0n/xgbfi
#  [3]: https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/lightgbm.pdf
#  [4]: https://github.com/Microsoft/LightGBM
#  [5]: https://www.kaggle.com/jtrotman/ieee-fraud-lgb-split-points
#  

# In[144]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from pandas.api.types import union_categoricals
import gc, os, sys, re, time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from IPython.display import display, Image, HTML


# In[3]:


DTYPE = {
    'TransactionID': 'int32',
    'isFraud': 'int8',
    'TransactionDT': 'int32',
    'TransactionAmt': 'float32',
    'ProductCD': 'category',
    'card1': 'int16',
    'card2': 'float32',
    'card3': 'float32',
    'card4': 'category',
    'card5': 'float32',
    'card6': 'category',
    'addr1': 'float32',
    'addr2': 'float32',
    'dist1': 'float32',
    'dist2': 'float32',
    'P_emaildomain': 'category',
    'R_emaildomain': 'category',
}

IDX = 'TransactionID'
TGT = 'isFraud'

CCOLS = [f'C{i}' for i in range(1, 15)]
DCOLS = [f'D{i}' for i in range(1, 16)]
MCOLS = [f'M{i}' for i in range(1, 10)]
VCOLS = [f'V{i}' for i in range(1, 340)]

DTYPE.update((c, 'float32') for c in CCOLS)
DTYPE.update((c, 'float32') for c in DCOLS)
DTYPE.update((c, 'float32') for c in VCOLS)
DTYPE.update((c, 'category') for c in MCOLS)


DTYPE_ID = {
    'TransactionID': 'int32',
    'DeviceType': 'category',
    'DeviceInfo': 'category',
}

ID_COLS = [f'id_{i:02d}' for i in range(1, 39)]
ID_CATS = [
    'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30',
    'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
]

DTYPE_ID.update(((c, 'float32') for c in ID_COLS))
DTYPE_ID.update(((c, 'category') for c in ID_CATS))

IN_DIR = '../input'

NR = None
NTRAIN = 590540

def read_both(t):
    df = pd.read_csv(f'{IN_DIR}/{t}_transaction.csv',
                     index_col=IDX,
                     nrows=NR,
                     dtype=DTYPE)
    df = df.join(
        pd.read_csv(f'{IN_DIR}/{t}_identity.csv',
                    index_col=IDX,
                    nrows=NR,
                    dtype=DTYPE_ID))
    print(t, df.shape)
    return df

def read_dataset():
    train = read_both('train')
    test = read_both('test')
    
    train.pop('isFraud')
    
    train['isTest'] = 0
    test['isTest'] = 1
    
    ntrain = train.shape[0]
    for c in train.columns:
        s = train[c]
        if hasattr(s, 'cat'):
            u = union_categoricals([train[c], test[c]], sort_categories=True)
            train[c] = u[:ntrain]
            test[c] = u[ntrain:]
    
    uni = train.append(test)
    return uni


# In[5]:


uni = read_dataset()
uni.shape


# No count features in this version, the kernel runs out of memory. Add some simple extra features:

# In[77]:


uni['TimeInDay'] = uni.TransactionDT % 86400
uni['Cents'] = uni.TransactionAmt % 1


# In[139]:


params = {
    'num_leaves': 64,
    'objective': 'binary',
    'min_data_in_leaf': 10,
    'learning_rate': 0.1,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'max_cat_to_onehot': 128,
    'metric': 'auc',
    'num_threads': 8,
    'seed': 42,
    'subsample_for_bin': uni.shape[0]
}


# # *FAST* Cross Validation with *Cached OOF*
# 
# Let's use (nearly) all input columns as features, and run a KFold CV. By passing a callback to `lgb.cv` we can store references to the trained models and save them, then query them for split point and feature importance information.
# 
# 

# In[112]:


class LightGbmSnoop:
    def __init__(self):
        self.train_logs = []
        self.valid_logs = []
    def _callback(self, env):
        self.model = env.model
        self.train_logs.append( [b.eval_train()[0][2] for b in self.model.boosters] )
        self.valid_logs.append( [b.eval_valid()[0][2] for b in self.model.boosters] )
    def train_log(self):
        return pd.DataFrame(self.train_logs).add_prefix('train_')
    def valid_log(self):
        return pd.DataFrame(self.valid_logs).add_prefix('valid_')
    def logs(self):
        return pd.concat((self.train_log(), self.valid_log()), 1)
    def get_oof(self, n):
        oof = np.zeros(n, dtype=float)
        for i, b in enumerate(self.model.boosters):
            vs = b.valid_sets[0]  # validation data
            idx = vs.used_indices
            # Note: this uses all trees, not the early stopping peak count.
            # You can use b.rollback_one_iter() to drop trees :)
            p = b._Booster__inner_predict(1) # 0 = train; 1 = valid
            oof[idx] = p
        return oof

TGT = 'isTest'
FEATS = uni.columns.tolist()
FEATS.remove(TGT)  
FEATS.remove('TransactionDT') # makes train/test trivially separable, remove it
print(len(FEATS), 'features')

folds = list(KFold(n_splits=4, shuffle=True, random_state=42).split(uni[FEATS]))
ds = lgb.Dataset(uni[FEATS], uni[TGT], params=params)
s = LightGbmSnoop()
res = lgb.cv(params,
             ds,
             folds=folds,
             num_boost_round=3000,
             early_stopping_rounds=100,
             verbose_eval=100,
             callbacks=[s._callback])


# Demo of how to save validation predictions, and get AUC of full validation:

# In[108]:


OOF = s.get_oof(uni.shape[0])
np.save('ieee_fraud_adversarial_lgb_oof', OOF)
roc_auc_score(uni[TGT], OOF)


# The models are normally quite confident:

# In[123]:


pd.Series(OOF).plot.hist(bins=100, title='Histogram of predictions of p(isTest)')


# And the predictions are generally lower for the training set (first half) and higher for the test set (2nd half).

# In[149]:


pd.Series(OOF).plot(figsize=(14,6), title='OOF Prediction of p(isTest)')


# But that is hard to see, so using smoothing helps, and reveals a **shelf** at about 300k rows where train suddenly becomes a bit more like test, the abrupt change to the test set is visible at about 590k rows, and after that the test set rows get a higher prediction as time goes on :)

# In[150]:


pd.Series(OOF).rolling(500).mean().plot(figsize=(14,6), title='Smoothed OOF prediction of p(isTest)')


# In[123]:


pd.Series(OOF[:NTRAIN]).plot.hist(bins=100, title='Histogram of predictions of p(isTest) - train set rows')


# In[123]:


pd.Series(OOF[NTRAIN:]).plot.hist(bins=100, title='Histogram of predictions of p(isTest) - test set rows')


# Save the models - LightGBM saves in an easy to parse text format. (The files won't be used here but it is useful in general to save.)

# In[15]:


for i, b in enumerate(s.model.boosters):
    b.save_model(f'ieee_fraud_adversarial_lgb_model_{i}.txt')


# Likewise for AUC training/validation logs.

# In[113]:


s.logs().to_csv(f'ieee_fraud_adversarial_lgb_auc_logs.csv', index_label='Round')


# In[138]:


logs = pd.DataFrame({'train':s.train_log().mean(1), 'valid':s.valid_log().mean(1)})
logs.train.plot(legend=True, title='Adversarial AUC Logs')
logs.valid.plot(legend=True)


# # Standard Feature Importances
# 
# Sum the usual feature importances from all models in our CV collection.

# In[140]:


def make_importances(clf, importance_type):
    return pd.Series(data=clf.feature_importance(importance_type), index=clf.feature_name())

IMPORTANCES = pd.concat((make_importances(b, 'gain')
                         for b in s.model.boosters), 1).sum(1).to_frame('Gain')
IMPORTANCES['Count'] = pd.concat((make_importances(b, 'split') for b in s.model.boosters), 1).sum(1)
IMPORTANCES.sort_values('Gain', ascending=False).head()


# In[141]:


IMPORTANCES.to_csv('ieee_fraud_adversarial_lgb_importances.csv')


# ## Standard Plot

# In[91]:


COLORS = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
]
toplot = IMPORTANCES.sort_values('Gain').tail(80)
toplot['Gain'].plot.barh(figsize=(12,20), color=COLORS, title='Adversarial Feature Gain')


# # Booster.dump_model()
# 
# The returned LightGBM model format is hierarchical, trees are nested `dict` objects containing `left_child` and `right_child` subtrees. Walking over the trees and summarizing the splits can be done with a short recursive function...
# 
#     tree_info  - list of dicts
#     (each contains):
#         tree_structure
#             left_child
#             right_child
# 
# The `dump_model()` information records 'gain' at each split, and we simply re-use that.

# In[156]:


# uncomment to see model structure
# clf.dump_model(num_iteration=2)['tree_info']


# In[95]:


# NOTE: lightgbm.Booster has a new get_split_value_histogram API which counts split points used.
# This code pre-dates that, and sums gain instead of counting appearances.
# Here it is adapted from the original to use a collection of models, and sum the overall data.
def get_split_point_stats_multi(clfs):
    split_points = defaultdict(Counter)

    def visit_node(d):
        if 'tree_info' in d:
            for tree in d['tree_info']: # a list of trees
                visit_node(tree)
        for k in ['tree_structure', 'left_child', 'right_child' ]:
            if k in d:
                visit_node(d[k])
        if 'split_feature' in d:
            split_points[names[d['split_feature']]] [d['threshold']] += d['split_gain']

    for clf in clfs:
        names = clf.feature_name()
        visit_node(clf.dump_model())
    return split_points


# In[20]:


split_points = get_split_point_stats_multi(s.model.boosters)


# Each feature indexes a Counter object in the `split_points` dict. In each Counter, the keys are feature values, and the values are sum of gain, for example, here are the most used values in feature `card1`, with the gain for each:

# In[69]:


split_points['card1'].most_common(5)


# Dump all the split point data to an xlsx file (can be opened with open-source *Open Office* or *[Libre Office][1]*)
# 
#  [1]: https://www.libreoffice.org/download/download/

# In[34]:


with pd.ExcelWriter('ieee_fraud_adversarial_split_points.xlsx') as writer:
    for feat in FEATS:
        counter = split_points[feat]
        df = pd.Series(counter, name=feat).sort_index().to_frame('GainSum')
        df.to_excel(writer, feat, index_label=feat)

    for sheet in writer.sheets.values():
        sheet.set_column(0, 0, 30)


# # Plotting Code
# 
# Warning: this only shows the 50 split points with the most gain, so the x-axis will be a bit nonlinear, some values won't appear. See the xlsx file for all the values.

# In[35]:


MAX_SHOW = 50


# In[115]:


ADJS = 'abundant:common:ubiquitous:omnipresent:rampant:rife:permeant:widespread:legendary:popular:fashionable:frequent:usual:useful:predominant:recurrent:repetitive:repetitious:marked:prevalent:prevalent:prevalent'.split(':')

np.random.seed(42)

def plot_it(col):
    counts = split_points[col]
    ser = pd.Series(dict(counts)).sort_values(ascending=False)
    total_gain = IMPORTANCES.loc[col, 'Gain']
    total_splits = IMPORTANCES.loc[col, 'Count']
    if hasattr(uni[col], 'cat'):
        # remap categories from int -> cat value
        try:
            ser.index = uni[col].cat.categories[ser.index.astype(int)]
        except:
            # e.g. TypeError: Cannot cast Index to dtype <class 'int'>
            # a categorical with many categories and '1||4||7' etc type splits
            # leave it as it is
            pass
    adj = np.random.choice(ADJS)
    display(
        HTML(
            f'<h1 id="plot_{col}">{col}</h1>'
            f'<p>Used {total_splits} times, total gain is {total_gain}.'
            f'<p>{len(ser)} split point values used. '
            f'Most {adj} is {ser.index[0]} with gain of {ser.values[0]}.'
        )
    )
    ser = ser.head(MAX_SHOW).sort_index()
    ax = ser.plot.bar(title=f'{col} — Adversarial split points by gain',
                      rot=90, fontsize=12, figsize=(15,5),
                      width=0.7, color=COLORS)
    plt.show()


# # Plots For IEEE Features
# 
# All the features with 2 or more unique split point values are shown.
# 
# ## Notes
# 
# Most of the split points have long decimal values like `379.00000000000006` - the LightGBM algorithm only sees binned data, so it sets split thresholds as values [halfway between neighbouring bin lower/upper edges][6], but bumped upwards a tiny fraction using `std::nextafter` in the [C++ standard library][5], resulting in strangely precise [floating point format][1] values :)
# 
# Zero is checked for using a [kZeroThreshold = 1e-35f][7] variable - this comes out of the model as a split point of 1.0000000180025095e-35 &mdash; a tiny number. When you see that, think *zero*.
# 
# Split points for categorical dtypes depends on the `max_cat_to_onehot` which I have set to 128 - so categoricals in this data set are treated with a one-vs-all split. This means `feature==value` in the node split test, instead of the usual `feature<=value`. `max_cat_to_onehot` is by default set to 4, meaning categories with more values than this use splits based on target statistics, and the resulting split points have values like `1||3||5||7||8||9` which indicate which category codes go down the *left* branch. (But this is hard to show in bar charts... hence I used *one-vs-all splits*.)
# 
# **Note**: in this adversarial version, *id* columns are included. Two (`DeviceInfo` and `id_30`) have more than 128 values, so their bar charts have these obscure very long axis labels :)
# 
# LightGBM keeps a separate bin for NaN values and at all node tests, records whether that bin goes left/right separately - this is not shown here.
# 
# ## What to Look For
# 
# With adversarial validation, one of the the aims is to detect differences in the train/test set features, and possibly alter the representations to make test look more like train, with the hope that this results in better model accuracy.
# 
# In some ways what we **don't** see is more interesting than what we **do**. As with normal feature importances: if we see a feature is not used at all it is clearly not useful in detecting train/test difference, so is probably a safe feature to predict isFraud.
# 
# If there is **one prominent peak** it means the train or test sets have values on one side of the split that are not present in the other set. It may make sense to cap the values in both sets.
# 
# ## Notes
# 
# Here are some quick observations:
# 
#  - <a href="#plot_ProductCD">ProductCD</a> differs most in the H, R and S values, as also seen in the [heatmap plots notebook][4].
# 
#  - <a href="#plot_id_30">id_30</a> has very clear peaks at `iOS 11.4.1` and the later values of `Mac OS X 10_13_4` onwards, which are obviously time-related, software versions released in the test set era.
# 
#  - <a href="#plot_Cents">Cents</a> has an interesting spike at about 0.989 - which might be tied in to [Chris Deotte's discussion post here][8].
# 
# 
# A further useful extension would be to look at how the leaf node values vary underneath each split in the left/right branches, similarly to [SHAP plots][9]...
# 
# 
#  [1]: https://en.wikipedia.org/wiki/Double-precision_floating-point_format
#  [2]: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43886
#  [3]: https://www.kaggle.com/tilii7
#  [4]: https://www.kaggle.com/jtrotman/ieee-fraud-time-series-heatmaps
#  [5]: https://en.cppreference.com/w/cpp/numeric/math/nextafter
#  [6]: https://github.com/microsoft/LightGBM/blob/master/src/io/bin.cpp
#  [7]: https://github.com/microsoft/LightGBM/blob/master/include/LightGBM/meta.h 
#  [8]: https://www.kaggle.com/c/ieee-fraud-detection/discussion/108467#624331
#  [9]: https://github.com/slundberg/shap

# In[90]:


for col in FEATS:
    counts = split_points[col]
    if len(counts) >= 2:
        plot_it(col)


# # Conclusions
# 
# Now we can inspect trained models to see **which points** in the feature space matter for train/test distinctions...
# 
# An interesting next step might be to use this information to build an an *auto-relaxing* function that buckets the data for us in a way that makes the train and test sets more similar, without any tedious manual inspection of plots :)
# 
# ___
# 
# A note to any newer Kagglers still reading: the original features used here are only a starting point, used just to demonstrate. If (say) `DeviceInfo` of `hi6210sft Build/MRA58K` comes along in the training set and makes a fast burst of transactions (all marked fraud), then appears in the test set but spread out and on many separate days, it does not make sense to predict a high fraud likelihood, simply because of that one feature. Features that capture *event* timing & behaviour are needed :)
# 
# For inspiration you should check out [an **extensive** index of **winning** and high ranking Kaggle **solutions** here][1], an auto-generated notebook that indexes the Kaggle Forums for post-competition write-ups by top teams, using the [Meta Kaggle][2] dataset ;)
# 
#  [1]: https://www.kaggle.com/jtrotman/high-ranking-solution-posts
#  [2]: https://www.kaggle.com/kaggle/meta-kaggle
#  
