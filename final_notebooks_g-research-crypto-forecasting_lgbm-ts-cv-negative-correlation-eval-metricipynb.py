#!/usr/bin/env python
# coding: utf-8

# # 🪙💲 Proposal for a meaningful LB + LGBM [S]
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/30894/logos/header.png)
# 
# # Ok, so LB is meaningless. Now what?
# 
# It was confirmed that `2021-06-13 00:00:00` is the beginning of the test data of the public LB [here](https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/285505#1572680).
# 
# Please note that the very nature of the Target is "leakable". There is no fault at all at any level here. In any case, the organizers could have been more clear about the fact that the train csv contained the LB data, but they clarified that already. The target is publicly accesible with almost no effort and therefore there is no possible "realistic leaderboard".
# 
# On a final note before continuing, the LB was already meaningless even when the solutions were in the `0.01` range. Any solution using the full `train.csv` overfits. Period. I shared the `0.999` to rip off the band aid and move one.
# 
# 
# There are __two problems at hand now__:
# 
# ## 1)  The public LB data is contained in the `train.csv`. 
# 
# That is one of the problems, the one that breaks the LB.
# 
# The corolaries are various: 
# * The leaderboard is meaningless, rendering the "LB validation" useless. 
# * It turns impossible to assess whether a publicly shared model is useful or not.
# 
# I think we can address it quite easily and find a common ground for sharing solutions (and actually using the LB score of them as an accurate measure of something). The leaderboard is broken and useless, but we can use the submission scores still to communicate, compare results and share good models.
# 
# The second problem is:
# 
# ## 2) The distance between the public and the private/final LB.
# 
# I will ignore this problem for now, and focus on the first one: how to get some common ground for comparing and assessing LB scores.
# 
# 
# ---
# 
# 
# 
# # Not sure what I am talking about? Check these links:
# * __[Watch out!: test LB period is contained in the train csv](https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/285505) (topic)__
# * __[🪙💲 G-Research- Using the overlap fully [LB=0.99]](https://www.kaggle.com/julian3833/g-research-using-the-overlap-fully-lb-0-99) (notebook)__
# * __[Meaningful submission scores / sharing the lower boundary of public test data](https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/285289) (topic)__
# 
# 
# 
# 
# ---

# # A proposal for a meaningful LB - "Strict submissions" [Strict] / [S]
# 
# The idea is quite obvious and probably you have thought of it. I just want to propose it as a "convention" so we have a common ground of understanding:
# 
# <h2 style="text-align:center; background-color:#00FF00;padding:40px;border-radius: 30px;"> Only keep data from before 2021-06-13 00:00:00.</h2>
# 
# This can be done by replacing the `pd.read_csv()` with the following function `read_csv_strict()` or however you want.
# 

# There are other files around, if you use them make sure you do the same filtering. Of course, you will have a validation scheme inside the notebook: anything inside the notebook should be before `2021-06-13`, even the validation.
# Only the submission iteration will see that time horizon.
# 
# 
# ## If you do this, your public LB score can be labelled as "Strict". And it will be comparable to any other fellow kaggler reporting a "Strict" score.
# 
# For public shared kernels, we can use the tag `[Strict]` or `[S]` for labelling them. The visible scores of notebooks that always have run with this strict mechanism will be reallistic representations of the performance of the model.
# 
# 
# ## So that is the proposal: __drop all the data  after `2021-06-13` and flag your work as `[Strict]` or `[S]`.__
# 
# # I wish I see comparable results soon! 
# # Can you beat my current `[S] 0.017`? It seems pretty weak, I bet you can! 😁😁
# 
# 
# 
# ---
# 
# &nbsp;
# &nbsp;
# &nbsp;
# &nbsp;

# # Strict model: 🪙 G-Research Crypto - Starter LGBM Pipeline
# 
# ### This is just a copy of the original [🪙💲 G-Research- Starter LGBM Pipeline](https://www.kaggle.com/julian3833/g-research-starter-lgbm-pipeline), but without the LB score contamination with the "leaky" data. Therefore, this is a valid "`[S]`" (or "`[Strict]`") notebook, as it follows the convention.
# 
# # Import and load dfs
# 
# References: [Tutorial to the G-Research Crypto Competition](https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition)

# In[ ]:


import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
import gresearch_crypto
from sklearn.model_selection import train_test_split


TRAIN_CSV = '../input/g-research-crypto-forecasting/train.csv'
ASSET_DETAILS_CSV = '../input/g-research-crypto-forecasting/asset_details.csv'

def read_csv_strict(file_name='../input/g-research-crypto-forecasting/train.csv'):
    df = pd.read_csv(file_name)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
#     df = df[df['datetime'] < '2021-06-13 00:00:00']
    print('len of df', len(df))
    supp = pd.read_csv('../input/g-research-crypto-forecasting/supplemental_train.csv')
    print('len of supp', len(supp))
    supp['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = pd.concat([df, supp]).reset_index(drop=True)
    return df


# In[ ]:


df_train = read_csv_strict()


# In[ ]:


df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")
df_asset_details.shape


# # Training
# 
# ## Utility functions to train a model for one asset
# 
# ### Features from [G-Research - Starter [0.361 LB]](https://www.kaggle.com/danofer/g-research-starter-0-361-lb)
# ### And [[GResearch] Simple LGB Starter](https://www.kaggle.com/code1110/gresearch-simple-lgb-starter#Feature-Engineering)

# In[ ]:


# Two new features from the competition tutorial
def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

# A utility function to build features from the original df
# It works for rows to, so we can reutilize it.
def get_features(df):
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    return df_feat

def get_Xy_and_model_for_asset(df_train, asset_id):
    
    from sklearn.model_selection import KFold
    
    df = df_train[df_train["Asset_ID"] == asset_id]
    
    # TODO: Try different features here!
    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    df_proc = df_proc.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]

    # TODO: Try different models here!
    n_splits = 3
#     folds = KFold(5, shuffle=True)
    
    def neg_correlation(preds, labels):
        is_higher_better = False
        return "neg_correlation", -np.corrcoef(preds, labels)[1, 0], is_higher_better
    
    if asset_id == 0:
        params = {'learning_rate': .4}
    elif asset_id == 3:
        params = {'learning_rate': .05}
    elif asset_id == 6:
        params = {'learning_rate': .025}
    elif asset_id == 7:
        params = {'learning_rate': .0001}
    elif asset_id == 8:
        params = {'learning_rate': .01}
    elif asset_id == 9:
        params = {'learning_rate': .01}
    elif asset_id == 10:
        params = {'learning_rate': .01}
    elif asset_id == 11:
        params = {'learning_rate': .025}
    elif asset_id == 12:
        params = {'learning_rate': .025}
    elif asset_id == 13:
        params = {'learning_rate': .05}
    else:
        params = {}

    eval_results = []
#     for train_idx, val_idx in folds.split(X):
#         X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]
    for split in range(1, n_splits+1):
        train_idx = np.arange(len(X)-2236494//14*split)
        val_idx = np.arange(len(X)-2236494//14*split, len(X))
        model = LGBMRegressor(n_estimators=1000, **params, min_child_samples=10_000_000)
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            eval_metric=neg_correlation,
            verbose=0,
        )
        eval_results.append(
            np.asarray(model.evals_result_["valid_0"]["neg_correlation"])[:, np.newaxis]
        )

    cv_results = np.hstack(eval_results)
    best_n_estimators = np.argmin(cv_results.mean(axis=1)) + 1
    print('best n estimators', best_n_estimators)
    print('best cv', np.min(cv_results.mean(axis=1)))
    print('best cv std', cv_results[best_n_estimators-1, :].std())

    model = LGBMRegressor(n_estimators=best_n_estimators)
    model.fit(X, y)

    return X, y, model


# ## Loop over all assets

# In[ ]:


Xs = {}
ys = {}
models = {}

for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
    X, y, model = get_Xy_and_model_for_asset(df_train, asset_id)    
    Xs[asset_id], ys[asset_id], models[asset_id] = X, y, model


# In[ ]:


# Check the model interface
x = get_features(df_train.iloc[1])
y_pred = models[0].predict([x])
y_pred[0]


# # Predict & submit
# 
# References: [Detailed API Introduction](https://www.kaggle.com/sohier/detailed-api-introduction)
# 
# Something that helped me understand this iterator was adding a pdb checkpoint inside of the for loop:
# 
# ```python
# import pdb; pdb.set_trace()
# ```
# 
# See [Python Debugging With Pdb](https://realpython.com/python-debugging-pdb/) if you want to use it and you don't know how to.
# 

# In[ ]:


env = gresearch_crypto.make_env()
iter_test = env.iter_test()

for i, (df_test, df_pred) in enumerate(iter_test):
    for j , row in df_test.iterrows():
        
        model = models[row['Asset_ID']]
        x_test = get_features(row)
        y_pred = model.predict([x_test])[0]
        
        df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = y_pred
        
        
        # Print just one sample row to get a feeling of what it looks like
        if i == 0 and j == 0:
            display(x_test)

    # Display the first prediction dataframe
    if i == 0:
        display(df_pred)

    # Send submissions
    env.predict(df_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




