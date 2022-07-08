#!/usr/bin/env python
# coding: utf-8

# This notebook is derived from Kaggle Grandmaster Bojan Tunguz's NB
# https://www.kaggle.com/tunguz/elo-with-h2o-automl This notebook merely builds on his fantastic feature engineering (194 features!) and tries to select the best features using the new python feature selection library: featurewiz
# # FeatureWiz
# <a href="https://github.com/AutoViML/featurewiz"><img src="https://i.ibb.co/ZLdZMZg/featurewiz-logos.png" alt="featurewiz-logos" border="0"></a>
# 

# ### We are first going to download some libraries

# In[ ]:


get_ipython().system('pip install lazytransform --ignore-installed --no-deps')


# In[ ]:


get_ipython().system('pip install featurewiz --ignore-installed --no-deps')
get_ipython().system('pip install xlrd --ignore-installed --no-deps')


# In[ ]:


### You need to install this on Kaggle notebooks since Kaggle has an incompatible version of Pillow ##
get_ipython().system('pip install Pillow==9.0.0')


# In[ ]:


### use only these 3 datasets ##
trainfile = '/kaggle/input/mitsuru-features-only-train-and-test/train_df_noindex.csv'
testfile = '../input/mitsuru-features-only-train-and-test/test_df_noindex.csv'
subfile = '/kaggle/input/elo-merchant-category-recommendation/sample_submission.csv'


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv(trainfile)
test = pd.read_csv(testfile)
sub = pd.read_csv(subfile)
print(train.shape, test.shape)
train.head()


# In[ ]:


target = ['target']
idvars = []
modeltype = 'Regression' ### This is needed to make it simpler for us to evaluate results


# In[ ]:


transform_target = False
imbalanced = False


# In[ ]:


#df[target].value_counts()


# # make sure you set the environment preferences to "latest" in the Kaggle menu options to the right -> see here

# In[ ]:


#!pip install scikit-learn --upgrade


# # Import the required libraries

# In[ ]:


from lazytransform import LazyTransformer


# In[ ]:


import featurewiz as FW


# #### We can try different types of category encoders to see which one works best for this dataset. We will start with a simple encoder and a Robust Scaler since we have many outliers in the dataset

# In[ ]:


encoders = 'auto' ### auto, label, woe encoders are all good ones to start with 
scalers = 'std' ### MinMax is also a good scaler to try ###


# ### We must always check the data types before applying any transforms

# In[ ]:





# In[ ]:


if isinstance(target, str):
    preds = [x for x in list(train) if x not in [target]+idvars]
else:
    preds = [x for x in list(train) if x not in target+idvars]
len(preds)


# #### Check if there are any mixed data types in data - are numeric features numeric? are string features object? These kind of questions should be answered and fixed immediately

# In[ ]:


#train[preds].dtypes


# In[ ]:


#train[''].value_counts()


# ### We need to split data before doing any feature engg or feature transformations.
# Otherwise there will be data leakage which is a major mistake newbies make

# In[ ]:


## we are going to use a random_state for now and see ########
from sklearn.model_selection import train_test_split
trainx, testx = train_test_split(train, train_size=0.80, random_state=99)
print(trainx.shape, testx.shape)


# #### If there are mixed datatypes in object column. We need to fix them using lazytransform which has a handy function to convert mixed data types to numeric ###

# In[ ]:


from lazytransform import convert_all_object_columns_to_numeric
trainx, testx = convert_all_object_columns_to_numeric(trainx, testx)


# ### Let's build a simple model with all the features to set a baseline

# In[ ]:


infcols=[]
infcols = FW.EDA_find_remove_columns_with_infinity(trainx)
print(len(infcols))
trainx.drop(infcols, axis=1, inplace=True)
testx.drop(infcols, axis=1, inplace=True)


# In[ ]:


if isinstance(target, str):
    preds = [x for x in list(trainx) if x not in [target]+idvars]
else:
    preds = [x for x in list(trainx) if x not in target+idvars]
len(preds)


# In[ ]:


outputs = FW.complex_XGBoost_model(X_train=trainx[preds], y_train=trainx[target], 
                        X_test=testx[preds], log_y=False, 
                        GPU_flag=False, scaler='', enc_method='label', n_splits=5, verbose=-1)


# In[ ]:


if modeltype != 'Regression':
    #y_preds = lazy.yformer.inverse_transform(y_preds)
    y_preds = outputs[0]
else:
    y_preds = outputs[0]
y_preds[:4]


# In[ ]:


if modeltype == 'Regression':
    from sklearn.metrics import r2_score, mean_squared_error
    print('R-Squared = %0.0f%%' %(100*r2_score(testx[target],y_preds)))
    print('RMSE = %0.2f' %np.sqrt(mean_squared_error(testx[target],y_preds)))
    #plot_scatter(test[target],testm[target+'_XGBoost_predictions'])
else:
    from sklearn.metrics import balanced_accuracy_score, classification_report
    if isinstance(target, str): 
        print('Bal accu %0.0f%%' %(100*balanced_accuracy_score(testx[target],y_preds)))
        print(classification_report(testx[target],y_preds))
    elif len(target) == 1:
            print('Bal accu %0.0f%%' %(100*balanced_accuracy_score(testx[target],y_preds)))
            print(classification_report(testx[target],y_preds))
    else:
        for each_i, target_name in enumerate(target):
            print('For %s:' %target_name)
            print('    Bal accu %0.0f%%' %(100*balanced_accuracy_score(testx[target].values[:,each_i],y_preds[:,each_i])))
            print(classification_report(testx[target].values[:,each_i],y_preds[:,each_i]))


# ### Let us do some feature engg to see if we can add groupby features

# In[ ]:


catcols = trainx.select_dtypes(include='object').columns.tolist()
catcols = [x for x in catcols if x not in [target]]
len(catcols)


# In[ ]:


numcols = trainx.select_dtypes(include='number').columns.tolist()
if isinstance(target, str):
    numcols = [x for x in numcols if x not in [target]+idvars]
else:
    numcols = [x for x in numcols if x not in target+idvars]
len(numcols)

### This is what we do for adding groupby features ###
rcc = FW.Groupby_Aggregator(categoricals=catcols,aggregates=['count','mean','min','max'], 
                            numerics=numcols)
rcc
# In[ ]:


#train = rcc.fit_transform(trainx)
#test = rcc.transform(testx)


# In[ ]:


if isinstance(target, str):
    train = trainx[preds+[target]]
else:
    train = trainx[preds+target]
test = testx[preds]


# In[ ]:


## check if the features look correct before and after ##
train.head()


# In[ ]:


print(train.shape, test.shape)
train.head(2)


# # Let us do some feature selection since it has too many correlated vars

# In[ ]:


#### Make sure you set correlation limit and other defaults as needed ###
trainm, testm = FW.featurewiz(train, target, corr_limit=0.70, verbose=2, sep=',', 
		header=0, test_data=test,feature_engg='', category_encoders='',
		dask_xgboost_flag=False, nrows=None)


# In[ ]:


print(testm.shape)
testm.isnull().sum().sum()


# In[ ]:


print(trainm.shape)
trainm.isnull().sum().sum()


# In[ ]:


len(train.columns)


# # We will now use the same featurewiz model to see if selected features are making the model better

# In[ ]:


select = testm.columns.tolist()
#select = oldselect
len(select)


# In[ ]:


outputs2 = FW.complex_XGBoost_model(trainm[select], trainm[target], 
                        testm[select],  log_y=False, 
                        GPU_flag=False, scaler='', 
                        enc_method='label', n_splits=5, verbose=-1)


# In[ ]:


#dicto = {1:'f',0:'t'}
#dicto ={}


# In[ ]:


if modeltype != 'Regression':
    y_preds = outputs2[0]
    #y_preds = pd.Series(y_preds).map(dicto)
else:
    y_preds = outputs2[0]
y_preds[:4]


# In[ ]:


if modeltype == 'Regression':
    from sklearn.metrics import r2_score, mean_squared_error
    print('R-Squared = %0.0f%%' %(100*r2_score(testx[target],y_preds)))
    print('RMSE = %0.2f' %np.sqrt(mean_squared_error(testx[target],y_preds)))
    #plot_scatter(test[target],testm[target+'_XGBoost_predictions'])
else:
    from sklearn.metrics import balanced_accuracy_score, classification_report
    if isinstance(target, str): 
        print('Bal accu %0.0f%%' %(100*balanced_accuracy_score(testx[target],y_preds)))
        print(classification_report(testx[target],y_preds))
    elif len(target) == 1:
            print('Bal accu %0.0f%%' %(100*balanced_accuracy_score(testx[target],y_preds)))
            print(classification_report(testx[target],y_preds))
    else:
        for each_i, target_name in enumerate(target):
            print('For %s:' %target_name)
            print('    Bal accu %0.0f%%' %(100*balanced_accuracy_score(testx[target].values[:,each_i],y_preds[:,each_i])))
            print(classification_report(testx[target].values[:,each_i],y_preds[:,each_i]))


# # Let us use LazyTransform to build a better pipeline 

# In[ ]:


x_train = trainx[select]
y_train = trainx[target]
x_test = testx[select]
y_test = testx[target].values


# In[ ]:


### If this appears to be an Imbalanced dataset. Let us set Imbalanced to True in LazyTransform
#pd.DataFrame(y_train.value_counts(1))


# ## Now we will try a simple model first in the lazytransform pipeline
# 1. We can set Imbalanced flag to be True and LazyTransform will automatically apply SMOTE to balance the data sets classes.
# 2. By consistently applying different encoders we were able to select the best encoder. That helps

# ## # let us try a model with lazytransform and see if it is better
# A stacking regressor or classifier typically gives better results than an individual estimator since it combines predictions from multiple estimators.

# In[ ]:


### Let us use a simple model ##
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from xgboost import XGBClassifier, XGBRegressor
if modeltype =='Regression':
    #lgb = LGBMRegressor(n_estimators=150, random_state=0)
    lgb = XGBRegressor(n_estimators=150, random_state=0, 
                        base_score=None, booster='gbtree')
    #lgb = FW.Stacking_Regressor()
    #lgb = FW.Blending_Regressor()
    if isinstance(target, list):
        if len(target) > 1:
            lgb = MultiOutputRegressor(lgb)
else:
    #lgb = LGBMClassifier(n_estimators=50, random_state=0)
    lgb = XGBClassifier(n_estimators=150, random_state=0, 
                        base_score=None, booster='gbtree')
    #lgb = FW.Stacking_Classifier()
    if isinstance(target, list):
        if len(target) > 1:
            lgb = MultiOutputClassifier(lgb)
lgb


# In[ ]:


y_train[:3]


# In[ ]:


### We need to make sure that we transform the target since HeartDisease is categorical
lazy = LazyTransformer(model=lgb, encoders=encoders, scalers=scalers, 
        date_to_string=False, transform_target=transform_target, imbalanced=imbalanced)


# In[ ]:


lazy.fit(x_train,y_train)


# In[ ]:


if modeltype != 'Regression' and transform_target:
    y_preds = lazy.predict(x_test)
    #y_preds = lazy.yformer.inverse_transform(y_preds)
else:
    y_preds = lazy.predict(x_test)
y_preds[:4]


# In[ ]:


if modeltype == 'Regression':
    from sklearn.metrics import r2_score, mean_squared_error
    print('R-Squared = %0.0f%%' %(100*r2_score(testx[target],y_preds)))
    print('RMSE = %0.2f' %np.sqrt(mean_squared_error(testx[target],y_preds)))
    #plot_scatter(test[target],testm[target+'_XGBoost_predictions'])
else:
    from sklearn.metrics import balanced_accuracy_score, classification_report
    if isinstance(target, str): 
        print('Bal accu %0.0f%%' %(100*balanced_accuracy_score(testx[target],y_preds)))
        print(classification_report(testx[target],y_preds))
    elif len(target) == 1:
            print('Bal accu %0.0f%%' %(100*balanced_accuracy_score(testx[target],y_preds)))
            print(classification_report(testx[target],y_preds))
    else:
        for each_i, target_name in enumerate(target):
            print('For %s:' %target_name)
            print('    Bal accu %0.0f%%' %(100*balanced_accuracy_score(testx[target].values[:,each_i],y_preds[:,each_i])))
            print(classification_report(testx[target].values[:,each_i],y_preds[:,each_i]))


# ## Summary: Performing feature engg and feature selection + a lazy pipeline will yield very good results.
# Additional: Remember to apply Imbalanced when using lazytransform for imbalanced datasets and also send in a more complex model such as Stacking models when dealing with difficult datasets.

# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




