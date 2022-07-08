#!/usr/bin/env python
# coding: utf-8

# ## Concepts and Examples Covered
# The aim of this notebook is to provide example code on how to define and use custom data transformations in scikit-learn pipelines (***spoiler alert: you don't need wrapper classes for your custom transformations!***). More specifically, the custom data transformations are as follows:
# 
# - Splitting a dataset into training and test sets using a random split or by year
# - Weight the training set based on label criteria
# - Train a machine learning model
# - Make predictions on the test set using the trained model
# 
# ## Define the Custom Transformations/Pipeline Steps

# In[ ]:


# imports
import numpy as np, pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

# load data, rename columns
# NOTE: columns are being renamed here only because they were not properly labelled in the original dataset
data= pd.read_csv('/kaggle/input/lower-back-pain-symptoms-dataset/Dataset_spine.csv')
data= data.iloc[:, :-1]
data.columns= ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis', 'pelvic_slope', 'direct_tilt', 'thoracic_slope', 'cervical_tilt', 'sacrum_angle', 'scoliosis_slope', 'label']

# to better illustrate one of the custom transformations, we're also going to add a Year column. Note that this column is not part of the original dataset and is only being added for illustrative purposes
data['year']= np.random.randint(2000, 2010, data.shape[0])

# very quickly, we'll label-encode the label column, so the data labels are numeric
data.loc[data.loc[data['label'] == 'Normal'].index, 'label']= 0
data.loc[data.loc[data['label'] == 'Abnormal'].index, 'label']= 1

print('Number of columns/rows with any missing values:', data.isnull().sum().sum())
print('Dataset size:', np.shape(data))
data.head()
# for this dataset, each row is 1 observation/spine and all except the last column are features


# Now that the data is loaded, we're ready to define our pipelines. For each custom transformation, we're going to define a function that fulfills the transformation. Then, we're going to use the FunctionTransformer() function from sklearn.preprocessing in our pipeline to use the custom transformation functions in our pipeline.
# 
# Here, we're jumping straight into writing the functions that define our custom data transformations, but you may want to write and test your transformations first, before using them in a pipeline. The key thing to remember about pipelines is that **the output of a pipeline step is the input to the next pipeline step**.
# 
# To conserve space, the code for the first pipeline step is displayed here. The code for the other pipeline steps is hidden.

# In[ ]:


# define the data split transformation
def data_split(df, split_list, features, label):
    """
    Split dataframe into train and test sets using either a random 80/20 split or by year
    
    Parameters
    ----------
    df : Pandas DataFrame. Columns include features and label/target
    
    split_list : either a list of integers or None. If None, train/test split is a random 80/20 split. If list of integers, the integers are the years to be used as the test set and all other years are used as train set
    
    features : list of strings specifying the column names in df to use as model features
    
    label : string specifying the column in df that is the label
    
    Returns
    ----------
    X_train : Pandas Dataframe. Columns are features, rows are train set observations
    
    X_test : Pandas Dataframe. Columms are features, rows are test set observations
    
    y_train : 1D numpy array of integers that are labels corresponding to observations in X_train
    
    y_test : 1D numpy array of integers that are labels corresponding to observations in X_test
    """
    if split_list == None: # split data using random 80/20 split
        X= df[features]
        y= df[label].values
        X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 3)
        X_train= pd.DataFrame(X_train, columns= features).reset_index(drop= True)
        X_test= pd.DataFrame(X_test, columns= features).reset_index(drop= True)
        
    else: # split data based on years
        train_df= df.loc[~df['year'].isin(split_list)]
        test_df= df.loc[df['year'].isin(split_list)]
        X_train= train_df[features]
        y_train= train_df[label].values
        X_test= test_df[features]
        y_test= test_df[label].values
    
    return X_train, X_test, y_train, y_test


# In[ ]:


# define the weight training set transformation
def weight_train(data_split_output, wgt, wgt_amt, lab_criteria):
    """
    Weight training examples belonging to a specific class by some amount (idea is model will learn more from these observations)
    
    Parameters
    ----------
    data_split_output : tuple. Output from data_split()
    
    wgt : Boolean specifying whether to add extra weight to training examples belonging to a certain class
    
    wgt_amt : integer specifying the amount to weight the training examples by (only applies if wgt == True)
    
    lab_criteria : integer specifying the class label to apply the weight to. Training observations belonging to all other classes will have a weight of 1 (only applies if wgt == True)
    
    Returns
    ----------
    X_train : Pandas Dataframe. Columns are features, rows are train set observations
    
    X_test : Pandas Dataframe. Columms are features, rows are test set observations
    
    y_train : 1D numpy array of integers that are labels corresponding to observations in X_train
    
    y_test : 1D numpy array of integers that are labels corresponding to observations in X_test
    
    weights : 1D numpy array of integers. Weights corresponding to observations in X_train
    """
    
    X_train, X_test, y_train, y_test= data_split_output
    
    weights= np.full(len(y_train), 1)
    
    if wgt == True:
        loc_wgt= np.where(y_train == lab_criteria)[0]
        weights[loc_wgt]= wgt_amt
    
    return X_train, X_test, y_train, y_test, weights


# In[ ]:


# define a custom pipeline step that trains a decision tree model
def train_dt(weight_train_output, model_hypers):
    """
    Train a scikit-learn decision tree model
    
    Parameters
    ----------
    weight_train_output : tuple. Output from weight_train()
    
    model_hypers : None or dictionary specifying the decision tree hyperparameter settings for the decision tree. If None, the default hyperparameter settings are used
    
    Returns
    ----------
    X_train : Pandas Dataframe. Columns are features, rows are train set observations
    
    X_test : Pandas Dataframe. Columms are features, rows are test set observations
    
    y_train : 1D numpy array of integers that are labels corresponding to observations in X_train
    
    y_test : 1D numpy array of integers that are labels corresponding to observations in X_test
    
    model : trained decision tree model (trained using X_train and y_train)
    
    """
    X_train, X_test, y_train, y_test, weights= weight_train_output
    
    if model_hypers == None: # if hyperparameters not specified, use the default settings
        model= DecisionTreeClassifier()
    else:
        model= DecisionTreeClassifier(**model_hypers)
    
    model.fit(X_train, y_train.astype(int), weights)
    
    return X_train, X_test, y_train, y_test, model


# In[ ]:


# define a custom pipeline step that makes predictions using the trained decision tree model
def make_predictions(train_dt_output, return_perf= True, display_perf= True):
    """
    Make predicions on the train and test sets using the trained decision tree model
    
    Parameters
    ----------
    train_dt_output : tuple. Output from train_dt
    
    return_perf : Boolean specifying whether model performance metrics should be returned from the function
    
    display_perf : Boolean specifying whether model performance metrics should be displayed within the function
    
    Returns
    ----------
    df_train : Pandas Dataframe. Columns include features, label, and predictions. Rows are train set observations
    
    df_test : Pandas Dataframe. Columns include feautres, label, and predictions. Rows are test set observations
    
    """
    X_train, X_test, y_train, y_test, model= train_dt_output
    
    # make predictions for the train and test sets
    df_test= X_test.copy()
    df_test['predicted']= model.predict(X_test)
    df_test['label']= y_test
    
    df_train= X_train.copy()
    df_train['predicted']= model.predict(X_train)
    df_train['label']= y_train
    
    # calculate performance metrics (optional)
    if (display_perf == True) | (return_perf == True) :
        acc_train= accuracy_score(df_train['label'].astype(int), df_train['predicted'].astype(int))
        acc_test= accuracy_score(df_test['label'].astype(int), df_test['predicted'].astype(int))
        recall_test= recall_score(df_test['label'].astype(int), df_test['predicted'].astype(int))
        precision_test= precision_score(df_test['label'].astype(int), df_test['predicted'].astype(int))
    
    # display performance metrics (optional)
    if display_perf == True:
        display(pd.DataFrame({'Train accuracy': acc_train, 'Test accuracy': acc_test, 'Test recall': recall_test, 'Test precision': recall_test}, index= [0]))
    
    if return_perf == True:
        return df_train, df_test, acc_train, acc_test, recall_test, precision_test
    
    else:
        return df_train, df_test


# ## Run the pipeline

# In[ ]:


# define variables to be used in pipeline function arguments
# modify these to change the model and model training
split_list= None
features= data.columns[:5].values # using just a subset of available features
label= 'label'
wgt= False
wgt_amt= 10
lab_criteria= 1
model_hypers= {'max_depth': 3, 'min_samples_split': 4}
return_perf= False
display_perf= True

# define the pipeline
# feed in arguments to specific functions using a dictionary to kw_args
p1= make_pipeline(FunctionTransformer(data_split, kw_args= {'split_list': split_list, 'features': features, 'label': label}),
                  FunctionTransformer(weight_train, kw_args= {'wgt': wgt, 'wgt_amt': wgt_amt, 'lab_criteria': lab_criteria}),
                  FunctionTransformer(train_dt, kw_args= {'model_hypers': model_hypers}),
                  FunctionTransformer(make_predictions, kw_args= {'return_perf': return_perf, 'display_perf': display_perf})
                 )

# run the pipeline, visualize results
df_train, df_test= p1.transform(data)


# ## Conclusions
# This notebook provided example code on defining custom data transformations to use in scikit-learn pipelines. Although they can be intimidating at first, pipelines are useful because they automate workflows, making work reproducible and easy to understand. Hopefully this sample script brings you one step closer to defining your own custom workflows! Post your feedback and questions in the comments!
