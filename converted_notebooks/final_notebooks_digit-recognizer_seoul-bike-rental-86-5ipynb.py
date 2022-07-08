#!/usr/bin/env python
# coding: utf-8

# **Import Libraries and set up Notebook**

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib
import warnings

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, accuracy_score

# Configure Jupyter Notebook
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 500) 
pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', -1)
display(HTML("<style>div.output_scroll { height: 35em; }</style>"))

reload(plt)
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")

warnings.filterwarnings('ignore')

plt.style.use('seaborn-white')
plt.rcParams['figure.figsize']=15,15 
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.labelsize']=20
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
plt.rcParams['legend.fontsize']=16


# # Data Understanding (Exploratory Data Analysis)

# ## Overview and Correlations

# In[ ]:


df = pd.read_csv('../input/seoul-bike-sharing-demand-prediction/SeoulBikeData.csv',encoding='cp1252')


# In[ ]:


df.info()


# In[ ]:


list(df)


# In[ ]:


df.head(10).T


# In[ ]:


df.describe(include='all').T


# In[ ]:


plt.style.use('seaborn-white')

df_cluster2 = df.corr()
plot_kws={"s": 1}
sns.clustermap(df_cluster2,
            cmap='RdYlBu',
            linewidths=0.1,
            figsize=(15,15),
               annot=True,
            linecolor='white')


# In[ ]:


plt.figure(figsize=(15,15))
threshold = 0.5
sns.set_style("whitegrid", {"axes.facecolor": ".0"})
df_cluster2 = df.corr()
mask = df_cluster2.where((abs(df_cluster2) >= threshold)).isna()
plot_kws={"s": 1}
sns.heatmap(df_cluster2,
            cmap='RdYlBu',
            annot=True,
            mask=mask,
            linewidths=0.2, 
            linecolor='lightgrey').set_facecolor('white')


# In[ ]:


df_num_corr = df.corr()["Rented Bike Count"][:-1]


# In[ ]:


# Features with high correlation
strong_features = df_num_corr[abs(df_num_corr) >= 0.5].index.tolist()
strong_features.append("Rented Bike Count")
df_strong_features = df.loc[:, strong_features]


# In[ ]:


from pandas_profiling import ProfileReport


# **Read the Report Carefully** - it is commented out to save space and time

# In[ ]:


# %%time
# profile = ProfileReport(df,
#                         explorative=True,
#                        )
# profile


# In[ ]:


list(df)


# In[ ]:


df[df['Rented Bike Count'].isna()]


# ## Add interpreted features

# In[ ]:


df['Date']=pd.to_datetime(df['Date'])
df['Year']=df['Date'].dt.year
df['Month']=df['Date'].dt.month
df['Day']=df['Date'].dt.day
df['WeekDay']=df['Date'].dt.day_name()
mapping_dictDay={'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
df['WeekDayEncoding']=df['WeekDay'].map(mapping_dictDay)


# In[ ]:


df['Functioning Day']=df['Functioning Day'].map({'Yes':1,'No':0})
df['IsHoliday']=df['Holiday'].map({'No Holiday':0,'Holiday':1})


# In[ ]:





# # Pre-processing and Feature Selection
# 

# ## Drop irrelevant or excess features
# 
# The first feature to drop is 'Id'. This feature is an index and not descriptive. Further the features with too much missing data are dropped. 

# In[ ]:


list_drop = ['Date',
            ]
df.drop(list_drop,axis=1,inplace=True)


# In[ ]:


## no rentals on a non functioning day, since it is not open for rentals. Ignore these entries
df=df[df['Functioning Day']!=0]


# ## Manage Missing Values

# In[ ]:


# show the numeric characters
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.describe(include='all').T


# In[ ]:





# ## Encode categorical features
# 
# The categorical features must be encoded to ensure that the models can interpret them. One-hot encoding is used since none of the categorical features are ordinal.  

# In[ ]:


df = pd.get_dummies(df,drop_first=True)


# In[ ]:


df_numeric = df.select_dtypes(include=[np.number])
df_cat = df.select_dtypes(exclude=[np.number])


# ## Split and select the Best Features
# 
# This section does an analysis (univariate statistical tests) to determine which features best predict the target feature. 

# In[ ]:


X = df.drop(['Rented Bike Count'],axis=1)
y = df['Rented Bike Count']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 0,
                                                   )


# In[ ]:


# awesome bit of code from https://www.kaggle.com/code/adibouayjan/house-price-step-by-step-modeling

Selected_Features = []
import statsmodels.api as sm

def backward_regression(X, y, initial_list=[], threshold_out=0.05, verbose=True):
    """To select feature with Backward Stepwise Regression 

    Args:
        X -- features values
        y -- target variable
        initial_list -- features header
        threshold_out -- pvalue threshold of features to drop
        verbose -- true to produce lots of logging output

    Returns:
        list of selected features for modeling 
    """
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f"worst_feature : {worst_feature}, {worst_pval} ")
        if not changed:
            break
    Selected_Features.append(included)
    print(f"\nSelected Features:\n{Selected_Features[0]}")


# Application of the backward regression function on our training data
backward_regression(X_train, y_train)


# In[ ]:


# Keep the selected features only
X_train = X_train.loc[:, Selected_Features[0]]
X_test = X_test.loc[:, Selected_Features[0]]


# ## Normalize features
# a minmax scaler is used on the features to put them all in the same order of size.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Import Metrics
# 
# Imports the libraries that will be used to evaluate the models later on

# In[ ]:


import time
model_performance = pd.DataFrame(columns=['r-Squared','RMSE','total time'])

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, accuracy_score

import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# # Modelling
# 

# In[ ]:


model_performance = pd.DataFrame(columns=['R2','RMSE', 'time to train','time to predict','total time'])


# ## Decision Tree

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.tree import DecisionTreeRegressor\nstart = time.time()\nmodel = DecisionTreeRegressor(min_samples_leaf=21).fit(X_train,y_train)\nend_train = time.time()\ny_predictions = model.predict(X_test) # These are the predictions from the test data.\nend_predict = time.time()\n\nmodel_performance.loc[\'Decision Tree\'] = [model.score(X_test,y_test), \n                                   mean_squared_error(y_test,y_predictions,squared=False),\n                                   end_train-start,\n                                   end_predict-end_train,\n                                   end_predict-start]\n\nprint(\'R-squared error: \'+ "{:.2%}".format(model.score(X_test,y_test)))\nprint(\'Root Mean Squared Error: \'+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False)))\n')


# In[ ]:


plt.style.use('seaborn-white')
plt.rcParams['figure.figsize']=10,10 
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.labelsize']=20
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
plt.rcParams['legend.fontsize']=16

fig,ax = plt.subplots()
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
g = sns.scatterplot(x=y_test,
                y=y_predictions,
                s=100,
                alpha=0.6,
                linewidth=1,
                edgecolor='black',
                ax=ax)
f = sns.lineplot(x=[min(y_test),max(y_test)],
             y=[min(y_test),max(y_test)],
             linewidth=4,
             color='gray',
             ax=ax)

plt.annotate(text=('R-squared error: '+ "{:.2%}".format(model.score(X_test,y_test)) +'\n' +
                  'Root Mean Squared Error: '+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False))),
             xy=(0,3000),
             size='medium')

xlabels = ['{:,.0f}'.format(x) for x in g.get_xticks()]
g.set_xticklabels(xlabels)
ylabels = ['{:,.0f}'.format(x) for x in g.get_yticks()]
g.set_yticklabels(ylabels)
sns.despine()


# In[ ]:


# import graphviz
# from sklearn import tree
# # DOT data
# dot_data = tree.export_graphviz(model, out_file=None, 
#                                 # feature_names=X.columns,  
#                                 # class_names=['Benign','Malignant'],
#                                 filled=True)

# # Draw graph
# graph = graphviz.Source(dot_data, format="png") 
# graph


# In[ ]:


R2E = []
RMSE = []

for nr in range(1,41):
    model = DecisionTreeRegressor(min_samples_leaf=nr
                                  ).fit(X_train,y_train)
    y_predictions = model.predict(X_test) # These are the predictions from the test data.
    R2E.append(model.score(X_test,y_test))
    RMSE.append(mean_squared_error(y_test,y_predictions,squared=False))
 
                                   


# In[ ]:


plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize']=10,10 
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size']=16

fig,ax = plt.subplots()
plt.title('R-Squared error')
plt.ylabel('error')
plt.xlabel('minimum number of samples per leaf node')

f = sns.lineplot(x=range(1,41),
             y=R2E,
             linewidth=3,
             color='gray',
             ax=ax,
                )


sns.despine()


# In[ ]:





# In[ ]:





# ## kNN

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.neighbors import KNeighborsRegressor\nstart = time.time()\nmodel = KNeighborsRegressor(n_neighbors=4).fit(X_train,y_train)\nend_train = time.time()\ny_predictions = model.predict(X_test) # These are the predictions from the test data.\nend_predict = time.time()\n\nmodel_performance.loc[\'kNN\'] = [model.score(X_test,y_test), \n                                   mean_squared_error(y_test,y_predictions,squared=False),\n                                   end_train-start,\n                                   end_predict-end_train,\n                                   end_predict-start]\n\nprint(\'R-squared error: \'+ "{:.2%}".format(model.score(X_test,y_test)))\nprint(\'Root Mean Squared Error: \'+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False)))\n')


# In[ ]:


plt.style.use('seaborn-white')
plt.rcParams['figure.figsize']=10,10 

fig,ax = plt.subplots()
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
sns.scatterplot(x=y_test,
                y=y_predictions,
                s=100,
                alpha=0.6,
                linewidth=1,
                edgecolor='black',
                ax=ax)
sns.lineplot(x=[min(y_test),max(y_test)],
             y=[min(y_test),max(y_test)],
             linewidth=4,
             color='gray',
             ax=ax)

plt.annotate(text=('R-squared error: '+ "{:.2%}".format(model.score(X_test,y_test)) +'\n' +
                  'Root Mean Squared Error: '+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False))),
             xy=(0,3000),
             size='medium')

xlabels = ['{:,.0f}'.format(x) for x in g.get_xticks()]
g.set_xticklabels(xlabels)
ylabels = ['{:,.0f}'.format(x) for x in g.get_yticks()]
g.set_yticklabels(ylabels)
sns.despine()


# In[ ]:


R2E = []
RMSE = []

for nr in range(1,41):
    model = KNeighborsRegressor(n_neighbors=nr
                                  ).fit(X_train,y_train)
    y_predictions = model.predict(X_test) # These are the predictions from the test data.
    R2E.append(model.score(X_test,y_test))
    RMSE.append(mean_squared_error(y_test,y_predictions,squared=False))
 
                                   


# In[ ]:


plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize']=10,10 

fig,ax = plt.subplots()
plt.title('R-Squared error')
plt.ylabel('min samples per leaf node')
plt.xlabel('minimum number of samples per leaf node')

f = sns.lineplot(x=range(1,41),
             y=R2E,
             linewidth=3,
             color='gray',
             ax=ax,
                )


sns.despine()


# In[ ]:





# ## Random Forest

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.ensemble import RandomForestRegressor\nstart = time.time()\nmodel = RandomForestRegressor(n_jobs=-1,\n                              n_estimators=500,\n                              min_samples_leaf=2,\n                             ).fit(X_train,y_train)\nend_train = time.time()\ny_predictions = model.predict(X_test) # These are the predictions from the test data.\nend_predict = time.time()\n\nmodel_performance.loc[\'Random Forest\'] = [model.score(X_test,y_test), \n                                   mean_squared_error(y_test,y_predictions,squared=False),\n                                   end_train-start,\n                                   end_predict-end_train,\n                                   end_predict-start]\n\nprint(\'R-squared error: \'+ "{:.2%}".format(model.score(X_test,y_test)))\nprint(\'Root Mean Squared Error: \'+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False)))\n')


# In[ ]:


plt.style.use('seaborn-white')
plt.rcParams['figure.figsize']=10,10 

fig,ax = plt.subplots()
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
g = sns.scatterplot(x=y_test,
                y=y_predictions,
                s=100,
                alpha=0.6,
                linewidth=1,
                edgecolor='black',
                ax=ax)
f = sns.lineplot(x=[min(y_test),max(y_test)],
             y=[min(y_test),max(y_test)],
             linewidth=4,
             color='gray',
             ax=ax)

plt.annotate(text=('R-squared error: '+ "{:.2%}".format(model.score(X_test,y_test)) +'\n' +
                  'Root Mean Squared Error: '+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False))),
             xy=(0,3000),
             size='medium')

xlabels = ['{:,.0f}'.format(x) for x in g.get_xticks()]
g.set_xticklabels(xlabels)
ylabels = ['{:,.0f}'.format(x) for x in g.get_yticks()]
g.set_yticklabels(ylabels)
sns.despine()


# In[ ]:


R2E = []
RMSE = []

for nr in range(1,41):
    model = RandomForestRegressor(n_jobs=-1,
                                  n_estimators=500,
                                  min_samples_leaf=nr
                                  ).fit(X_train,y_train)
    y_predictions = model.predict(X_test) # These are the predictions from the test data.
    R2E.append(model.score(X_test,y_test))
    RMSE.append(mean_squared_error(y_test,y_predictions,squared=False))
 
                                   


# In[ ]:


plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize']=10,10 

fig,ax = plt.subplots()
plt.title('R-Squared error')
plt.ylabel('error')
plt.xlabel('minimum number of samples per leaf node')

f = sns.lineplot(x=range(1,41),
             y=R2E,
             linewidth=3,
             color='gray',
             ax=ax,
                )


sns.despine()


# In[ ]:





# # Evaluate
# 
# The models are compared in this chapter to determine which give the best performance.

# In[ ]:


# model_performance


# In[ ]:


# model_performance.fillna(.90,inplace=True)
model_performance.style.background_gradient(cmap='RdYlBu_r').format({'R2': '{:.2%}',
                                                                     'RMSE': '{:.2f}',
                                                                     'time to train':'{:.3f}',
                                                                     'time to predict':'{:.3f}',
                                                                     'total time':'{:.3f}',
                                                                     })


# In[ ]:





# # Deploy

# In[ ]:





# In[ ]:




