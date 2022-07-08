#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

import plotly
plotly.offline.init_notebook_mode(connected=True)

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("/kaggle/input/energy-anomaly-detection/train.csv", delimiter=',', encoding='utf8')
pd.set_option('display.max_columns', None)
df.tail()


# In[ ]:


df.isnull().sum()


# #Beware, since head hour was 00:00:00 adter the next snippet it seemed that it was cleaned hr/min.sec.
# 
# Though when we run tail, we say that they are still there.

# In[ ]:


#Code by Dasmehdixtr https://www.kaggle.com/code/dasmehdixtr/energy-anomally-detection-65-test-set-accuracy/notebook

df['timestamp'] = pd.to_datetime(df['timestamp'])
df.tail()


# In[ ]:


#Code by Dasmehdixtr https://www.kaggle.com/code/dasmehdixtr/energy-anomally-detection-65-test-set-accuracy/notebook

df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['second'] = df['timestamp'].dt.second
df = df.drop(['timestamp'], axis=1)
df.tail()


# #Energy Consumption by month

# In[ ]:


df["month"].plot(figsize=(20,4));


# #I have No clue how to interpret the anomalies on the scatter bellow.

# In[ ]:


fig = px.scatter(df, x="meter_reading", y="anomaly",color_discrete_sequence=['#4257f5'], title="Energy Anomaly Detection" )
fig.show()


# In[ ]:


#Code by Abhishek Sharma https://www.kaggle.com/code/anotherbadcode/apriori-analysis

import scipy.stats as ss
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import os
import json
from datetime import datetime
from itertools import combinations, groupby
from collections import Counter

import itertools
import networkx as nx


# In[ ]:


#Code by Abhishek Sharma https://www.kaggle.com/code/anotherbadcode/apriori-analysis

def cramers_v(confusion_matrix):
    """ 
    Calculate Cramers V statistic for categorial-categorial association.
    Considering items & their occurences here as categories.
    
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


# In[ ]:


#Code by Abhishek Sharma https://www.kaggle.com/code/anotherbadcode/apriori-analysis

def find_association(method, data, columns, plot = True, returnCorr= False):
    """
    Finding associations b/w different metrics based on Cramer's rule.
    """
    
    df = data
    cols = columns
    corrM = np.zeros((len(cols),len(cols)))
    np.fill_diagonal(corrM, 1)

    for col1, col2 in itertools.combinations(cols, 2):
        idx1, idx2 = cols.index(col1), cols.index(col2)
        corrM[idx1, idx2] = cramers_v((pd.crosstab(df[col1], df[col2])).values)
        corrM[idx2, idx1] = corrM[idx1, idx2]

    corr = pd.DataFrame(corrM, index=cols, columns=cols)
    
    corr = round(corr, 2)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    if plot:
        fig, ax = plt.subplots(figsize=(20, 20))
        ax = sns.heatmap(corr, cmap='coolwarm', annot=True, ax=ax, mask=mask, 
                         vmax=1.0, vmin=-1.0, linewidths=0.1,
                        annot_kws={"size": 9}, square=True, cbar=True); 
        ax.set_title("Correlation between Variables");

        plt.savefig("association.png", dpi = 300);
        plt.show()
    
    if returnCorr:
        return corr


# In[ ]:


#Code by Abhishek Sharma https://www.kaggle.com/code/anotherbadcode/apriori-analysis

def drawNetworkGraph(df, startNode, nCols = 9):
    """
    Takes input DataFrame & plots network graph based on Association.
    Edges are weighted using correlation.
    df : Transactional DataFrame
    StartNode : 
    
    """
    weightDict = {}
    cols = (df.sum() / df.shape[0]).sort_values(ascending=False).head(9).index.tolist()
    corrDf = find_association("cramers_v", df, cols, plot=False, returnCorr=True)
    weights = corrDf[startNode].sort_values(ascending=False)

    for key, value in zip(weights.index.tolist(), weights.values.tolist()):
        weightDict[key] = value

    combs = [x for x in itertools.combinations(weightDict.keys(), 2)]
    
    edge_width = [round(45 * corrDf[u][v] , 3) for u, v in combs]
    
    G = nx.Graph()
    plt.figure(figsize =(10, 7))
    G.add_node(startNode)
    for items in weightDict.keys():
        G.add_node(items)

    for items in combs:
        G.add_edge(items[0], items[1])
    
    pos = nx.fruchterman_reingold_layout(G)

    nx.draw_networkx(G, pos, with_labels = True, arrowstyle='-|>',
                     alpha = 0.7, width = edge_width,
                     arrows=True, node_size = weights * 500,
                     edge_color ='.7', cmap = plt.cm.Blues, node_color ='green') 
    
    plt.axis('off')
    plt.tight_layout();
    
drawNetworkGraph(df, startNode = 'anomaly', nCols = 9)


# #Replacing Missing Values.

# In[ ]:


# Lets first handle numerical features with nan value
numerical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes!='O']
numerical_nan


# In[ ]:


df[numerical_nan].isna().sum()


# In[ ]:


## Replacing the numerical Missing Values

for feature in numerical_nan:
    ## We will replace by using median since there are outliers
    median_value=df[feature].median()
    
    df[feature].fillna(median_value,inplace=True)
    
df[numerical_nan].isnull().sum()


# In[ ]:


#imports 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor

from sklearn import metrics

from sklearn.model_selection import cross_validate


# In[ ]:


X = df["meter_reading"]  # numpy array
y = df["anomaly"] # numpy array


# In[ ]:


#Code by Julie Tian  https://www.kaggle.com/julietian/predicting-steam-usage-per-location

model = LinearRegression() #instantiate model

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) #20% for testing 

#required reshaping to fit and predict for the model
X_train_reshape = X_train.values.reshape(-1,1)
X_test_reshape = X_test.values.reshape(-1,1)


# In[ ]:


#proper shape for modelling and predicting 
print(X_train_reshape.shape)
print(y_train.shape)

print(X_test_reshape.shape)
print(y_test.shape)


# In[ ]:


model.fit(X_train_reshape, y_train)


# In[ ]:


predictions = model.predict(X_test_reshape)


# In[ ]:


for y, y_pred in list(zip(y_test, predictions))[:5]:
    print("Real value: {:.3f} Estimated value: {:.5f}".format(y, y_pred))


# #The Competition metric is Area Under Receiver Operating Characteristic Curve (AUC-ROC). 
# 
# However, I didn't find yet another script with AUC/ROC to help me.

# In[ ]:


#Code by Julie Tian  https://www.kaggle.com/julietian/predicting-steam-usage-per-location

rsq = metrics.r2_score(y_test, predictions)
print(f"the R-squaared score is {rsq}")

mse = metrics.mean_squared_error(y_test, predictions)
print(f"the Mean Absolute Error is {mse}")

rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
print(f"the Root Mean Squared Error is {rmse}")


# In[ ]:


results = {} #to store results 


# In[ ]:


#Code by Julie Tian  https://www.kaggle.com/julietian/predicting-steam-usage-per-location

def evaluate_model(estimator, X, y):
    cv_results = cross_validate(estimator,
                    X=X,
                    y=y,
                    scoring="neg_mean_squared_error",
                          n_jobs=-1, cv=50,
                     return_train_score=True)
    return pd.DataFrame(cv_results).abs().mean().to_dict()


# In[ ]:


linreg  = LinearRegression()
dtree   = DecisionTreeRegressor()
elastic = ElasticNet()
lasso   = Lasso()
ridge   = Ridge()


# In[ ]:


#Code by Julie Tian  https://www.kaggle.com/julietian/predicting-steam-usage-per-location

results["linear reg"] = evaluate_model(linreg, X_train_reshape, y_train)
results["tree"] = evaluate_model(dtree, X_train_reshape, y_train)
results["elasticnet"] = evaluate_model(elastic, X_train_reshape, y_train)
results["lasso"] = evaluate_model(lasso, X_train_reshape, y_train)
results["ridge"] = evaluate_model(ridge, X_train_reshape, y_train)

pd.DataFrame.from_dict(results).T


# In[ ]:


from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


# In[ ]:


sns.scatterplot(df["anomaly"],df["meter_reading"] );


# In[ ]:


train_model_data = df[["anomaly"]]
train = train_model_data.iloc[0:(len(train_model_data)-53)].copy()
test = train_model_data.iloc[len(train):(len(train_model_data) -1)].copy()


# In[ ]:


train.shape


# #And more charts that I don't know what's their meaning.

# In[ ]:


ax = train.plot(figsize=(25,4))
test.plot(ax=ax);


# In[ ]:


sm.graphics.tsa.plot_pacf(train,lags=30)
plt.show()


# In[ ]:


#It took so long that I commented it.

#sm.graphics.tsa.plot_acf(train,lags=50)
#plt.show()


# #Acknowledgements: 
# 
# Abhishek Sharma https://www.kaggle.com/code/anotherbadcode/apriori-analysis
# 
# Dasmehdixtr https://www.kaggle.com/code/dasmehdixtr/energy-anomally-detection-65-test-set-accuracy/notebook
# 
# Julie Tian  https://www.kaggle.com/julietian/predicting-steam-usage-per-location
# 
# Clayton Miller https://www.kaggle.com/claytonmiller/weather-influence-on-energy-consumption-example
