#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Periodic Table
# 
# **The periodic table is the table developed for the classification of chemical elements. This table is an order of all known elements in ascending order of atomic number. Although there have been studies in this direction before the periodic table, its inventor is generally accepted as Russian chemist Dmitri Mendeleev. In 1869, when Mendeleev arranged atoms in order of increasing atomic weight, he noticed that certain properties were repeated. He stacked elements with repeating properties and called it a period.**

# # Importing

# In[ ]:


import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as ex
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyo

pyo.init_notebook_mode()
sns.set_style('darkgrid')

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix
import scikitplot as skplt

plt.rc('figure',figsize=(18,9))

get_ipython().run_line_magic('pip', 'install imbalanced-learn')
from imblearn.over_sampling import SMOTE


# In[ ]:


df = pd.read_csv("../input/periodic/periodic_table.csv")
print("Number of datapoints:", len(df))
df


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# In[ ]:


fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['AtomicRadius'],name='AtomicRadius Box Plot',boxmean=True)
tr2=go.Histogram(x=df['AtomicRadius'],name='AtomicRadius Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of AtomicRadius")
fig.show()


# In[ ]:


ex.pie(df,names='Phase',title='Propotion Of Phase',hole=0.33)


# In[ ]:


ex.pie(df,names='Type',title='Propotion Of Different Types',hole=0.33)


# In[ ]:


fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['FirstIonization'],name='First Ionization Box Plot',boxmean=True)
tr2=go.Histogram(x=df['FirstIonization'],name='First Ionization Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of First Ionization (close family size)")
fig.show()


# In[ ]:


fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Electronegativity'],name='Electronegativity Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Electronegativity'],name='Electronegativity Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of Electronegativity")
fig.show()


# In[ ]:


fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Density'],name='Density Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Density'],name='Density Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of Density")
fig.show()


# In[ ]:


fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['NumberofShells'],name='Number of Shells Box Plot',boxmean=True)
tr2=go.Histogram(x=df['NumberofShells'],name='Number of Shells Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of Number of Shells")
fig.show()


# In[ ]:


fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Period'],name='Period Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Period'],name='Period Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of Period")
fig.show()


# In[ ]:


fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Group'],name='Group Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Group'],name='Group Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of Group")
fig.show()


# In[ ]:


fig = make_subplots(rows=2, cols=1,shared_xaxes=True,subplot_titles=('Perason Correaltion',  'Spearman Correaltion'))
colorscale=     [[1.0              , "rgb(165,0,38)"],
                [0.8888888888888888, "rgb(215,48,39)"],
                [0.7777777777777778, "rgb(244,109,67)"],
                [0.6666666666666666, "rgb(253,174,97)"],
                [0.5555555555555556, "rgb(254,224,144)"],
                [0.4444444444444444, "rgb(224,243,248)"],
                [0.3333333333333333, "rgb(171,217,233)"],
                [0.2222222222222222, "rgb(116,173,209)"],
                [0.1111111111111111, "rgb(69,117,180)"],
                [0.0               , "rgb(49,54,149)"]]

s_val =df.corr('pearson')
s_idx = s_val.index
s_col = s_val.columns
s_val = s_val.values
fig.add_trace(
    go.Heatmap(x=s_col,y=s_idx,z=s_val,name='pearson',showscale=False,xgap=0.7,ygap=0.7),
    row=1, col=1
)


s_val =df.corr('spearman')
s_idx = s_val.index
s_col = s_val.columns
s_val = s_val.values
fig.add_trace(
    go.Heatmap(x=s_col,y=s_idx,z=s_val,xgap=0.7,ygap=0.7),
    row=2, col=1
)
fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell"
    )
)
fig.update_layout(height=700, width=900, title_text="Numeric Correaltions")
fig.show()

