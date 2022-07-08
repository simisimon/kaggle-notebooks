#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

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


# <center style="font-family:verdana;"><h1 style="font-size:200%; padding: 10px; background: #001f3f;"><b style="color:orange;">Grammar of Graphics. L.Wilkinson. GGPlot. H.Wickham. W.Chang</b></h1></center>
# 
# The Grammar of Graphics
# 
# "The grammar of graphics is a way of thinking about how graphs are constructed that allows data analysts to move beyond thinking about a small number of graph types (like bar graphs, line graphs, scatter plots, etc)."
# 
# "Think about the grammar of graphics just like you would about the grammar of sentence structure in language. Think beyond understanding language as just sentence types, like declarative sentences (statements of fact), imperative sentences (statement of request), or exclamatory sentences (statements of excitement/emotion)."
# 
# https://murraylax.org/rtutorials/gog.html
# 
# "Statistical graphics is a mapping from data to aesthetic attributes (colour, shape, size) of geometric objects (points, lines, bars)"
# 
# "These are basic building blocks according to the Grammar of Graphics:"
# 
# "data The data + a set of aesthetic mappings that describing variables mapping"
# 
# "geom Geometric objects, represent what you actually see on the plot: points, lines, polygons, etc."
# 
# "stats Statistical transformations, summarise data in many useful ways."
# 
# "scale The scales map values in the data space to values in an aesthetic space"
# 
# "coord A coordinate system, describes how data coordinates are mapped to the plane of the graphic."
# 
# "facet A faceting specification describes how to break up the data into subsets for plotting individual set"
# 
# https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/
# 
# 
# Without any of these three components, plotnine wouldn’t know how to draw the graphic: Data
# Aesthetics, Geometric objects.
# 
# "Data is the information to use when creating the plot."
# 
# "Aesthetics (aes) provides a mapping between data variables and aesthetic, or graphical, variables used by the underlying drawing system. In the previous section, you mapped the date and pop data variables to the x- and y-axis aesthetic variables."
# 
# "Geometric objects (geoms) defines the type of geometric object to use in the drawing. You can use points, lines, bars, and many others."
# 
# https://realpython.com/ggplot-python/

# ![](https://miro.medium.com/max/1200/1*8hEQcRXPyGH4tDHUI3VOoA.png)towardsdatascience.com

# #The Grammar of Graphics. 
# 
# By Thomas de Beus
# 
# "This Grammar of Graphics framework is built by statistician and computer scientist Leland Wilkinson. Its popularity got an enormous boost when Hadley Wickham created the immense popular package ggplot2 within the statistical programming language R. This package is based on the Grammar of Graphics, and the code you write follows its layers."
# 
# ![](https://miro.medium.com/max/1400/1*MMZuYgeC_YjXNC1r4D4sog.png)https://medium.com/tdebeus/think-about-the-grammar-of-graphics-when-improving-your-graphs-18e3744d8d18

# In[ ]:


from plotnine import *
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


df = pd.read_csv('../input/student-mental-health/Student Mental health.csv',delimiter=',', encoding='ISO-8859-2')
df.head()


# In[ ]:


#Code by Dasmehdixtr https://www.kaggle.com/code/dasmehdixtr/energy-anomally-detection-65-test-set-accuracy/notebook

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.tail()


# In[ ]:


#Code by Dasmehdixtr https://www.kaggle.com/code/dasmehdixtr/energy-anomally-detection-65-test-set-accuracy/notebook

df['year'] = df['Timestamp'].dt.year
df['month'] = df['Timestamp'].dt.month
df['day'] = df['Timestamp'].dt.day
df['hour'] = df['Timestamp'].dt.hour
df['minute'] = df['Timestamp'].dt.minute
df['second'] = df['Timestamp'].dt.second
df = df.drop(['Timestamp'], axis=1)
df.tail()


# In[ ]:


#Code by Rohannanaware (Yoda) https://www.kaggle.com/code/rohannanaware/decision-tree-basics-us-income-dataset/notebook

(ggplot(df, aes(x = "Choose your gender", fill = "What is your course?"))
 + geom_bar(position="fill")
 + theme(axis_text_x = element_text(angle = 60, hjust = 1))
)


# In[ ]:


#Code by Rohannanaware (Yoda) https://www.kaggle.com/code/rohannanaware/decision-tree-basics-us-income-dataset/notebook

(ggplot(df, aes(x = "Do you have Depression?", fill = "Did you seek any specialist for a treatment?"))
 + geom_bar(position="fill")
 + theme(axis_text_x = element_text(angle = 60, hjust = 1))
 + facet_wrap('Age')
)


# In[ ]:


#Code by Rohannanaware (Yoda) https://www.kaggle.com/code/rohannanaware/decision-tree-basics-us-income-dataset/notebook

(ggplot(df, aes(x = "Your current year of Study", fill = "What is your CGPA?"))
 + geom_bar(position="fill")
 + theme(axis_text_x = element_text(angle = -90, hjust = 1))
 + facet_wrap('Choose your gender')#Original was ~Sex
)


# #Cumulative Grade Point Average (CGPA) is an assessment tool used to evaluate your academic performance. In the Bachelor of Engineering programs, CGPA is calculated to determine a student's current standing overall (this includes all courses counting towards the degree) and if applicable, in a minor.
# 
# https://carleton.ca/engineering-design/current-students/undergrad-academic-support/cumulative-grade-point-average-cgpa/#:~:text=Cumulative%20Grade%20Point%20Average%20(CGPA)%20is%20an%20assessment%20tool%20used,if%20applicable%2C%20in%20a%20minor.

# In[ ]:


#Code by Rohannanaware (Yoda) https://www.kaggle.com/code/rohannanaware/decision-tree-basics-us-income-dataset/notebook

(ggplot(df, aes(x = "Marital status", fill = "What is your CGPA?"))
 + geom_bar(position="fill")
 + theme(axis_text_x = element_text(angle = -90, hjust = 1))
 + facet_wrap('Choose your gender')
)


# In[ ]:


#Code by Rohannanaware (Yoda) https://www.kaggle.com/code/rohannanaware/decision-tree-basics-us-income-dataset/notebook

(ggplot(df, aes(x="What is your CGPA?", y="Age"))
 + geom_jitter(position=position_jitter(0.1), color= "red")
)


# In[ ]:


#Code by Rohannanaware (Yoda) https://www.kaggle.com/code/rohannanaware/decision-tree-basics-us-income-dataset/notebook

(ggplot(df, aes(x="Choose your gender", y="Do you have Depression?"))
 + geom_jitter(position=position_jitter(0.1), color="blue")
)


# In[ ]:


#Code by https://realpython.com/ggplot-python/

from plotnine.data import mpg
from plotnine import ggplot, aes, geom_bar

(ggplot(df, aes(x = "Do you have Panic attack?", color="Choose your gender"))
 + geom_bar(alpha = 0.1, color = "blue")#I couldn't change the grey color. Only the contour
)


# In[ ]:


#Code by https://realpython.com/ggplot-python/

from plotnine.data import mpg
from plotnine import ggplot, aes, geom_bar

(ggplot(df, aes(x = "Do you have Depression?"))
 + geom_bar(alpha = 0.1, color = "red")
)


# In[ ]:


from plotnine.data import huron
from plotnine import ggplot, aes, geom_histogram

(ggplot (df, aes(x="month"))
 + geom_histogram(bins=10, color = "blue")+ huron
)


# In[ ]:


#Code by https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

(ggplot(df, aes(x='Your current year of Study', y='month',
    size = 'Do you have Panic attack?')) + geom_point(alpha = 0.1, color = "blue"))


# In[ ]:


#Code by https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

(ggplot(df, aes(x='Marital status', y='month', 
    size = 'Age', color = 'What is your course?')) + geom_point())


# In[ ]:


#Code by https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

(ggplot(df, aes(x='month', y='What is your course?')) + \
    geom_boxplot())


# "Produce a plot comparing the number of observations for each species at each site. The plot should have site_id on the x axis, ideally as categorical data. (HINT: You can convert a column in a DataFrame df to the 'category' type using: df['some_col_name'] = df['some_col_name'].astype('category'))"
# 
# "Create a boxplot of hindfoot_length across different species (species_id column) (HINT: There's a list of geoms available for plotnine in the docs - instead of geom_bar, which one should you use ?)"
# 
# https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

# In[ ]:


#Why are we not seeing mulitple boxplots, one for each year? This is because year variable is continuous in our data frame, but for this purpose we want it to be categorical.

df['month'] = df['month'].astype("category")

(ggplot(df, aes(x='month', y='Age')) + \
    geom_boxplot())


# #If you notice that the x-axis labels are overlapped. To flip them 90-degrees we can apply a theme so they look less cluttered. 
# 
# https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

# In[ ]:


#Code by https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

(ggplot(df, aes(x='month', y='Age')) + \
    geom_boxplot() + \
    theme(axis_text_x = element_text(angle=90, hjust=1)))


# #To save some typing, let's define this x-axis label rotating theme as a short variable name that we can reuse:

# In[ ]:


#Code by https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

flip_xlabels = theme(axis_text_x = element_text(angle=90, hjust=1))

(ggplot(df, aes(x='month', y='Age')) + \
    geom_violin() + \
    flip_xlabels)


# #Faceting
# 
# "ggplot has a special technique called faceting that allows to split one plot into multiple plots based on a factor included in the dataset. We will use it to make one plot for a time series for each species."
# 
# https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

# In[ ]:


#Code by  https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

(ggplot(df, aes(x='month', y='Age')) + \
    geom_boxplot() + \
    facet_wrap(['What is your course?']) + \
    flip_xlabels + \
    theme(axis_text_x = element_text(size=6)))


# In[ ]:


#Code by  https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

(ggplot(df, aes(x='month', y='Age')) + \
    geom_boxplot() + \
    facet_wrap(['Did you seek any specialist for a treatment?']) + \
    flip_xlabels + \
    theme(axis_text_x = element_text(size=6)))


# In[ ]:


#Code by https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

ggplot(df, aes(x='month', y='Age')) + \
    geom_boxplot() + \
    facet_wrap(['Your current year of Study']) + \
    theme_xkcd() + \
    theme(axis_text_x = element_text(size=4, angle=90, hjust=1))


# In[ ]:


#Code by  https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

(ggplot(df, aes(x='month', y='Age')) + \
    geom_boxplot() + \
    facet_wrap(['Your current year of Study']) + \
    flip_xlabels + \
    theme(axis_text_x = element_text(size=6)))


# In[ ]:


#Code by  https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

(ggplot(df, aes(x='month', y='Age')) + \
    geom_boxplot() + \
    theme(axis_text_x = element_text(size=4)) + \
    facet_wrap(['Do you have Depression?']) + \
    flip_xlabels)


# In[ ]:


#Code by  https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

(ggplot(df, aes(x='month', y='Age')) + \
    geom_boxplot() + \
    theme(axis_text_x = element_text(size=4)) + \
    facet_wrap(['Marital status']) + \
    flip_xlabels)


# #Theming
# 
# plotnine allows pre-defined 'themes' to be applied as aesthetics to the plot.
# 
# A list available theme you may want to experiment with is here: https://plotnine.readthedocs.io/en/stable/api.html#themes
# 
# https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

# In[ ]:


#Code by  https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/ 

ggplot(df, aes(x='month', y='Age')) + \
    geom_boxplot() + \
    theme_bw() + \
    flip_xlabels


# #Extra bits 
# 
# This is a different way to look at your data

# In[ ]:


#Code by  https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/

ggplot(df, aes("month", "Age")) + \
    stat_summary(fun_y = np.mean, fun_ymin=np.min, fun_ymax=np.max) + \
    theme(axis_text_x = element_text(angle=90, hjust=1))

ggplot(df, aes("month", "Age")) + \
    stat_summary(fun_y = np.median, fun_ymin=np.min, fun_ymax=np.max) + \
    theme(axis_text_x = element_text(angle=90, hjust=1))

ggplot(df, aes("month", "Age")) + \
    stat_summary(fun_y = np.mean, fun_ymin=np.min, fun_ymax=np.max) + \
    theme(axis_text_x = element_text(angle=90, hjust=1))


# #Acknowledgements:
# 
#  https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/
#  
#  https://realpython.com/ggplot-python/
#  
#  Rohannanaware (Yoda) https://www.kaggle.com/code/rohannanaware/decision-tree-basics-us-income-dataset/notebook
# 
