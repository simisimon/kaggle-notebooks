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


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/hr-analytics/HR_comma_sep.csv')


# # First look

# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# We don't have missing values

# In[ ]:


df.describe()


# #  Data preparation 

# In[ ]:


df.head()


# We have strings in Department and salary columns, let's change them into numerical values

# In[ ]:


df.nunique()


# In[ ]:


df.groupby(['Department']).count()


# **The story of numbers in Department column will be the next:** 
# * IT - 0, 
# * RandD - 1,
# * accounting - 2, 
# * hr - 3, 
# * management - 4, 
# * marketing - 5, 
# * product_mng - 6, 
# * sales - 7, 
# * support - 8, 
# * technical - 9

# In[ ]:


mapping_dep = {'IT': 0, 'RandD': 1, 'accounting': 2, 'hr': 3, 'management': 4, 'marketing': 5, 'product_mng': 6, 'sales': 7, 'support': 8, 'technical': 9}


# In[ ]:


df = df.replace({'Department': mapping_dep})
df.head()


# **The same operation will be performed for salary column**

# In[ ]:


df.groupby(['salary']).count()


# **New variables:**
# * low - 0,
# * medium - 1,
# * high - 2

# In[ ]:


mapping_sal = {'low': 0, 'medium': 1, 'high': 2}


# In[ ]:


df = df.replace({'salary': mapping_sal})
df.head()


# Also we have a big difference in average montly hours column, we could separate it into a few groups

# In[ ]:


sns.histplot(df['average_montly_hours'], color = 'red', bins = 30)


# Here making 4 groups is a good decision.

# In[ ]:


df['productivity_group'] = pd.cut(df['average_montly_hours'], 4)


# In[ ]:


df[['productivity_group', 'salary']].groupby(['productivity_group'], as_index = False).mean().sort_values(by = 'productivity_group', ascending = True)


# So the features for avg monthly hours will be the next:
# * **Group 0** - from 95.786h to 149.5h
# * **Group 1** - 149.5h to 203.0h
# * **Group 2** - 203.0h to 256.5h
# * **Group 3** - 256.5h to 310.0h

# More logical was to make it with universally recognized values as 100h-150h, 150h-200g etc. but we will try to make our model more accurate.

# In[ ]:


df.loc[ df['average_montly_hours'] <= 149.5, 'average_montly_hours'] = 0
df.loc[(df['average_montly_hours'] > 149.5) & (df['average_montly_hours'] <= 203.0), 'average_montly_hours'] = 1
df.loc[(df['average_montly_hours'] > 203.0) & (df['average_montly_hours'] <= 256.5), 'average_montly_hours'] = 2
df.loc[ df['average_montly_hours'] > 256.5, 'average_montly_hours'] = 3


# In[ ]:


sns.histplot(df['average_montly_hours'], color = 'red', bins = 4)


# Now after changes in our avg monthly hours columns we can drop the column with productivity groups

# In[ ]:


df = df.drop(['productivity_group'], axis = 1)
df.head()


# # Correlations and visualisations

# **Let's take a look on a correlation heatmap to understand which data can give us some important information and must be explained better**

# In[ ]:


corrmat = df.corr()
plt.figure(figsize = (15,13))
sns.heatmap(corrmat, square = True, annot = True)
plt.show()


# Here is a good correlation between **number of projects** and **average monthly hours**, maybe it will give us some valuable information later. Also there is a strong negative correlation between **satisfaction level** and status of the employee (**left**) - did he/she left their work and currently are not a part of the company. So we can understand from negative correlation that satisfaction level was very low at the moment of leaving the company. This is a strong parameter for us because we are trying to predict this value according to other parameters. Between **left** and the **time spent in company**, and **left** with **salary**. The last good correlation for this dataset I would like to point out is **last evaluation** and **satisfaction level**, lets try to analyse it better.

# In[ ]:


graph_age = sns.FacetGrid(df, col = 'number_project')
graph_age.map(plt.hist, 'average_montly_hours', bins = 4, color = 'orange', alpha = 0.6)


# Quite logical output, if employees have more projects - they have more hours. On these graphs we can see that if employees have more than 5 projects they will more likely be in 2nd and 3rd productivity group.

# In[ ]:


graph_age = sns.FacetGrid(df, col = 'left')
graph_age.map(plt.hist, 'satisfaction_level', bins = 30, color = 'red', alpha = 0.6)


# So it's easy to see that the less satisfied an employee is with his job, the more likely he is to leave it, and vice versa.

# In[ ]:


graph_age = sns.FacetGrid(df, col = 'time_spend_company')
graph_age.map(plt.hist, 'satisfaction_level', bins = 8, color = 'purple', alpha = 0.6)


# In[ ]:


graph_age = sns.FacetGrid(df, col = 'left')
graph_age.map(plt.hist, 'time_spend_company', bins = 8, color = 'purple', alpha = 0.6)


# By these 2 graphs it's possible to say that after 3 years in company employees, whose satisfaction level is lower than others are more possible to leave the company, but also if employee spent more than 6 years in company this chance is lower

# In[ ]:


graph_age = sns.FacetGrid(df, col = 'left')
graph_age.map(plt.hist, 'salary', bins = 3, color = 'green', alpha = 0.6)


# In[ ]:


graph_age = sns.FacetGrid(df, col = 'salary')
graph_age.map(plt.hist, 'left', bins = 2, color = 'green', alpha = 0.6)


# In[ ]:


graph_age = sns.FacetGrid(df, col = 'salary')
graph_age.map(plt.hist, 'time_spend_company', bins = 3, color = 'green', alpha = 0.6)


# After these 3 visualisations we can say that the logic 'experience=salary' doesn't work for this company and employees may leave their jobs because they are not paid good as for their big experience. Also the chance of leaving job with high salary is lower than with low salary.

# In[ ]:


graph_age = sns.FacetGrid(df, col = 'Work_accident')
graph_age.map(plt.hist, 'time_spend_company', bins = 10, color = 'red')


# In[ ]:


graph_age = sns.FacetGrid(df, col = 'Work_accident')
graph_age.map(plt.hist, 'left', bins = 2, color = 'red')


# According to these 2 graphs it's hard to say that employee will leave his job because of work accident.

# In[ ]:


graph_age = sns.FacetGrid(df, col = 'promotion_last_5years')
graph_age.map(plt.hist, 'time_spend_company', bins = 8, color = 'orange')


# In[ ]:


sns.scatterplot(data = df, x = 'time_spend_company', y = 'satisfaction_level', hue = 'promotion_last_5years')


# It could be possible to say that satisfaction level is decreasing because of zero promotions and lower salary, but this hypothesis is not correct. Promotion is a rare occurrence in this company.

# In[ ]:


sns.set(rc = {'figure.figsize':(15,8)})
sns.scatterplot(data = df, x = 'last_evaluation', y = 'satisfaction_level', hue = 'time_spend_company', size = "time_spend_company", sizes=(50, 250))


# To begin with, we can see a few clusters on this scatter. This plot also shows that there is a big difference between employees satisfaction and his real productivity. For employees, who spent 9-10 years in this company it's hard to find any good relationship, they might have normal distribution, anyway it's not typical for a few employees with such a big experience to have bad points. The biggest cluster (according to mean of spent time and whole population amount) is mainly applied to employees who spent 3-4 years in this company, the range is quite big but mostly their satisfaction level is on the same degree which is >40% and also their points since last evaluation are in the same range. The next cluster of employees I'd like to admit are for employees with 6-7 years spent in company, whose satisfaction level is low (0-20%), but evaluation points are in a wide range (50-100%). The final cluster is related to employees with 6-7 years spent in company, with high satisfaction level (70-90%) and with high evaluation results.
# * The minority of old employees with different satisfaction level and points
# * The majority of new, mostly satisfied employees with medium and good points
# * Experienced good and satisfied employees
# * Experienced good and dissatisfied employees
# 
# Let's look does it work for employees who left their job

# In[ ]:


sns.set(rc = {'figure.figsize':(15,8)})
sns.scatterplot(data = df, x = 'last_evaluation', y = 'satisfaction_level', hue = 'left', size = "time_spend_company", sizes=(25, 150))


# This scatter helped us to visualise our clusters in better way to predict employees who can potentially leave their job. So we will use 3 clusters and try to predict it using k-means

# # Last steps

# So we performed this data analysis to try to predict will the employee leave his job. It will be done by k-means according to our previous calculations.

# In[ ]:


from sklearn.cluster import KMeans

# Graph and create 3 clusters of Employee Turnover
kmeans = KMeans(n_clusters = 3)
kmeans.fit(df[df.left == 1][["last_evaluation", "satisfaction_level"]])

kmeans_colors = ['green' if c == 0 else 'purple' if c == 2 else 'orange' for c in kmeans.labels_]

fig = plt.figure(figsize = (15, 8))
plt.xlabel("Last evaluation")
plt.ylabel("Satisfaction level")
plt.scatter(x = "last_evaluation",y = "satisfaction_level", data = df[df.left == 1],
            alpha = 0.25, color = kmeans_colors)
plt.title("Clusters")
plt.show()


# So after clustering by 3 main parameters we have 3 clusters of employees to predict will they leave their job:
# * Satisfied, experienced and well-evaluated employees **(Yellow)**
# * Dissatisfied, experienced and well-evaluated employees **(Purple)**
# * Low/medium-satisfied, not experienced and medium-evaluated eployees **(Green)**
# 
# Lets create our logistic regression model

# # Logistic Regression

# In[ ]:


df_model = df[['satisfaction_level', 'last_evaluation', 'time_spend_company', 'salary']]
left = df[df.left == 1]
df_model.head()


# First of all we have to train our model on the most interesting parameters

# In[ ]:


X = df_model
y = df.left


# Fitting 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[ ]:


logreg.fit(X_train,y_train)


# In[ ]:


y_pred = logreg.predict(X_test)
y_pred


# # Scores

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# In[ ]:


accuracy_score(y_test, y_pred)


# Quite good result for our model, could be better using more parameters with the same test sample (20%)

# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
clf = SVC(random_state=0)
clf.fit(X_train, y_train)
SVC(random_state=0)
predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
disp.plot()
plt.show()

