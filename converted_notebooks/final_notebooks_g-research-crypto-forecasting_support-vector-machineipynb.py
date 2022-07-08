#!/usr/bin/env python
# coding: utf-8

# <center><img src="https://raw.githubusercontent.com/fukashi-hatake/kaggle_notebooks/main/images/banner_svc.png" 
#      width="100%" 
#      height="100%" />

# ### Importing libraries 

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

### SVM library from sklearn 
from sklearn.svm import SVC 

import warnings
warnings.filterwarnings('ignore') 


# ### Loading dataset 

# In[ ]:


## for task a 
X_1 = pd.read_csv("https://raw.githubusercontent.com/fukashi-hatake/kaggle_notebooks/main/svc_data/X_1.csv", names=['x1', 'x2'], header=None)
y_1 = pd.read_csv("https://raw.githubusercontent.com/fukashi-hatake/kaggle_notebooks/main/svc_data/y_1.csv", names=['y1'], header=None)

## for task b  
X_2 = pd.read_csv("https://raw.githubusercontent.com/fukashi-hatake/kaggle_notebooks/main/svc_data/X_2.csv", names=['x1', 'x2'], header=None) 
y_2 = pd.read_csv("https://raw.githubusercontent.com/fukashi-hatake/kaggle_notebooks/main/svc_data/y_2.csv", names=['y1'], header=None) 

## for task c  
X_3 = pd.read_csv("https://raw.githubusercontent.com/fukashi-hatake/kaggle_notebooks/main/svc_data/X_3.csv", names=['x1', 'x2'], header=None) 
y_3 = pd.read_csv("https://raw.githubusercontent.com/fukashi-hatake/kaggle_notebooks/main/svc_data/y_3.csv", names=['y1'], header=None)  


# ---

# ### Part a 

# **a)** Plot X_1.csv and y_1.csv (use different color for different classes. Apply
# linear SVM and draw decision boundary (solid line) and margins (dashed line) in the
# plot. Provide all support vectors and circle them in the plot.

# In[ ]:


## First let's merge X and y 
data_a = X_1.copy()
data_a['y'] = y_1 


# ##### 1) Plotting 

# In[ ]:


plt.figure(figsize=(8, 6))

sns.scatterplot(data=data_a, x="x1", y="x2", hue='y', s=50) 


# ##### 2) Apply linear SVM and draw decision boundary (solid line) and margins (dashed line) in the plot.  
# ##### 3) Provide all support vectors and circle them in the plot.

# In[ ]:


X = data_a.drop(['y'], axis=1)  
y = data_a['y'] 

svc_model = SVC(kernel='linear', random_state=42).fit(X, y) 


# In[ ]:


svc_model.support_vectors_


# In[ ]:


plt.figure(figsize=(8, 6))

# Plotting
sns.scatterplot(data=data_a, x="x1", y="x2", hue='y', s=50)

# Constructing a hyperplane using a formula.
w = svc_model.coef_[0]           # w consists of 2 elements
b = svc_model.intercept_[0]      # b consists of 1 element
x_points = np.linspace(-2, 5)    # generating x-points from -2 to 5 (based on our data)
y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points

# Plotting a red hyperplane
plt.plot(x_points, y_points, c='r')

# Getting unit-vector:
w_hat = svc_model.coef_[0] / (np.sqrt(np.sum(svc_model.coef_[0] ** 2)))

# Getting margin:
margin = 1 / np.sqrt(np.sum(svc_model.coef_[0] ** 2))

# Calculating points of the margin lines:
decision_boundary_points = np.array(list(zip(x_points, y_points)))
points_of_line_upper = decision_boundary_points + w_hat * margin
points_of_line_lower = decision_boundary_points - w_hat * margin

# Plot margin lines
# Blue margin line upper
plt.plot(points_of_line_upper[:, 0], points_of_line_upper[:, 1], 'b--', linewidth=2)

# Green margin line lower
plt.plot(points_of_line_lower[:, 0], points_of_line_lower[:, 1], 'g--', linewidth=2)

# Circling support vectors 
plt.scatter(svc_model.support_vectors_[:, 0],
            svc_model.support_vectors_[:, 1], 
            s=100, 
            facecolors='none', 
            edgecolors='k', 
            alpha=.5); 


# ### Part b 

# **b)** Plot X_2.csv and y_2.csv (use different color for different classes. Apply kernel SVM (using RBF kernel) and draw decision boundary (solid line) and margins (dashed line) in the plot. Provide all support vectors and circle them in the plot. 

# In[ ]:


## First let's merge X and y 
data_b = X_2.copy()
data_b['y'] = y_2 


# ##### 1) Plotting 

# In[ ]:


plt.figure(figsize=(8, 6))

sns.scatterplot(data=data_b, x="x1", y="x2", hue='y', palette="vlag", s=50) 


# ##### 2) Apply kernel SVM (using RBF kernel) and draw decision boundary (solid line) and margins (dashed line) in the plot. 
# ##### 3) Provide all support vectors and circle them in the plot.

# In[ ]:


X = data_b.drop(['y'], axis=1)  
y = data_b['y'] 

svc_model = SVC(kernel='rbf', random_state=42).fit(X, y)  


# Note **plot_svc_decision_function** is taken from this book (book's webpage): https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html 

# In[ ]:


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# In[ ]:


plt.figure(figsize=(8, 6))

sns.scatterplot(data=data_b, x="x1", y="x2", hue='y', palette="vlag", s=50) 

# 3) Provide all support vectors and circle them in the plot. 
plt.scatter(svc_model.support_vectors_[:, 0],
            svc_model.support_vectors_[:, 1], 
            s=100, 
            facecolors='none', 
            edgecolors='k',
            lw=1.5, 
            alpha=0.5); 

plot_svc_decision_function(svc_model)


# ### Part c 

# **c)** Plot X_3.csv and y_3.csv (use different color for different classes. Apply linear SVM with soft margin and draw decision boundary (solid line) and margins (dashed line) in the plot. Provide all support vectors and circle them in the plot. Try at least 3 different hyperparameters C.

# In[ ]:


## First let's merge X and y 
data_c = X_3.copy()
data_c['y'] = y_3 


# ##### 1) Plotting 

# In[ ]:


plt.figure(figsize=(8, 6))

sns.scatterplot(data=data_c, x="x1", y="x2", hue='y', palette="vlag", s=50) 


# #### 2) Apply linear SVM with soft margin and draw decision boundary (solid line) and margins (dashed line) in the plot. 
# #### 3) Provide all support vectors and circle them in the plot. 
# #### 4) Try at least 3 different hyperparameters C.

# ##### Hyperparameters C = 0.1

# In[ ]:


X = data_c.drop(['y'], axis=1)  
y = data_c['y'] 

svc_model = SVC(kernel='linear', C=0.1, random_state=42).fit(X, y)  


# In[ ]:


plt.figure(figsize=(8, 6))

# Plotting
sns.scatterplot(data=data_c, x="x1", y="x2", hue='y', palette="vlag", s=50)

# Constructing a hyperplane using a formula.
w = svc_model.coef_[0]           # w consists of 2 elements
b = svc_model.intercept_[0]      # b consists of 1 element
x_points = np.linspace(-2, 5)    # generating x-points from -2 to 5 (based on our data)
y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points

# Plotting a red hyperplane
plt.plot(x_points, y_points, c='r')

# Getting unit-vector:
w_hat = svc_model.coef_[0] / (np.sqrt(np.sum(svc_model.coef_[0] ** 2)))

# Getting margin:
margin = 1 / np.sqrt(np.sum(svc_model.coef_[0] ** 2))

# Calculating points of the margin lines:
decision_boundary_points = np.array(list(zip(x_points, y_points)))
points_of_line_upper = decision_boundary_points + w_hat * margin
points_of_line_lower = decision_boundary_points - w_hat * margin

# Plot margin lines
# Blue margin line upper
plt.plot(points_of_line_upper[:, 0], points_of_line_upper[:, 1], 'b--', linewidth=2)

# Green margin line lower
plt.plot(points_of_line_lower[:, 0], points_of_line_lower[:, 1], 'g--', linewidth=2)

# Circling support vectors 
plt.scatter(svc_model.support_vectors_[:, 0],
            svc_model.support_vectors_[:, 1], 
            s=100, 
            facecolors='none', 
            edgecolors='k', 
            alpha=.5); 


# ##### Hyperparameters C = 1  

# In[ ]:


X = data_c.drop(['y'], axis=1)  
y = data_c['y'] 

svc_model = SVC(kernel='linear', C=1, random_state=42).fit(X, y)  


# In[ ]:


plt.figure(figsize=(8, 6))

# Plotting
sns.scatterplot(data=data_c, x="x1", y="x2", hue='y', palette="vlag", s=50)

# Constructing a hyperplane using a formula.
w = svc_model.coef_[0]           # w consists of 2 elements
b = svc_model.intercept_[0]      # b consists of 1 element
x_points = np.linspace(-2, 5)    # generating x-points from -2 to 5 (based on our data)
y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points

# Plotting a red hyperplane
plt.plot(x_points, y_points, c='r')

# Getting unit-vector:
w_hat = svc_model.coef_[0] / (np.sqrt(np.sum(svc_model.coef_[0] ** 2)))

# Getting margin:
margin = 1 / np.sqrt(np.sum(svc_model.coef_[0] ** 2))

# Calculating points of the margin lines:
decision_boundary_points = np.array(list(zip(x_points, y_points)))
points_of_line_upper = decision_boundary_points + w_hat * margin
points_of_line_lower = decision_boundary_points - w_hat * margin

# Plot margin lines
# Blue margin line upper
plt.plot(points_of_line_upper[:, 0], points_of_line_upper[:, 1], 'b--', linewidth=2)

# Green margin line lower
plt.plot(points_of_line_lower[:, 0], points_of_line_lower[:, 1], 'g--', linewidth=2)

# Circling support vectors 
plt.scatter(svc_model.support_vectors_[:, 0],
            svc_model.support_vectors_[:, 1], 
            s=100, 
            facecolors='none', 
            edgecolors='k', 
            alpha=.5); 


# ##### Hyperparameters C = 100  

# In[ ]:


svc_model = SVC(kernel='linear', C=100, random_state=42).fit(X, y)  


# In[ ]:


plt.figure(figsize=(8, 6))

# Plotting
sns.scatterplot(data=data_c, x="x1", y="x2", hue='y', palette="vlag", s=50)

# Constructing a hyperplane using a formula.
w = svc_model.coef_[0]           # w consists of 2 elements
b = svc_model.intercept_[0]      # b consists of 1 element
x_points = np.linspace(-2, 5)    # generating x-points from -2 to 5 (based on our data)
y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points

# Plotting a red hyperplane
plt.plot(x_points, y_points, c='r')

# Getting unit-vector:
w_hat = svc_model.coef_[0] / (np.sqrt(np.sum(svc_model.coef_[0] ** 2)))

# Getting margin:
margin = 1 / np.sqrt(np.sum(svc_model.coef_[0] ** 2))

# Calculating points of the margin lines:
decision_boundary_points = np.array(list(zip(x_points, y_points)))
points_of_line_upper = decision_boundary_points + w_hat * margin
points_of_line_lower = decision_boundary_points - w_hat * margin

# Plot margin lines
# Blue margin line upper
plt.plot(points_of_line_upper[:, 0], points_of_line_upper[:, 1], 'b--', linewidth=2)

# Green margin line lower
plt.plot(points_of_line_lower[:, 0], points_of_line_lower[:, 1], 'g--', linewidth=2)

# Circling support vectors 
plt.scatter(svc_model.support_vectors_[:, 0],
            svc_model.support_vectors_[:, 1], 
            s=100, 
            facecolors='none', 
            edgecolors='k', 
            alpha=.5); 


# > **Summary**: By using different hyperparameter C, we can see that when the C is small (C=0.1), the distance between upper and below margin lines is high. On the other hand when the C is high, the distance between upper and below margin lines is smaller.  

# #### Thank you for reading. 
# #### If you found this notebook useful, please upvote it! 
