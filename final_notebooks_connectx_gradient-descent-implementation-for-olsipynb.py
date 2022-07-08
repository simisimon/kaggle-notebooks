#!/usr/bin/env python
# coding: utf-8

# <h1> Gradient Descent
# 
# [cs229_code_example](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://cs229.stanford.edu/notes-spring2019/Gradient_Descent_Viz.pdf)

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# # Load Dataset

# In[ ]:


from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)
pass


# In[ ]:


# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


# standard sclaer
from sklearn.preprocessing import StandardScaler
def std_scaler(X_train,X_test):
  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train_scale=scaler.transform(X_train)
  X_test_scale=scaler.transform(X_test)
  # add a column of 1 for X
  X_train_scale=np.append(np.ones(shape=(X_train.shape[0],1)),X_train_scale,axis=1)
  X_test_scale=np.append(np.ones(shape=(X_test.shape[0],1)),X_test_scale,axis=1)
  return X_train_scale, X_test_scale

X_train,X_test=std_scaler(X_train,X_test)


# # Gradient Descent

# * step 1 : define the cost function
#   * Cost Function = MSE/2 = 1/2m * (X * Theta_T - y) ^ 2
# *   step 2: calculate the Gradient for cost function
#   * Gradient = (1/m) * X_T * (X * Theta_T - y)
# *   step 3: update theta iteratively 
#   * Theta = Theta - alpha * Gradient
#   * alpha =  step size
# ![link text](https://cs217.stanford.edu/assets/img/grad_descent.png)

# In[ ]:


# function for: define cost function ; calculate the gradient
def cost_function(X,theta,y):
  m=y.size
  y_pred=np.matmul(X,theta.T)
  error=y_pred-y
  cost = 0.5 * m * np.dot(error.T, error)
  gradient = (1/m)* np.dot(X.T,error)
  return cost,gradient


# In[ ]:


# function for: using iteration calculate theta
def gradient_descent(X, y, theta_init, alpha, iters):
  cost_array={}
  theta=theta_init
  m=y.size
  for i in range(iters):
    cost,gradient=cost_function(X,theta,y)
    theta=theta-alpha * gradient
    cost_array[i]=cost
  return theta, cost_array


# In[ ]:


# Set hyperparameters
alpha = 0.01
iterations = 1500

# Initialize Theta Values to 0
theta_init = np.zeros(X_train.shape[1])

# Run Gradient Descent
theta, cost_array = gradient_descent(X_train, y_train, theta_init, alpha, iterations)


# In[ ]:


# function for: display the cost-iter trend
def plot_cost_iter(cost_array):
  iters=cost_array.keys()
  cost=cost_array.values()
  fig=sns.scatterplot(x=iters,y=cost)
  fig.set_xlabel('iters')
  fig.set_ylabel('cost')

# display
plot_cost_iter(cost_array)


# # Plot the Result (MSE + scatterplot)

# In[ ]:


# scatterplot the y_pred and y_actual, display the MSE, theta
def plot_pred_actual(theta,X_test,y_test):
  num_of_pt=y_test.size
  y_test_pred=np.matmul(X_test,theta.T)
  mse=mean_squared_error(y_test,y_test_pred)
  # plot the graph
  fig, ax1 = plt.subplots(1,1,constrained_layout=True)
  sns.scatterplot(x=y_test_pred,y=y_test,ax=ax1)
  xpoints = ypoints = plt.xlim()
  plt.plot(xpoints, ypoints, linestyle='--', color='k')
  ax1.set_xlabel('y_pred')
  ax1.set_ylabel('y_actual')
  ax1.text(0,0.92,f'The MSE = {mse}',transform=ax1.transAxes)
  ax1.text(0,0.85,f'The num of Point = {num_of_pt}',transform=ax1.transAxes)
  return mse


# In[ ]:


# test set result
mse_grad_test=plot_pred_actual(theta,X_test,y_test)

# train set result
mse_grad_train=plot_pred_actual(theta,X_train,y_train)


# # Compare to sklearn package : linear_model.LinearRegression
# 
# * sklearn linear_model web link: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression

# In[ ]:


from sklearn import linear_model
# Create linear regression object
regr = linear_model.LinearRegression(fit_intercept=False)

# Train the model using the training sets
regr.fit(X_train,y_train)

# plot the y_test - y_actual for scikit regr model
mse_sklearn_test=plot_pred_actual(theta=regr.coef_,
                 X_test=X_test,
                 y_test=y_test)


# In[ ]:


# compare the mse for our gradient descent regression VS sklearn package
fig=sns.barplot(x=[mse_sklearn_test,mse_grad_test],y=['sklearn_mse','own_grad_des_mse'])
fig.set_title('sklearn VS own_gradient_descent_regression MSE Compare')

