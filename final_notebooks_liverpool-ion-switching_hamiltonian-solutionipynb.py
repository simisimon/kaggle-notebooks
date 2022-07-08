#!/usr/bin/env python
# coding: utf-8

# ## Derive the equations of motion y(t) and y'(t)
# 
# This problem might be solved using optimal control mathematics. We want to control a detector of the system's output states (how many open channels) so that the f1 score of the detector is maximized. The Kalman filter is based on this approach yet it has a slightly different measure of performance **J(x)**.
# 
# The Hamiltonian equation offers a solution. The solution of the minumized partial derivative of the Hamiltonian is the solution for this problem. It tries to provide an estimate of "open_channels" at the highest possible f1 score. This requires that our performance score is diffenciable on both time and x.
# 
# ![A System Model](https://www.elmtreegarden.com/wp-content/uploads/2020/04/ion-channel-system-model.jpg)
# 
# ## The HJB equation, now called the Hamiltonian
# 
# In this case the Hamiltonian is the energy equation of the system. H = T - V = kinetic energy - potential energy.
# 
# ![The Hamiltonian](https://www.elmtreegarden.com/wp-content/uploads/2020/05/pontryagin.jpg)
# 
# Let,
# 
# x0(t) = signal(t) which has been comb filtered to remove noise<br>
# x1(t) = energy of the signal = r x0(t)^2 * dt<br>
# x2(t) = injected energy to probe = a1 * x1(t-Ta:t).min() * dt<br>
# y(t) = open_channels<br>
# 
# The energy of open channels is k * open_channels * dt. This energy is the energy required to transport the ion through the electric field in the probe.
# 
# to conserve enery we know that total energy is zero, so
# 
#     0 = Signal Energy + Injected Energy - energy of transition
#     
#     0 = x1(t) dt + x2(t) dt - k y'(t)
#     y(t) = 1/k * (x1(t) dt + x2(t) dt) = 
#     y(t) = 1/k * ( x1(t) + a1 * x1(t-Ta:t).min)) *dt
#     y(t) = 1/k * ( r x0(t)^2 + a1 r x0(t-Ta:t)^2.min()) dt
#     
#     dy(t)/dt = r/k ( x0(t)^2 + a1 x0(t-Ta:t)^2.min() )
# 
# 
# 
# ### Measure of performance
# 
# The performance function, **J(t)** , is the f_1 score, f_1(y, y_true). It is a measure of energy. Unfortunately the f1 score abruptly changes. It is not differentiable with time. A solution is to 'soften this function' into a measurement score that is differentiable. Here is [a link to the code](https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d). <br>
# 
#   **J**(y(t), t) = soft_f1_score(y(t),y_true(t))
# 
# 
# 
# ### Initial conditions
# Initial conditions will be assumed are<br>
# X0(0) = X0_dot(0) = 0<br>
# y'(0) = 0
# 
# ### The old HJB Equation (as done 40 years ago)
# 
# The HJB equation it's partial derivatives of d**J**/dt = **J<sub>t</sub>(t)**, and d**J**/dx = **J<sub>x</sub>(t)** is:
# 
# Per Donald Kirk's book, the text I learned 40 years ago:<br>
# 0 = **J<sub>t</sub>(t)**(**y**, t) + **H**{**x**, **u'**{**x**, **J<sub>x</sub>(x)**(**y**,t), t},**J<sub>x</sub>(x)**(**y**, t), t}<br>
# The **u'** that solves this diffential equation is the optimal control for the History.
# 
# 
# 
# References:
# 1. Kirk, "Optimal Control Theory", 1970 Chapters 1-3<br>
# 2. [E.T. Jaynes: Minimum Entropy Principle](https://pdfs.semanticscholar.org/b326/6b25cb2ff34634aff48434652bacb3fede9c.pdf)
# 3. [Toward Data Science Blog](https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d)

# In[ ]:


import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import scipy as sp
import scipy.fftpack
from scipy import signal
from pykalman import KalmanFilter
from sklearn import tree
import tensorflow as tf
import tensorflow_hub as hub

import gc

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


DATA_PATH = "../input/liverpool-ion-switching"

x = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
#test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
#submission_df = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))


# ## Reduce known non-gaussian noise.
# 
# To eliminate low frequency waveform noise, the filter should block DC and VLF. In train data from 365s to 385s exists obvious 100Hz and harmonics of 100Hz (n x 100Hz).
# 
# Tho remove both of these noises I'll first try a Comb filter using it's difference equation:
# 
# x<sub>1</sub>(t) = a0 x<sub>0</sub>(t) + a1 x<sub>0</sub>(t-K)<br>
# K = fs/f0<br>
# a0 = 1<br>
# a1 = -0.99

# In[ ]:


x = x.rename(columns = {'open_channels':'y'})

fs = 10000
f0 = 100

K = np.int(fs/f0)
a1 = -0.99 
x['x0'] = 0.
x['x0'] = x['signal'] + a1 * x['signal'].shift(K)
x.loc[0:K-1,'x0'] = x.loc[0:K-1,'signal']


# ## Add features

# In[ ]:


# Energy of a signal is i^2 * time
dt = 0.0001

x.loc[:,'energy'] = x['x0']**2 * dt

# energy of our signal = energy of measurement + injection energy
# measurement energy
x['x1'] = x['energy'] - x['energy'].rolling(window=7500,min_periods=5).mean()
x.loc[0:4,'x1'] = x.loc[0:4,'energy']

# The energy_floor over 7500 periods (0.75s) is
# is it injection energy ??
x['x2']  = - x['x1'].rolling(window=7500, min_periods=5).min()
x.loc[0:4,'x2'] = 0.    #   - x.loc[0:4,'x1']

# injection current
x['x2'] = np.sqrt(x['x2']) / dt

# x2 will denote the mode of operation. Mode changes very infrequently


# ## Plot Features

# In[ ]:


examples = ['signal','y','x0','x1','x2']

fig, ax = plt.subplots(nrows=len(examples), ncols=1, figsize=(25, 3.5*len(examples)))
fig.subplots_adjust(hspace = .5)
ax = ax.ravel()
colors = plt.rcParams["axes.prop_cycle"]()

for i in range(len(examples)):
    
    c = next(colors)["color"]
    ax[i].grid()
    if examples[i] in ['x0','x2','signal']:
        ax[i].plot(x['time'], x[examples[i]],color=c, linewidth= 2)
        ax[i].set_ylabel('current (pA)', fontsize=14)
        
    if examples[i] in ['y']:
        ax[i].plot(x['time'], x[examples[i]],color=c, linewidth= .1)
        ax[i].set_ylabel('Open Channels', fontsize=14)
    if examples[i] in ['x1']:
        ax[i].plot(x['time'], x[examples[i]],color=c, linewidth= .1)
        ax[i].set_ylabel('Energy 10^-24 W-s', fontsize=14)                     
    ax[i].plot(x['time'], x[examples[i]],color=c, linewidth=.5)
    ax[i].set_title(examples[i], fontsize=24)
    ax[i].set_xlabel('Time (seconds)', fontsize=14)
    #ax[i].set_ylabel('current (pA)', fontsize=24)
    #ax[i].set_ylim(0,5)


# ## Injection current determines mode
# 
# x2 above, the injection current (IC), frees ion channels to open. When IC is lowest (1/2 pA), it is rare for more than 1 channel to be open. As IC increase, so does ion channel freedom to open. At IC = 2.5pA, we see up to 10 open channels.
# 
# Other notebooks have shown that given the mode, the state of open channels are gaussian distributions. We also make the assumption that after initial comb filtering (Drift removal) that our signal also has a gaussian distribution. Gaussians have continuous first and second derivatives (a HJB requirement). This also inspires me to try to imagine the "True" reciever's transfer function H(z) as a gaussian and as the product of two gaussians:
# 
#      gaussian(y) = gaussian(mode) * gaussian(signal)
# I may be wrong that y is a gaussian. y should take on the distribution that will produce highest entropy.

# In[ ]:


from collections import namedtuple
gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])


# In[ ]:


def update(prior, measurement):
    x, P = prior        # mean and variance of prior of x (system)
    z, R = measurement  # mean and variance of measurement (open_channels) with ion probe
    
    J = 1 - f1_score(z,y)        # residual - This is error we want to minumize
    K = P / (P + R)              # Kalman gain

    x = x + K*J      # posterior
    P = (1 - K) * P  # posterior variance
    return gaussian(x, P)

def predict(posterior, movement):
    x, P = posterior # mean and variance of posterior
    dx, Q = movement # mean and variance of movement
    x = x + dx
    P = P + Q
    return gaussian(x, P)


# In[ ]:


P = np.eye(3) * 2
R = [1,1]
dim_x = 3
for i in range(1000):
    
    for j,k in enumerate(['x0','x1','x2']):
        measurement = gaussian(x.loc[i,'y'],R)
        
        x[k], P[:,j] = update(P[:,j],x.loc[i,'y'])
        x[k], P[:,j] = predict(P[:,j],x[k])
    x.loc[i,'y_pred'] = x.mean
    


# In[ ]:


from itertools import islice

def window(seq, n=2):
    "Sliding window width n from seq.  From old itertools recipes."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
        
pairs = pd.DataFrame(window(x.loc[:,'y']), columns=['state1', 'state2'])
counts = pairs.groupby('state1')['state2'].value_counts()
alpha = 1 # Laplacian smoothing is when alpha=1
counts = counts + 1
#counts = counts.fillna(0)
P = ((counts + alpha )/(counts.sum()+alpha)).unstack()
P


# In[ ]:


# Reference https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost


# In[ ]:


train_df['work'] = train_df['signal']**2 - (train_df['signal']**2).mean()
pairs = pd.DataFrame(window(train_df.loc[:,'work']), columns=['state1', 'state2'])
means = pairs.groupby('state1')['state2'].mean()
alpha = 1 # Laplacian smoothing is when alpha=1
means = means.unstack()
means


# In[ ]:





# In[ ]:


print('Occurence Table of State Transitions')
ot = counts.unstack().fillna(0)
ot


# In[ ]:


P = (ot)/(ot.sum())
Cal = - P * np.log(P)
Cal


# In[ ]:


Caliber = Cal.sum().sum()
Caliber


# In[ ]:


# reference https://www.kaggle.com/friedchips/on-markov-chains-and-the-competition-data
def create_axes_grid(numplots_x, numplots_y, plotsize_x=6, plotsize_y=3):
    fig, axes = plt.subplots(numplots_y, numplots_x)
    fig.set_size_inches(plotsize_x * numplots_x, plotsize_y * numplots_y)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig, axes


# In[ ]:


fig, axes = create_axes_grid(1,1,10,5)
axes.set_title('Markov Transition Matrix P for all of train')
sns.heatmap(
    P,
    annot=True, fmt='.3f', cmap='Blues', cbar=False,
    ax=axes, vmin=0, vmax=0.5, linewidths=2);


# In[ ]:


eig_values, eig_vectors = np.linalg.eig(np.transpose(P))
print("Eigenvalues :", eig_values)


# Constrains on our system will define it's behavior. The system is constrained by the law of conservation of Energy. Input is signal x and output is open channels y.
# 
# Our signal x is current in pico-amperes (pA or 10^-12 Amps). The current squared x(t)^2 is proportional to instintainious Power is R * x(t)^2, where R is the resistance of the circuit. Transition Energy = TE = R*{x(t)^2-x(t-1)^2}* dt
# 
# Assuming that each Ion that transitions through the channel requires energy to make this transition, eT. The energy to make n state transitions is n * eT, where -10 <= n <= 10  so our constraint is:
# 
# TE - n * eT = 0   or
# 
# R*{x(t)^2-x(t-1)^2} * dt - n * eT = 0
# 
# Lagrangian analysis seeks to find minimums and maxima for prediction purposes. First find the Lagrangian L such that:
# 
# f(x,y) = L g(x,y)
# 

# In[ ]:


# reference: http://kitchingroup.cheme.cmu.edu/blog/2013/02/03/Using-Lagrange-multipliers-in-optimization/
def func(X):
    x = X[0]
    y = X[1]
    L = X[2] 
    return x + y + L * (x**2 + k * y)

def dfunc(X):
    dL = np.zeros(len(X))
    d = 1e-4 
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = d
        dL[i] = (func(X+dX)-func(X-dX))/(2*d);
    return dL


# In[ ]:


from scipy.optimize import fsolve

# this is the max
X1 = fsolve(dfunc, [1, 1, 0])
print(X1, func(X1))

# this is the min
X2 = fsolve(dfunc, [-1, -1, 0])
print(X2, func(X2))

