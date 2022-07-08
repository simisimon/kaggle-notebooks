#!/usr/bin/env python
# coding: utf-8

# ![https://2.imimg.com/data2/EI/YM/MY-4632297/sun-dried-raisins-250x250.jpg](https://2.imimg.com/data2/EI/YM/MY-4632297/sun-dried-raisins-250x250.jpg)
# 
# # **Raisin Dataset**
# 
# * **Machine learning algorithm written in Python to distinguish between two raisin types (Kecimen, Besni).**
# * **Data set containts 900 rows, 450 rows of each class.**
# * **Classification problem solved by Support-vector machine (SVM) algorithm.**
# 
# *Origin of dataset: https://www.kaggle.com/datasets/muratkokludataset/raisin-dataset*

# **All required libraries. Numpy for vectors, pandas for data frames, matplotlib for plots. Sklearn for machine learning.**
# 
# **Internet enabled in kaggle notebook required to download openpyxl for pandas.**

# In[ ]:


# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SVM libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report

# Install to use read_excel
get_ipython().system('pip install openpyxl')


# **Reading and displaying the data set.**
# 
# **All 7 raisin features are numerical. Class vector has to be changed to numbers (0 and 1).**

# In[ ]:


# Read data set
data = pd.read_excel('../input/raisin-dataset/Raisin_Dataset/Raisin_Dataset.xlsx', sheet_name = 'Raisin_Dataset')
data


# **Structure of data set displayed. Almost all columns have correct type.**

# In[ ]:


# Data frame structure
data.info()


# **Converting class values to 0 and 1 and making sure every column is numerical.**

# In[ ]:


# Convert columns to numeric
data['Class'] = data['Class'].str.replace('Kecimen', '0')
data['Class'] = data['Class'].str.replace('Besni', '1')

cols = data.columns
data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')

# Data frame structure
data.info()


# **Need to check if values are distributed evenly.**

# In[ ]:


# Classes are distrubuted evenly
data['Class'].value_counts()


# **Preparing data for SVM algorithm.**
# 
# **Spliting class column from others.**

# In[ ]:


# Data for SVM
X = data
X = X.drop(['Class'], axis = 1)
Y = data.Class # class


# # **Polynomial kernel**
# **Parameters for polynomial kernel SVM model. Polynomial of multiple degrees is tested.**

# In[ ]:


# Parameters for classificator - polynomial kernel
degree = [1,2,3,4,5,6,7,8,9]
degree = np.array(degree)

# Tables to save accuracy - polynomial kernel
Accuracy_CV_poly = np.zeros((10,1))
Accuracy_poly = np.zeros(len(degree))


# **SVM model training with polynomial kernel.**
# 
# **Splitting data into 10% test size and rest for the training.**
# 
# **Data is standarized after splitting.**
# 
# **After model creation, it's fitted and tested.**
# 
# **Later the accuracy is calculated.**

# In[ ]:


# SVM - polynomial kernel
for i in range(0, len(degree)): # degree
    for k in range(1, 10): # Crossvalidation
        # Split data into test and train sets
        X_train_poly, X_test_poly, Y_train_poly, Y_test_poly = train_test_split(X, Y, test_size = 0.10) # 10 times cs so test_size is 10% of data set 
            
        # Standarization
        sc = StandardScaler()
        sc.fit(X_train_poly)
        X_train_poly = sc.transform(X_train_poly)
        X_test_poly = sc.transform(X_test_poly)
        X_train_poly = pd.DataFrame(X_train_poly)
        X_test_poly = pd.DataFrame(X_test_poly)
            
        # Model
        svclassifier = SVC(kernel='poly', degree = degree[i])
        svclassifier.fit(X_train_poly, Y_train_poly)
        
        y_pred_poly = svclassifier.predict(X_test_poly)
            
        # Accuracy - how many values from y_pred are equal to Y_test
        Accuracy_CV_poly[k] = sum(y_pred_poly == Y_test_poly)/len(Y_test_poly)

    Accuracy_poly[i] = np.mean(Accuracy_CV_poly) # rows - gamma, columns - C
    
Accuracy_poly # display accuracy table


# **Plotting accuracy.**
# 
# **Change of degree is axis X.**

# In[ ]:


# polynomial kernel accuracy plot
# axis X - degree
plt.plot(Accuracy_poly)
plt.xlabel('degree')
plt.ylabel('Accuracy')
#plt.xticks(np.arange(len(degree)), degree) # correct axis X ticks


# **As we can see, the model build with polynomial kernel doesn't have the best accuracy.**

# # **RBF kernel**
# **Creating parameters for classificator.**
# 
# **Multiple gamma and C values to check which one should be used in model for best accuracy.**
# 
# **A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly. gamma defines how much influence a single training example has. The larger gamma is, the closer other examples must be to be affected.**
# 
# **Creating accuracy tables to save their values. Accuracy_CV is 10x1 because of 10-fold cross-validation.**

# In[ ]:


# Parameters for classificator - rbf kernel
gamma = [0.0005, 0.005, 0.01, 0.05, 0.2, 0.8, 1.5, 2.5, 5, 10, 20, 50, 100]
gamma = np.array(gamma)
C = [1, 10, 100, 1000, 10000, 100000]
C = np.array(C)


# **SVM model training with rbf kernel.**
# 
# **Splitting data into 10% test size and rest for the training.**
# 
# **Data is standarized after splitting.**
# 
# **After model creation, it's fitted and tested.**
# 
# **Later the accuracy is calculated.**

# In[ ]:


# Tables to save accuracy - rbf kernel
Accuracy_CV = np.zeros((10,1))
Accuracy = np.zeros((len(gamma), len(C)))

# SVM - rbf kernel
for i in range(0, len(C)): # C
    for j in range(0, len(gamma)): # gamma
        for k in range(1, 10): # Crossvalidation
            # Split data into test and train sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10) # 10 times cs so test_size is 10% of data set 
            
            # Standarization
            sc = StandardScaler()
            sc.fit(X_train)
            X_train = sc.transform(X_train)
            X_test = sc.transform(X_test)
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            
            # Model
            svclassifier = SVC(kernel = 'rbf', C = C[i], gamma = gamma[j])
            
            svclassifier.fit(X_train, Y_train)
            y_pred = svclassifier.predict(X_test)
            
            # Accuracy - how many values from y_pred are equal to Y_test
            Accuracy_CV[k] = sum(y_pred == Y_test)/len(Y_test)

        Accuracy[j,i] = np.mean(Accuracy_CV) # rows - gamma, columns - C
        
Accuracy # display accuracy table


# **Plotting accuracy.**
# 
# **Each line is model with different C value. Change of gamma is axis X.**

# In[ ]:


# rbf kernel accuracy plot
# axis X - gamma, different lines - C     
for p in range(0, len(C)): # number of lines = number of C values 
    plt.plot(Accuracy[:,p], label = C[p]) # every line plotted separately in order to have a name
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.xticks(np.arange(len(gamma)), gamma) # correct axis X ticks
plt.legend(title = 'C')


# **Accuracy falls fown rapidly after gamma = 5.0. Bigger C values doesn't mean better accuracy as it seems**

# **RBF kernel will be tested once again. This time data will come from previously splitted data set (X_train, Y_train) and will be splitted into new sets. The model will be trained on newly created train samples (X_train_rbf, Y_train_rbf) but previously created X_test will be predicted.**

# In[ ]:


Accuracy_CV_rbf = np.zeros((10,1))
Accuracy_rbf = np.zeros((len(gamma), len(C)))

# SVM - rbf kernel - testing on test
for i in range(0, len(C)): # C
    for j in range(0, len(gamma)): # gamma
        for k in range(1, 10): # Crossvalidation
            # Split data into test and train sets
            X_train_rbf, X_test_rbf, Y_train_rbf, Y_test_rbf = train_test_split(X_train, Y_train, test_size = 0.10) # 10 times cs so test_size is 10% of data set 
            
            # Standarization
            sc = StandardScaler()
            sc.fit(X_train_rbf)
            X_train_rbf = sc.transform(X_train_rbf)
            X_test_rbf = sc.transform(X_test_rbf)
            X_train_rbf = pd.DataFrame(X_train_rbf)
            X_test_rbf = pd.DataFrame(X_test_rbf)
            
            # Model
            svclassifier = SVC(kernel = 'rbf', C = C[i], gamma = gamma[j])
            
            svclassifier.fit(X_train_rbf, Y_train_rbf)
            y_pred_rbf = svclassifier.predict(X_test)
            
            # Accuracy - how many values from y_pred are equal to Y_test
            Accuracy_CV_rbf[k] = sum(y_pred_rbf == Y_test)/len(Y_test)

        Accuracy_rbf[j,i] = np.mean(Accuracy_CV_rbf) # rows - gamma, columns - C
        
#Accuracy # display accuracy table


# In[ ]:


# rbf kernel accuracy plot
# axis X - gamma, different lines - C     
for p in range(0, len(C)): # number of lines = number of C values 
    plt.plot(Accuracy_rbf[:,p], label = C[p]) # every line plotted separately in order to have a name
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.xticks(np.arange(len(gamma)), gamma) # correct axis X ticks
plt.legend(title = 'C')


# **gamma = 0.05 and C = 100 looks like the best option - building a model with those parameters**

# In[ ]:


# getting data from original X and Y
X_train_rbf_o, X_test_rbf_o, Y_train_rbf_o, Y_test_rbf_o = train_test_split(X, Y, test_size = 0.10)
            
# Standarization
sc = StandardScaler()
sc.fit(X_train_rbf_o)
X_train_rbf_o = sc.transform(X_train_rbf_o)
X_test_rbf_o = sc.transform(X_test_rbf_o)
X_train_rbf_o = pd.DataFrame(X_train_rbf_o)
X_test_rbf_o = pd.DataFrame(X_test_rbf_o)
            
# Model
svclassifier = SVC(kernel = 'rbf', C = 100, gamma = 0.05)
            
svclassifier.fit(X_train_rbf_o, Y_train_rbf_o)
y_pred_rbf_o = svclassifier.predict(X_test_rbf_o)
            
# Accuracy - how many values from y_pred are equal to Y_test
Accuracy_CV_rbf_o = sum(y_pred_rbf_o == Y_test_rbf_o)/len(Y_test_rbf_o)
Accuracy_CV_rbf_o


# In[ ]:


print(classification_report(Y_test_rbf_o, y_pred_rbf_o, target_names=['Besni', 'Kecimen']))


# # **Accuracy looks okay - 88%**
