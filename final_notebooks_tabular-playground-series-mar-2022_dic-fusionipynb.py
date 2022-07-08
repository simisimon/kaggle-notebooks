#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import shutil
import glob
import json
import pickle
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import cv2
import skimage.io as io
from skimage.transform import resize
import scipy
from skimage.filters import sobel
from skimage.feature import graycomatrix , graycoprops
from skimage.measure import shannon_entropy
import lightgbm as lgbm
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score,
                             auc, roc_curve, roc_auc_score, cohen_kappa_score, plot_confusion_matrix, plot_roc_curve,
                             plot_precision_recall_curve, precision_recall_fscore_support, precision_recall_curve)
from sklearn.utils import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, LinearClassifierMixin
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import lightgbm as lgbm

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator, save_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential, Model, load_model, model_from_json
from tensorflow.keras.layers import (Input, InputLayer, Conv2D, MaxPooling2D, Dense, Concatenate, BatchNormalization, Dropout, Activation, GlobalAveragePooling2D, InputSpec, Flatten, Concatenate)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.metrics import Accuracy, AUC, Precision, Recall
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


get_ipython().system('pip install mrmr_selection')
from mrmr import mrmr_classif


# In[ ]:


BASE_DIR = "../input/dic-ensemble/ML-Classify"

classes = ['pituitary', 'meningioma', 'glioma']
print(classes)


# In[ ]:


class2label = {}

for i in range(len(classes)):
    class2label[classes[i]] = i
    
label2class = {v:k for k, v in class2label.items()}

print(class2label)
print("--"*25)
print(label2class)


# In[ ]:


cr_indexes = list(class2label.keys())
labelling = list(class2label.values())

print(cr_indexes)
print(labelling)


# In[ ]:


model = load_model("../input/dic-ensemble/ML-Classify/Densenet.h5")


# In[ ]:


mean_val = []
median_val = []
std_val = []
skew_val = []
kurt_val = []
entropy_val = []
shannon_val = []

med_abs_dev = []
root_mean_sq_val = []
int_quart_rng = []
variation = []
trim_std = []

imagesData = []
labels = []

dissimilarity = [] 
contrast = [] 
homogeneity = [] 
energy = [] 
correlation = []
asm = []

dissimilarity1 = [] 
contrast1 = [] 
homogeneity1 = [] 
energy1 = [] 
correlation1 = []
asm1 = []

dissimilarity2 = [] 
contrast2 = [] 
homogeneity2 = [] 
energy2 = [] 
correlation2 = []
asm2 = []

dissimilarity3 = [] 
contrast3 = [] 
homogeneity3 = [] 
energy3 = [] 
correlation3 = []
asm3 = []

dissimilarity4 = [] 
contrast4 = [] 
homogeneity4 = [] 
energy4 = [] 
correlation4 = []
asm4 = []

dissimilarity5 = [] 
contrast5 = [] 
homogeneity5 = [] 
energy5 = [] 
correlation5 = []
asm5 = []

dissimilarity6 = [] 
contrast6 = [] 
homogeneity6 = [] 
energy6 = [] 
correlation6 = []
asm6 = []

dissimilarity7 = [] 
contrast7 = [] 
homogeneity7 = [] 
energy7 = [] 
correlation7 = []
asm7 = []

props = ['dissimilarity', 'contrast', 'homogeneity', 'energy', 'correlation']
for cat in classes:
    path = os.path.join(BASE_DIR, cat, "*")
    data = glob.glob(path)

    for i in range(len(data)):
        img_arr = cv2.imread(data[i], 0)
        labels.append(class2label[cat])
        
        arr = img_to_array(load_img(data[i], target_size=(256, 256)))/255.0
        imagesData.append(arr)
        
        mean_val.append(img_arr.mean())
        median_val.append(np.median(img_arr))
        std_val.append(img_arr.std())
        skew_val.append(scipy.stats.skew(img_arr.flatten()))
        kurt_val.append(scipy.stats.kurtosis(img_arr.flatten()))
        entropy_val.append(scipy.stats.entropy(img_arr.flatten()))
        shannon_val.append(shannon_entropy(img_arr))
        
        med_abs_dev.append(scipy.stats.median_abs_deviation(img_arr.flatten()))
        root_mean_sq_val.append(np.sqrt(np.mean(img_arr**2)))
        int_quart_rng.append(scipy.stats.iqr(img_arr.flatten()))
        variation.append(scipy.stats.variation(img_arr.flatten()))
        trim_std.append(scipy.stats.tstd(img_arr.flatten()))
                                                                                                                                                                                            
        glcm = graycomatrix(img_arr, [1], [0], 256, symmetric=True, normed=True)
        dissimilarity.append(graycoprops(glcm, 'dissimilarity')[0,0])
        contrast.append(graycoprops(glcm, 'contrast')[0,0])
        homogeneity.append(graycoprops(glcm, 'homogeneity')[0,0])
        energy.append(graycoprops(glcm, 'energy')[0,0])
        correlation.append(graycoprops(glcm, 'correlation')[0,0])
        asm.append(graycoprops(glcm, 'ASM')[0,0])

        glcm = graycomatrix(img_arr, [1], [np.pi/2], 256, symmetric=True, normed=True)
        dissimilarity1.append(graycoprops(glcm, 'dissimilarity')[0,0])
        contrast1.append(graycoprops(glcm, 'contrast')[0,0])
        homogeneity1.append(graycoprops(glcm, 'homogeneity')[0,0])
        energy1.append(graycoprops(glcm, 'energy')[0,0])
        correlation1.append(graycoprops(glcm, 'correlation')[0,0])
        asm1.append(graycoprops(glcm, 'ASM')[0,0])
        
        glcm = graycomatrix(img_arr, [2], [2*np.pi/3], 256, symmetric=True, normed=True)
        dissimilarity2.append(graycoprops(glcm, 'dissimilarity')[0,0])
        contrast2.append(graycoprops(glcm, 'contrast')[0,0])
        homogeneity2.append(graycoprops(glcm, 'homogeneity')[0,0])
        energy2.append(graycoprops(glcm, 'energy')[0,0])
        correlation2.append(graycoprops(glcm, 'correlation')[0,0])
        asm2.append(graycoprops(glcm, 'ASM')[0,0])
        
        glcm = graycomatrix(img_arr, [2], [3*np.pi/4], 256, symmetric=True, normed=True)
        dissimilarity3.append(graycoprops(glcm, 'dissimilarity')[0,0])
        contrast3.append(graycoprops(glcm, 'contrast')[0,0])
        homogeneity3.append(graycoprops(glcm, 'homogeneity')[0,0])
        energy3.append(graycoprops(glcm, 'energy')[0,0])
        correlation3.append(graycoprops(glcm, 'correlation')[0,0])
        asm3.append(graycoprops(glcm, 'ASM')[0,0])
        
        glcm = graycomatrix(img_arr, [4], [3*np.pi/4], 256, symmetric=True, normed=True)
        dissimilarity4.append(graycoprops(glcm, 'dissimilarity')[0,0])
        contrast4.append(graycoprops(glcm, 'contrast')[0,0])
        homogeneity4.append(graycoprops(glcm, 'homogeneity')[0,0])
        energy4.append(graycoprops(glcm, 'energy')[0,0])
        correlation4.append(graycoprops(glcm, 'correlation')[0,0])
        asm4.append(graycoprops(glcm, 'ASM')[0,0])
        
        glcm = graycomatrix(img_arr, [4], [4*np.pi/5], 256, symmetric=True, normed=True)
        dissimilarity5.append(graycoprops(glcm, 'dissimilarity')[0,0])
        contrast5.append(graycoprops(glcm, 'contrast')[0,0])
        homogeneity5.append(graycoprops(glcm, 'homogeneity')[0,0])
        energy5.append(graycoprops(glcm, 'energy')[0,0])
        correlation5.append(graycoprops(glcm, 'correlation')[0,0])
        asm5.append(graycoprops(glcm, 'ASM')[0,0])
        
        glcm = graycomatrix(img_arr, [6], [np.pi/4], 256, symmetric=True, normed=True)
        dissimilarity6.append(graycoprops(glcm, 'dissimilarity')[0,0])
        contrast6.append(graycoprops(glcm, 'contrast')[0,0])
        homogeneity6.append(graycoprops(glcm, 'homogeneity')[0,0])
        energy6.append(graycoprops(glcm, 'energy')[0,0])
        correlation6.append(graycoprops(glcm, 'correlation')[0,0])
        asm6.append(graycoprops(glcm, 'ASM')[0,0])
        
        glcm = graycomatrix(img_arr, [6], [5*np.pi/6], 256, symmetric=True, normed=True)
        dissimilarity7.append(graycoprops(glcm, 'dissimilarity')[0,0])
        contrast7.append(graycoprops(glcm, 'contrast')[0,0])
        homogeneity7.append(graycoprops(glcm, 'homogeneity')[0,0])
        energy7.append(graycoprops(glcm, 'energy')[0,0])
        correlation7.append(graycoprops(glcm, 'correlation')[0,0])
        asm7.append(graycoprops(glcm, 'ASM')[0,0])


# In[ ]:


df = pd.DataFrame(list(zip(labels, imagesData, mean_val, median_val, std_val, skew_val, kurt_val, entropy_val, shannon_val, med_abs_dev, root_mean_sq_val, int_quart_rng, variation, trim_std, dissimilarity, energy, correlation, homogeneity, contrast, asm, dissimilarity1, energy1, correlation1, homogeneity1, contrast1, asm1, dissimilarity2, energy2, correlation2, homogeneity2, contrast2, asm2, dissimilarity3, energy3, correlation3, homogeneity3, contrast3, asm3, dissimilarity4, energy4, correlation4, homogeneity4, contrast4, asm4, dissimilarity5, energy5, correlation5, homogeneity5, contrast5, asm5, dissimilarity6, energy6, correlation6, homogeneity6, contrast6, asm6, dissimilarity7, energy7, correlation7, homogeneity7, contrast7, asm7)))
df.columns = ['labels', 'imagesData', 'mean_val', 'median_val', 'std_val', 'skew_val', 'kurt_val', 'entropy_val', 'shannon_val', 'median_absolute_deviation', 'root_mean_square', 'internal_quartile_rng', 'variation', 'trimmed_std', 'dissimilarity', 'energy', 'correlation', 'homogeneity', 'contrast', 'asm', 'dissimilarity1', 'energy1', 'correlation1', 'homogeneity1', 'contrast1', 'asm1', 'dissimilarity2', 'energy2', 'correlation2', 'homogeneity2', 'contrast2', 'asm2', 'dissimilarity3', 'energy3', 'correlation3', 'homogeneity3', 'contrast3', 'asm3', 'dissimilarity4', 'energy4', 'correlation4', 'homogeneity4', 'contrast4', 'asm4', 'dissimilarity5', 'energy5', 'correlation5', 'homogeneity5', 'contrast5', 'asm5', 'dissimilarity6', 'energy6', 'correlation6', 'homogeneity6', 'contrast6', 'asm6', 'dissimilarity7', 'energy7', 'correlation7', 'homogeneity7', 'contrast7', 'asm7']
df.sample(10 , random_state=42)


# In[ ]:


df.shape


# In[ ]:


if df.isnull().sum().sum() == 0:
    print('No Null Values found in dataset')
else:
    print(f'{df.isnull().sum().sum()} null values found in dataset')


# In[ ]:


x = df.iloc[:,1:].values
y = df.iloc[:,0].values
print(x.shape , y.shape)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
print(x_train.shape, x_test.shape , y_train.shape, y_test.shape)


# In[ ]:


print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))


# In[ ]:


x_trainML , x_testML = x_train[:,1:] , x_test[:,1:]
x_testDL = x_test[:,0]

print(x_trainML.shape , x_testML.shape)
print(x_testDL.shape)


# In[ ]:


dl_xtest = []
for idx in range(len(x_testDL)):
    img_array = x_testDL[idx]
    dl_xtest.append(img_array)


# In[ ]:


dl_xtest = np.array(dl_xtest)
print(dl_xtest.shape)


# In[ ]:


preds = model.predict(dl_xtest).argmax(axis=1)
accuracy_score(y_test, preds)


# In[ ]:


ss = StandardScaler()
x_trainML = ss.fit_transform(x_trainML)
x_testML = ss.transform(x_testML)


# In[ ]:


n_iter_search = 20
param_dist = {
    'C': [1, 10, 100, 1000, 10000],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
}

# random search
estimator = SVC(kernel='rbf', decision_function_shape='ovr')
random_search = RandomizedSearchCV(estimator = estimator,

                                   param_distributions = param_dist,
                                   n_iter = n_iter_search,
                                   n_jobs = -1,
                                   refit = True,
                                   cv = 3,
                                   verbose = 2
                                  )
random_search.fit(x_trainML, y_train)
bp = random_search.best_params_

# model fit
svc = SVC(C = bp['C'], kernel = 'rbf', gamma = bp['gamma'], decision_function_shape = "ovr")
svc.fit(x_trainML, y_train)

# prediction
preds = svc.predict(x_testML)
print(classification_report(y_test, preds))

acc = accuracy_score(y_test, preds)
print(acc)


# In[ ]:


# constants
n_iter_search = 15
param_dist = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 19, 21, 23]
}

estimator = KNeighborsClassifier()
random_search = RandomizedSearchCV(estimator = estimator,
                                   param_distributions = param_dist,
                                   n_iter = n_iter_search,
                                   n_jobs = -1,
                                   refit = True,
                                   cv = 2,
                                   verbose = 2
                                  )
random_search.fit(x_trainML, y_train)
bp = random_search.best_params_
print("\nBest Params: ", bp)

# model fit
rfc = KNeighborsClassifier(n_neighbors = bp['n_neighbors'])
rfc.fit(x_trainML, y_train)

# prediction
preds = rfc.predict(x_testML)
print(classification_report(y_test, preds))

acc = accuracy_score(y_test, preds)
print(acc)


# In[ ]:


n_iter_search = 10
param_dist = {
    'max_depth': [10, 20, 30, 40],
    'min_samples_leaf': [2, 4, 5],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400]
}

estimator = RandomForestClassifier()
random_search = RandomizedSearchCV(estimator = estimator,
                                   param_distributions = param_dist,
                                   n_iter = n_iter_search,
                                   n_jobs = -1,
                                   refit = True,
                                   cv = 2,
                                   verbose = 2
                                  )
random_search.fit(x_trainML, y_train)
bp = random_search.best_params_
print("\nBest Params: ", bp)

# model fit
rfc = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=bp['max_depth'], max_features='auto', min_samples_leaf=bp['min_samples_leaf'],
                       min_samples_split=bp['min_samples_split'], n_estimators=bp['n_estimators'], n_jobs=-1, random_state=42, verbose=0)
rfc.fit(x_trainML, y_train)

# prediction
preds = rfc.predict(x_testML)
print(classification_report(y_test, preds))

acc = accuracy_score(y_test, preds)
print(acc)


# In[ ]:




