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


BASE_DIR = "../input/ml-dl-fusion-brain/full"
_, _, testData, trainData = [os.path.join(BASE_DIR, sub_dir) for sub_dir in os.listdir(BASE_DIR)]

print(trainData, '\n', testData)


# In[ ]:


classes = ['pituitary', 'meningioma', 'glioma']


# In[ ]:


class2label = {'pituitary': 0, 'meningioma': 1, 'glioma': 2}
label2class = {0: 'pituitary', 1: 'meningioma', 2: 'glioma'}

cr_indexes = list(class2label.keys())
labelling = list(class2label.values())

print(cr_indexes)
print(labelling)


# In[ ]:


distribution = {}
print("TRAIN-SET DISTRIBUTION\n")
for cat in classes:
    path = os.path.join(trainData, cat, "*")
    data = glob.glob(path)
    print(f"Number of {cat} Images: {len(data)}")
    distribution[cat] = len(data)
    
distribution = {}
print("\n\nTEST-SET DISTRIBUTION\n")
for cat in classes:
    path = os.path.join(testData, cat, "*")
    data = glob.glob(path)
    print(f"Number of {cat} Images: {len(data)}")
    distribution[cat] = len(data)


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
    path = os.path.join(trainData, cat, "*")
    data = glob.glob(path)

    for i in range(len(data)):
        img_arr = cv2.imread(data[i], 0)
        labels.append(class2label[cat])
        
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


train_data = pd.DataFrame(list(zip(labels, mean_val, median_val, std_val, skew_val, kurt_val, entropy_val, shannon_val, med_abs_dev, root_mean_sq_val, int_quart_rng, variation, trim_std, dissimilarity, energy, correlation, homogeneity, contrast, asm, dissimilarity1, energy1, correlation1, homogeneity1, contrast1, asm1, dissimilarity2, energy2, correlation2, homogeneity2, contrast2, asm2, dissimilarity3, energy3, correlation3, homogeneity3, contrast3, asm3, dissimilarity4, energy4, correlation4, homogeneity4, contrast4, asm4, dissimilarity5, energy5, correlation5, homogeneity5, contrast5, asm5, dissimilarity6, energy6, correlation6, homogeneity6, contrast6, asm6, dissimilarity7, energy7, correlation7, homogeneity7, contrast7, asm7)))
train_data.columns = ['labels', 'mean_val', 'median_val', 'std_val', 'skew_val', 'kurt_val', 'entropy_val', 'shannon_val', 'median_absolute_deviation', 'root_mean_square', 'internal_quartile_rng', 'variation', 'trimmed_std', 'dissimilarity', 'energy', 'correlation', 'homogeneity', 'contrast', 'asm', 'dissimilarity1', 'energy1', 'correlation1', 'homogeneity1', 'contrast1', 'asm1', 'dissimilarity2', 'energy2', 'correlation2', 'homogeneity2', 'contrast2', 'asm2', 'dissimilarity3', 'energy3', 'correlation3', 'homogeneity3', 'contrast3', 'asm3', 'dissimilarity4', 'energy4', 'correlation4', 'homogeneity4', 'contrast4', 'asm4', 'dissimilarity5', 'energy5', 'correlation5', 'homogeneity5', 'contrast5', 'asm5', 'dissimilarity6', 'energy6', 'correlation6', 'homogeneity6', 'contrast6', 'asm6', 'dissimilarity7', 'energy7', 'correlation7', 'homogeneity7', 'contrast7', 'asm7']
train_data.sample(10 , random_state = 42)


# In[ ]:


train_data.shape


# In[ ]:


if train_data.isnull().sum().sum() == 0:
    print('No Null Values found in dataset')
else:
    print(f'{train_data.isnull().sum().sum()} null values found in dataset')


if train_data.duplicated().sum() == 0:
    print('No Duplicate Values found in dataset')
else:
    print(f'{train_data.duplicated().sum()} Duplicate values found in dataset')


# In[ ]:


x_train = train_data.iloc[:,1:].values
y_train = train_data.iloc[:,0].values

combined = list(zip(x_train, y_train))
random.shuffle(combined)
x_train, y_train = zip(*combined)
x_train, y_train = np.array(x_train) , np.array(y_train)
print(x_train.shape , y_train.shape)


# In[ ]:


ss = StandardScaler()
x_train = ss.fit_transform(x_train)
print(x_train.shape , y_train.shape)


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
random_search.fit(x_train, y_train)
bp = random_search.best_params_
print("\nBest Params: ", bp)

# model fit
svc = SVC(C = bp['C'], kernel = 'rbf', gamma = bp['gamma'], decision_function_shape = "ovr")
svc.fit(x_train, y_train)

# prediction
preds = svc.predict(x_train)
print(classification_report(y_train, preds))
acc = accuracy_score(y_train, preds)
print(acc)


# In[ ]:


n_iter_search = 10
param_dist = {
    'n_neighbors': [1, 3, 5, 7, 11, 15, 17, 19, 23, 27]
}

print(x_train.shape , y_train.shape)
estimator = KNeighborsClassifier()
random_search = RandomizedSearchCV(estimator = estimator,
                                   param_distributions = param_dist,
                                   n_iter = n_iter_search,
                                   n_jobs = -1,
                                   refit = True,
                                   cv = 2,
                                   verbose = 2
                                  )
random_search.fit(x_train, y_train)
knn_bp = random_search.best_params_
print("\nBest Params:", knn_bp)

# model fit
knn = KNeighborsClassifier(n_neighbors = knn_bp['n_neighbors'])
knn.fit(x_train, y_train)

# prediction
preds = knn.predict(x_train)
print(classification_report(y_train, preds))
acc = accuracy_score(y_train, preds)
print(acc)


# In[ ]:


n_iter_search = 10
param_dist = {
    'max_depth': [10, 20, 30, 40],
    'min_samples_leaf': [2, 4, 5],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400]
}

print(x_train.shape, y_train.shape)

estimator = RandomForestClassifier()
random_search = RandomizedSearchCV(estimator = estimator,
                                   param_distributions = param_dist,
                                   n_iter = n_iter_search,
                                   n_jobs = -1,
                                   refit = True,
                                   cv = 2,
                                   verbose = 2
                                  )
random_search.fit(x_train, y_train)
rfc_bp = random_search.best_params_
print("\nBest Params: ", rfc_bp)

# model fit
rfc = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=rfc_bp['max_depth'], 
                             max_features='auto', min_samples_leaf=rfc_bp['min_samples_leaf'], 
                             min_samples_split=rfc_bp['min_samples_split'], n_estimators=rfc_bp['n_estimators'], 
                             n_jobs=-1, random_state=42, verbose=0)
rfc.fit(x_train, y_train)

# prediction
preds = rfc.predict(x_train)
print(classification_report(y_train, preds))

acc = accuracy_score(y_train, preds)
print(acc)


# In[ ]:


print(x_train.shape, y_train.shape)
lda = LinearDiscriminantAnalysis(solver = "eigen")
lda.fit(x_train, y_train)

# prediction
preds = lda.predict(x_train)
print(classification_report(y_train, preds))
acc = accuracy_score(y_train, preds)
print(acc)


# In[ ]:


n_iter_search = 10
param_dist = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

print(x_train.shape, y_train.shape)

estimator = XGBClassifier(learning_rate=0.02, n_estimators=600, n_jobs=-1)
random_search = RandomizedSearchCV(estimator = estimator,
                                   param_distributions = param_dist,
                                   n_iter = n_iter_search,
                                   n_jobs = -1,
                                   refit = True,
                                   cv = 2,
                                   verbose = 2
                                  )
random_search.fit(x_train, y_train)
xgb_bp = random_search.best_params_
print("\nBest params:", xgb_bp)

# model fit
# rfc = XGBClassifier(learning_rate=0.02, n_estimators=600, min_child_weight=5, max_depth=5, gamma=1, 
#                     subsample=0.6, colsample_bytree=0.6, n_jobs=-1, random_state=42)
xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, min_child_weight=xgb_bp['min_child_weight'], max_depth=xgb_bp['max_depth'], gamma=xgb_bp['gamma'], 
                    subsample=xgb_bp['subsample'], colsample_bytree=xgb_bp['colsample_bytree'], n_jobs=-1, random_state=42)
xgb.fit(x_train, y_train)

# prediction
preds = xgb.predict(x_train)
print(classification_report(y_train, preds))
acc = accuracy_score(y_train, preds)
print(acc)


# In[ ]:


print(x_train.shape, y_train.shape)

mlp = MLPClassifier(beta_1=0.99, beta_2=0.995, epsilon=1e-09, solver='lbfgs')
mlp.fit(x_train, y_train)

# prediction
preds = mlp.predict(x_train)
print(classification_report(y_train, preds))

acc = accuracy_score(y_train, preds)
print(acc)


# In[ ]:


model = load_model("../input/ml-dl-fusion-brain/full/Densenet.h5")
# model.summary()


# In[ ]:


test_mean_val = []
test_median_val = []
test_std_val = []
test_skew_val = []
test_kurt_val = []
test_entropy_val = []
test_shannon_val = []

test_med_abs_dev = []
test_root_mean_sq_val = []
test_int_quart_rng = []
test_variation = []
test_trim_std = []

test_images = []
test_labels = []

test_dissimilarity = [] 
test_contrast = [] 
test_homogeneity = [] 
test_energy = [] 
test_correlation = []
test_asm = []

test_dissimilarity1 = [] 
test_contrast1 = [] 
test_homogeneity1 = [] 
test_energy1 = [] 
test_correlation1 = []
test_asm1 = []

test_dissimilarity2 = [] 
test_contrast2 = [] 
test_homogeneity2 = [] 
test_energy2 = [] 
test_correlation2 = []
test_asm2 = []

test_dissimilarity3 = [] 
test_contrast3 = [] 
test_homogeneity3 = [] 
test_energy3 = [] 
test_correlation3 = []
test_asm3 = []

test_dissimilarity4 = [] 
test_contrast4 = [] 
test_homogeneity4 = [] 
test_energy4 = [] 
test_correlation4 = []
test_asm4 = []

test_dissimilarity5 = [] 
test_contrast5 = [] 
test_homogeneity5 = [] 
test_energy5 = [] 
test_correlation5 = []
test_asm5 = []

test_dissimilarity6 = [] 
test_contrast6 = [] 
test_homogeneity6 = [] 
test_energy6 = [] 
test_correlation6 = []
test_asm6 = []

test_dissimilarity7 = [] 
test_contrast7 = [] 
test_homogeneity7 = [] 
test_energy7 = [] 
test_correlation7 = []
test_asm7 = []

props = ['dissimilarity', 'contrast', 'homogeneity', 'energy', 'correlation']
for cat in classes:
    path = os.path.join(testData, cat, "*")
    data = glob.glob(path)

    for i in range(len(data)):
        arr = img_to_array(load_img(data[i], target_size=(256, 256)))
        test_images.append(arr)
        test_labels.append(class2label[cat])
        
        img_arr = cv2.imread(data[i], 0)
        test_mean_val.append(img_arr.mean())
        test_median_val.append(np.median(img_arr))
        test_std_val.append(img_arr.std())
        test_skew_val.append(scipy.stats.skew(img_arr.flatten()))
        test_kurt_val.append(scipy.stats.kurtosis(img_arr.flatten()))
        test_entropy_val.append(scipy.stats.entropy(img_arr.flatten()))
        test_shannon_val.append(shannon_entropy(img_arr))
        
        test_med_abs_dev.append(scipy.stats.median_abs_deviation(img_arr.flatten()))
        test_root_mean_sq_val.append(np.sqrt(np.mean(img_arr**2)))
        test_int_quart_rng.append(scipy.stats.iqr(img_arr.flatten()))
        test_variation.append(scipy.stats.variation(img_arr.flatten()))
        test_trim_std.append(scipy.stats.tstd(img_arr.flatten()))
                                                                                                                                                                                            
        test_glcm = graycomatrix(img_arr, [1], [0], 256, symmetric=True, normed=True)
        test_dissimilarity.append(graycoprops(glcm, 'dissimilarity')[0,0])
        test_contrast.append(graycoprops(glcm, 'contrast')[0,0])
        test_homogeneity.append(graycoprops(glcm, 'homogeneity')[0,0])
        test_energy.append(graycoprops(glcm, 'energy')[0,0])
        test_correlation.append(graycoprops(glcm, 'correlation')[0,0])
        test_asm.append(graycoprops(glcm, 'ASM')[0,0])

        test_glcm = graycomatrix(img_arr, [1], [np.pi/2], 256, symmetric=True, normed=True)
        test_dissimilarity1.append(graycoprops(glcm, 'dissimilarity')[0,0])
        test_contrast1.append(graycoprops(glcm, 'contrast')[0,0])
        test_homogeneity1.append(graycoprops(glcm, 'homogeneity')[0,0])
        test_energy1.append(graycoprops(glcm, 'energy')[0,0])
        test_correlation1.append(graycoprops(glcm, 'correlation')[0,0])
        test_asm1.append(graycoprops(glcm, 'ASM')[0,0])
        
        test_glcm = graycomatrix(img_arr, [2], [2*np.pi/3], 256, symmetric=True, normed=True)
        test_dissimilarity2.append(graycoprops(glcm, 'dissimilarity')[0,0])
        test_contrast2.append(graycoprops(glcm, 'contrast')[0,0])
        test_homogeneity2.append(graycoprops(glcm, 'homogeneity')[0,0])
        test_energy2.append(graycoprops(glcm, 'energy')[0,0])
        test_correlation2.append(graycoprops(glcm, 'correlation')[0,0])
        test_asm2.append(graycoprops(glcm, 'ASM')[0,0])
        
        test_glcm = graycomatrix(img_arr, [2], [3*np.pi/4], 256, symmetric=True, normed=True)
        test_dissimilarity3.append(graycoprops(glcm, 'dissimilarity')[0,0])
        test_contrast3.append(graycoprops(glcm, 'contrast')[0,0])
        test_homogeneity3.append(graycoprops(glcm, 'homogeneity')[0,0])
        test_energy3.append(graycoprops(glcm, 'energy')[0,0])
        test_correlation3.append(graycoprops(glcm, 'correlation')[0,0])
        test_asm3.append(graycoprops(glcm, 'ASM')[0,0])
        
        test_glcm = graycomatrix(img_arr, [4], [3*np.pi/4], 256, symmetric=True, normed=True)
        test_dissimilarity4.append(graycoprops(glcm, 'dissimilarity')[0,0])
        test_contrast4.append(graycoprops(glcm, 'contrast')[0,0])
        test_homogeneity4.append(graycoprops(glcm, 'homogeneity')[0,0])
        test_energy4.append(graycoprops(glcm, 'energy')[0,0])
        test_correlation4.append(graycoprops(glcm, 'correlation')[0,0])
        test_asm4.append(graycoprops(glcm, 'ASM')[0,0])
        
        test_glcm = graycomatrix(img_arr, [4], [4*np.pi/5], 256, symmetric=True, normed=True)
        test_dissimilarity5.append(graycoprops(glcm, 'dissimilarity')[0,0])
        test_contrast5.append(graycoprops(glcm, 'contrast')[0,0])
        test_homogeneity5.append(graycoprops(glcm, 'homogeneity')[0,0])
        test_energy5.append(graycoprops(glcm, 'energy')[0,0])
        test_correlation5.append(graycoprops(glcm, 'correlation')[0,0])
        test_asm5.append(graycoprops(glcm, 'ASM')[0,0])
        
        test_glcm = graycomatrix(img_arr, [6], [np.pi/4], 256, symmetric=True, normed=True)
        test_dissimilarity6.append(graycoprops(glcm, 'dissimilarity')[0,0])
        test_contrast6.append(graycoprops(glcm, 'contrast')[0,0])
        test_homogeneity6.append(graycoprops(glcm, 'homogeneity')[0,0])
        test_energy6.append(graycoprops(glcm, 'energy')[0,0])
        test_correlation6.append(graycoprops(glcm, 'correlation')[0,0])
        test_asm6.append(graycoprops(glcm, 'ASM')[0,0])
        
        test_glcm = graycomatrix(img_arr, [6], [5*np.pi/6], 256, symmetric=True, normed=True)
        test_dissimilarity7.append(graycoprops(glcm, 'dissimilarity')[0,0])
        test_contrast7.append(graycoprops(glcm, 'contrast')[0,0])
        test_homogeneity7.append(graycoprops(glcm, 'homogeneity')[0,0])
        test_energy7.append(graycoprops(glcm, 'energy')[0,0])
        test_correlation7.append(graycoprops(glcm, 'correlation')[0,0])
        test_asm7.append(graycoprops(glcm, 'ASM')[0,0])


# In[ ]:


test_data = pd.DataFrame(list(zip(test_labels, test_mean_val, test_median_val, test_std_val, test_skew_val, test_kurt_val, 
                                  test_entropy_val, test_shannon_val, test_med_abs_dev, test_root_mean_sq_val, test_int_quart_rng, 
                                  test_variation, test_trim_std, test_dissimilarity, test_energy, test_correlation, test_homogeneity, 
                                  test_contrast, test_asm, test_dissimilarity1, test_energy1, test_correlation1, test_homogeneity1, 
                                  test_contrast1, test_asm1, test_dissimilarity2, test_energy2, test_correlation2, test_homogeneity2, 
                                  test_contrast2, test_asm2, test_dissimilarity3, test_energy3, test_correlation3, 
                                  test_homogeneity3, test_contrast3, test_asm3, test_dissimilarity4, test_energy4, 
                                  test_correlation4, test_homogeneity4, test_contrast4, test_asm4, test_dissimilarity5, 
                                  test_energy5, test_correlation5, test_homogeneity5, test_contrast5, test_asm5, 
                                  test_dissimilarity6, test_energy6, test_correlation6, test_homogeneity6, test_contrast6, 
                                  test_asm6, test_dissimilarity7, test_energy7, test_correlation7, test_homogeneity7, 
                                  test_contrast7, test_asm7)))
test_data.columns = ['tlabels', 'tmean_val', 'tmedian_val', 'tstd_val', 'tskew_val', 'tkurt_val', 'tentropy_val', 'tshannon_val', 
                'tmedian_absolute_deviation', 'troot_mean_square', 'tinternal_quartile_rng', 'tvariation', 'ttrimmed_std', 
                'tdissimilarity', 'tenergy', 'tcorrelation', 'thomogeneity', 'tcontrast', 'tasm', 'tdissimilarity1', 'tenergy1', 
                'tcorrelation1', 'thomogeneity1', 'tcontrast1', 'tasm1', 'tdissimilarity2', 'tenergy2', 'tcorrelation2', 
                'thomogeneity2', 'tcontrast2', 'tasm2', 'tdissimilarity3', 'tenergy3', 'tcorrelation3', 'thomogeneity3', 
                'tcontrast3', 'tasm3', 'tdissimilarity4', 'tenergy4', 'tcorrelation4', 'thomogeneity4', 'tcontrast4', 'tasm4', 
                'tdissimilarity5', 'tenergy5', 'tcorrelation5', 'thomogeneity5', 'tcontrast5', 'tasm5', 'tdissimilarity6', 
                'tenergy6', 'tcorrelation6', 'thomogeneity6', 'tcontrast6', 'tasm6', 'tdissimilarity7', 'tenergy7', 
                'tcorrelation7', 'thomogeneity7', 'tcontrast7', 'tasm7']
test_data.sample(10 , random_state = 42)


# In[ ]:


test_data.shape


# In[ ]:


if test_data.isnull().sum().sum() == 0:
    print('No Null Values found in dataset')
else:
    print(f'{test_data.isnull().sum().sum()} null values found in dataset')

if test_data.duplicated().sum() == 0:
    print('No Duplicate Values found in dataset')
else:
    print(f'{test_data.duplicated().sum()} Duplicate values found in dataset')


# In[ ]:


x_test = test_data.iloc[:,1:].values
y_test = test_data.iloc[:,0].values

combined = list(zip(x_test, y_test))
random.shuffle(combined)
x_test, y_test = zip(*combined)
x_test, y_test = np.array(x_test) , np.array(y_test)

x_test = ss.fit_transform(x_test)
print(x_test.shape, y_test.shape)


# In[ ]:


# prediction
preds = svc.predict(x_test)
print(classification_report(y_test, preds))

acc = accuracy_score(y_test, preds)
print(acc)


# In[ ]:


predictions = model.predict(np.array(test_images))
predictions = np.argmax(predictions, axis=1)

(predictions == y_test).sum()/y_test.shape[0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


train_datagen  = ImageDataGenerator(rescale = 1./255)

train_dataset  = train_datagen.flow_from_directory(directory = trainData,
                                               target_size = (256, 256),
                                               color_mode = "grayscale",
                                               class_mode = "categorical",
                                               batch_size = 1,
                                               shuffle = False) 


test_datagen  = ImageDataGenerator(rescale = 1./255)

test_dataset  = test_datagen.flow_from_directory(directory = testData,
                                               target_size = (256, 256),
                                               color_mode = "grayscale",
                                               class_mode = "categorical",
                                               batch_size = 1,
                                               shuffle = False)


# In[ ]:


start = time.time()

dl_images, ml_images = [], []
targets = []

for idx in range(len(train_dataset)):
    img, target = train_dataset[idx][0][0], train_dataset[idx][1].argmax(axis=1)[0]
    ml_images.append(img)
    dl_images.append(np.concatenate((img, img, img), axis=2))
    targets.append(target)
    
print(time.time()-start)


# In[ ]:


plt.imshow(dl_images[-1])
plt.show()

plt.imshow(ml_images[-1], cmap = 'gray')
plt.show()


# In[ ]:


start = time.time()

combined = list(zip(ml_images, targets))
random.shuffle(combined)
ml_images, targets = zip(*combined)
ml_images, targets = np.array(ml_images) , np.array(targets)
print(ml_images.shape , targets.shape)

print(time.time()-start)


# In[ ]:


start = time.time()

dl_images_test, ml_images_test = [], []
targets_test = []

for idx in range(len(test_dataset)):
    img, target = test_dataset[idx][0][0], test_dataset[idx][1].argmax(axis=1)[0]
    ml_images_test.append(img)
    dl_images_test.append(np.concatenate((img, img, img), axis=2))
    targets_test.append(target)
    
print(time.time()-start)


# In[ ]:


plt.imshow(dl_images_test[-1])
plt.show()

plt.imshow(ml_images_test[-1], cmap = 'gray')
plt.show()


# In[ ]:


# start = time.time()

# combined = list(zip(ml_images_test, targets_test))
# random.shuffle(combined)
# ml_images_test, targets_test = zip(*combined)
ml_images_test, targets_test = np.array(ml_images_test) , np.array(targets_test)
print(ml_images_test.shape , targets_test.shape)

# print(time.time()-start)


# In[ ]:


model = load_model("../input/ml-dl-fusion-brain/full/Densenet.h5")
model.summary()


# In[ ]:


dl_preds = []

for i in range(len(dl_images_test)):
    dl_preds.append(model.predict(dl_images_test[i].reshape((1, *dl_images_test[i].shape))).argmax(axis=1)[0])


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

for i in range(len(ml_images)):
    img_arr = np.squeeze(ml_images[i]).astype('int')

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


data = pd.DataFrame(list(zip(mean_val, median_val, std_val, skew_val, kurt_val, entropy_val, shannon_val, med_abs_dev, root_mean_sq_val, int_quart_rng, variation, trim_std, dissimilarity, energy, correlation, homogeneity, contrast, asm, dissimilarity1, energy1, correlation1, homogeneity1, contrast1, asm1, dissimilarity2, energy2, correlation2, homogeneity2, contrast2, asm2, dissimilarity3, energy3, correlation3, homogeneity3, contrast3, asm3, dissimilarity4, energy4, correlation4, homogeneity4, contrast4, asm4, dissimilarity5, energy5, correlation5, homogeneity5, contrast5, asm5, dissimilarity6, energy6, correlation6, homogeneity6, contrast6, asm6, dissimilarity7, energy7, correlation7, homogeneity7, contrast7, asm7)))
data.columns = ['mean_val', 'median_val', 'std_val', 'skew_val', 'kurt_val', 'entropy_val', 'shannon_val', 'median_absolute_deviation', 'root_mean_square', 'internal_quartile_rng', 'variation', 'trimmed_std', 'dissimilarity', 'energy', 'correlation', 'homogeneity', 'contrast', 'asm', 'dissimilarity1', 'energy1', 'correlation1', 'homogeneity1', 'contrast1', 'asm1', 'dissimilarity2', 'energy2', 'correlation2', 'homogeneity2', 'contrast2', 'asm2', 'dissimilarity3', 'energy3', 'correlation3', 'homogeneity3', 'contrast3', 'asm3', 'dissimilarity4', 'energy4', 'correlation4', 'homogeneity4', 'contrast4', 'asm4', 'dissimilarity5', 'energy5', 'correlation5', 'homogeneity5', 'contrast5', 'asm5', 'dissimilarity6', 'energy6', 'correlation6', 'homogeneity6', 'contrast6', 'asm6', 'dissimilarity7', 'energy7', 'correlation7', 'homogeneity7', 'contrast7', 'asm7']
data.sample(10 , random_state = 42)


# In[ ]:


data.shape


# In[ ]:


if data.isnull().sum().sum() == 0:
    print('No Null Values found in dataset')
else:
    print(f'{data.isnull().sum().sum()} null values found in dataset')

# checking for any duplicate values
if data.duplicated().sum() == 0:
    print('No Duplicate Values found in dataset')
else:
    print(f'{data.duplicated().sum()} Duplicate values found in dataset')


# In[ ]:




