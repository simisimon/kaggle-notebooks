#!/usr/bin/env python
# coding: utf-8

# # **Credit Card fraud Detection**
# 
# I'm using JOPARGA3's 'In depth skewed data classif. (93% recall acc now)'.
# 
# and I share the learning code of library errors, SVMs and decision trees that I experienced during learning.

# In[ ]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

os.getcwd()


# In[ ]:


data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.head(3)


# In[ ]:


# checking Target Class

count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.xlabel("Class")
plt.ylabel("Frequency")

# label = 0 (No Fraud), label = 1 (Fraud)


# In[ ]:


from sklearn.preprocessing import StandardScaler

# data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1)) -> Series hadn't reshape
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data.head(2)


# In[ ]:


data = data.drop(['Time','Amount'],axis=1)
data.head(2)


# In[ ]:


# 데이터분리
# X = data.ix[:, data.columns != 'Class']
# y = data.ix[:, data.columns == 'Class']

# Delete 'ix' attribute in Dataframe
x = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']


# In[ ]:


# undersampling

# 데이터 지정
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = data[data.Class == 0].index

# normal_indices 를 랜덤(x)으로 추출
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# under dataset 생성
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
under_sample_data = data.iloc[under_sample_indices,:]

# X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
# y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

x_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']

# 비율점검
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))


# In[ ]:


from sklearn.model_selection import train_test_split

# Whole dataset
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)
print("[row dataset]")
print("Number transactions train dataset: ", len(x_train))
print("Number transactions test dataset: ", len(x_test))
print("Total number of transactions: ", len(x_train)+len(x_test))

# Undersampled dataset
x_train_undersample, x_test_undersample, y_train_undersample, y_test_undersample = train_test_split(x_undersample
                                                                                                   ,y_undersample
                                                                                                   ,test_size = 0.3
                                                                                                   ,random_state = 0)
print(" ")
print("[under dataset]")
print("Number transactions train dataset: ", len(x_train_undersample))
print("Number transactions test dataset: ", len(x_test_undersample))
print("Total number of transactions: ", len(x_train_undersample)+len(x_test_undersample))


# In[ ]:


print('ratio of train data label value')
print(y_train_undersample.value_counts()/y_train.shape[0] * 100)

print('ratio of test data label value')
print(y_test_undersample.value_counts()/y_test.shape[0] * 100)


# Logistic Regression : under Dataset

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 

logisticRegr = LogisticRegression()


# In[ ]:


logisticRegr.fit(x_train_undersample, y_train_undersample)


# In[ ]:


import itertools # 반복자를 만드는 모듈

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


best_c = 0.01 # c_param
lr = LogisticRegression(C = best_c, penalty = 'l2')
lr.fit(x_train_undersample,y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(x_test_undersample.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# In[ ]:


# 모델 반영
best_c = 0.01
lr = LogisticRegression(C = best_c, penalty = 'l2')
lr.fit(x_train_undersample,y_train_undersample.values.ravel())
y_pred = lr.predict(x_test.values)

# Compute confusion matrix 적용 (row data)
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# In[ ]:


# ROC CURVE : 이진분류기의 성능을 표현하는는 그래프로 가능한 모든 threshhold에 대해 FPR과 TRP의 비율을 표현한것
# 설명 url : https://angeloyeo.github.io/2020/08/05/ROC.html
# FPR(False Positive Rate) : 양성률 class 1을 1로 맞게 예측한 비율
# TPR(True Positive Rate) : 위양성률 class 0을 1로 잘못 예측한 비율

lr = LogisticRegression(C = best_c, penalty = 'l2')
y_pred_undersample_score = lr.fit(x_train_undersample,y_train_undersample.values.ravel()).decision_function(x_test_undersample.values)

fpr, tpr, thresholds = roc_curve(y_test_undersample.values.ravel(),y_pred_undersample_score)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# **SVM**

# In[ ]:


from sklearn import svm
classifier = svm.SVC(kernel='linear')

classifier.fit(x_train_undersample,y_train_undersample.values.ravel())


# In[ ]:


y_pred_undersample = classifier.predict(x_test_undersample)

cm = confusion_matrix(y_test_undersample, y_pred_undersample)
plot_confusion_matrix(cm,class_names)


# In[ ]:


y_pred = classifier.predict(x_test.values)

cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm,class_names)


# In[ ]:


y_pred_undersample_score = classifier.fit(x_train_undersample,y_train_undersample.values.ravel()).decision_function(x_test_undersample.values)

fpr, tpr, thresholds = roc_curve(y_test_undersample.values.ravel(),y_pred_undersample_score)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# **Decision Tree**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train_undersample,y_train_undersample.values.ravel())


# In[ ]:


y_pred_undersample = dt.predict(x_test_undersample.values)

cm = confusion_matrix(y_test_undersample, y_pred_undersample)
plot_confusion_matrix(cm,class_names)


# In[ ]:


y_pred = dt.predict(x_test.values)

cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm,class_names)


# In[ ]:


from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import f1_score, classification_report
from sklearn import metrics

def display_scores(y_test, y_pred):
    '''
    Display ROC-AUC score, f1 score and classification report of a model.
    '''
    print(f"F1 Score: {round(f1_score(y_test, y_pred)*100,2)}%") 
    print("\n\n")
    print(f"Classification Report: \n {classification_report(y_test, y_pred)}")


# In[ ]:


display_scores(y_test, y_pred)


# In[ ]:


# y_pred_undersample_score = dt.fit(x_train_undersample,y_train_undersample.values.ravel()).decision_function(x_test_undersample.values)


fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# I found the best model that was logistic regression.
# Among logistic regression, svm, and decision tree, logistic regression showed the best performance.

# In[ ]:




