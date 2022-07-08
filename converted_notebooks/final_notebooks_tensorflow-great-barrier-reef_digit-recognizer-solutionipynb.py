#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Libraries

# In[ ]:


# Importing Libraries
import numpy as np 
import pandas as pd
from collections import Counter
import math
# Importing Library for Data Visualization
import matplotlib.pyplot as plt

import sklearn
# Importing Algorithms for Model Training
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve


# # 2. Data Exploration

# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
training_data = pd.read_csv('../input/digit-recognizer/train.csv')


# In[ ]:


train_label = train['label']
train


# In[ ]:


# No. of Rows & Columns
train.shape


# In[ ]:


#Creation of Pie Chart
def create_pie(df, target_variable, figsize=(10, 10)):
    print(df[target_variable].value_counts())
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(df[target_variable].value_counts().values, labels=df[target_variable].value_counts().index, autopct='%1.2f%%', textprops={'fontsize': 10})
    ax.axis('equal')
    plt.title(target_variable)
    plt.show()


# In[ ]:


plt.figure(figsize=(25, 25))
create_pie(train, 'label')


# In[ ]:


# drop - for dropping entire columns/row of specific attribute value
cp = train.drop(['label'], axis=1)
ratio = int(math.sqrt(cp.shape[1]))


# In[ ]:


train = train.drop(['label'], axis=1)


# In[ ]:


first_image = train.iloc[0]
first_image = np.array(first_image).reshape(ratio, ratio)
plt.imshow(first_image);


# In[ ]:


# trying to see 9 first images
plt.figure(figsize=(25, 25))
columns = 3
firsts_image = train.iloc[:10]


for i in range(0, 9):
    image = np.array(train.iloc[i]).reshape(ratio, ratio)
    
    plt.subplot(int( firsts_image.shape[0]/ columns + 1), columns, i + 1)
    plt.imshow(image, cmap='Greens')


# # 3. Model Preparation

# In[ ]:


X = train
y = train_label.to_list()

X = train[:1000]
y = train_label[:1000].to_list()

X = X / 255.0

# split our data into train and test
X_train, X_test, y_train, y_test = train_test_split( X, y,
                                                    test_size=0.3,
                                                    random_state=42)

# and we also need a validation set 
X_test, X_val, y_test, y_val = train_test_split( X_test, y_test,
                                                    test_size=0.3,
                                                    random_state=42)
        
# normalized paramas
# {'coef0': 1, 'degree': 2, 'gamma': 0.1, 'kernel': 'poly'}

SVM = svm.SVC(gamma=0.1, coef0=1, kernel='poly', degree=2)
SVM.fit(X_train, y_train) 
index = 1
#image = np.array(X_train.iloc[i]).reshape(ratio, ratio)
x_pred = SVM.predict(X_test)
print('prediction: \n', x_pred)

X_prediction = SVM.predict(X_test)
print(f'test accuracy: \t{accuracy_score(y_test,X_prediction)}')
print(f'{classification_report(y_test, X_prediction)}')

# Plot Actual vs. Predicted 
y_test = y_test
predicted = SVM.predict(X_test)
'''
plt.figure(figsize=(28, 8))
plt.title("all images -- Actual vs. Predicted SVM model -- normalized ", fontsize=20)
plt.scatter(y_test, predicted,
            color="purple", marker="o", facecolors="none")

plt.plot([0, 9], [0, 9], "darkorange", lw=2)
plt.xlim(0, 9)
plt.ylim(0, 9)
plt.xlabel("Y test", fontsize=16)
plt.ylabel("Predicted Y", fontsize=16)
plt.show()


N,train_score, val_score = learning_curve(model, X_train, y_train,
                                   train_sizes=np.linspace(0.2,1.0,10 ),cv=5)

plt.figure(figsize=(28, 8))
plt.title("all images -- Learning Curve -- normalized ", fontsize=20)
plt.plot(N, train_score.mean(axis = 1), label ='Train')
plt.plot(N, val_score.mean(axis = 1), label ='Validation')
plt.xlabel('train sizes')
plt.legend();'''

# Parameter Grid
param_grid = {'kernel':['linear', 'poly'],# 'rbf', 'sigmoid'],
              'degree':[1, 2,],
              'gamma': [0.01, 0.1],
              'coef0': [0.5, 1]
             }
grid = GridSearchCV(svm.SVC(), param_grid, cv=3)
grid.fit(X_train, y_train)
model = grid.best_estimator_
model
model.score(X_test, y_test)

confusion_matrix(y_test, model.predict(X_test))


# In[ ]:


# Data Expansion
from sklearn.datasets import load_digits
from tensorflow.keras.datasets import mnist

digits = load_digits()
(X_train, y_train), (X_test, y_test) = mnist.load_data()

train = pd.read_csv('../input/digit-recognizer/train.csv')

train_label = train['label']


# In[ ]:


# sklearn train image 7 X 7 px
# so for using it with my actual dataset I need: resize this dataset to 28 X 28   or rezize to 7 X 7 

digit_data = digits.data
digit_target = digits.target
'''print(len(digit_data))
print(digit_data.shape)
print(digit_data[0])
print(len(digit_target))'''

'''ratio = int(math.sqrt(digit_data[0].shape[0]))

# trying to see 9 first images
plt.figure(figsize=(25, 25))
columns = 3
firsts_image = digit_data[:10]


for i in range(0, 9):
    image = np.array(digit_data[i]).reshape(ratio, ratio)
    
    plt.subplot(int( firsts_image.shape[0]/ columns + 1), columns, i + 1)
    plt.imshow(image, cmap='Purples')'''


# In[ ]:


# Keras train image 28 X 28 px
# I could use it with my actual dataset 


digit_data = X_train
digit_target = y_train
'''print(len(digit_data))
print(digit_data.shape)
print(digit_data[0])
print(len(digit_target))'''

'''# trying to see 9 first images
plt.figure(figsize=(25, 25))
columns = 3
firsts_image = digit_data[:10]


for i in range(0, 9):
    image = np.array(digit_data[i])#.reshape(ratio, ratio)
    
    plt.subplot(int( firsts_image.shape[0]/ columns + 1), columns, i + 1)
    plt.imshow(image, cmap='Purples')'''


# In[ ]:


#(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)

X_trainB = np.append(X_train, X_test).reshape(-1, 28 * 28)
print(X_trainB.shape)

y_trainB = np.append(y_train, y_test)
print(y_trainB.shape)

'''train_label = train['label']
train = pd.read_csv('../input/digit-recognizer/train.csv')
train'''


# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')
train_label = train['label']
train = train.drop(['label'], axis=1)

X_train = np.append(X_trainB, train).reshape(-1, 28 * 28)
print(X_train.shape)

y_train = np.append(y_trainB, train_label)
print(y_train.shape)


# In[ ]:


X = X_train
y = y_train

X = X / 255.0

# split our data into train and test
'''X_train, X_test, y_train, y_test = train_test_split( X, y,
                                                    test_size=0.3,
                                                    random_state=42)'''

# and we also need a validation set 
'''X_test, X_val, y_test, y_val = train_test_split( X_test, y_test,
                                                    test_size=0.3,
                                                    random_state=42)'''
        
# normalized paramas
# {'coef0': 1, 'degree': 2, 'gamma': 0.1, 'kernel': 'poly'}

SVM = svm.SVC(gamma=0.1, coef0=1, kernel='poly', degree=2)
SVM.fit(X, y) 
'''index = 1
#image = np.array(X_train.iloc[i]).reshape(ratio, ratio)
x_pred = SVM.predict(X_test)
print('prediction: \n', x_pred)

X_prediction = SVM.predict(X_test)
print(f'test accuracy: \t{accuracy_score(y_test,X_prediction)}')
print(f'{classification_report(y_test, X_prediction)}')'''

# Plot Actual vs. Predicted 
'''y_test = y_test
predicted = SVM.predict(X_test)

plt.figure(figsize=(28, 8))
plt.title("all images -- Actual vs. Predicted SVM model -- normalized ", fontsize=20)
plt.scatter(y_test, predicted,
            color="purple", marker="o", facecolors="none")

plt.plot([0, 9], [0, 9], "darkorange", lw=2)
plt.xlim(0, 9)
plt.ylim(0, 9)
plt.xlabel("Y test", fontsize=16)
plt.ylabel("Predicted Y", fontsize=16)
plt.show()'''


'''N,train_score, val_score = learning_curve(model, X_train, y_train,
                                   train_sizes=np.linspace(0.2,1.0,10 ),cv=3)

plt.figure(figsize=(28, 8))
plt.title("all images -- Learning Curve -- normalized ", fontsize=20)
plt.plot(N, train_score.mean(axis = 1), label ='Train')
plt.plot(N, val_score.mean(axis = 1), label ='Validation')
plt.xlabel('train sizes')
plt.legend();'''


#confusion_matrix(y_test, model.predict(X_test))


# In[ ]:


# trying to see 9 first images
plt.figure(figsize=(25, 25))
columns = 3
firsts_image = test.iloc[:10]


for i in range(0, 9):
    image = np.array(test.iloc[i]).reshape(ratio, ratio)
    
    plt.subplot(int( firsts_image.shape[0]/ columns + 1), columns, i + 1)
    plt.imshow(image, cmap='Purples')


# **Normalization Submission**

# In[ ]:


x_submit = test.copy()
x_submit = x_submit / 255.0


# # 4. Prediction & Submission

# In[ ]:


X_prediction = SVM.predict(x_submit)
print('prediction: \n', X_prediction)


# In[ ]:


pred = pd.Series(X_prediction, name='Label')
pred.head()

index_list = []

for i, item in enumerate(pred):
    index_list.append(i+1)
    
image_id = pd.Series(index_list, name='ImageId')


# In[ ]:


submit = pd.concat([image_id, pred], axis=1)
submit.to_csv('submission.csv', index=False)
submit.tail()
sub = pd.read_csv('./submission.csv')
sub

