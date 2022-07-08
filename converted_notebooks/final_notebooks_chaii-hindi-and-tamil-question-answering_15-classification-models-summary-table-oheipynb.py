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


# # Importing Data

# In[ ]:


data=pd.read_csv('/kaggle/input/students-adaptability-level-in-online-education/students_adaptability_level_online_education.csv')
data.head()


# In[ ]:


col=data.columns
data.shape


# In[ ]:


data.info()


# ![image.png](attachment:4abd6e56-65d8-4163-ada7-51aecfaadea6.png)

# In[ ]:


for i in col:
    print(i,"-",data[i].isna().sum())


# # Visualization

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[ ]:


for i in col:
    fig=px.pie(data,values=data[i].value_counts(),
               names=data[i].value_counts().index,template ="simple_white",
               color_discrete_sequence = px.colors.diverging.Tropic,hole=.3)
    fig.update_layout(title_text="Distribution Of The"+i,
                      title_x=0.5,
                      font_size=15)
    fig.show()


# In[ ]:


i = 1
plt.figure(figsize = (15,25))
for feature in col:
    plt.subplot(6,3,i)
    sns.countplot(x = feature ,  data = data)
    i +=1


# In[ ]:


i = 1
plt.figure(figsize = (15,25))
for feature in col:
    plt.subplot(6,3,i)
    sns.countplot(x = feature , hue='Adaptivity Level', data = data)
    i +=1


# # Model building

# ## It is a classification Problem . The Models we can consider are:
# 
# ### 1)Logistic Regression
# ### 2)Naive Bayes
#         * MultinomialNB
#         * ComplementNB
#         * CategoricalNB
# ### 3)Support vector machine
#         * SVC
#         * LinearSVC
# ### 4)KNN
#         * KNeighborsClassifier
#         * RadiusNeighborsClassifier
# ### 5)DecisionTreeClassifier
#         * (GridSearchCV)
# ### 6)RandomForestClassifier 
#         * (RandomizedSearchCV,GridSearchCV)
# ### 7)Ensemble
#         * ExtraTreesClassifier
#         * AdaBoostClassifier
# ### 9)XGBClassifier
#         * (GridSearchCV) 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report,ConfusionMatrixDisplay




# In[ ]:


col


# ## Using One Hot Encoding here
# ## To see how Ordinal Encoding performs see another notebook

# In[ ]:


one_hot_encoded_data = pd.get_dummies(data, columns = ['Gender', 'Age', 'Education Level', 'Institution Type', 'IT Student',
       'Location', 'Load-shedding', 'Financial Condition', 'Internet Type',
       'Network Type', 'Class Duration', 'Self Lms', 'Device'])
one_hot_encoded_data.head()


# In[ ]:


one_hot_encoded_data['Adaptivity Level']=one_hot_encoded_data['Adaptivity Level'].map(
{'Moderate':2,'Low':1,'High':0})


# In[ ]:


X=one_hot_encoded_data.drop(['Adaptivity Level'],axis=1)
y=one_hot_encoded_data[['Adaptivity Level']]


# In[ ]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# # Logistic Multiclass

# In[ ]:


clf=LogisticRegression(random_state=42,multi_class='multinomial',solver='lbfgs', max_iter=500,penalty='l2',warm_start=True)
clf.fit(X_train,y_train.values.ravel())


# In[ ]:


clf.get_params()


# In[ ]:


pred=clf.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


cm = confusion_matrix(pred,y_test)
disp = ConfusionMatrixDisplay(cm, display_labels=["High","Low","Moderate"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()


# # Naive bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB,ComplementNB,CategoricalNB


# In[ ]:


clf = MultinomialNB()
clf.fit(X_train, y_train.values.ravel())
pred=clf.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


cm = confusion_matrix(pred,y_test)
disp = ConfusionMatrixDisplay(cm, display_labels=["High","Low","Moderate"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()


# In[ ]:


clf = ComplementNB()
clf.fit(X_train, y_train.values.ravel())
pred=clf.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


cm = confusion_matrix(pred,y_test)
disp = ConfusionMatrixDisplay(cm, display_labels=["High","Low","Moderate"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()


# In[ ]:


clf = CategoricalNB()
clf.fit(X_train, y_train.values.ravel())
pred=clf.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


cm = confusion_matrix(pred,y_test)
disp = ConfusionMatrixDisplay(cm, display_labels=["High","Low","Moderate"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()


# # SVM

# In[ ]:


from sklearn import svm


# In[ ]:


clf = svm.SVC(C=10,kernel='poly')
clf.fit(X_train, y_train.values.ravel())
pred=clf.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


clf = svm.LinearSVC()
clf.fit(X_train, y_train.values.ravel())
pred=clf.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


clf = svm.SVC(decision_function_shape='ovo',C=10,kernel='poly')
clf.fit(X_train, y_train.values.ravel())
pred=clf.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# # KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train.values.ravel())
pred = knn.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


accuracy_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,X,y.values.ravel(),cv=10)
    accuracy_rate.append(score.mean())


# In[ ]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,X,y.values.ravel(),cv=10)
    error_rate.append(1-score.mean())


# In[ ]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i,weights='distance')
    knn.fit(X_train,y_train.values.ravel())
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test.values.ravel()))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
#plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
 #        markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[ ]:


# NOW WITH K=4
knn = KNeighborsClassifier(n_neighbors=4,weights='distance')

knn.fit(X_train,y_train.values.ravel())
pred = knn.predict(X_test)

print('WITH K=4')
print('\n')
print(confusion_matrix(y_test.values.ravel(),pred))
print('\n')
print(classification_report(y_test.values.ravel(),pred))


# # Decision Tree

# In[ ]:


from sklearn import tree
from sklearn.model_selection import GridSearchCV


# In[ ]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train.values.ravel())
pred = clf.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


params = {
    'max_depth': range(1,20,2),
    'min_samples_leaf': range(1,100,5),
    'min_samples_split': range(2,10),
    'criterion': ["gini", "entropy"],
    'splitter':['best', 'random'],
    'max_features': ['auto']
}


# In[ ]:


grid_search = GridSearchCV(estimator=clf, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy",error_score='raise')


# In[ ]:


grid_search.fit(X_train, y_train.values.ravel())


# In[ ]:


score_df = pd.DataFrame(grid_search.cv_results_)
score_df.head()


# In[ ]:


score_df.nlargest(5,"mean_test_score")


# In[ ]:


grid_search.best_estimator_


# In[ ]:


dt_best = grid_search.best_estimator_


# In[ ]:


def evaluate_model(dt_classifier):
    print("Train Accuracy :", accuracy_score(y_train, dt_classifier.predict(X_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, dt_classifier.predict(X_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, dt_classifier.predict(X_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, dt_classifier.predict(X_test)))


# In[ ]:


evaluate_model(dt_best)


# In[ ]:


pred = dt_best.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


#https://www.researchgate.net/publication/355891881_Students'_Adaptability_Level_Prediction_in_Online_Education_using_Machine_Learning_Approaches


# # Random Forest
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf = RandomForestClassifier(random_state=0)
clf.fit(X_train,y_train.values.ravel())
pred = dt_best.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[ ]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
rf_random.fit(X_train,y_train.values.ravel())


# In[ ]:


pred = rf_random.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


from sklearn.model_selection import GridSearchCV# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}# Create a based model
rf = RandomForestClassifier()# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train,y_train.values.ravel())


# In[ ]:


pred = grid_search.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# # Ensemble

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X_train,y_train.values.ravel())


# In[ ]:


pred = clf.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X_train,y_train.values.ravel())


# In[ ]:


pred = clf.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


ab_clf = AdaBoostClassifier(random_state=0)
parameters = {
    'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30],
    'learning_rate': [(0.97 + x / 100) for x in range(0, 8)],
    'algorithm': ['SAMME', 'SAMME.R'],
}
clf = GridSearchCV(ab_clf, parameters, cv=5,  n_jobs = -1, verbose = 2)
clf.fit(X_train,y_train.values.ravel())


# In[ ]:


pred = clf.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# # XGB

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgb_model = XGBClassifier()
xgb_model.fit(X_train,y_train.values.ravel())


# In[ ]:


pred = clf.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# In[ ]:


#param_grid ={'gamma': [0.1,0.4,0.8],
 #             'learning_rate': [0.01,0.1, 0.300000012],
  #            'max_depth': [6,8,10],
   #           'n_estimators': [65,100,130],
    #          'reg_alpha': [0,0.2,0.4],
     #         'reg_lambda': [0.2,0.8,1.6]}


# In[ ]:


param_grid = {'gamma': [0.1,0.4,0.8,1.6,3.2],
              'learning_rate': [ 0.01,0.1, 0.300000012, 0.2, 0.6],
              'max_depth': [6,8,10,12],
              'n_estimators': [65,100,130],
              'reg_alpha': [0,0.2,0.4,0.8,1.6,3.2],
              'reg_lambda': [0.2,0.4,0.8,1.6]}


# In[ ]:


clf_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid, verbose=2, cv=3,n_jobs=-1)
clf_xgb.fit(X_train,y_train.values.ravel())


# In[ ]:


pred = clf_xgb.predict(X_test)
print(classification_report(y_test,pred, zero_division=1))


# # Conclusion
# 
# ### I Know you are too tired of scrolling this many cell.
# ## But do give a glance at the table.
# ### Also consider Looking another notebook with ordinal encoding and how models performed there

# ![one-min.jpg](attachment:dd7da087-f9a9-4a35-9ac5-3e1509ba321b.jpg)

# ![two_page-0001-min.jpg](attachment:3a48dbf0-4704-4b73-883b-ddb62004c142.jpg)

# ## Might be if both notebooks gets a bronze medal will try AutoML

# cite the original research paper. Students' Adaptability Level Prediction in Online Education using Machine Learning Approaches 
# https://www.researchgate.netpublication355891881_Students'_Adaptability_Level_Prediction_in_Online_Education_using_Machine_Learning_Approaches
# or DOI: 10.1109/ICCCNT51525.2021.9579741
