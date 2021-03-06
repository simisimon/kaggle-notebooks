#!/usr/bin/env python
# coding: utf-8

# # Contents
# 1. [Importing Libraries and Packages](#p1)
# 2. [Loading and Viewing Data Set](#p2)
# 3. [Dealing with NaN Values (Imputation)](#p3)
# 4. [Feature Engineering](#p4)
# 5. [Modeling and Predicting with sklearn](#p5)
# 6. [Evaluating Model Performances](#p6)
# 7. [Tuning Parameters with GridSearchCV](#p7)
# 8. [Submission](#p8)

# <a id="p1"></a>
# # 1. Importing Libraries and Packages
# 

# In[ ]:


import numpy as np 
import pandas as pd 

import seaborn as sns 
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# setting seaborn style into whitegrid
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings("ignore")


# <a id="p2"></a>
# # 2. Loading and Viewing Data Set
# With Pandas, we can load both the training and testing set that we wil later use to train and test our model. Before we begin, we should take a look at our data table to see the values that we'll be working with. We can use the head and describe function to look at some sample data and statistics. We can also look at its keys and column names.

# In[ ]:


training = pd.read_csv("../input/train.csv")
testing = pd.read_csv("../input/test.csv")


# In[ ]:


training.head()


# In[ ]:


training.describe()


# In[ ]:


print(training.keys())
print(testing.keys())


# <a id="p3"></a>
# # 3. Dealing with NaN Values (Imputation)
# There are NaN values in our data set in the age column. Furthermore, the Cabin column has too many missing values and isn't useful to be used in predicting survival. We can just drop the column as well as the NaN values which will get in the way of training our model. We also need to fill in the NaN values with replacement values in order for the model to have a complete prediction for every row in the data set.

# In[ ]:


def null_table(training, testing):
    print("Training Data Frame")
    print(pd.isnull(training).sum()) 
    print(" ")
    print("Testing Data Frame")
    print(pd.isnull(testing).sum())

null_table(training, testing)


# In[ ]:


training.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)
testing.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)

null_table(training, testing)


# We take a look at the distribution of the Age column to see if it's skewed or symmetrical. This will help us determine what value to replace the NaN values.

# We can fill in the null values with the median for the most accuracy.

# In[ ]:


#the median will be an acceptable value to place in the NaN cells
training["Age"].fillna(training["Age"].median(), inplace = True)
testing["Age"].fillna(testing["Age"].median(), inplace = True) 
training["Embarked"].fillna("S", inplace = True)
testing["Fare"].fillna(testing["Fare"].median(), inplace = True)

null_table(training, testing)


# <a id="p4"></a>
# # 4. Feature Engineering
# Because values in the Sex and Embarked columns are categorical values, we have to represent these strings as numerical values in order to perform our classification with our model. We can also do this process through **One-Hot-Encoding**.

# In[ ]:


training.sample(5)


# In[ ]:


testing.sample(5)


# We change Sex to binary, as either 1 for female or 0 for male. We do the same for Embarked. We do this same process on both the training and testing set to prepare our data for Machine Learning.

# In[ ]:


training.loc[training["Sex"] == "male", "Sex"] = 0
training.loc[training["Sex"] == "female", "Sex"] = 1

training.loc[training["Embarked"] == "S", "Embarked"] = 0
training.loc[training["Embarked"] == "C", "Embarked"] = 1
training.loc[training["Embarked"] == "Q", "Embarked"] = 2

testing.loc[testing["Sex"] == "male", "Sex"] = 0
testing.loc[testing["Sex"] == "female", "Sex"] = 1

testing.loc[testing["Embarked"] == "S", "Embarked"] = 0
testing.loc[testing["Embarked"] == "C", "Embarked"] = 1
testing.loc[testing["Embarked"] == "Q", "Embarked"] = 2


# In[ ]:


testing.sample(10)


# We can combine SibSp and Parch into one synthetic feature called family size, which indicates the total number of family members on board for each member. 

# In[ ]:


training["FamSize"] = training["SibSp"] + training["Parch"] + 1
testing["FamSize"] = testing["SibSp"] + testing["Parch"] + 1


# <a id="p6"></a>
# # 6. Model Fitting and Predicting
# Now that our data has been processed and formmated properly, and that we understand the general data we're working with as well as the trends and associations, we can start to build our model. We can import different classifiers from sklearn. We will try different types of models to see which one gives the best accuracy for its predictions.

# **sklearn Models to Test**

# In[ ]:


from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# To evaluate our model performance, we can use the make_scorere and accuracy_score function from sklearn metrics.

# In[ ]:


from sklearn.metrics import make_scorer, accuracy_score 


# We can also use a GridSearch cross validation to find the optimal parameters for the model we choose to work with and use to predict on our testing set.

# In[ ]:


from sklearn.model_selection import GridSearchCV


# **Defining Features in Training/Test Set**

# In[ ]:


features = ["Pclass", "Sex", "Age", "Embarked", "Fare", "FamSize"]
X_train = training[features] #define training features set
y_train = training["Survived"] #define training label set
X_test = testing[features] #define testing features set


# **Validation Data Set**
# 
# Although we already have a test set, it is generally easy to overfit the data with these classifiers. It is therefore useful to have a third data set called the validation data set to ensure that our model doesn't overfit with the data. We can make this third data set with sklearn's train_test_split function. We can also use the validation data set to test the general accuracy of our model.

# In[ ]:


from sklearn.model_selection import train_test_split #to create validation data set

X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0) #X_valid and y_valid are the validation sets


# **SVC Model**

# In[ ]:


svc_clf = SVC() 
svc_clf.fit(X_training, y_training)
pred_svc = svc_clf.predict(X_valid)
acc_svc = accuracy_score(y_valid, pred_svc)

print(acc_svc)


# **LinearSVC Model**

# In[ ]:


linsvc_clf = LinearSVC()
linsvc_clf.fit(X_training, y_training)
pred_linsvc = linsvc_clf.predict(X_valid)
acc_linsvc = accuracy_score(y_valid, pred_linsvc)

print(acc_linsvc)


# **RandomForest Model**

# In[ ]:


rf_clf = RandomForestClassifier()
rf_clf.fit(X_training, y_training)
pred_rf = rf_clf.predict(X_valid)
acc_rf = accuracy_score(y_valid, pred_rf)

print(acc_rf)


# **LogisiticRegression Model**

# In[ ]:


logreg_clf = LogisticRegression()
logreg_clf.fit(X_training, y_training)
pred_logreg = logreg_clf.predict(X_valid)
acc_logreg = accuracy_score(y_valid, pred_logreg)

print(acc_logreg)


# **KNeighbors Model**

# In[ ]:


knn_clf = KNeighborsClassifier()
knn_clf.fit(X_training, y_training)
pred_knn = knn_clf.predict(X_valid)
acc_knn = accuracy_score(y_valid, pred_knn)

print(acc_knn)


# **GaussianNB Model**

# In[ ]:


gnb_clf = GaussianNB()
gnb_clf.fit(X_training, y_training)
pred_gnb = gnb_clf.predict(X_valid)
acc_gnb = accuracy_score(y_valid, pred_gnb)

print(acc_gnb)


# **DecisionTree Model**

# In[ ]:


dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_training, y_training)
pred_dt = dt_clf.predict(X_valid)
acc_dt = accuracy_score(y_valid, pred_dt)

print(acc_dt)


# **XGBoost Model**

# In[ ]:


from xgboost import XGBClassifier

xg_clf = XGBClassifier(objective="binary:logistic", n_estimators=10, seed=123)
xg_clf.fit(X_training, y_training)
pred_xg = xg_clf.predict(X_valid)
acc_xg = accuracy_score(y_valid, pred_xg)

print(acc_xg)


# <a id="p7"></a>
# # 7. Evaluating Model Performances
# After making so many models and predictions, we should evaluate and see which model performed the best and which model to use on our testing set.

# In[ ]:


model_performance = pd.DataFrame({
    "Model": ["SVC", "Linear SVC", "Random Forest", 
              "Logistic Regression", "K Nearest Neighbors", "Gaussian Naive Bayes",  
              "Decision Tree", "XGBClassifier"],
    "Accuracy": [acc_svc, acc_linsvc, acc_rf, 
              acc_logreg, acc_knn, acc_gnb, acc_dt, acc_xg]
})

model_performance.sort_values(by="Accuracy", ascending=False)


# It appears that the Random Forest model works the best with our data so we will use it on the test set.

# <a id="p8"></a>
# # 8. Tuning Parameters with GridSearchCV

# We can improve the accuracy of our model by turning the hyperparameters of our Random Forest model. We will run a GridSearchCV to find the best parameters for the model and use that model to train and test our data.

# In[ ]:


# Uncomment below if you use GridSearchCV
"""rf_clf = RandomForestClassifier()

parameters = {"n_estimators": [4, 5, 6, 7, 8, 9, 10, 15], 
              "criterion": ["gini", "entropy"],
              "max_features": ["auto", "sqrt", "log2"], 
              "max_depth": [2, 3, 5, 10], 
              "min_samples_split": [2, 3, 5, 10],
              "min_samples_leaf": [1, 5, 8, 10]
             }

grid_cv = GridSearchCV(rf_clf, parameters, scoring = make_scorer(accuracy_score))
grid_cv = grid_cv.fit(X_train, y_train)

print("Our optimized Random Forest model is:")
grid_cv.best_estimator_
"""


#  <a id="p9"></a>
# # 9. Tuning Parameters with RandomSearchCV

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


rf_clf = RandomForestClassifier()

parameters = {"n_estimators": [4, 5, 6, 7, 8, 9, 10, 15], 
              "criterion": ["gini", "entropy"],
              "max_features": ["auto", "sqrt", "log2"], 
              "max_depth": [2, 3, 5, 10], 
              "min_samples_split": [2, 3, 5, 10],
              "min_samples_leaf": [1, 5, 8, 10]
             }

grid_cv = RandomizedSearchCV(rf_clf, parameters, scoring = make_scorer(accuracy_score))
grid_cv = grid_cv.fit(X_train, y_train)

print("Our optimized Random Forest model is:")
grid_cv.best_estimator_


# In[ ]:


rf_clf = grid_cv.best_estimator_

rf_clf.fit(X_train, y_train)


# <a id="p8"></a>
# # 10. Submission

# Let's create a dataframe to submit to the competition with our predictions of our model.

# In[ ]:


submission_predictions =rf_clf.predict(X_test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": testing["PassengerId"],
        "Survived": submission_predictions
    })

submission.to_csv("titanic.csv", index=False)
print(submission.shape)

