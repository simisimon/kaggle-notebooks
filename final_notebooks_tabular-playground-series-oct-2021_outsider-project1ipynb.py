#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[ ]:


from scipy import stats
# from scipy.stats import pearsonr
# from scipy.stats import ttest_ind
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_learning_curves
from sklearn.metrics import accuracy_score


# In[ ]:


dataset_path = "/kaggle/input/loan-data-set/"
dataset_filename = "loan_data_set.csv"


# In[ ]:


# --------- load data + data overview ---------- 
dataset = pd.read_csv(dataset_path+dataset_filename)
print(dataset.head())


# In[ ]:


print("The dimensions of the dataset are:", dataset.shape)


# In[ ]:


print("Male vs Female entry points:")
dataset.Gender.value_counts(dropna=False)


# In[ ]:


sns.countplot(x="Loan_Status", data=dataset).set(title="The distribution of Loan_Status")


# ## Data processing

# ## We need to handle NaN values
# ### Using data imputation technique we will replace all nan values with the mode value -> the most common value of each column

# In[ ]:


print("Check whether there are nan values:")
dataset.isnull().sum()


# In[ ]:


# --------- treat nan values by setting them to mode value of distribution for each category --------- 
def nan_to_mode(df):
    for col in df.columns:
        temp = df[col]
        df.loc[temp.isna(), col] = temp.mode()[0]
        # or
        #dataset[col].fillna(dataset[col].mode()[0], inplace=True)
    return df


# In[ ]:


dataset = nan_to_mode(dataset)


# ## We need to handle outliers

# In[ ]:


# def cap_outliers(df):
#     for col in df.columns:
#         if (df[col].dtype != 'object'):
#             tmp = df[col]
#             q1 = tmp.quantile(.05)
#             q3 = tmp.quantile(.95)
#             df.loc[tmp < q1, col] = q1
#             df.loc[tmp > q3, col] = q3
#     return df


# In[ ]:


# --------- treat outliers by setting them to mode value of distribution for each category --------- 
def outliers_to_mode(df):
    for col in df.columns:
        if (df[col].dtype != 'object'): # data needs to be non-categorical
            tmp = df[col]
            q1 = tmp.quantile(.05)
            q3 = tmp.quantile(.95)
            df.loc[tmp < q1, col] = tmp.mode()[0]
            df.loc[tmp > q3, col] = tmp.mode()[0]
    return df


# In[ ]:


sns.histplot(data=dataset, x="ApplicantIncome", kde=True, color='green').set(title="sample data distribution before treating outliers");


# In[ ]:


dataset = outliers_to_mode(dataset)


# In[ ]:


sns.histplot(data=dataset, x="ApplicantIncome", kde=True, color='green').set(title="sample data distribution after treating outliers");


# ## Drop unused columns and encode categorical data

# In[ ]:


# --------- drop unused columns and encode categorical data ---------
dataset = dataset.drop(columns=['Loan_ID'])


# In[ ]:


def encode_categorical_data(df):
    for col in df.columns:
        if(df[col].dtype == 'object'):
            lbl = LabelEncoder()
            lbl.fit(list(df[col].values))
            df[col] = lbl.transform(list(df[col].values))
    return df


# In[ ]:


# one hot gives better results
def one_hot_encoding(df):
    df = pd.get_dummies(dataset)
    # Drop extra binary columns
    df = df.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 
                  'Self_Employed_No', 'Loan_Status_N'], axis = 1)

    # Rename the positive class
    new = {'Gender_Male': 'Gender', 'Married_Yes': 'Married', 
           'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed',
           'Loan_Status_Y': 'Loan_Status'}
       
    df.rename(columns=new, inplace=True)
    return df


# In[ ]:


# dataset.head()


# In[ ]:


# dataset = encode_categorical_data(dataset)
dataset = one_hot_encoding(dataset)
dataset.head(10)


# In[ ]:


# ---------- split the dataset inputs and outputs ----------
X = dataset.drop(columns=['Loan_Status'])
y = dataset.Loan_Status


# In[ ]:


# oversample in order to avoid overfitting due to the unbalanced dataset
X, y = SMOTE().fit_resample(X, y)


# In[ ]:


X = StandardScaler().fit_transform(X)
X = minmax_scale(X)


# In[ ]:


# ---------- split the dataset to train and test samples ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Logistic Regression

# In[ ]:


LRclassifier = LogisticRegression(solver='saga', max_iter=500, random_state=1)
LRclassifier.fit(X_train, y_train)

y_pred = LRclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


LRAcc = accuracy_score(y_pred,y_test)
print('LR accuracy: {:.2f}%'.format(LRAcc*100))


# # K-NN

# In[ ]:


scoreListknn = []
for i in range(1,21):
    KNclassifier = KNeighborsClassifier(n_neighbors = i)
    KNclassifier.fit(X_train, y_train)
    scoreListknn.append(KNclassifier.score(X_test, y_test))
    
plt.plot(range(1,21), scoreListknn)
plt.xticks(np.arange(1,21,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()
KNAcc = max(scoreListknn)
print("KNN best accuracy: {:.2f}%".format(KNAcc*100))


# # SVM

# In[ ]:


SVCclassifier = SVC(kernel='rbf', max_iter=500)
SVCclassifier.fit(X_train, y_train)

y_pred = SVCclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
SVCAcc = accuracy_score(y_pred,y_test)
print('SVC accuracy: {:.2f}%'.format(SVCAcc*100))


# # Categorical NB

# In[ ]:


NBclassifier1 = CategoricalNB()
NBclassifier1.fit(X_train, y_train)

y_pred = NBclassifier1.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
NBAcc1 = accuracy_score(y_pred,y_test)
print('Categorical Naive Bayes accuracy: {:.2f}%'.format(NBAcc1*100))


# # Gaussian NB

# In[ ]:


NBclassifier2 = GaussianNB()
NBclassifier2.fit(X_train, y_train)

y_pred = NBclassifier2.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
NBAcc2 = accuracy_score(y_pred,y_test)
print('Gaussian Naive Bayes accuracy: {:.2f}%'.format(NBAcc2*100))


# # Decision Tree

# In[ ]:


scoreListDT = []
for i in range(2,21):
    DTclassifier = DecisionTreeClassifier(max_leaf_nodes=i)
    DTclassifier.fit(X_train, y_train)
    scoreListDT.append(DTclassifier.score(X_test, y_test))
    
plt.plot(range(2,21), scoreListDT)
plt.xticks(np.arange(2,21,1))
plt.xlabel("Leaf")
plt.ylabel("Score")
plt.show()
DTAcc = max(scoreListDT)
print("Decision Tree Accuracy: {:.2f}%".format(DTAcc*100))


# # Random Forest

# In[ ]:


scoreListRF = []
bestModelListRF = []
for i in range(2,25):
    RFclassifier = RandomForestClassifier(n_estimators = 1000, random_state = 1, max_leaf_nodes=i)
    RFclassifier.fit(X_train, y_train)
    scoreListRF.append(RFclassifier.score(X_test, y_test))
    bestModelListRF.append(RFclassifier)
    
plt.plot(range(2,25), scoreListRF)
plt.xticks(np.arange(2,25,1))
plt.xlabel("RF Value")
plt.ylabel("Score")
plt.show()
RFAcc = max(scoreListRF)
RFBestModel = bestModelListRF[np.argmax(scoreListRF)]
print("Random Forest Accuracy:  {:.2f}%".format(RFAcc*100))
plot_learning_curves(X_train, y_train, X_test, y_test, RFBestModel);


# In[ ]:


# plt.savefig("learning_curve")


# # Gradient Boosting

# In[ ]:


paramsGB={'n_estimators':[100,200,300,400,500],
      'max_depth':[1,2,3,4,5],
      'subsample':[0.5,1],
      'max_leaf_nodes':[2,5,10,20,30,40,50]}


# In[ ]:


GB = RandomizedSearchCV(GradientBoostingClassifier(), paramsGB, cv=20)
GB.fit(X_train, y_train)


# In[ ]:


print(GB.best_estimator_)
print(GB.best_score_)
print(GB.best_params_)
print(GB.best_index_)


# In[ ]:


GBclassifier = GradientBoostingClassifier(subsample=0.5, n_estimators=400, max_depth=4, max_leaf_nodes=10)
GBclassifier.fit(X_train, y_train)

y_pred = GBclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
GBAcc = accuracy_score(y_pred,y_test)
print('Gradient Boosting accuracy: {:.2f}%'.format(GBAcc*100))


# In[ ]:




