#!/usr/bin/env python
# coding: utf-8

# # **Importing libraries**

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  plot_confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# In[ ]:


Data = pd.read_csv('/kaggle/input/personal-key-indicators-of-heart-disease/heart_2020_cleaned.csv')


# **View summary of Data**

# In[ ]:


Data.info()


# In[ ]:


Data.head()


# **Data preprocessing**

# In[ ]:


#categorial
le = LabelEncoder()
col = Data[['HeartDisease', 'Smoking', 'AlcoholDrinking','AgeCategory', 'Stroke', 'DiffWalking','Race', 'Sex','PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer','GenHealth' ,'Diabetic']]
for i in col:
  Data[i] = le.fit_transform(Data[i])
Data.head()


# In[ ]:


#numeric
num_cols = ['MentalHealth', 'BMI', 'PhysicalHealth', 'SleepTime'  ]
Scaler = StandardScaler()
Data[num_cols] = Scaler.fit_transform(Data[num_cols])


# In[ ]:


import plotly.express as px
fig = px.imshow(Data.corr(),color_continuous_scale="Blues")
fig.update_layout(height=800)
fig.show()


# In[ ]:


corr_matrix = Data.corr()
corr_matrix["HeartDisease"].sort_values(ascending=False)


# **Declare feature vector and target variable**

# In[ ]:


X = Data.drop(columns =['HeartDisease'], axis = 1)
Y = Data['HeartDisease']


# **Split data into separate training and test set**

# In[ ]:


X_train, x_test, y_train, y_test = train_test_split(X,Y,shuffle = True, test_size = .2, random_state = 42 )
y_train.value_counts()


# As you can see , our data is unbalanced data (234055 '0' , 21781 '1' , so lets try to oversample our data

# In[ ]:


ros = RandomOverSampler(random_state =42)
X_train_resampled , y_train_resampled , = ros.fit_resample(X_train , y_train)
y_train_resampled.value_counts()


# # ***Training with Naive Bayes***

# ***Bernoulli***

# In[ ]:


model_0 = BernoulliNB()
model_0.fit(X_train_resampled,y_train_resampled)

pred_0 = model_0.predict(x_test)
print(classification_report(y_test,pred_0))
cm = confusion_matrix(y_test,pred_0)
sns.set_context ("poster") 
dispo = plot_confusion_matrix(model_0, x_test , y_test  , colorbar= False )


# ***Gaussian***

# In[ ]:


model_1 = BernoulliNB()
model_1.fit(X_train_resampled,y_train_resampled)

pred_1 = model_1.predict(x_test)
print(classification_report(y_test,pred_1))
cm = confusion_matrix(y_test,pred_1)
sns.set_context ("poster") 
dispo = plot_confusion_matrix(model_1, x_test , y_test  , colorbar= False )


# # **Random forest**

# In[ ]:


rf=RandomForestClassifier( n_estimators = 9 )
rf.fit(X_train_resampled , y_train_resampled)

y_pred_3 = rf.predict(x_test)

print(classification_report(y_test,y_pred_3))
from sklearn.metrics import  plot_confusion_matrix
sns.set_context ("poster") 
dispo = plot_confusion_matrix(rf, x_test , y_test  , colorbar= False )


# # **Voting**

# In[ ]:


ber_clf = BernoulliNB()
rnd_clf = RandomForestClassifier()
gss_clf = GaussianNB()
voting_clf = VotingClassifier(
estimators=[('Bernoulli', ber_clf), ('rf', rnd_clf), ('gss', gss_clf)],
voting='hard')
voting_clf.fit(X_train_resampled,y_train_resampled)

for clf in (ber_clf, rnd_clf, gss_clf, voting_clf):
    clf.fit(X_train_resampled,y_train_resampled)
    y_pred_4 = clf.predict(x_test)
    print(clf.__class__.__name__, recall_score(y_test, y_pred_4))
print(classification_report(y_test,y_pred_4))
sns.set_context ("poster") 
dispo = plot_confusion_matrix(voting_clf, x_test , y_test  , colorbar= False )


# # **Bagging**

# In[ ]:


bag_clf = BaggingClassifier(
DecisionTreeClassifier(), n_estimators=500,
max_samples=150, bootstrap=False, n_jobs=-1)
bag_clf.fit(X_train_resampled,y_train_resampled)
y_pred_5 = bag_clf.predict(x_test)
print(classification_report(y_test,y_pred_5))
sns.set_context ("poster") 
dispo = plot_confusion_matrix(bag_clf, x_test , y_test  , colorbar= False )


# # **Train the XGBoost classifier**

# In[ ]:


# declare parameters
params = {
            'objective':'binary:logistic',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':9
        }
                      
# instantiate the classifier 
xgb_clf = XGBClassifier(**params)

# fit the classifier to the training data
xgb_clf.fit(X_train_resampled,y_train_resampled)

# make predictions on test data
y_pred_6 = xgb_clf.predict(x_test)

print(classification_report(y_test,y_pred_6))
sns.set_context ("poster") 
dispo = plot_confusion_matrix(xgb_clf, x_test , y_test  , colorbar= False )


# # **LogisticRegression**

# In[ ]:


LR=LogisticRegression( )
LR.fit(X_train_resampled , y_train_resampled)

y_pred_6 = LR.predict(x_test)

print(classification_report(y_test,y_pred_6))
sns.set_context ("poster") 
dispo = plot_confusion_matrix(LR, x_test , y_test  , colorbar= False )


# ***this work will be updated, please upvote it to encourage us***
