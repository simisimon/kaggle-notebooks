#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install openpyxl')


# In[ ]:


# All the imports needed

# Data Manipulation
import numpy as np
import pandas as pd 

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#Train Test Split
from sklearn.model_selection import train_test_split

#Scaling
from sklearn.preprocessing import StandardScaler

#Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#File Directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#Reading the file
dates = pd.read_excel('/kaggle/input/date-fruit-datasets/Date_Fruit_Datasets/Date_Fruit_Datasets.xlsx')


# In[ ]:


#Groupby based on each class
datesgp = dates.groupby(by='Class').mean()
datesgp


# In[ ]:


#Describe the original dataset
dates.describe()


# In[ ]:


#Checking data types and null values
dates.info()


# In[ ]:


#Heatmap for all correlations
plt.figure(figsize=(24,12))
sns.heatmap(dates.corr(),annot=False,cmap='copper_r')


# In[ ]:


#Scatterplot comparing the Area and Perimeter based on each class
plt.figure(figsize=(16,8))
sns.scatterplot(x=dates['AREA'],y=dates['PERIMETER'],hue=dates['Class'],palette='copper_r',legend='auto')


# In[ ]:


#Count of each class
sns.countplot(data=dates, x='Class', palette='copper_r')


# In[ ]:


#Barplot for the mean of area of each class
sns.barplot(x=dates['Class'],y=dates['AREA'],palette='copper_r')


# In[ ]:


#Scatterplot for the area of each class
sns.scatterplot(x=dates['AREA'],y=dates['Class'],hue=dates['Class'],palette='copper_r')


# In[ ]:


#Scatterplot for the perimeter of each class
sns.scatterplot(x=dates['PERIMETER'],y=dates['Class'],hue=dates['Class'],palette='copper_r')


# In[ ]:


#Printing swarm and boxplot in one plot based on the following features
feat = ["AREA","PERIMETER","MAJOR_AXIS","MINOR_AXIS","EQDIASQ","SOLIDITY", "CONVEX_AREA"]
for feature in feat: 
    plt.figure(figsize=(12,6))
    sns.swarmplot(x=dates["Class"], y=dates[feature], color="black", alpha=0.7)
    sns.boxplot(x=dates["Class"], y=dates[feature], palette='copper_r')
    plt.show()


# # Dropping outliers

# In[ ]:


dates = dates[dates['SOLIDITY'] > 0.93]


# In[ ]:


dates = dates[dates['MAJOR_AXIS'] < 1000]


# # Feature Selection

# In[ ]:


Features = dates.drop('Class',axis=1)
Label = dates['Class']


# In[ ]:


#Features to drop based on feature importance figure down
todrop = ['EXTENT','COMPACTNESS','StdDevRR','EntropyRG','KurtosisRB','ECCENTRICITY','MeanRB','SkewRR',
           'ASPECT_RATIO','SOLIDITY','SHAPEFACTOR_3','SHAPEFACTOR_4','KurtosisRR']
Features.drop(todrop,axis=1,inplace=True)


# In[ ]:


# Scaling the data using standard scaler
scaler = StandardScaler()
scaler.fit(Features)
scaled = scaler.transform(Features)


# In[ ]:


#Train test split
X = scaled
y = Label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


# Show which features has the most effect on our results so we can modify and tune our features
# I used Random Forest Classifier to determine the feature importances

plt.figure(figsize=(10,10))
rfc = RandomForestClassifier(n_estimators=70,max_features='auto', max_depth=8, n_jobs=-1)
rfc.fit(X_train,y_train)
rfcpred = rfc.predict(X_test)
importance = pd.Series(rfc.feature_importances_,index=Features.columns)
importance.nsmallest(15).plot(kind='bar',color='brown')
plt.show()


# # Training Models

# ### KNN

# In[ ]:


err_rate = [] # Array to save all error rates

for i in range(1,40): # loop to try all error rates from 1 to 40
    knn = KNeighborsClassifier(n_neighbors=i) # create a knn object with number of neighbours with value i
    knn.fit(X_train,y_train) # fit the model
    pred_i = knn.predict(X_test) # predict the value
    err_rate.append(np.mean(pred_i != y_test)) #add the value to the array
    
    # Plotting the value of k error rate using the method we created above to make it easier to choose a k value
plt.figure(figsize=(20,10)) # size of the figure
plt.plot(range(1,40),err_rate,color='blue',linestyle='dotted',marker='o',markerfacecolor='red',markersize=8)#plotting the values
plt.title = 'K Values VS Error Rates' #title
plt.xlabel = 'K Value' #x label
plt.ylabel= 'Error Rate' # y label
plt.show()


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=4,p=2,n_jobs=-1)
knn.fit(X_train,y_train)
knnpred = knn.predict(X_test)
print('KNN Classifier' + '\n')
print(classification_report(y_test,knnpred))
sns.heatmap(confusion_matrix(y_test,knnpred),cmap='copper',annot=True,linewidths=2,linecolor='white')


# ### SVC

# In[ ]:


sup = SVC(C=10,gamma='auto',kernel='rbf')
sup.fit(X_train,y_train)
svcpred = sup.predict(X_test)
print('Support Vector Classifier' + '\n')
print(classification_report(y_test,svcpred))
sns.heatmap(confusion_matrix(y_test,svcpred),cmap='copper',annot=True,linewidths=2,linecolor='white')


# ### Logistic regression

# In[ ]:


logr = LogisticRegression(C=1,max_iter=150,multi_class='auto')
logr.fit(X_train,y_train)
logpred = logr.predict(X_test)
print('Logistic Regression' + '\n')
print(classification_report(y_test,logpred))
sns.heatmap(confusion_matrix(y_test,logpred),cmap='copper',annot=True,linewidths=2,linecolor='white')


# ### Random forest classifier

# In[ ]:


err = [] # Array to save all error rates

for i in range(1,20): # Loop to try all error rates from 1 to 20
    rfe = RandomForestClassifier(n_estimators=i*10) # Create rfc with number of estimators with value i*10
    rfe.fit(X_train,y_train) # Fit the model
    errpred = rfe.predict(X_test) # Predict the value
    err.append(np.mean(errpred != y_test)) #Add the value to the array
    
    
# Plotting the value of estimators error rate using the method we created above to make it easier to choose an estimator value
plt.figure(figsize=(20,10)) # Size of the figure
plt.plot(range(1,20),err,color='blue',linestyle='dotted',marker='x',markerfacecolor='red',markersize=10)#plotting the values
plt.title = 'Number of estimators VS Error Rates' #title
plt.xlabel = 'Estimators' #X label
plt.ylabel= 'Error Rate' # Y label
plt.show()


# In[ ]:


rfc = RandomForestClassifier(n_estimators=70,max_features='auto',max_depth=8)
rfc.fit(X_train,y_train)
rfcpred = rfc.predict(X_test)
print('Random Forest' + '\n')
print(classification_report(y_test,rfcpred))
sns.heatmap(confusion_matrix(y_test,rfcpred),cmap='copper',annot=True,linewidths=2,linecolor='white')


# # Gradient Boosting Classifier

# In[ ]:


err = [] # Array to save all error rates

for i in range(1,20): # Loop to try all error rates from 1 to 20
    clf = GradientBoostingClassifier(n_estimators=i*10, learning_rate=1.0,max_depth=1)
    clf.fit(X_train,y_train)
    errpred = clf.predict(X_test)
    err.append(np.mean(errpred != y_test)) #Add the value to the array
    
    
# Plotting the value of estimators error rate using the method we created above to make it easier to choose an estimator value
plt.figure(figsize=(20,10)) # Size of the figure
plt.plot(range(1,20),err,color='blue',linestyle='dotted',marker='*',markerfacecolor='red',markersize=10)#plotting the values
plt.title = 'Number of estimators VS Error Rates' #title
plt.xlabel = 'Estimators' #X label
plt.ylabel= 'Error Rate' # Y label
plt.show()


# In[ ]:


GBC = GradientBoostingClassifier(n_estimators=80, learning_rate=1.0,max_depth=1)
GBC.fit(X_train,y_train)
GBCpred = clf.predict(X_test)


# In[ ]:


print('Gradient Boosting' + '\n')
print(classification_report(y_test,GBCpred))
sns.heatmap(confusion_matrix(y_test,GBCpred),cmap='copper',annot=True,linewidths=2,linecolor='white')


# ### Thank you !, feel free to add any notes or comments
