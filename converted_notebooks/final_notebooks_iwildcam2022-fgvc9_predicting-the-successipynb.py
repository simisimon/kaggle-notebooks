#!/usr/bin/env python
# coding: utf-8

# 
# # Obtaining and Understanding Data

# ## Importing Libraries

# In[ ]:


#Downloading the required version of plotly for graphs
get_ipython().system('pip install plotly==4.8.2')


# In[ ]:


#Importing the requied packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import time
from pandas import Grouper
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score


# ## Loading dataframe

# In[ ]:


#Reading the data file
ks18 = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv', encoding="ISO-8859-1", low_memory=False)
ks18.sample(15)


# ## Basic Information of Data

# In[ ]:


#Checking original shape of the dataframe 
ks18.shape


# In[ ]:


#Checking Columns of the dataframe
ks18.columns


# In[ ]:


#Displaying information of the dataset
ks18.info()


# # Data Visualisation

# In[ ]:


#Creating a df for generating graph
df_state = ks18[ks18['state'].isin(['failed', 'successful'])]
df_state.sample(5)


# In[ ]:


#Extracting the year of the deadline and launched column
df_state['deadline_year'] = pd.DatetimeIndex(df_state['deadline']).year
df_state['launched_year'] = pd.DatetimeIndex(df_state['launched']).year
df_state.head()


# In[ ]:


#Converting the datatype of deadline and launched column
df_state['deadline'] = pd.to_datetime(df_state['deadline'])
df_state['launched'] = pd.to_datetime(df_state['launched'])

#Generating the difference of deadline and launched
df_state['launch_to_deadline'] =  df_state['deadline'] - df_state['launched'] 
df_state.head()


# In[ ]:


#Writing the dataframe into csv
df_state.to_csv('df_state_days.csv')


# In[ ]:


#Reading the data
df_state_days = pd.read_excel('../input/kickstarter-projects/df_state_days.csv')
df_state_days.head()


# In[ ]:


#Plot for number of projects based on duration
def return_week_bins(val):
    if val < 7:
        return "< 1 week"
    elif val < 14:
        return "1-2 weeks"
    elif val < 28:
        return "2-4 weeks"
    elif val < 35:
        return "4-5 weeks"
    elif val < 42:
        return "5-6 weeks"
    elif val < 56:
        return "6-8 weeks"
    else:
        return "8+ weeks"

successful_state_series = df_state_days[df_state_days['state'] == "successful"]['launch_to_deadline_days']
successful_state_series.map(return_week_bins).value_counts().plot(kind='bar')


# In[ ]:


#Plotting a graph for checking number of projects in each category which are successful or failed
plt.figure(figsize=(20,20))
sns.catplot(x='main_category', hue='state', kind='count', data=df_state.sort_values('main_category'), palette=sns.color_palette(['coral','cornflowerblue']))
plt.xticks(rotation = 90)
plt.title("Count and Status for different categories")
plt.xlabel('Category')
plt.ylabel('Number of Projects')
plt.show()


# In[ ]:


#Graph for displaying information about each category based on country
px.sunburst(ks18, values= 'usd_pledged_real' , path = ["country","main_category"], 
            color_continuous_scale='RdBu',width=750, height=750,title="USD Pledged Amount Per Category Per Country")


# In[ ]:


#Converting the datatype of the launched column
df_state.launched = pd.to_datetime(df_state.launched)
df_state['launched(y)']=pd.to_datetime(df_state.launched).dt.year 


# In[ ]:


#Plotting project count against the years
plt.figure(figsize=(10,5))
df_state.groupby("launched(y)").main_category.count().plot(kind = 'bar', color = 'c')
plt.title("Number of projects Per Year")
plt.ylabel('Number of Projects')
plt.xlabel('Launch Year')
plt.show()


# In[ ]:


#Getting the count based on state
successful_kickstarter = df_state_days[df_state_days['state'] == "successful"]
failed_kickstarter = df_state_days[df_state_days['state'] == "failed"]
cancelled_kickstarter = df_state_days[df_state_days['state'] == "canceled"]
print(len(successful_kickstarter)," successful campaigns")
print(len(failed_kickstarter)," failed campaigns")
print(len(cancelled_kickstarter)," cancelled campaigns")


# In[ ]:


#Plot for Distribution of time between launch and deadline for successful and failed campaigns
n = 'launch_to_deadline_days'
compare_L2D = pd.concat([df_state_days[n].apply(return_week_bins).value_counts(),successful_kickstarter[n].apply(return_week_bins).value_counts(),failed_kickstarter[n].apply(return_week_bins).value_counts()], axis=1)
compare_L2D.columns = ['all','successful','failed']
compare_L2D.loc['total'] = compare_L2D.sum(axis=0)

def get_perc(val, col):
    return (float(val) / float(compare_L2D[col]['total']))*100

compare_L2D['all_perc'] = compare_L2D['all'].apply(get_perc, args=('all',))
compare_L2D['success_perc'] = compare_L2D['successful'].apply(get_perc, args=('successful',))
compare_L2D['failed_perc'] = compare_L2D['failed'].apply(get_perc, args=('failed',))
#compare_L2D['cancelled_perc'] = compare_L2D['cancelled'].apply(get_perc, args=('cancelled',))

compare_L2D[['success_perc','failed_perc']][:6].sort_index().plot(kind='bar', figsize=(8,8))
plt.xticks(size = 12 )
plt.yticks(size = 12 )
plt.title("Distribution of time between launch and deadline for successful and failed campaigns", fontdict = {'fontsize' : 12})
plt.show()


# In[ ]:


#Plot for project count based on main category
df_state_days['main_category'].value_counts().plot(kind = 'bar', figsize=(8,8))


# # Data preprocessing

# ## Checking Null Values

# In[ ]:


# Checking for null values in all the columns
ks18.isnull().sum()


# In[ ]:


# Checking null values in 'name' column
ks18[ks18['name'].isnull()]


# ## Filling Null Values

# In[ ]:


#Filling the null values with Dummy name
ks18['name'].fillna('Dummy', inplace = True)


# In[ ]:


#Rechecking the null values
ks18.isnull().sum()


# First, we considered to separate the values based on usd pledged column but then in the end we dropped this column, Since, the corrected values are presened in the usd_pledged_real column. Therefore, commenting the code

# In[ ]:


# #Checking null values in 'usd pledged' column
# df_usd_pl_null = ks18[ks18['usd pledged'].isnull()]
# df_usd_pl_null


# In[ ]:


# df_usd_pl_null[df_usd_pl_null.state != 'undefined']


# In[ ]:


# df_usd_pl_not_null = ks18[~ks18['usd pledged'].isnull()]


# In[ ]:


# df_usd_pl_not_null.isnull().sum()


# In[ ]:


# df_usd_pl_not_null.dtypes


# In[ ]:


# df_usd_pl_not_null[df_usd_pl_not_null['usd pledged'] != df_usd_pl_not_null['usd_pledged_real']].sample(10)


# ## Droppping Unnecessary Columns

# In[ ]:


# Droppping columns = 'name' & 'category' 
# Dropping 'usd pledged' as the correct data is recorded in 'usd_pledged_real'
 
ks18.drop(columns=['ID','name','category','usd pledged','currency','goal','pledged'], axis = 1, inplace = True)


# In[ ]:


#Printing the sample of the dataframe
ks18.sample(10)


# In[ ]:


#Crosschecking for null values
ks18.isnull().sum()


# ## Cleaning the Target Variable

# In[ ]:


#Determing the unique values of state column
ks18.state.unique()


# In[ ]:


#Printing the shape of the dataframe
ks18.shape


# In[ ]:


# Discarding records for 'live', 'canceled', 'undefined', 'suspended' projects

ks18 = ks18[ks18['state'].isin(['failed', 'successful'])]
ks18.sample(10)


# ## Converting the Categorical Variables

# In[ ]:


#Assigning Successful as 1 and failed as 0
ks18 = ks18.assign(state_output = (ks18['state'] == 'successful').astype(int))
ks18


# In[ ]:


#Dropping the converted column
ks18.drop(columns = ['state'], inplace = True)
ks18


# In[ ]:


#Label encoding the Country Column
country_encoder= LabelEncoder().fit(ks18['country'])
ks18['country'] = country_encoder.transform(ks18['country'])

#Label Encoding the Main Category Column
main_category_encoder= LabelEncoder().fit(ks18['main_category'])
ks18['main_category'] = main_category_encoder.transform(ks18['main_category'])

ks18.sample(10)


# In[ ]:


#Parsing deadline and launched date for better results
ks18.deadline = pd.to_datetime(ks18.deadline)
ks18.launched = pd.to_datetime(ks18.launched)
#Number of days the campaign was running
ks18['duration(days)'] = (ks18['deadline'] - ks18.launched).dt.days 
#Deadline year of the campaign 
ks18['deadline(y)']=pd.to_datetime(ks18.deadline).dt.year 
#Deadline month of the campaign
ks18['deadline(m)']=pd.to_datetime(ks18.deadline).dt.month 
#Launched year of the campaign
ks18['launched(y)']=pd.to_datetime(ks18.launched).dt.year 
#Launched month of the campaign
ks18['launched(m)']=pd.to_datetime(ks18.launched).dt.month


# In[ ]:


#Dropping the parsed columns
ks18.drop(['deadline','launched'], axis = 1, inplace=True)
ks18


# ## Defining Dependent and Independent Variables

# In[ ]:


#Defining X as dataframe for Independent features
X = ks18.drop("state_output",axis=1)
#Defining y as Dependent Variable
y = ks18["state_output"] 


# In[ ]:


print(X)


# In[ ]:


print(y)


# ## Handling the Imbalance in the Data

# In[ ]:


#Using smote to balance the data
smt = SMOTE(random_state=0)
X_train_sm , y_train_sm = smt.fit_resample(X, y)


# ## Splitting the dataset into the Training set and Test set

# In[ ]:


#Splitting the dataset into train and test set with the ratio of 80:20

X_train, X_test, y_train, y_test = train_test_split(X_train_sm , y_train_sm, test_size = 0.2, random_state = 0)


# In[ ]:


print(X_train)


# In[ ]:


print(y_train)


# In[ ]:


# Checking Correlations Heatmap for X_train variables
plt.subplots (figsize = (16,9))
sns.heatmap (X_train.corr(), square = True, cbar = True, annot = True, 
             annot_kws = {'size': 10}, fmt = '0.2f',linewidths=.5, cmap='Blues')
plt.show()


# In[ ]:


#Selecting the columns based on correlation values. Eleminating the columns with higher correlation
corr = X_train.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False

selected_columns = X_train.columns[columns]
selected_columns


# In[ ]:


#Selecting the columns which are required
X_train = X_train[selected_columns]
X_test = X_test[selected_columns]


# In[ ]:


X_train


# ## Scaling the Features

# In[ ]:


# Applying feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


print(X_train)


# In[ ]:


print(X_test)


# # Model Implementation

# ## Decision Tree Model

# In[ ]:


#Checking the optimal number of parameters to avoid overfitting
param_grid = {
    "max_depth": [3,5,10,15,20,None],
    "min_samples_split": [2,5,7,10],
    "min_samples_leaf": [1,2,5]
}

clf = DecisionTreeClassifier(random_state=42)
grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc", n_jobs=-1, cv=3).fit(X_train, y_train)

print("Param for GS", grid_cv.best_params_)
print("CV score for GS", grid_cv.best_score_)
print("Train AUC ROC Score for GS: ", roc_auc_score(y_train, grid_cv.predict(X_train)))
print("Test AUC ROC Score for GS: ", roc_auc_score(y_test, grid_cv.predict(X_test)))


# In[ ]:


#Modeling with obtained parameters
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2)

#Fitting the Model
dt_model.fit(X_train, y_train)


# In[ ]:


#Predicting the test set values
y_pred_dt = dt_model.predict(X_test)


# In[ ]:


#Predicting the train set for calculating the accuracy
y_pred_train_dt = dt_model.predict(X_train)


# In[ ]:


# Confusion Matrix for evaluating the model
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_dt))
print("Accuracy Score for Test Set: ",accuracy_score(y_test, y_pred_dt)) 


# In[ ]:


#Printing the accuracy for the training data
print("Accuracy Score for Training Set: ",accuracy_score(y_train, y_pred_train_dt))


# In[ ]:


#Calculating the mean squared error for training set
training_error = mean_squared_error(y_train, y_pred_train_dt)
training_error


# In[ ]:


#Calculating the mean squared error for test set
test_error = mean_squared_error(y_test, y_pred_dt)
test_error


# ## XG Boost

# In[ ]:


#Creating an object of XGBoost Classifier
xgbc = XGBClassifier()

#Fitting the model on the training set
xgbc.fit(X_train, y_train)


# In[ ]:


#Predicting the test set
y_pred_xgb = xgbc.predict(X_test)


# In[ ]:


#Predicting for Training Set
y_pred_train_xgb = xgbc.predict(X_train)


# In[ ]:


# Confusion Matrix for evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_xgb))
print("Accuracy Score for Test Set: ",accuracy_score(y_test, y_pred_xgb))  


# In[ ]:


print("Accuracy Score for Training Set: ",accuracy_score(y_train, y_pred_train_xgb))


# ## K-Nearest Neighbors

# In[ ]:


#Checking the optimal K value
error_rate = []
for i in range(1,15):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


#Plotting the error rate for each K value
plt.figure(figsize=(10,6))
plt.plot(range(1,15),error_rate,color='blue', linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
req_k_value = error_rate.index(min(error_rate))+1
print("Minimum error:-",min(error_rate),"at K =",req_k_value)


# Commenting the code because it takes a lot of time.

# In[ ]:


# #List Hyperparameters to tune
# from sklearn.model_selection import GridSearchCV
# leaf_size = list(range(1,10))
# n_neighbors = list(range(1,15))
# p=[1,2]
# #convert to dictionary
# hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
# #Making model
# clf = GridSearchCV(knn, hyperparameters, cv=10)
# best_model = clf.fit(X_train,y_train)
# #Best Hyperparameters Value
# print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
# print('Best p:', best_model.best_estimator_.get_params()['p'])
# print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
# #Predict testing set
# y_pred = best_model.predict(X_test)
# #Check performance using accuracy
# print(accuracy_score(y_test, y_pred))
# #Check performance using ROC
# roc_auc_score(y_test, y_pred)


# In[ ]:


#Creating the Object for KNN
KNclassifier = KNeighborsClassifier(n_neighbors = 3)
#Fitting the data
KNclassifier.fit(X_train, y_train)


# In[ ]:


#Predicting the Test Set
y_pred_KN = KNclassifier.predict(X_test)


# In[ ]:


#Predicting the Train set
y_pred_train_KN = KNclassifier.predict(X_train)


# In[ ]:


# Confusion Matrix for evaluating the model
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_KN))
print("Accuracy Score for Test Set: ",accuracy_score(y_test, y_pred_KN))


# In[ ]:


#Printing accuracy for Training set
print("Accuracy Score for Training: ",accuracy_score(y_train, y_pred_train_KN))


# ## Logistic Regression

# In[ ]:


#Using Logistic Regression
lr = LogisticRegression()

#Training the model
lr.fit(X_train, y_train)


# In[ ]:


#Predicting the test set
y_pred_LR = lr.predict(X_test)


# In[ ]:


#Predicting the training set
y_pred_train_lr = lr.predict(X_train)


# In[ ]:


# Confusion Matrix for evaluating the model
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_LR))
print("Accuracy Score for Test Set: ",accuracy_score(y_test, y_pred_LR))


# In[ ]:


#Printing accuracy for training set
print("Accuracy Score for Training: ",accuracy_score(y_train, y_pred_train_lr))


# ## Random Forest Regression

# In[ ]:


#Using Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)


# In[ ]:


#Predicting the values for test set
y_pred_rf = random_forest.predict(X_test)


# In[ ]:


#Predicting for train set
y_pred_train_rf = random_forest.predict(X_train)


# In[ ]:


#Confusion Matrix for evaluating the model
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_rf))
print("Accuracy Score for Test Set: ",accuracy_score(y_test, y_pred_rf))


# In[ ]:


#Printing accuracy for training set
print("Accuracy Score for Training: ",accuracy_score(y_train, y_pred_train_rf))


# ## Naive Bayes Classifier 

# In[ ]:


#Using Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)


# In[ ]:


#Predicting the values for testing set
y_pred_nb = nb.predict(X_test)


# In[ ]:


#Predicting the values for train set
y_pred_train_nb = nb.predict(X_train)


# In[ ]:


#Confusion Matrix for evaluating the model
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_nb))
print("Accuracy Score: ",accuracy_score(y_test, y_pred_nb))  


# In[ ]:


#Printing accuracy for training set
print("Accuracy Score for Training: ",accuracy_score(y_train, y_pred_train_nb))


# ## AdaBoost Classifier

# In[ ]:


#Creating an object
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Train Adaboost Classifer
abc.fit(X_train, y_train)


# In[ ]:


#Predict the response for test dataset
y_pred_abc = abc.predict(X_test)


# In[ ]:


#Predict the response for train dataset
y_pred_train_abc = abc.predict(X_train)


# In[ ]:


# Confusion Matrix for evaluating the model
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_abc))
print("Accuracy Score: ",accuracy_score(y_test, y_pred_abc))  


# In[ ]:


#Printing accuracy for training set
print("Accuracy Score for Training: ",accuracy_score(y_train, y_pred_train_abc))


# ## Support Vector Machine

# Commented the code for SVM, as it takes time to execute.

# In[ ]:


# from sklearn.svm import SVC
# from sklearn.model_selection import cross_validate
# from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


# In[ ]:


# classifier = SVC(kernel = 'linear', random_state = 0)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)


# In[ ]:


# SVM = SVC(probability = True)
# scoring = ['accuracy','precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
# scores = cross_validate(SVM, X_train, y_train, scoring=scoring, cv=20)

# sorted(scores.keys())
# SVM_fit_time = scores['fit_time'].mean()
# SVM_score_time = scores['score_time'].mean()
# SVM_accuracy = scores['test_accuracy'].mean()
# SVM_precision = scores['test_precision_macro'].mean()
# SVM_recall = scores['test_recall_macro'].mean()
# SVM_f1 = scores['test_f1_weighted'].mean()
# SVM_roc = scores['test_roc_auc'].mean()


# # Evaluation Metrics

# ## Accuracy

# In[ ]:


print("Accuracy Score for KNN: ",accuracy_score(y_test, y_pred_KN))
print("Accuracy Score for Logistic Regression: ",accuracy_score(y_test, y_pred_LR))
print("Accuracy Score for AdaBoost Classifier: ",accuracy_score(y_test, y_pred_abc)) 
print("Accuracy Score for Decision Tree Classifier: ",accuracy_score(y_test, y_pred_dt)) 


# In[ ]:


accuracy_KN = accuracy_score(y_test, y_pred_KN)
accuracy_LR = accuracy_score(y_test, y_pred_LR)
accuracy_abc = accuracy_score(y_test, y_pred_abc)
accuracy_dt = accuracy_score(y_test, y_pred_dt)


# ## Confusion Matrix

# In[ ]:


#Function to calculate the confusion matrix and plot it for each model.

def get_conf_matrix(y_pred, model_name):
  conmat = confusion_matrix(y_test, y_pred)
  val = np.mat(conmat) 

  classnames = list(set(y_train))

  df_cm = pd.DataFrame(

          val, index=classnames, columns=classnames, 

      )

  print(df_cm)

  plt.figure()

  heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues")

  heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')

  heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')

  plt.ylabel('True label')

  plt.xlabel('Predicted label')

  plt.title(model_name+' Model Results')

  plt.show()


# In[ ]:


#Confusion Matrix for Decision Tree
get_conf_matrix(y_pred_dt, 'Decision Tree ')


# In[ ]:


#Confusion Matrix for ADABoost Classifier
get_conf_matrix(y_pred_abc, 'ADABoost Classifier ')


# In[ ]:


#Confusion Matrix for KNN
get_conf_matrix(y_pred_KN, 'KNN ')


# In[ ]:


#Confusion Matrix for Logistic Regression
get_conf_matrix(y_pred_LR, 'Logistic Regression ')


# ## F1, Recall, Precision Score

# In[ ]:


#Function to get the different scores for each model
def get_scores(y_prediction):
  f1 = f1_score(y_test, y_prediction)
  recall = recall_score(y_test, y_prediction)
  precision = precision_score(y_test, y_prediction)
  return f1, recall, precision


# In[ ]:


#Scores for Decision Tree
f1_dt, recall_dt, precision_dt = get_scores(y_pred_dt)
print('For Decision Tree : \n')
print ('F1 score:',f1_dt )
print ('Recall:',recall_dt)
print ('Precision:', precision_dt)


# In[ ]:


#Scores for ADABoost Classifier
f1_abc, recall_abc, precision_abc = get_scores(y_pred_abc)
print('For Adaptive Boosting : \n')
print ('F1 score:',f1_abc )
print ('Recall:',recall_abc)
print ('Precision:', precision_abc)


# In[ ]:


#Scores for KNN
f1_KN, recall_KN, precision_KN = get_scores(y_pred_KN)
print('For KNN : \n')
print ('F1 score:',f1_KN)
print ('Recall:',recall_KN)
print ('Precision:', precision_KN)


# In[ ]:


#Scores for Logistic Regression
f1_LR, recall_LR, precision_LR = get_scores(y_pred_LR)
print('For Logistic Regression : \n')
print ('F1 score:',f1_LR)
print ('Recall:',recall_LR)
print ('Precision:', precision_LR)


# ## Plot for Accuracy

# In[ ]:


#Plot for Accuracy
x = ['KNN','Logistic Regression','ADABoost','Decision Tree']
y = [accuracy_KN,accuracy_LR,accuracy_abc,accuracy_dt]

plt.plot(x,y, marker = 'o' )
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Score Comparison for Different Models')
plt.show()


# ## Plot for Precision

# In[ ]:


#Plot for Precision
x = ['KNN','Logistic Regression','ADABoost','Decision Tree']
y = [precision_KN, precision_LR, precision_abc, precision_dt]

plt.plot(x,y, marker = 'o' )
plt.xlabel('Models')
plt.ylabel('Precision Score')
plt.title('Precision Score Comparison for Different Models')
plt.show()


# ## Plot for Recall

# In[ ]:


#Plot for Recall
x = ['KNN','Logistic Regression','ADABoost','Decision Tree']
y = [recall_KN, recall_LR, recall_abc, recall_dt]

plt.plot(x,y, marker = 'o' )
plt.xlabel('Models')
plt.ylabel('Recall Score')
plt.title('Recall Score Comparison for Different Models')
plt.show()


# ## Plot for F1 Score

# In[ ]:


#Plot for F1
x = ['KNN','Logistic Regression','ADABoost','Decision Tree']
y = [f1_KN, f1_LR, f1_abc, f1_dt]

plt.plot(x,y, marker = 'o' )
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison for Different Models')
plt.show()

