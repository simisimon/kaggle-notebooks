#!/usr/bin/env python
# coding: utf-8

# # Introduction
# #### In this notebook, we dived into **Exploratory Data Analysis** for the stroke prediction dataset.
# 
# #### We also **built Supervised Learning and Unsupervised Learning models** and evaluated their performance. Finally, we made some **interesting conclusions.**
# 
# #### Let's start!
# (Kindly **upvote my notebook** if you find it helpful. Honest feedback is welcome)

# # **Import libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# # **Load the data**

# In[ ]:


data = pd.read_csv('../input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv')
data.head(20)


# In[ ]:


data.info()


# In[ ]:


data.describe()
#some values are missing for bmi
#age values has decimals
#we need to do some data preprocessing


# # **Data preprocessing**

# **Replace NAN values in 'BMI' column with mean of column (BMI) values**

# In[ ]:


mean_value = data['bmi'].mean()
data['bmi'].fillna(value = mean_value, inplace = True) 


# **Approximate age values to the nearest whole number**

# In[ ]:


data['age'] = (data['age'].apply(np.ceil)).astype('int64')


# **Delete single data with gender 'other'**

# In[ ]:


data.drop(data[data.gender == 'Other'].index, inplace = True)


# **Delete data with smoking_status 'unknown'**

# In[ ]:


data.drop(data[data.smoking_status == 'Unknown'].index, inplace = True)


# In[ ]:


data.tail(20)


# # Encode data

# - ever_married (0-no, 1-yes)
# - Residence_type (0 - rural, 1-urban)
# - gender (0 - female, 1-male)
# - smoking status (formally_smoked-0, never_smoked-1, smokes-2)
# - work_type (Govt_job-0, never_worked-1 private-2,  self_employed-3,  children-4)

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
data['gender'] = data[['gender']].apply(le.fit_transform)
data['ever_married'] = data[['ever_married']].apply(le.fit_transform)
data ['Residence_type'] = data[['Residence_type']].apply(le.fit_transform)
data['work_type'] = data[['work_type']].apply(le.fit_transform)
data ['smoking_status'] = data[['smoking_status']].apply(le.fit_transform)


# In[ ]:


data.tail(20)


# In[ ]:


#data.describe()


# # **Exploratory Data Analysis (EDA)**

# ## **Visualize the dataset**

# ### **Stroke vs No-stroke Observations**
# 
# Observation: The dataset is imbalanced for stroke and no_stroke observations.
# 
# The ratio for Stroke Vs NoStroke is 202:3363 which is approximately 1:17
# i.e 1 in 17 people are affected by stroke in the dataset

# In[ ]:


#Split the data into those with stroke and without stroke
no_stroke = data[data['stroke'] == 0] #extract the information of those without stroke
no_stroke_extract = no_stroke['stroke'] #extract one column 

yes_stroke = data[data['stroke'] == 1] #extract the information of those with stroke
yes_stroke_extract = yes_stroke['stroke'] #extract one column

count = [yes_stroke_extract.count(), no_stroke_extract.count()]
labels = ('stoke', 'no stroke')

#plot stroke vs no stroke observations
plt.bar(labels, count)
print ("number with stroke: ",yes_stroke_extract.count())
print ("number without stroke: ",no_stroke_extract.count())
print ("\n")


# In[ ]:


#same plot as above: Using seaborn to visualize stroke vs no stroke observations
import seaborn as sns
g = sns.countplot(data['stroke'])
g.set_xticklabels(['no stroke','stroke'])
plt.show()


# ### **Feature analysis**

# In[ ]:


yes_stroke.describe() #observations with stroke
#minimum age is 32


# In[ ]:


no_stroke.describe() #observations without stroke


# **Age vs stroke**

# In[ ]:


#Stroke chances increases with age, starting from age 32 
sns.lineplot(data=data, x="stroke", y="age")
sns.catplot(data=data, kind = "swarm", x="stroke", y="age")


# **Gender vs stroke**

# In[ ]:


#The dataset have a good gender distribution
sns.catplot(data=data, kind="bar", x="gender", y="stroke") 


# **Body health statistics vs stroke**

# In[ ]:


#bmi, stroke
#plot shows that people with stroke have higher bmi values

sns.lineplot(data=data, x="stroke", y="bmi", hue="stroke") 


# In[ ]:


#line plot for stroke, bmi vs heart diesase
sns.lineplot(data=data, x="bmi", y="heart_disease", hue='stroke')


# **Disease vs stroke**

# In[ ]:


#hypertension and heart disease increases the risk of an individual having a stroke
sns.catplot(data=data, kind="bar", x="stroke", y="heart_disease", hue="stroke") #heart disease vs stroke
sns.catplot(data=data, kind="bar", x="stroke", y="hypertension", hue="stroke") #hypertension vs stroke


# **Habits vs stroke**

# In[ ]:


#Those who never smoked have almost equal observations with current smokers. We can safely conclude that smoking status has little or no risk effect for stroke
sns.catplot(data=data, kind="bar", x="smoking_status", y="stroke")


# In[ ]:


#smoking status vs age
#plot tells us there are a number of people below 30 who smoke, but they do not have stroke

sns.catplot(data=data, kind="swarm", x="smoking_status", y="age", hue = "stroke")


# In[ ]:


#avg_glucose_level, smoking status, stroke 
sns.catplot(data=data, kind="swarm", x="smoking_status", y="avg_glucose_level", hue = "stroke")


# **Other features analysis**

# In[ ]:


#ever married, age and stroke
#plot shows that a higher number of married people have stroke compared to the single folks
sns.catplot(data=data, kind="swarm", x="ever_married", y="age", hue="stroke") 


# In[ ]:


sns.catplot(data=data, kind="bar", x="ever_married", y="stroke") #ever married, age, stroke


# In[ ]:


#residence type, gender, stroke
#plot show that residence type have no effect on the risk of stroke

sns.catplot(data=data, kind="bar", x="Residence_type", y="gender", hue="stroke")


# In[ ]:


#residence type, work type, stroke
#work type private (encoded as 1) have no observations for those who have stroke, the rest shows pretty similar plots. 
#We could assume two things. The first is there was no data collected for private workers who had stroke. 
#Secondly, there might be the possibility that the risk of private workers having a stroke is low

sns.catplot(data=data, kind="bar", x="work_type", y="Residence_type", hue="stroke")


# ### **Visualize stroke cases only**

# In[ ]:


###Analysis of yes-stroke_cases 'actual stroke cases' by age and gender
#From age 30, there is an increased risk for stroke
#Women in their 30s and early 40s are likely to have stroke than men

#plot of age and gender
g = sns.catplot(data=yes_stroke, kind="swarm",x="stroke", y="age", hue="gender")
g.set_axis_labels("stroke cases", "age")
g.legend.set_title("0-female, 1-male")


# In[ ]:


sns.catplot(data=yes_stroke, kind="swarm",x="gender", y="age")


# In[ ]:


#a higher number of married people have stroke compared to the single folks
sns.catplot(data=yes_stroke, kind="swarm",x="hypertension", y="bmi", hue="ever_married")


# In[ ]:


#catplot of hypertension, age, gender
sns.catplot(data=yes_stroke, kind="swarm",x="hypertension", y="age", hue="gender")


# In[ ]:


#compared to hypertension, there are less number of stroke patients with heart disease
sns.catplot(data=yes_stroke, kind="swarm",x="heart_disease", y="age", hue="gender")


# ## **Conclusions of EDA**
# **From the dataset,**
# 
# - The risk of stroke increases with age, starting from age 30.
# 
# - Women in their early 30s have a higher risk for stroke compared to men. The risk for stroke for men starts from age 45.
# 
# - Hypertension and heart disease increases the risk of an individual having a stroke. However, there were more stroke cases with hypertension than with heart disease.
# 
# - Living conditions or work history does not affect the risk for stroke.
# 
# - Married people had higher risk of stroke than the unmarried ones
# 
# 
# 
# **Other risk factors apart from age and gender?** 
# - There were more observations of stroke cases for people who formally smoked and current smokers. Although those who had never smoked had almost equal observations with current smokers. We can safely conclude that smoking status has little or no risk effect for stroke.

# # **Sampling (Undersampling)**
# The records of the stroke cases compared to no stroke cases in the dataset shows an imbalance. To ensure optimal performance of the machine learning models, we would use undersampling technique (note that there are several other techniques to consider)

# **Delete the 'Children' observations**

# In[ ]:


#Extract the data containing children information
nos_of_children = data[data['work_type'] == 4]
nos_of_children_count = nos_of_children['work_type'].count()
print (nos_of_children_count)

nos_of_children.describe()
#There are 69 children in total in the dataset, in age range 10-16.


# In[ ]:


data.drop(data[data.work_type == 4].index, inplace = True) #delete the records


# In[ ]:


#split the data into those with stroke and without stroke
no_stroke = data[data['stroke'] == 0] #extract the information of those without stroke
no_stroke_extract = no_stroke['stroke'] #extract one column 

yes_stroke = data[data['stroke'] == 1] #extract the information of those with stroke
yes_stroke_extract = yes_stroke['stroke'] #extract one column

count = [yes_stroke_extract.count(), no_stroke_extract.count()]
labels = ('stoke', 'no stroke')

plt.bar(labels, count)
print ("number with stroke: ",yes_stroke_extract.count())
print ("number without stroke: ",no_stroke_extract.count())


# **Extract few data from the no-stroke observations**
# 
# We make the ratio between stroke and no-stroke observations in the dataset to be:1:4 
# i.e 1 in 4 people are affected by stroke in the sampled dataset

# In[ ]:


#undersampling
class_no_stroke_extract = no_stroke.sample(800) 
sampled_data = pd.concat([class_no_stroke_extract, yes_stroke], axis=0)

#shuffle the data
sampled_data = sampled_data.sample(frac = 1)


# In[ ]:


# visualize the output
g = sns.countplot(sampled_data['stroke'])
g.set_xticklabels(['no stroke','stroke'])
plt.show()


# # **Goal 1: Supervised Learning**
# - Build regression and classification models to predict a stroke
# - Evaluate the models
# 
#   

# # **A - Model (selected features) & Model Evaluation**
# - Regression model: linear regression
# - Classification models: logistic regression, multi-layer perceptron and KNN

# ## **Feature selection**
# 
# To acheive lower computational cost and improved model performance, feature selection is advised. 

# In[ ]:


#The selected features based on the feature analysis and visualizations
X = sampled_data[['age', 'hypertension', 'heart_disease', 'bmi', 'avg_glucose_level','smoking_status']] 
y = sampled_data['stroke']


# ## **Normalize the data**

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# ## **Split the dataset into train and test**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3) 


# # **Linear regression model**

# In[ ]:


from sklearn.linear_model import LinearRegression
linear_reg_model = LinearRegression(normalize=True) #model

linear_reg_model.fit(X_train,y_train)

train_predictions_linear = linear_reg_model.predict(X_train) 
test_predictions_linear = linear_reg_model.predict(X_test)   


# ### **Performance metrics - linear regression**

# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
print('Linear regression model evaluation:')
print('Precision: ', precision_score(y_test, test_predictions_linear.round()))
print('Recall: ', recall_score(y_test, test_predictions_linear.round()))

print('\nAccuracy: ',accuracy_score(y_test, test_predictions_linear.round()))
print('f1_score: ', f1_score(y_test, test_predictions_linear.round())) 


# In[ ]:


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_predictions_linear.round())
f = sns.heatmap(cm, annot=True, fmt='d')


# # **Logistic regression model**

# In[ ]:


from sklearn.linear_model import LogisticRegression
log_reg_model = LogisticRegression(max_iter=1000) #model
log_reg_model.fit(X_train,y_train) 

train_predictions_log = log_reg_model.predict(X_train) #train the model
test_predictions_log = log_reg_model.predict(X_test) #test the model


# ### **Performance metrics - logistic regression**

# In[ ]:


print('Logistic regression model evaluation:')
print('Precision: ', precision_score(y_test,test_predictions_log))
print('Recall: ', recall_score(y_test,test_predictions_log))
print('\nAccuracy: ',accuracy_score(y_test,test_predictions_log))
print('f1_score: ', f1_score(y_test,test_predictions_log)) 


# In[ ]:


#confusion matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(log_reg_model,X_test,y_test)


# # **Multi-Layer Perceptron (MLP) model**

# In[ ]:


from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(activation='logistic',solver='lbfgs',hidden_layer_sizes=(15),max_iter=8000) 
#solver can be sgd, lbfgs or adam; lbfgs performed better
mlp_model.fit(X_train, y_train) 


# In[ ]:


test_predictions_mlp = mlp_model.predict(X_test)
train_predictions_mlp = mlp_model.predict(X_train)


# ### **Performance metrics - MLP**

# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
print('MLP model evaluation:')
print('Precision: ', precision_score(y_test,test_predictions_mlp))
print('Recall: ', recall_score(y_test, test_predictions_mlp))
print('\nAccuracy: ',accuracy_score(y_test, test_predictions_mlp))
print('f1_score: ',f1_score(y_test, test_predictions_mlp))


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(mlp_model,X_test,y_test)


# # **KNN model**

# ### **KNN Model**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
test_predictions_knn =knn_model.predict(X_test)


# ### **Performance metrics - KNN**

# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
print('KNN model evaluation:')
print('Precision: ', precision_score(y_test,test_predictions_knn))
print('Recall: ', recall_score(y_test,test_predictions_knn))

print('\nAccuracy: ',accuracy_score(y_test,test_predictions_knn))
print('f1_score: ',f1_score(y_test,test_predictions_knn))


# In[ ]:


plot_confusion_matrix(knn_model,X_test,y_test)


# # **B - Model (all features) & Model Evaluation**

# In[ ]:


all_features_data = sampled_data.drop(columns='stroke') 
result = sampled_data['stroke']


# In[ ]:


from sklearn.preprocessing import StandardScaler
scalerr = StandardScaler()
scalerr.fit(all_features_data)
all_features_scaled = scalerr.transform(all_features_data)


# In[ ]:


X_training, X_testing, y_training, y_testing = train_test_split(all_features_scaled, result, test_size=0.3)  #same test size as the datasplit with selected features


# # **Linear regression model**

# In[ ]:


linear_reg_model_all = LinearRegression(normalize=True)
linear_reg_model_all.fit(X_training,y_training)

train_predictions_linear_all = linear_reg_model_all.predict(X_training) 
test_predictions_linear_all = linear_reg_model_all.predict(X_testing) 


# In[ ]:


#Evaluation metric
print('Linear regression model evaluation:')
print('Precision: ', precision_score(y_testing, test_predictions_linear_all.round()))
print('Recall: ', recall_score(y_testing, test_predictions_linear_all.round()))

print('\nAccuracy: ',accuracy_score(y_testing, test_predictions_linear_all.round()))
print('f1_score: ', f1_score(y_testing, test_predictions_linear_all.round()))
print ("\n") 

from sklearn.metrics import confusion_matrix
cma = confusion_matrix(y_testing, test_predictions_linear_all.round())
f = sns.heatmap(cma, annot=True, fmt='d')


# # **Logistic regression model**

# In[ ]:


log_reg_model_all = LogisticRegression(max_iter=1000) #model
log_reg_model_all.fit(X_training,y_training) 

train_predictions_log_all = log_reg_model_all.predict(X_training) 
test_predictions_log_all = log_reg_model_all.predict(X_testing) 

print('Logistic regression model evaluation:')
print('Precision: ', precision_score(y_testing,test_predictions_log_all))
print('Recall: ', recall_score(y_testing,test_predictions_log_all))
print('\nAccuracy: ',accuracy_score(y_testing,test_predictions_log_all))
print('f1_score: ', f1_score(y_testing,test_predictions_log_all))
print("\n") 

plot_confusion_matrix(log_reg_model_all,X_testing,y_testing)


# # **Multi-Layer Perceptron model**

# In[ ]:


mlp_model_all = MLPClassifier(activation='logistic',solver='lbfgs',hidden_layer_sizes=(15),max_iter=8000)
mlp_model_all.fit(X_training, y_training) 

test_predictions_mlp_all = mlp_model_all.predict(X_testing)
train_predictions_mlp_all = mlp_model_all.predict(X_training)

print('MLP model evaluation:')
print('Precision: ', precision_score(y_testing, test_predictions_mlp_all))
print('Recall: ', recall_score(y_testing, test_predictions_mlp_all))
print('\nAccuracy: ',accuracy_score(y_testing, test_predictions_mlp_all))
print('f1_score: ',f1_score(y_testing, test_predictions_mlp_all))
print("\n")

plot_confusion_matrix(mlp_model_all,X_testing,y_testing)


# # **KNN model**

# In[ ]:


knn_model_all = KNeighborsClassifier(n_neighbors=5)
knn_model_all.fit(X_training, y_training)
test_predictions_knn_all =knn_model_all.predict(X_testing)

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
print('KNN model evaluation:')
print('Precision: ', precision_score(y_testing,test_predictions_knn_all))
print('Recall: ', recall_score(y_testing,test_predictions_knn_all))

print('\nAccuracy: ',accuracy_score(y_testing,test_predictions_knn_all))
print('f1_score: ',f1_score(y_testing,test_predictions_knn_all))
print("\n")

plot_confusion_matrix(knn_model_all,X_testing,y_testing)


# # Goal 2: Unsupervised Learning
# - Build unsupervised models to predict a stroke
#   - KNN, DBSCAN
# - Evaluate the models

# # **KNN Model (Unsupervised)**

# In[ ]:


from sklearn.cluster import KMeans
knn_unsupervised_model = KMeans(n_clusters=2,init='random') #n_clusters selected as 2 since the classes in the labelled data are 2

#dataset with selected features
knn_unsupervised_model = knn_unsupervised_model.fit(X_scaled) 

#Get the cluster labels
clusters_knn = knn_unsupervised_model.predict(X_scaled) #assigned clusters i.e. predicted result


#dataset with all features
knn_unsupervised_model2 = KMeans(n_clusters=2,init='random') 
knn_unsupervised_model2 = knn_unsupervised_model2.fit(all_features_scaled) 
clusters_knn_2 = knn_unsupervised_model2.predict(all_features_scaled) #assigned clusters i.e. predicted result


# ### **Perfomance metrics**

# In[ ]:


#performance of the unsupervised KNN model (selected features)
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
print('KNN (UNSUPERVISED) model evaluation (selected features):')
print('Precision: ', precision_score(y,clusters_knn))
print('Recall: ', recall_score(y,clusters_knn))

print('\nAccuracy: ',accuracy_score(y,clusters_knn))
print('f1_score: ',f1_score(y,clusters_knn))
#performs really good on the recall metric


# In[ ]:


#confusion matrix for the unsupervised KNN model (selected features)
cm_uknn = confusion_matrix(y, clusters_knn)
f = sns.heatmap(cm_uknn, annot=True, fmt='d')


# In[ ]:


#performance of the unsupervised KNN model (all features)
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
print('KNN (UNSUPERVISED) model evaluation (all features):')
print('Precision: ', precision_score(y,clusters_knn_2))
print('Recall: ', recall_score(y,clusters_knn_2))

print('\nAccuracy: ',accuracy_score(y,clusters_knn_2))
print('f1_score: ',f1_score(y,clusters_knn_2))


# # **DBSCAN** 
# - not suitable for this dataset but we attempted it anyways

# In[ ]:


#DBSCAN model
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps = 2, min_samples = 5)
dbscan.fit(X_test)

clusters_dbscan = dbscan.labels_
clusters_dbscan
#DBSCAN is not suitable for this dataset as we cannot retrict the number of clusters to 2 (since we are working with labelled data); here it generated 3 classes - 0,1,2


# # **Conclusion**
# **Models Performance analysis:**
# 
# - Overall, the learning models performed better with fewer features when compared to the performance of the models built with all features. We can conclude that feature selection benefits learning models and the selected features adequately captures the information contained in the original set of features.
# 
# 
# **Model Selection**
# - Because of the sensitivity (i.e. health) of the dataset, recall and precision would be a better metric in the final choice of the model. in addition, evaluation metric like accuracy is not well-suited for imbalanced datasets.
# 
# - Recall evaluates the stroke cases that were predicted by the model while precision looks at the total stroke cases that were predicted and evaluates the percentage that are actually stroke cases. The goal is to choose a model that will not make egregious prediction errors  i.e making a wrong prediction of the absence of stroke or poorly identifying stroke cases.
