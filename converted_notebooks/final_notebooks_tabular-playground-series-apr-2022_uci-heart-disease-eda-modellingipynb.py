#!/usr/bin/env python
# coding: utf-8

# # **UCI Heart Disease Data Set: Exploration and ML Modelling** 

# *Note:*
# 
# *Thanks to author of https://www.kaggle.com/code/nareshbhat/eda-classification-ensemble-92-accuracy notebook, NARESH BHAT. This notebook was a copy of that original version, but will differ significantly when its completed. Please upvote my notebook as well as Naresh's one, if you find below contents useful.*

# ## **Background**
# 
# The original dataset is sourced from the following four healthcare institutes. They are listed below, along with the contributors:
# Source:
# 
# 1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
# 2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
# 3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
# 4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
# 
# Donor is mentioned as follows in the original website:
# 
# David W. Aha (aha '@' ics.uci.edu) (714) 856-8779
# 
# Regarding dataset characteristics, I will just quote the original description in the UCI machine learning repository website. (we'll explore more when we do Exploratory Data Analysis(EDA)):
# 
# 
# >*This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to
# this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.*
# 
# 
# ### Regarding Heart Disease 
# 
# The term 'heart disease' usually refers to "coronary heart disease", a clinical condition caused by obstruction in vessels that supply blood to heart(called 'coronary blood vessels'). Critical obstructions, usually 100% of a blockage or multiple blockages, will cause heart attacks.Here are some other points that I thought important to get a feel for the importance of tackling this challenge.
# 
# - Heart disease is part of a group of conditions known as Cardiovascular diseases (CVDs), which is the leading cause of death globally. 
# - They all have a similar pathological background: gradual accumilation of cholesterol deposits or calcifications in blood vessels leads to inadequate blood supply to vital organs, followed by lack of oxygen, which can result in 'death' of certain parts of these organs. 
# - Lack of blood supply to brain causes stroke, same for legs and arms cause peripheral vascular disease. 
# - According to World Health Organization(WHO),CVDs represent 32% of all global deaths. Of these deaths, 85% were due to heart attack and stroke.  And Over three quarters of CVD deaths take place in low- and middle-income countries.
# 
# Also,
# 
# - Heart disease can be prevented and controlled by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol.
# - Therefore, it is important to detect it as early as possible so that management with counselling and medicines can begin.
# 
# ### Target variable
# 
# The challenge here is to train a ML model to predict the presence(denoted as '1' in the dataset), or absence('0') of heart disease, using other variables in this tabluar dataset. This is called the 'Target' or 'Ground truth' or 'Label' in ML lingo. 
# 
# According to the authors, the presence of heart disease was defined according to the angiography finding. Coronary angiogram is the gold-standard test to detect coronary heart disease. It's performed by passing a catheter via a guided wire to the heart, and injecting a dye, which will pass through the catheter to your coronary blood vessels(specifically arteries). Then series of X-rays are taken, and the blockages are identified by observing the places where either the dye doesn't passthrough or the flow is significantly reduced.  
# 
# More than 50% of a block in angiogram is taken as 'presence' ('1') of heart disease in the dataset. 
# 
# 
# 

# ## **Exploratory Data Analysis** 

# ### Importing the required libraries and packages
# 
# I will import some of the libraries at the start as below, but we'll import some others as we need them later.
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# We'll get the dataset into a pandas dataframe so that we can explore it easily.

# In[ ]:


heart_df = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')
heart_df.head(10) #gives us the first 10 rows of the dataset


# By the look of it, all of the columns or features are numerical. This is important since ML algorithms can't train with string/object data. You need to represent them in numbers. But we don't have to worry about it here. 'Target' column, is well, the target or label we train the model to predict. This is therefore a [supervised machine learning](https://www.geeksforgeeks.org/supervised-unsupervised-learning/) problem. (geeksforgeeks.org is an awesome resource for data science beginners btw!)

# Most feature names(eg: cp, trestbps etc.) are abbreviated. So we'll very briefly try to understand what they mean (only the ones are not obvious):
# 
# <div>
# <img src="attachment:6fed7ca6-aba7-4637-8909-0d7944ffc436.png" width="700"/>
# </div>
# 
# 
# ['sex', 'cp', 'restecg','exang', 'slope', 'ca', 'thal' ]

# 
# 
# 

# Let's describe the dataset to understand the nature of it:

# In[ ]:


heart_df.describe()


# So you can see there are 303 patients, and ranging from age 29 to 77. Looking at the 'count', it seems that every feature has values for all the patients. Meaning, no null values! Let's be certain, since some ML models can't handle null values.

# In[ ]:


heart_df.isnull().sum()


# Yeah there are no null values. 

# In[ ]:


heart_df.info()


# `info()` method gives us the data types of each feature. int means integer, and float means floating points. But, as we explored, some of these are better represented as categorical variables, because the categories have a meaning, and an 'order'. For example, cp or chest pain type has 4 values: '1' says typical angina. Angina refers to the chest pain that  occurs due to a blockage in heart vessels. So '1' is most indicative of heart disease, and '4' is least indicative. 

# Next, we'll explore whether there's any relationship between the features, using pearson's correlation coefficient. Now, this is a topic in itself, but to put it simply, this statistic, denoted by 'r', measures the linear relationship between two continous variables. 
# - it spans from -1 to +1. 
# - -1 means there's strong inverse relationship(or negative correlation)(as one variable increases, other decreases by the same proportion), and 
# - +1 means there's a strong positive correlation. 0 means there's no linear relationship. 
# - Remember, it tests only 'linear' relationships, meaning the relationship desribed by a line. There may be different correlations between variables that are missed by r.

# In[ ]:


pearson = heart_df.corr()

cmap = sns.diverging_palette(220, 20, as_cmap=True)
_, ax = plt.subplots(figsize=(14,8))
ax = sns.heatmap (pearson, annot=True, linewidth=2, cmap=cmap)


# Sources helped me to draw the above heatmap(from seaborn docs):
# 
# https://seaborn.pydata.org/generated/seaborn.diverging_palette.html
# 
# https://seaborn.pydata.org/tutorial/color_palettes.html

# In[ ]:


cat_features = ['sex', 'cp','fbs', 'restecg','exang', 'slope', 'ca', 'thal' ]
num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
label = 'target'


# In[ ]:


heart_df[cat_features] = heart_df[cat_features].astype('category')


# In[ ]:


heart_df.dtypes


# In[ ]:


heart_df.cp.value_counts()


# In[ ]:


heart_df.exang.unique()


# In[ ]:


datatype = heart_df.exang.dtype

print(datatype)


# 

# - cholesterol level, strangely doesn't seam to differ between two target groups, we'll analyze further via a boxplot
# - st depression induced by exercise(`oldpeak`) seems to be clearly more in heart disease patients
# 

# In[ ]:


heart_df.groupby(heart_df.target).chol.mean()


# https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.GroupBy.mean.html

# In[ ]:


sns.boxplot(x='target', y='chol', data=heart_df)


# Despite having a person with a cholesterol level of above 500, heart disease group shows no significant difference in the mean cholesteroal levels compared to no heart disase group. Remember, outliers like this 500+ value can sway the mean towards a higher value. 

# In[ ]:


heart_pt_percentage = len(heart_df.loc[heart_df['target']==1])/len(heart_df)*100
heart_pt_percentage


# In[ ]:


#plotting target:
ax = sns.countplot(x='target', data=heart_df)
ax.set_title(f'Heart Disease % is:{heart_pt_percentage: .2f}')


# In[ ]:





# In[ ]:


plt.figure(figsize = (16,10))
for i,col in enumerate(cat_features):
    plt.subplot(4,4,i+1)
    sns.countplot(data = heart_df , x = col, palette = "husl", hue = 'target' )
    plt.title(col,weight = 'bold', color = 'black')
    plt.legend(['No heart disease','Heart Disease'])
    plt.ylabel(" ")
    plt.xlabel(" ")
    plt.tight_layout()


# Now now, there are much more insights in the categorical variables:
# 
# -  Presence of chest pain, ST changes in rest ECG, downward slope of the ST segment in ECG, obstructed vessels in fluroscopy and presence of a fixed defect are all more in heart disease group, which is compatible with research evidence on coronary heart disease
# - strangely. exercise induced angina is more in no heart disease group, but it's not clear whether heart disease patients who had angina at rest, were considered 'negative'.
# 

# Now it's time for modelling! From above exploration and medical domain knowledge, I decided to rearrange the 'order' of the variables `thal` and `restecg`,
# because in thal, fixed defect carries the most wieght in terms of predicting heart disease, and ST changes in restecg carries the most weight in a similar manner. 

# In[ ]:


#value counts before reordering
heart_df.restecg.value_counts()


# In[ ]:


heart_df['restecg'] = heart_df['restecg'].replace({0:0, 1:2, 2:1})


# In[ ]:


#let's check after reordering:
heart_df.restecg.value_counts()


# In[ ]:


heart_df.thal.value_counts()


# In[ ]:


heart_df['thal'] = heart_df['thal'].replace({1:1, 2:3, 3:2})


# In[ ]:


heart_df.thal


# In[ ]:


from sklearn.preprocessing import OrdinalEncoder
ordinal = OrdinalEncoder()
heart_df[cat_features] = ordinal.fit_transform(heart_df[cat_features])


# In[ ]:


heart_df.thal.dtype


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, test_index in split.split(heart_df, heart_df["target"]):
    strat_train = heart_df.loc[train_index]
    strat_test = heart_df.loc[test_index]


# In[ ]:


X_train = strat_train.drop('target', axis=1)
y_train = strat_train['target']
x_test = strat_test.drop('target', axis=1)
y_test = strat_test['target']


# In[ ]:


X_train


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


def rf(xs, y, n_estimators=40,
max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,
    max_features=max_features,
    min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)


# In[ ]:


X_train.shape, y_train.shape


# Number of rows match!

# In[ ]:


rf_model = rf(X_train, y_train)


# In[ ]:


#predicting on validation set and evaluating
from sklearn.metrics import classification_report
train_preds = rf_model.predict(X_train)
print(classification_report(train_preds, y_train))
y_preds = rf_model.predict(x_test)
print(classification_report(y_preds,y_test))


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = [
{'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 600], 'max_features': [0.5, 'sqrt', 'log2'], 'min_samples_leaf':[4], 
 'n_jobs':[-1]},
{'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500], 'max_features': [0.5, 'sqrt', 'log2'],  
 'n_jobs':[-1]}, 
   ]

rf = RandomForestClassifier()
grid_search = GridSearchCV(rf, param_grid, cv=5,
scoring='accuracy',
return_train_score=True)
grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_


# In[ ]:


rf_model2 = RandomForestClassifier( oob_score=True, n_estimators=2000, max_features='sqrt', n_jobs=-1, max_depth=5, min_samples_leaf=4)
rf_model2.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import classification_report
train_preds = rf_model2.predict(X_train)
print(classification_report(train_preds, y_train))
y_preds = rf_model2.predict(x_test)
print(classification_report(y_preds,y_test))


# In[ ]:


m2 = 'Naive Bayes'
nb = GaussianNB()
nb.fit(X_train,y_train)
nbpred = nb.predict(X_test)
nb_conf_matrix = confusion_matrix(y_test, nbpred)
nb_acc_score = accuracy_score(y_test, nbpred)
print("confussion matrix")
print(nb_conf_matrix)
print("\n")
print("Accuracy of Naive Bayes model:",nb_acc_score*100,'\n')
print(classification_report(y_test,nbpred))


# In[ ]:


imp_feature = pd.DataFrame({'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'], 'Importance': xgb.feature_importances_})
plt.figure(figsize=(10,4))
plt.title("barplot Represent feature importance ")
plt.xlabel("importance ")
plt.ylabel("features")
plt.barh(imp_feature['Feature'],imp_feature['Importance'],color = 'rgbkymc')
plt.show()


# In[ ]:


lr_false_positive_rate,lr_true_positive_rate,lr_threshold = roc_curve(y_test,lr_predict)
nb_false_positive_rate,nb_true_positive_rate,nb_threshold = roc_curve(y_test,nbpred)
rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(y_test,rf_predicted)                                                             
xgb_false_positive_rate,xgb_true_positive_rate,xgb_threshold = roc_curve(y_test,xgb_predicted)
knn_false_positive_rate,knn_true_positive_rate,knn_threshold = roc_curve(y_test,knn_predicted)
dt_false_positive_rate,dt_true_positive_rate,dt_threshold = roc_curve(y_test,dt_predicted)
svc_false_positive_rate,svc_true_positive_rate,svc_threshold = roc_curve(y_test,svc_predicted)


sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(lr_false_positive_rate,lr_true_positive_rate,label='Logistic Regression')
plt.plot(nb_false_positive_rate,nb_true_positive_rate,label='Naive Bayes')
plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Random Forest')
plt.plot(xgb_false_positive_rate,xgb_true_positive_rate,label='Extreme Gradient Boost')
plt.plot(knn_false_positive_rate,knn_true_positive_rate,label='K-Nearest Neighbor')
plt.plot(dt_false_positive_rate,dt_true_positive_rate,label='Desion Tree')
plt.plot(svc_false_positive_rate,svc_true_positive_rate,label='Support Vector Classifier')
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.legend()
plt.show()


# # **Model Evaluation**

# In[ ]:


model_ev = pd.DataFrame({'Model': ['Logistic Regression','Naive Bayes','Random Forest','Extreme Gradient Boost',
                    'K-Nearest Neighbour','Decision Tree','Support Vector Machine'], 'Accuracy': [lr_acc_score*100,
                    nb_acc_score*100,rf_acc_score*100,xgb_acc_score*100,knn_acc_score*100,dt_acc_score*100,svc_acc_score*100]})
model_ev


# In[ ]:


colors = ['red','green','blue','gold','silver','yellow','orange',]
plt.figure(figsize=(12,5))
plt.title("barplot Represent Accuracy of different models")
plt.xlabel("Accuracy %")
plt.ylabel("Algorithms")
plt.bar(model_ev['Model'],model_ev['Accuracy'],color = colors)
plt.show()


# ## **Ensembling**
# 
# > In order to increase the accuracy of the model we use ensembling. Here we use stacking technique.
# 
# ### About Stacking 
# 
# > Stacking or Stacked Generalization is an ensemble machine learning algorithm. It uses a meta-learning algorithm to learn how to best combine the predictions from two or more base machine learning algorithms. The base level often consists of different learning algorithms and therefore stacking ensembles are often heterogeneous.The stacking ensemble is illustrated in the figure below
# 
# > <img style="float: centre;" src="https://mlfromscratch.com/content/images/2020/01/image-2.png" width="400px"/>

# In[ ]:


scv=StackingCVClassifier(classifiers=[xgb,knn,svc],meta_classifier= svc,random_state=42)
scv.fit(X_train,y_train)
scv_predicted = scv.predict(X_test)
scv_conf_matrix = confusion_matrix(y_test, scv_predicted)
scv_acc_score = accuracy_score(y_test, scv_predicted)
print("confussion matrix")
print(scv_conf_matrix)
print("\n")
print("Accuracy of StackingCVClassifier:",scv_acc_score*100,'\n')
print(classification_report(y_test,scv_predicted))

