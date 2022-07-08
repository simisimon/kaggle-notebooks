#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn Prediction - Starter Notebook
# 
# **Author:** Akshay Sehgal (www.akshaysehgal.com)

# The goal of this notebook is to provide an overview of how write a notebook and create a submission file that successfully solves the churn prediction problem. Please download the datasets, unzip and place them in the same folder as this notebook.
# 
# We are going to follow the process called CRISP-DM.
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/639px-CRISP-DM_Process_Diagram.png" style="height: 400px; width:400px;"/>
# 
# After Business and Data Understanding via EDA, we want to prepare data for modelling. Then evaluate and submit our predictions.

# # 0. Problem statement
# 
# In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.
# 
# For many incumbent operators, retaining high profitable customers is the number one business
# goal. To reduce customer churn, telecom companies need to predict which customers are at high risk of churn. In this project, you will analyze customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn, and identify the main indicators of churn.
# 
# In this competition, your goal is *to build a machine learning model that is able to predict churning customers based on the features provided for their usage.*
# 
# **Customer behaviour during churn:**
# 
# Customers usually do not decide to switch to another competitor instantly, but rather over a
# period of time (this is especially applicable to high-value customers). In churn prediction, we
# assume that there are three phases of customer lifecycle :
# 
# 1. <u>The ‘good’ phase:</u> In this phase, the customer is happy with the service and behaves as usual.
# 
# 2. <u>The ‘action’ phase:</u> The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the ‘good’ months. It is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)
# 
# 3. <u>The ‘churn’ phase:</u> In this phase, the customer is said to have churned. In this case, since you are working over a four-month window, the first two months are the ‘good’ phase, the third month is the ‘action’ phase, while the fourth month (September) is the ‘churn’ phase.

# # 1. Loading dependencies & datasets
# 
# Lets start by loading our dependencies. We can keep adding any imports to this cell block, as we write mode and mode code.

# In[ ]:


#Data Structures
import pandas as pd
import numpy as np
import re
import os

### For installing missingno library, type this command in terminal
#pip install missingno

import missingno as msno

#Sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score

#Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#Others
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# Next, we load our datasets and the data dictionary file.
# 
# The **train.csv** file contains both dependent and independent features, while the **test.csv** contains only the independent variables. 
# 
# So, for model selection, I will create our own train/test dataset from the **train.csv** and use the model to predict the solution using the features in unseen test.csv data for submission.

# In[ ]:


#COMMENT THIS SECTION INCASE RUNNING THIS NOTEBOOK LOCALLY

#Checking the kaggle paths for the uploaded datasets
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#INCASE RUNNING THIS LOCALLY, PASS THE RELATIVE PATH OF THE CSV FILES BELOW
#(e.g. if files are in same folder as notebook, simple write "train.csv" as path)

data = pd.read_csv("../input/telecom-churn-case-study-hackathon-c36/train.csv")
unseen = pd.read_csv("../input/telecom-churn-case-study-hackathon-c36/test.csv")
sample = pd.read_csv("../input/telecom-churn-case-study-hackathon-c36/sample.csv")
data_dict = pd.read_csv("../input/telecom-churn-case-study-hackathon-c36/data_dictionary.csv")

print(data.shape)
print(unseen.shape)
print(sample.shape)
print(data_dict.shape)


# 1. Lets analyze the data dictionary versus the churn dataset.
# 2. The data dictonary contains a list of abbrevations which provide you all the information you need to understand what a specific feature/variable in the churn dataset represents
# 3. Example: 
# 
# > "arpu_7" -> Average revenue per user + KPI for the month of July
# >
# > "onnet_mou_6" ->  All kind of calls within the same operator network + Minutes of usage voice calls + KPI for the month of June
# >
# >"night_pck_user_8" -> Scheme to use during specific night hours only + Prepaid service schemes called PACKS + KPI for the month of August
# >
# >"max_rech_data_7" -> Maximum + Recharge + Mobile internet + KPI for the month of July
# 
# Its important to understand the definitions of each feature that you are working with, take notes on which feature you think might impact the churn rate of a user, and what sort of analysis could you do to understand the distribution of the feature better.

# In[ ]:


data_dict


# For the purpose of this **starter notebook**, we I will restrict the dataset to only a small set of variables. 
# 
# The approach I use here is to understand each Acronym, figure our what variable might be important and filter out variable names based on the combinations of acrynoms using REGEX. So, if I want the total minutes a person has spent on outgoing calls, I need acronyms, TOTAL, OG and MOU. So corresponding regex is ```total.+og.+mou```

# In[ ]:


ids = ['id','circle_id']
total_amounts = [i for i in list(data.columns) if re.search('total.+amt',i)]
total_outgoing_minutes = [i for i in list(data.columns) if re.search('total.+og.+mou',i)]
offnetwork_minutes = [i for i in list(data.columns) if re.search('offnet',i)]
average_revenue_3g = [i for i in list(data.columns) if re.search('arpu.+3g',i)]
average_revenue_2g = [i for i in list(data.columns) if re.search('arpu.+2g',i)]
volume_3g = [i for i in list(data.columns) if re.search('vol.+3g',i)]
volume_2g = [i for i in list(data.columns) if re.search('vol.+2g',i)]
age_on_network = [i for i in list(data.columns) if re.search('aon',i)]

#Storing them in a single flat list
variables = [*ids, 
             *total_amounts, 
             *total_outgoing_minutes, 
             *offnetwork_minutes, 
             *average_revenue_3g, 
             *average_revenue_2g,
             *volume_3g,
             *volume_2g,
             *age_on_network, 
             'churn_probability']

data = data[variables].set_index('id')


# In[ ]:


data.head()


# Let's look at each variable's datatype:

# In[ ]:


data.info(verbose=1)


# Let's also summarize the features using the df.describe method:

# In[ ]:


data.describe(include="all")


# # 2. Create X, y and then Train test split
# 
# Lets create X and y datasets and skip "circle_id" since it has only 1 unique value

# In[ ]:


data['circle_id'].unique()


# In[ ]:


X = data.drop(['circle_id'],1).iloc[:,:-1]
y = data.iloc[:,-1]

X.shape, y.shape


# Splitting train and test data to avoid any contamination of the test data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


X_train.head()


# # 3. Handling Missing data
# 
# First lets analyse the missing data. We can use missingno library for quick visualizations.

# In[ ]:


msno.bar(X_train)


# In[ ]:


msno.matrix(X_train)


# Lets also calculate the % missing data for each column:

# In[ ]:


missing_data_percent = 100*X_train.isnull().sum()/len(y_train)
missing_data_percent


# Since too much missing information would make a column not really a great predictor for churn, we drop these columns and keep only the ones which have less than 40% missing data.

# In[ ]:


new_vars = missing_data_percent[missing_data_percent.le(40)].index
new_vars


# In[ ]:


X_train_filtered = X_train[new_vars]
X_train_filtered.shape


# Next, we try imputation on variables with any amount of missing data still left. There are multiple ways of imputing data, and each will require a good business understanding of what the missing data is and how you may handle it.
# 
# Some tips while working with missing data - 
# 
# 1. Can simply replace missing values directly with a constant value such as 0
# 2. In certain cases you may want to replace it with the average value for each column respectively
# 3. For timeseries data, you may consider using linear or spline interplolation between a set of points, if you have data available for some of the months, and missing for the others.
# 4. You can consider more advance methods for imputation such as MICE.
# 
# In our case, I will just demostrate a simple imputation with constant values as zeros.

# In[ ]:


missing_data_percent = X_train_filtered.isnull().any()
impute_cols = missing_data_percent[missing_data_percent.gt(0)].index
impute_cols


# In[ ]:


imp = SimpleImputer(strategy='constant', fill_value=0)
X_train_filtered[impute_cols] = imp.fit_transform(X_train_filtered[impute_cols])


# In[ ]:


msno.bar(X_train_filtered)


# In[ ]:


X_train_filtered.describe()


# # 4. Exploratory Data Analysis & Preprocessing
# 
# Lets start by analysing the univariate distributions of each feature.

# In[ ]:


plt.figure(figsize=(15,8))
plt.xticks(rotation=45)
sns.boxplot(data = X_train_filtered)


# ### 4.1 Handling outliers
# 
# The box plots of these features show there a lot of outliers. These can be capped with k-sigma method.

# In[ ]:


def cap_outliers(array, k=3):
    upper_limit = array.mean() + k*array.std()
    lower_limit = array.mean() - k*array.std()
    array[array<lower_limit] = lower_limit
    array[array>upper_limit] = upper_limit
    return array


# In[ ]:


X_train_filtered1 = X_train_filtered.apply(cap_outliers, axis=0)

plt.figure(figsize=(15,8))
plt.xticks(rotation=45)
sns.boxplot(data = X_train_filtered1)


# ### 4.2 Feature scaling
# 
# Lets also scale the features by scaling them with Standard scaler (few other alternates are min-max scaling and Z-scaling).

# In[ ]:


scale = StandardScaler()
X_train_filtered2 = scale.fit_transform(X_train_filtered1)


# In[ ]:


plt.figure(figsize=(15,8))
plt.xticks(rotation=45)
sns.boxplot(data = pd.DataFrame(X_train_filtered2, columns=new_vars))


# You can perform feature transformations at this stage. 
# 
# 1. **Positively skewed:** Common transformations of this data include square root, cube root, and log.
# 2. **Negatively skewed:** Common transformations include square, cube root and logarithmic.
# 
# Please read the following link to understand how to perform feature scaling and preprocessing : https://scikit-learn.org/stable/modules/preprocessing.html
#  
# Lets also plot the correlations for each feature for bivariate analysis.

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(pd.DataFrame(X_train_filtered2, columns=new_vars).corr())


# In[ ]:


#Distribution for the churn probability
sns.histplot(y_train)


# # 5. Feature engineering and selection
# 
# Let's understand feature importances for raw features as well as components to decide top features for modelling.

# In[ ]:


rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_train_filtered2, y_train)


# In[ ]:


feature_importances = pd.DataFrame({'col':new_vars, 'importance':rf.feature_importances_})


# In[ ]:


plt.figure(figsize=(15,8))
plt.xticks(rotation=45)
plt.bar(feature_importances['col'], feature_importances['importance'])


# At this step, you can create a bunch of features based on business understanding, such as 
# 1. "average % gain of 3g volume from month 6 to 8" - (growth or decline of 3g usage month over month?)
# 2. "ratio of total outgoing amount and age of user on network" - (average daily usage of a user?)
# 3. "standard deviation of the total amount paid by user for all services" - (too much variability in charges?)
# 4. etc..
# 
# Another way of finding good features would be to project them into a lower dimensional space using PCA. PCA creates components which are a linear combination of the features. This then allows you to select components which explain the highest amount of variance.
# 
# Lets try to project the data onto 2D space and plot. **Note:** you can try TSNE, which is another dimensionality reduction approach as well. Check https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html for moree details.

# In[ ]:


pca = PCA()
pca_components = pca.fit_transform(X_train_filtered2)
sns.scatterplot(x=pca_components[:,0], y=pca_components[:,1], hue=y_train)


# In[ ]:


sns.scatterplot(x=pca_components[:,1], y=pca_components[:,2], hue=y_train)


# Let's also check which of the components have high feature importances towards the end goal of churn prediction.

# In[ ]:


rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(pca_components, y_train)

feature_importances = pd.DataFrame({'col':['component_'+str(i) for i in range(16)], 
                                    'importance':rf.feature_importances_})

plt.figure(figsize=(15,8))
plt.xticks(rotation=45)
plt.bar(feature_importances['col'], feature_importances['importance'])


# # 6. Model building
# 
# Let's build a quick model with logistic regression and the first 2 PCA components.

# In[ ]:


lr = LogisticRegression(max_iter=1000, tol=0.001, solver='sag')
lr.fit(pca_components[:,:2], y_train)


# In[ ]:


lr.score(pca_components[:,:2], y_train)


# The model has 89.8% accuracy, but let's build a pipeline to fit and score the model faster.
# 
# The steps of this pipeline would be the following, but this is only one type of pipeline -
# 1. Imputation
# 2. Scaling
# 3. PCA
# 4. Classification model
# 
# You can change this pipeline, add addition transformations, change models, use cross validation or even use this pipeline to work with a Gridsearch.

# In[ ]:


imp = SimpleImputer(strategy='constant', fill_value=0)
scale = StandardScaler()
pca = PCA(n_components=10)
lr = LogisticRegression(max_iter=1000, tol=0.001)


# In[ ]:


pipe = Pipeline(steps = [('imputation',imp),
                         ('scaling',scale),
                         ('pca',pca),
                         ('model',lr)])


# In[ ]:


pipe.fit(X_train[new_vars], y_train)


# In[ ]:


train_score = pipe.score(X_train[new_vars], y_train)
print("Training accuracy:", train_score)


# In[ ]:


test_score = pipe.score(X_test[new_vars], y_test)
print("Test accuracy:", test_score)


# Let's make a confusion matrix to analyze how each class is being predicted by the model.

# In[ ]:


confusion_matrix(y_train, pipe.predict(X_train[new_vars]))


# In[ ]:


confusion_matrix(y_test, pipe.predict(X_test[new_vars]))


# We can see a high amount of type 2 error. Due to class imbalance, the model is clearly trying to predict majority of the cases as class 0. Understanding how to handle class imbalance in classification models might be the key to winning this competition :) (hint!)

# In[ ]:


precision_score(y_test, pipe.predict(X_test[new_vars]))


# In[ ]:


recall_score(y_test, pipe.predict(X_test[new_vars]))


# # 7. Creating submission file
# 
# For submission, we need to make sure that the format is exactly the same as the sample.csv file. It contains 2 columns, id and churn_probability

# In[ ]:


sample.head()


# The submission file should contain churn_probability values that have to be predicted for the unseen data provided (test.csv)

# In[ ]:


unseen.head()


# Lets first select the columns that we want to work with (or create them, if you have done any feature engineering)

# In[ ]:


submission_data = unseen.set_index('id')[new_vars]
submission_data.shape


# Next, lets create a new column in the unseen dataset called churn_probability and use the model pipeline to predict the probabilities for this data

# In[ ]:


unseen['churn_probability'] = pipe.predict(submission_data)
output = unseen[['id','churn_probability']]
output.head()


# Finally, lets create a csv file out of this dataset, ensuring to set index=False to avoid an addition column in the csv.

# In[ ]:


output.to_csv('submission_pca_lr_13jul.csv',index=False)


# You can now take this file and upload it as a submission on Kaggle.

# In[ ]:




