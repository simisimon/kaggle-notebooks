#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import  LinearRegression
sns.set()
import os


# In[ ]:


data=pd.read_csv('../input/insurance/insurance.csv')


# In[ ]:


data


# <b> Data Exploration

# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# In[ ]:


## there are no null values and the dataset has been provided clean.


# In[ ]:


# Let's visualize the Probability Density Function for each independent variable.


# In[ ]:


sns.displot(data['age'],kde=True)
plt.xlabel('age',fontsize=20)
plt.ylabel('density',fontsize=20)
plt.title('age distribution',fontsize=20)
plt.show()


# In[ ]:


sns.displot(data['sex'])
plt.xlabel('sex',fontsize=20)
plt.ylabel('density',fontsize=20)
plt.title('sex distribution',fontsize=20)
plt.show()


# In[ ]:


sns.displot(data['bmi'],kde=True)
plt.xlabel('bmi',fontsize=20)
plt.ylabel('density',fontsize=20)
plt.title('bmi distribution',fontsize=20)
plt.show()


# In[ ]:


sns.displot(data['children'])
plt.xlabel('children',fontsize=20)
plt.ylabel('density',fontsize=20)
plt.title('children distribution',fontsize=20)
plt.show()


# In[ ]:


sns.displot(data['smoker'])
plt.xlabel('smoker',fontsize=20)
plt.ylabel('density',fontsize=20)
plt.title('smoker distribution',fontsize=20)
plt.show()


# In[ ]:


sns.displot(data['region'])
plt.xlabel('region',fontsize=20)
plt.ylabel('density',fontsize=20)
plt.title('region distribution',fontsize=20)
plt.show()


# <b> Introducing the quantile, in case I need to remove outliers

# In[ ]:


q=data['charges'].quantile(0.98)
data_1= data[data['charges']<q]


# <b> Convert dummy Variables into 0 and 1 

# In[ ]:


data['sex']= data['sex'].map({'male':1,'female':0})


# In[ ]:


data['smoker']= data['smoker'].map({'yes':1,'no':0})


# In[ ]:


data


# <b> Create the Linear Regrssion Model

# <b> Declaring the Dependent & Independent Variables 

# In[ ]:


x=data[['age','sex','bmi','children','smoker']]
y=data['charges']


# <b> The Regression

# In[ ]:


reg= LinearRegression()
reg.fit(x,y)


# In[ ]:


reg.coef_


# In[ ]:


reg.intercept_


# In[ ]:


reg.score(x,y)


# In[ ]:


x.shape


# In[ ]:


r2=reg.score(x,y)

n=x.shape[0]
p=x.shape[1]

adj_r2=1-(1-r2)*(n-1)/(n-p-1)

print(adj_r2)


# In[ ]:


## the adjusted R^2 is 0.75, just using raw data, without any modification.


# <b> Feature Selection with F regressor

# In[ ]:


from sklearn.feature_selection import f_regression


# In[ ]:


f_regression(x,y)


# In[ ]:


p_values= f_regression(x,y)[1].round(3)


# In[ ]:


p_values


# <b> Creating the Summary Table

# In[ ]:


reg_summary=pd.DataFrame(data=x.columns,columns=['Features'])


# In[ ]:


reg_summary['Coefficients']=f_regression(x,y)[0]


# In[ ]:


reg_summary['p_values']=f_regression(x,y)[1].round(3)


# In[ ]:


reg_summary


# In[ ]:


## All the dependent variables are stastically significant


# In[ ]:


## let's Standardize the variables, in order to give them the same weight and see which one is relevant or not in the model.


# <b> Declaring the Dependent & Independent Variables 

# In[ ]:


x=data[['age','sex','bmi','children','smoker']]
y=data['charges']


# <b> Standardizing the independent variables

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()


# In[ ]:


scaler.fit(x)


# In[ ]:


scaled_x= scaler.transform(x)


# In[ ]:


scaled_x


# <b> Creating regression with scaled features

# In[ ]:


scaled_reg= LinearRegression()


# In[ ]:


scaled_reg.fit(scaled_x,y)


# In[ ]:


scaled_reg.coef_


# In[ ]:


scaled_reg.intercept_


# <b> Summary Table for scaled Features

# In[ ]:


scaled_summary= pd.DataFrame(['Bias','age','sex','bmi','children','smoker'], columns=['Features'])


# In[ ]:


scaled_summary['Weights']=scaled_reg.intercept_,scaled_reg.coef_[0],scaled_reg.coef_[1],scaled_reg.coef_[2],scaled_reg.coef_[3],scaled_reg.coef_[4]


# In[ ]:


scaled_summary


# In[ ]:


percentage=sum(map(abs, scaled_summary.Weights[1:6]))


# In[ ]:


scaled_summary['absolute']=scaled_summary['Weights'][1:6].abs()


# In[ ]:


scaled_summary['percentage_weight']=(scaled_summary['Weights'][1:6].abs()/percentage*100).round(1)


# In[ ]:


scaled_summary


# In[ ]:


## As we can see, those who smoke(60.7%) and the age (22.9) have the highest influence on the model 


# ## Plotting the dependent Variable against each feature

# In[ ]:


f, (ax1,ax2,ax3,ax4)= plt.subplots(1,4, sharey=True, figsize=(20,4))

ax1.scatter(data['age'],data['charges'])
ax1.set_title('charges & age')
ax2.scatter(data['bmi'],data['charges'])
ax2.set_title('charges & bmi')
ax3.scatter(data['children'],data['charges'])
ax3.set_title('charges & children')
ax4.scatter(data['smoker'],data['charges'])
ax4.set_title('charges & smoker')


# In[ ]:


## even if the bmi is statistically significant, with a relative high weight, we cannot include in the model due to the fact
## it doesn't follow a linear distirbution


# In[ ]:


plt.title("Correlation Matrix")
sns.heatmap(data.corr(),annot=True,cmap='coolwarm',linewidths=0.1)


# In[ ]:


## checking for possible correlations and avoid multicollinearity among independent variables


# In[ ]:


## Looking at the 1st graph (charges and age), we can spot 3 different clusters. Thus, we can see if inside these clusters
## we can spot other insights


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kmeans=KMeans(3)


# In[ ]:


x=data[['charges','age']]


# In[ ]:


kmeans.fit(x)


# In[ ]:


identified_clusters=kmeans.fit_predict(x)


# In[ ]:


identified_clusters


# In[ ]:


data_with_clusters=data


# In[ ]:


data_with_clusters['Clusters']=identified_clusters


# In[ ]:


data_with_clusters['name Clusters']=data_with_clusters['Clusters']
data_with_clusters['name Clusters']=data_with_clusters['Clusters'].apply(str)
data_with_clusters['name Clusters']= data_with_clusters['name Clusters'].map({'0':'low_charges','1':'hig_charges','2':'medium_charges'})


# In[ ]:


data_with_clusters


# In[ ]:


data_with_clusters.info()


# In[ ]:


fig, ax = plt.subplots()
scatter=plt.scatter(data_with_clusters['age'],data_with_clusters['charges'], c=data_with_clusters['Clusters'], cmap='rainbow')

legend1 = ax.legend(*scatter.legend_elements(),
                     title="Clusters")


# In[ ]:


## We can clearly see that these 3 groups share different charges even if they have the same age. Let's graphically 
## plot if smoking, sex or region has something to do with this. 


# In[ ]:


fig, ax = plt.subplots()
scatter=plt.scatter(data_with_clusters['age'],data_with_clusters['charges'], c=data_with_clusters['smoker'], cmap='rainbow')

legend1 = ax.legend(*scatter.legend_elements(),
                     title="Clusters")


# In[ ]:


## 0 = Non Smoker
## 1 = Smoker


# In[ ]:


# We can clearly see that the 3 clusters can be split as:
# Low Charges= Non Smokers
# High Charges= Smokers
# Medium Charges= both smokers and non smokers, which can probably have other characteristics (such as genetically inherited diseases) 
# which are not taken into account in the model.


# In[ ]:


fig, ax = plt.subplots()
scatter=plt.scatter(data_with_clusters['age'],data_with_clusters['charges'], c=data_with_clusters['sex'], cmap='rainbow')

legend1 = ax.legend(*scatter.legend_elements(),
                     title="Clusters")


# In[ ]:


## 0 = Female
## 1 = Male


# In[ ]:


sns.stripplot(x=data_with_clusters['age'], y=data_with_clusters['charges'], hue=data_with_clusters['region'],size=5)




# <b> Create the model with Dummies 

# In[ ]:


data_with_dummies= pd.get_dummies(data_with_clusters, drop_first=True)

data_with_dummies.drop(['sex','bmi','Clusters','children'],axis=1, inplace=True)


# In[ ]:


data_with_dummies


# In[ ]:


data_with_dummies.columns


# In[ ]:


x_final=data_with_dummies[['age', 'smoker','region_northwest', 'region_southeast',
       'region_southwest','name Clusters_low_charges','name Clusters_medium_charges']]

y_final=data_with_dummies['charges']


# In[ ]:


reg_final=LinearRegression()


# In[ ]:


reg_final.fit(x_final,y_final)


# In[ ]:


reg_final.score(x_final,y_final)


# In[ ]:


x_final.shape


# In[ ]:


final_r2=reg_final.score(x_final,y_final)
n_=x_final.shape[0]
p_=x_final.shape[1]

final_adj_r2=1-(1-final_r2)*(n_-1)/(n_-p_-1)

print(final_adj_r2, final_r2)


# In[ ]:


## Adding dummy variables and only including relevant factors suchs as smoking and age
## increase the explenatory power of the model to an adjusted R^2, of 0.93 (which is way improved compared to the 0.75)


# <b> Train & Test Split

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_final,y_final, test_size=0.2, random_state=42)


# In[ ]:


reg_train=LinearRegression()


# In[ ]:


reg_train.fit(x_train,y_train)


# In[ ]:


y_hat= reg_train.predict(x_train)


# In[ ]:


plt.scatter(y_train,y_hat)
plt.xlabel('targets y_train', size=18)
plt.ylabel('predictions y_hat', size=18)
plt.xlim(0,60000)
plt.ylim(0,60000)


# In[ ]:


sns.displot(y_train-y_hat, kde=True )
plt.title("Residual PDF",size=15)


# In[ ]:


# We can see that the probability density function of the residuals follows a normal distribution
# and most of the time, the predictions overestimate the targets


# In[ ]:


reg_train.score(x_train,y_train)


# In[ ]:


regression_summary=pd.DataFrame(x_train.columns.values, columns=['features'])
regression_summary['weights']=reg_train.coef_


# In[ ]:


regression_summary


# <b> Testing

# In[ ]:


y_hat_testing= reg_train.predict(x_test)


# In[ ]:


plt.scatter(y_test,y_hat_testing, alpha=0.2)
plt.xlabel('targets y_test', size=18)
plt.ylabel('predictions y_hat_testing', size=18)
plt.xlim(0,60000)
plt.ylim(0,60000)


# In[ ]:


# Plotting targets and predictions against each other, we can clearly see that the concentration is on the lower part,
# meaning that the model is good at predicting those with lower charges, where is the higher concentration
# of data points in the model; while if we trace a 45° line on the graph, we can spot that for those with
# medium and high charges, the predictions attribute higher charges compared to those observed.


# In[ ]:


df_performance= pd.DataFrame(y_hat_testing, columns=['Predictions'])


# In[ ]:


df_performance['target']=y_test


# In[ ]:


df_performance


# In[ ]:


y_test= y_test.reset_index(drop=True)


# In[ ]:


df_performance['target']=y_test


# In[ ]:


df_performance


# In[ ]:


df_performance['Delta']= df_performance['Predictions']-df_performance['target']


# In[ ]:


df_performance['Delta%']= df_performance['Delta']/df_performance['target']*100


# In[ ]:


df_performance


# In[ ]:


df_performance.describe()


# In[ ]:


# As result, the testing model predicts on average, charges 14% higher than the target, while the median delta is 4% higher.

