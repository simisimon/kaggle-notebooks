#!/usr/bin/env python
# coding: utf-8

# # Restaurant Revenue Prediction Using With Random Forest Regressor
# 
# 
# -Agit Çelik
# -Oğulcan Gök
# -Selim Kundakçıoğlu
# 
# 
# 
# 
# ## Contents
# - [Pre-Processing](#pre)
# 
# - [Classification](#class)
# 
# - [Standartizaiton](#std)
# 
# - [PCA](#pca)
# - [RBF](#rbf)
# - [RandomForestRegressor is used to predict "revenues"](#rnd)
# - [Score of worst model](#scr)
# - [Conclution](#conc)

# ### Importing required libraries

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np 
import pandas as pd 

#reading the csv data
trainData = pd.read_csv('../input/train.csv')
trainData.info()

trainData.head(5)


# ## PRE-PROCESSING  & SOME  ANALYSIS <a name="pre"></a>

# ### Converting Open Date column to Open Days; day count of the restaurant since the beginning and dropping the Open Date Columns

# In[ ]:


trainData['Open Date'] = pd.to_datetime(trainData['Open Date'], format='%m/%d/%Y')   
trainData['OpenDays']=""

dateLastTrain = pd.DataFrame({'Date':np.repeat(['01/01/2018'],[len(trainData)]) })
dateLastTrain['Date'] = pd.to_datetime(dateLastTrain['Date'], format='%m/%d/%Y')  

trainData['OpenDays'] = dateLastTrain['Date'] - trainData['Open Date']
trainData['OpenDays'] = trainData['OpenDays'].astype('timedelta64[D]').astype(int)

trainData = trainData.drop('Open Date', axis=1)


# ### Comparing the revenues of big cities and other cities

# In[ ]:


cityPerc = trainData[["City Group", "revenue"]].groupby(['City Group'],as_index=False).mean()

sns.barplot(x='City Group', y='revenue', data=cityPerc)


# # Plots
# ### Sorting the cities by revenue; getting the max earned cities

# In[ ]:


cityPerc = trainData[["City", "revenue"]].groupby(['City'],as_index=False).mean()

newDF = cityPerc.sort_values(["revenue"],ascending= False)
sns.barplot(x='City', y='revenue', data=newDF.head(10))


# In[ ]:


cityPerc = trainData[["City", "revenue"]].groupby(['City'],as_index=False).mean()
newDF = cityPerc.sort_values(["revenue"],ascending= True)
sns.barplot(x='City', y='revenue', data=newDF.head(10))


# ### Getting an insight of which restaurant type earns more

# In[ ]:


cityPerc = trainData[["Type", "revenue"]].groupby(['Type'],as_index=False).mean()
sns.barplot(x='Type', y='revenue', data=cityPerc)


# ### Plot about working days of specific restaurant types

# In[ ]:


cityPerc = trainData[["Type", "OpenDays"]].groupby(['Type'],as_index=False).mean()
sns.barplot(x='Type', y='OpenDays', data=cityPerc)


# ### Dropping the Id and Type columns since they are irrevelant for our predictions

# In[ ]:


trainData = trainData.drop('Id', axis=1)
trainData = trainData.drop('Type', axis=1)


# ### Creating dummy variables to represent City Groups. After doing dummy variables for City Group we dropped it

# In[ ]:


citygroupDummy = pd.get_dummies(trainData['City Group'])
trainData = trainData.join(citygroupDummy)


trainData = trainData.drop('City Group', axis=1)

trainData = trainData.drop('City', axis=1)

tempRev = trainData['revenue']
trainData = trainData.drop('revenue', axis=1)


trainData = trainData.join(tempRev)


# In[ ]:


trainData.head(10)


# # Train and  Test Split for RandomForestClassifier <a name="class"></a>
# 
# ### Using SKLEARN's train test split library for splitting train data

# In[ ]:


from sklearn.model_selection import train_test_split

X, y = trainData.iloc[:, 1:40].values, trainData.iloc[:, 40].values

X_trainForBestFeatures, X_testForBestFeatures, y_trainForBestFeatures, y_testForBestFeatures =\
    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                )
    
X_trainForBestFeatures.shape, X_testForBestFeatures.shape, y_trainForBestFeatures.shape, y_testForBestFeatures.shape


# In[ ]:


y[:20]


# In[ ]:


y_trainForBestFeatures[:20]


# ### For finding best features among others. We used random forest classifier in order to get the best features. We observed that using the first 19 features give us the best results

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

#To label our features form best to wors 
feat_labels = trainData.columns[1:40]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)
forest.fit(X_trainForBestFeatures, y_trainForBestFeatures)



importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_trainForBestFeatures.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))
    


# ### Plotting the importance of the features in a barplot

# In[ ]:


plt.title('Feature Importance')
plt.bar(range(X_trainForBestFeatures.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_trainForBestFeatures.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_trainForBestFeatures.shape[1]])
plt.tight_layout()

plt.show()


# In[ ]:


trainData[feat_labels[indices[0:39]]].head()


# ### We take the natural logarithm of the OpenDays column in order to make it more easy for model to predict

# In[ ]:


import numpy as numpy 
openDaysLog = trainData[feat_labels[indices[0:1]]].apply(numpy.log)
openDaysLog.head()


# ### Test and Train model created over best 19 features.
# 

# In[ ]:


bestDataFeaturesTrain = trainData[feat_labels[indices[1:19]]]

#insert after takeing log of OpenDays feature.
bestDataFeaturesTrain.insert(loc=0, column='OpenDays', value=openDaysLog)

bestDataFeaturesTrain.head()


# # Model will predict output by using best 19 features.
# train_test_split method of sklearn is used to split data into %30 of Test and %70 of Train data

# In[ ]:


# take the natural logarithm of the 'revenue' column in order to make it more easy for model to predict
y = trainData['revenue'].apply(numpy.log)

from sklearn.model_selection import train_test_split

X, y = trainData.iloc[:, 1:40].values, trainData.iloc[:, 40].values

X_train, X_test, y_train, y_test =\
    train_test_split(bestDataFeaturesTrain, y, 
                     test_size=0.3, 
                     random_state=0, 
                )



    
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Standardize features by removing the mean and scaling to unit variance
# <a name="std"></a>
# ### Standart Scaling for model efficiency

# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler(with_std  = True ,with_mean = True, copy = True)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# In[ ]:


X_train_std[:1]


# # PCA is used due to dimension reduction <a name="pca"></a>
# ### Applied PCA in order to make it more efficient and reducing te dimentions

# In[ ]:


from sklearn.decomposition import PCA,KernelPCA

pca = PCA(n_components=2,svd_solver='full')
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
pca.explained_variance_ratio_

kpca = KernelPCA(kernel="rbf", gamma=1)
X_kpca_train = kpca.fit_transform(X_train_pca)
X_kpca_test = kpca.transform(X_test_pca)




# # RBF is applied for linearity (since we have non-linear data) <a name="rbf"></a>
# 
# ### After RBF, increase in sample variance can be observed (blue plot)

# In[ ]:


X_train_pca[:1]
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,5))
ax[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1],color='red',marker='o')
ax[1].scatter(X_kpca_train[:, 0], X_kpca_train[:, 1])
ax[0].set_xlabel('Before RBF')
ax[1].set_yticks([])
ax[1].set_xlabel('After RBF')


# In[ ]:


X_test_pca[:1]


# In[ ]:


X_train.head()


# In[ ]:


X_train_std[:1]


# In[ ]:


X_test.head()


# In[ ]:


X_test.head()


# In[ ]:


y_test[:5]


# # RandomForestRegressor is used to predict "revenues" <a name="rnd"></a>
# 
# ### Finally after pre-processing now we can begin to predict with RandomForestRegressor.
# 
# #### Our model works on 86% accuracy.

# In[ ]:


import numpy
from sklearn import linear_model
cls = RandomForestRegressor(n_estimators=250, criterion='mse', max_depth=30)#cls = RandomForestRegressor(n_estimators=150)

cls.fit(X_kpca_train, y_train)#We are training the model with RBF'ed data

scoreOfModel = cls.score(X_kpca_train, y_train)


print("Score is calculated as: ",scoreOfModel)


# In[ ]:


pred = cls.predict(X_kpca_test)

pred


# # Our prediction and actual value with their differences and their score

# In[ ]:


for z in zip(y_test, pred):
    print(z, (z[0]-z[1]) /z[0] )


# ## Plot Revenues(orange line) and Predicted Revenues(blue line)
# 
# ### plotting the real revenues and the predicted values

# In[ ]:


r = []
for pair in  zip(pred, y_test):
    r.append(pair)

plt.plot(r)


# ## Effect of estimators on score 
# 
# ### With np.range we tried to find the most efficient estimator value. As can be seen on the plot, around 160 is the highest

# In[ ]:


estimators = np.arange(10, 250, 10) # 10 to 250 increased with 10
scores = []
for n in estimators:
    cls.set_params(n_estimators=n)
    cls.fit(X_kpca_train, y_train)
    scores.append(cls.score(X_kpca_train, y_train))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)


# In[ ]:


estimators = np.arange(10, 250, 10) # 10 to 250 increased with 10
scores = []
for n in estimators:
    cls.set_params(n_estimators=n)
    cls.fit(X_kpca_train, y_train)
    scores.append(cls.score(X_kpca_test, y_test))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)


# In[ ]:


pred[:20]


# # Score of worst model <a name="scr"></a>
# ### We took the worst 19 feautres and buiilt a model accordingly to see the effect to the model score.

# In[ ]:


worstDataFeaturesTrain = trainData[feat_labels[indices[19:39]]]
worstDataFeaturesTrain.head()


# ### Splitting the data with respect to worst features
# train_test_split method of sklearn is used to split data into %30 of Test and %70 of Train data

# In[ ]:


y = trainData['revenue'].values

from sklearn.model_selection import train_test_split

X, y = trainData.iloc[:, 1:40].values, trainData.iloc[:, 40].values

X_trainWorst, X_testWorst, y_trainWorst, y_testWorst =\
    train_test_split(worstDataFeaturesTrain, y, 
                     test_size=0.3, 
                     random_state=0, 
                )



    
X_trainWorst.shape, X_testWorst.shape, y_trainWorst.shape, y_testWorst.shape


# ### Fitting the model.
# ### As can be seen below, model score is 31%. Less than 50% so it is worse.

# In[ ]:


import numpy
from sklearn import linear_model
cls = RandomForestRegressor(n_estimators=250, criterion='mse', max_depth=30)#cls = RandomForestRegressor(n_estimators=150)

cls.fit(X_trainWorst, y_trainWorst)

scoreOfModel = cls.score(X_trainWorst, y_trainWorst)

print("Score is calculated as: ",scoreOfModel)


# ### Predicted values from the 'worse' model

# In[ ]:


pred = cls.predict(X_testWorst)

pred


# ### Again Trying to find the best estimator values for the model

# In[ ]:


estimators = np.arange(10, 250, 10) # 10 to 250 increased with 10
scores = []
for n in estimators:
    cls.set_params(n_estimators=n)
    cls.fit(X_trainWorst, y_trainWorst)
    scores.append(cls.score(X_trainWorst, y_trainWorst))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)


# In[ ]:


estimators = np.arange(10, 250, 10) # 10 to 250 increased with 10
scores = []
for n in estimators:
    cls.set_params(n_estimators=n)
    cls.fit(X_trainWorst, y_trainWorst)
    scores.append(cls.score(X_testWorst, y_testWorst))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)


# In[ ]:


import numpy as numpy 
openDaysLog = trainData[feat_labels[indices[0:1]]].apply(numpy.log)
openDaysLog.head()


# In[ ]:


bestDataFeaturesTrain = trainData[feat_labels[indices[1:2]]]

#insert after takeing log of OpenDays feature.
bestDataFeaturesTrain.insert(loc=0, column='OpenDays', value=openDaysLog)

bestDataFeaturesTrain.head()


# In[ ]:


# take the natural logarithm of the 'revenue' column in order to make it more easy for model to predict
y = trainData['revenue'].apply(numpy.log)

from sklearn.model_selection import train_test_split

X, y = trainData.iloc[:, 1:40].values, trainData.iloc[:, 40].values

X_train, X_test, y_train, y_test =\
    train_test_split(bestDataFeaturesTrain, y, 
                     test_size=0.3, 
                     random_state=0, 
                )



    
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


from sklearn import linear_model


# Create linear regression object
regr = linear_model.LinearRegression()


# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
linear_predictions = regr.predict(X_test)

linear_predictions


# In[ ]:


regr.score(X_test,y_test)


# In[ ]:


r = []
for pair in  zip(linear_predictions, y_test):
    r.append(pair)

plt.plot(r)


# # Conclusion <a name="conc"></a>
# 
# During this project we learned a lot of skills of Data Science and Machine Learning with using different techniques. 
# We observed the effects of PCA and RBF on our model. Also we had a chance to decide which technique to use for prediction.
# We choose random forest regressor because it is more usefull in our case of Data among other algorithms. 
# We saw the advantages of standartization on model score also the logarithm. 
# We achived 86% accuracy but we belive that we can improve it.
