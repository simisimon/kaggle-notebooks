#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # (1) Predicting Walmart Sales

# (2) Sources: https://www.kaggle.com/code/travisfeldman/walmart-store-sales-prediction-a-simple-approach. Basically all the matplotlib, pandas, and sklearn documentation

# (3) Model: The final model is a Linear Regression model. It works by finding the line of best fit of all the variables and then makes its predictions on that line. I measured the accuracy with a variance score and r-squared (how much variability in the y values is explained by the x values). Loss was measured by mean absolute percentage error (the percent difference between the models prediction and the actual y value). 

# (4) Data: This data set gathers information about 45 Walmart stores around the world. It takes week by week measures of sales, temperature, fuel price, CPI, unemployment, and existence of holiday season. I learned a lot about Walmart sales from exploring this data. It was especially intersting to learn that only the holidays had any real effect on sales; the other variables showed no influence on them at all. I used Pandas to explore and restructure the data significantly. For example, I made data frames that showed each store's basic statistics: minimum, maximum, average, and median sales. As to preprocessing, I added dummy variables that would allow the model to take into account the store number and the presence of a holiday. 

# (5) The aim of this model is to provide a tool for Walmart store managers to predict their sales for the week based on holiday season, temperature, fuel price, CPI, and unemployment. Unfortunately, the model works best only when the store number is included, so it cannot be used by any store other than the 45 included in the data set. The idea of this notebook is that a Walmart store manager would input the expected conditions for future weeks, and the model would return a sales prediction. 

# In[ ]:


walmart = pd.read_csv("../input/walmart-dataset/Walmart.csv")
walmart.head()


# ## Data exploration

# In[ ]:


walmart.dtypes


# I have to fix store and holiday_flag data types because they are categories, not numerical values

# In[ ]:


walmart.Store = pd.Categorical(walmart.Store)
walmart.Holiday_Flag = pd.Categorical(walmart.Holiday_Flag)

walmart.dtypes


# In[ ]:


#check a summary of the numerical values
walmart.describe()


# Let's look at how sales compare across stores

# In[ ]:


stores_stats = walmart.groupby('Store').Weekly_Sales.agg([min, max, 'mean', 'median'])
stores_stats.head()


# That looks very confusing, I'll set the units to millions of sales

# In[ ]:


# dividing sales by 1,000,000 to make the numbers easier to read
stores_stats = np.round(stores_stats / 1000000, 2)
stores_stats.head()


# Let's graph it so we can visualize it better

# In[ ]:


# comparing best sale to mean sale per store
plt.figure(figsize = (15,5))
plt.title('Maxmimum and Average Sales by Store')
plt.bar(walmart.Store.unique(), stores_stats['max'], width=.4, label='maximum')
plt.bar(walmart.Store.unique(), stores_stats['mean'], width=.4, label='mean')
plt.legend(['Maximum sale', 'Average sale'], fontsize='x-large')
plt.show()


# Median and max sales follow the same trend across stores, i.e. most stores with high maximum sales also have high median sales. It's also clear that the stores with the highest sales also have a bigger difference between its maximum and median sales. So we can conclude that stores with low sales also have a closer distribution than stores with large sales. 

# We can now add a line that shows whether stores are performing better or worse than average

# In[ ]:


plt.figure(figsize = (15,5))
plt.title('Maxmimum and Average Sales by Store')
plt.bar(walmart.Store.unique(), stores_stats['max'], width=.4, label='maximum')
plt.bar(walmart.Store.unique(), stores_stats['mean'], width=.4, label='median')
plt.axhline(y=walmart.Weekly_Sales.mean() / 1000000, color = 'red')
plt.legend(['Total average sale', 'Store maximum sale', 'Store average sale'], fontsize='x-large')
plt.show()


# ## What affects sales performance the most?

# Holiday markdowns probably have a lot to do with it. 

# In[ ]:


holiday_stats = walmart.groupby(['Store','Holiday_Flag']).Weekly_Sales.agg('mean')
holiday_stats = np.round(holiday_stats / 1000000, 2).reset_index()
holiday_stats.head()


# In[ ]:


plt.figure(figsize = (18,5))
plt.title('Holiday and Non-holiday Weekly Sales by Store')
# x needs to be an integer so we can do a side by side chart
x = np.arange(len(walmart.Store.unique())) 
plt.bar(x - .1, holiday_stats[holiday_stats.Holiday_Flag == 0].Weekly_Sales, width=.2, label='No holiday')
plt.bar(x + .1, holiday_stats[holiday_stats.Holiday_Flag == 1].Weekly_Sales, width=.2, label='Holiday')
plt.legend(['No Holiday', 'Holiday'], fontsize='x-large')
plt.show()


# At most stores, average sales during holiday weeks are higher than average sales during non-holiday weeks. 

# How do temperature, fuel price, CPI, and unemployment rate affect sales?

# In[ ]:


fig, axs = plt.subplots(2, 2, figsize=(14,10))

y = walmart.Weekly_Sales 

axs[0, 0].scatter(walmart.Temperature, y)
axs[0, 0].set_title('Temperature')
axs[0, 1].scatter(walmart.Fuel_Price, y, c='orange')
axs[0, 1].set_title('Fuel Price')
axs[1, 0].scatter(walmart.CPI, y, c='green')
axs[1, 0].set_title('CPI')
axs[1, 1].scatter(walmart.Unemployment, y, c='gray')
axs[1, 1].set_title('Unemployment')
fig.tight_layout()

plt.show()


# None of these factors seem to have any significant relationship with sales.
# 
# Before getting started with the actual machine learning, it's important to add dummy variables so that our model takes the holiday into consideration.

# In[ ]:


walmart_dummies = pd.get_dummies(walmart,columns=['Store', 'Holiday_Flag'])
print(walmart_dummies.columns.to_list())


# These (except for date) are all the variables that the model will adjust to

# ## ML Model

# In[ ]:


x = walmart_dummies.drop(["Date", "Weekly_Sales"], axis=1)
y = walmart_dummies.Weekly_Sales

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


# I tried two models: a knn and a linear regression. I'll start with the knn model because it didn't end up working

# #### K-Nearest Neighbor Model

# In[ ]:


model = KNeighborsRegressor(n_neighbors = 5)

knn = model.fit(x_train, y_train)


# In[ ]:


print('r-squared value:', metrics.r2_score(y_train, knn.predict(x_train)))
print('mean absolute percentage error: ', np.mean(np.abs((y_test - knn.predict(x_test)) / y_test)) * 100)


# r-squared measures how much of the variability in weekly sales is caused by the input variables.
# MAPE measures the percent difference between the model's predicted output and the actual output.
# 
# 
# Since knn would not create a good enough model, I found out that a simple linear regression might.

# #### Linear Regression Model

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr_model = LinearRegression()

lr_model.fit(x_train,y_train)


# In[ ]:


lr_model.score(x_test,y_test)


# This is a variance score, and it basically scores the model's predictions. 1 would be a perfect model, so .91 is a pretty good one. We can look at r-squared and MAPE to double check. 

# In[ ]:


print(x_test)
y_pred = lr_model.predict(x_test)
print("r-squared: ", metrics.r2_score(y_test, y_pred))
print("Mean Absolute Percentage Error: ", metrics.mean_absolute_percentage_error(y_test, y_pred))


# #### Same thing but without taking store number into account

# Both look great. Now let's see if our model works when we do not take the store number as part of our input. If this works, the model could be applied to any Walmart store in the US. 

# In[ ]:


walmart_dummies2 = pd.get_dummies(walmart,columns=['Holiday_Flag'])
print(walmart_dummies2.columns.to_list())

x2 = walmart_dummies2.drop(["Store", "Date", "Weekly_Sales"], axis=1)
y2 = walmart_dummies2.Weekly_Sales

x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.20)


# In[ ]:


model2 = KNeighborsRegressor(n_neighbors = 5)

knn = model2.fit(x_train2, y_train2)

print('r-squared value:', metrics.r2_score(y_train2, knn.predict(x_train2)))
print('mean absolute percentage error: ', np.mean(np.abs((y_test2 - knn.predict(x_test2)) / y_test2)) * 100)


# Nope, it's a much worse model.

# In[ ]:


lr_model2 = LinearRegression()

lr_model2.fit(x_train2,y_train2)

y_pred2 = lr_model2.predict(x_test2)
print("r-squared: ", metrics.r2_score(y_test2, y_pred2))
print("Mean Absolute Percentage Error: ", metrics.mean_absolute_percentage_error(y_test2, y_pred2))


# Yeah, it's terrible. Unfortunately, this model can only be used by the 45 stores that the dataset is made from. However, if more Walmart stores start collecting and adding to the set, it will be very easy to predict their weekly sales. 

# The user can now upload their data. It requires 51 features: temperature, fuel price, CPI, unemployment rate, Store 1-45 (with a 1 indicating the user's store and 0 in all the others), Holiday Flag 0 (with a 1 if the week is not a holiday week), and Holiday Flag 1 (with a 1 if the week is a holiday week). Preferrably, the user would upload their own .csv file with feature names; however, it's possible to use the model by inputting the data manually. 

# In the following cell, I gave the model a try. I used this week's conditions in Wallingford, CT and randomly chose to be in Store 1. For the second row, I arbitrarily changed the numbers from row 1. The model predicted that this hypothetical store would make 1,770,950 and 1,754,310 sales for each of the weeks.

# In[ ]:


user_input = [[72, 4.49, 289.109, 3.6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [83, 4.67, 289.109, 3.6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
user_preds = lr_model.predict(user_input)


# In[ ]:


print("Sales Predictions by Week")
for i in range(len(user_preds)):
    print("Week " + str(i + 1) + ": ", user_preds[i], "sales")


# If I could keep working on this project, I would continue to develop the user interface, making it easier for the user to input their data. Also, I would look at collecting more data, especially more features, and integrating them into this model.
