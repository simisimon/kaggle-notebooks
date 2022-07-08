#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#standard libraries
import pandas as pd
import numpy as np

#visualization tools
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_theme(style="whitegrid") 

#scikit-learn 
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# xgboost

from xgboost import XGBRegressor


# In[ ]:


# datasetni yuklab df ga yuklab olaman
df = pd.read_csv("../input/aviachipta-narxini-bashorat-qilish/train_data.csv")
df


# In[ ]:


# dataframe ning ustunlari haqidagi ma'lumot 
df.info()


# In[ ]:


# nechta aviakompaniya bor va ularning datasetdagi soni
df['airline'].value_counts()


# In[ ]:


#vizualizatsiya
comps = list(df['airline'].unique())
vals = list(df['airline'].value_counts())
exp = [0.2, 0, 0, 0, 0, 0]
plt.figure(figsize=(10, 8))
plt.pie(vals, labels=comps, shadow=True, explode=exp)
plt.show()


# In[ ]:


# samolyotlarni ham tekshiraman\
df['flight'].value_counts()


# In[ ]:


# samolyotning modeli: MASALAN Vistara = UK-xxx.
df['flight'].apply(lambda x: x[:2]).value_counts()


# In[ ]:


# qolgan matnli ustunlar uchun vizualizatsiya
fig, ax = plt.subplots(3, 2, figsize=(16, 14))

sns.countplot(ax=ax[0, 0], data=df, x='source_city')
sns.countplot(ax=ax[0, 1], data=df, x='destination_city')
sns.countplot(ax=ax[1, 0], data=df, x='departure_time')
sns.countplot(ax=ax[1, 1], data=df, x='arrival_time')
sns.countplot(ax=ax[2, 0], data=df, x='stops')
sns.countplot(ax=ax[2, 1], data=df, x='class');


# In[ ]:


# datasetni source_city va destination_city ustunlarini tekshiruvdan o'tkazib olaman
# ya'ni 'Mumbai' => 'Dehli'(to'g'ri) | 'Mumbai' => 'Mumbai'(noto'g'ri)
for i in range(len(df)):
    if df['source_city'][i] == df['destination_city'][i]:
        print('Xatolik bor!')


# In[ ]:


df.describe()


# In[ ]:


df.drop(['id','flight'], axis=1, inplace=True)


# # Machine Learning StratifiedShuffleSplit orqali

# In[ ]:


strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=15)
for tr_idx, te_idx in strat_split.split(df, df['class']):
    st_trainset = df.loc[tr_idx]
    st_testset = df.loc[te_idx]


# In[ ]:


st_trainset.head()


# In[ ]:


st_testset.head()


# In[ ]:


X_train = st_trainset.drop('price', axis=1)
y = st_trainset['price'].copy()

X_num = X_train[['duration', 'days_left']]


# In[ ]:


# NUM PIPELINE
num_pipeline = Pipeline([
            ('std_scaler', StandardScaler())
])


# In[ ]:


# FULL PIPELINE

num_attribs = list(X_num)
cat_attribs = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']

full_pipeline = ColumnTransformer([
              ('num', num_pipeline, num_attribs),
              ('cat', OneHotEncoder(), cat_attribs)
])


# In[ ]:


X_prepared = full_pipeline.fit_transform(X_train)


# ## Linear Regression

# In[ ]:


LR_model = LinearRegression()

LR_model.fit(X_prepared, y)
X_test = st_testset.drop('price', axis=1)
y_test = st_testset['price'].copy()
X_test_prepared = full_pipeline.transform(X_test)
y_predicted = LR_model.predict(X_test_prepared)

mae = mean_absolute_error(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)

print("MAE=", np.around(mae))
print("RMSE=", np.around(np.sqrt(mse)))


# ## DecisionTree Regressor

# In[ ]:


DT_model = DecisionTreeRegressor()

DT_model.fit(X_prepared, y)
y_predicted = DT_model.predict(X_test_prepared)

mae = mean_absolute_error(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)

print("MAE=", np.around(mae))
print("RMSE=", np.around(np.sqrt(mse)))


# ## RandomForest Regressor

# In[ ]:


RF_model = RandomForestRegressor()

RF_model.fit(X_prepared, y)
y_predicted = RF_model.predict(X_test_prepared)

mae = mean_absolute_error(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)

print("MAE=", np.around(mae))
print("RMSE=", np.around(np.sqrt(mse)))


# ## XGBoost Regressor

# In[ ]:


XGB_model = XGBRegressor()

XGB_model.fit(X_prepared, y)
y_predicted = XGB_model.predict(X_test_prepared)

mae = mean_absolute_error(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)

print("MAE=", np.around(mae))
print("RMSE=", np.around(np.sqrt(mse)))


# # Submission

# In[ ]:


df2 = pd.read_csv("../input/aviachipta-narxini-bashorat-qilish/test_data.csv")
df2


# In[ ]:


for i in range(len(df2)):
    if df2['source_city'][i] == df2['destination_city'][i]:
        print('Xatolik bor!')


# In[ ]:


df2.drop(['id', 'flight'], axis=1, inplace=True)
df2.head()


# In[ ]:


df2.describe()


# In[ ]:


forsub = full_pipeline.transform(df2)

# Submission uchun RandomForest Regressorni tanladim
sub_predicted = RF_model.predict(forsub)
sub_predicted


# In[ ]:


len(sub_predicted)


# In[ ]:


ss = pd.read_csv("../input/aviachipta-narxini-bashorat-qilish/sample_solution.csv")
ss.head()


# In[ ]:


ss['price'] = sub_predicted

ss.to_csv("submission.csv", index=False)


# In[ ]:


pd.read_csv("./submission.csv")


# In[ ]:




