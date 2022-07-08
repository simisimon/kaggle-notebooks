#!/usr/bin/env python
# coding: utf-8

# # Fertilizer Type Prediction

# ## Importing the libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ## Load the dataset

# In[ ]:


df = pd.read_csv("../input/fertilizer-prediction/Fertilizer Prediction.csv")
df


# In[ ]:


df.sample(10)


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df["Fertilizer Name"].value_counts()


# In[ ]:


df.nunique()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# ## EDA

# In[ ]:


df.hist(figsize=(10,10))


# ## Train Test Split

# In[ ]:


y = df["Fertilizer Name"]
X = df.drop(["Fertilizer Name"],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


# In[ ]:


print('X_train shape:',X_train.shape)
print('y_train shape:',y_train.shape)
print('X_test shape:',X_test.shape)
print('y_test shape:',y_test.shape)


# ## Checking numerical and categorical columns

# In[ ]:


num_cols = [col for col in X_train.columns if X_train[col].dtypes!='O']
num_cols


# In[ ]:


cat_cols = [col for col in X_train.columns if X_train[col].dtypes=='O']
cat_cols


# ## Scaling numerical columns

# In[ ]:


X_train[num_cols].head()


# In[ ]:


from sklearn.preprocessing import RobustScaler
r = RobustScaler()
r.fit(X_train[num_cols])


# In[ ]:


X_train_num_scaled = r.transform(X_train[num_cols])
X_train_num_scaled 


# In[ ]:


X_test_num_scaled = r.transform(X_test[num_cols])
X_test_num_scaled


# ## Encoding categorical columns

# In[ ]:


X_train[cat_cols].head()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
o = OneHotEncoder(sparse=False, handle_unknown='ignore')
o.fit(X_train[cat_cols])


# In[ ]:


X_train_cat_encoded = o.transform(X_train[cat_cols])
X_train_cat_encoded


# In[ ]:


X_test_cat_encoded = o.transform(X_test[cat_cols])
X_test_cat_encoded


# In[ ]:


X_train = pd.DataFrame(np.concatenate((X_train_num_scaled, X_train_cat_encoded), axis=1))
X_train


# In[ ]:


X_test = pd.DataFrame(np.concatenate((X_test_num_scaled, X_test_cat_encoded), axis=1))
X_test


# ## Encoding the target

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit(y_train)
y_train_encoded = le.transform(y_train)


# In[ ]:


y_test_encoded = le.transform(y_test)


# In[ ]:


y_train_encoded


# In[ ]:


y_test_encoded


# ## Model Training and Evaluation

# In[ ]:


## model 1
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train_encoded)
print('Training accuracy: ', model.score(X_train,y_train_encoded))     


# In[ ]:


y_pred = model.predict(X_test)
y_pred


# In[ ]:


from sklearn.metrics import plot_confusion_matrix, classification_report

print("Plot confusion matrix",plot_confusion_matrix(model,X_test,y_test_encoded))


# In[ ]:


print("Classification Report",classification_report(y_test_encoded,y_pred))


# In[ ]:


## model 2

from sklearn.tree import DecisionTreeClassifier
model_2 = DecisionTreeClassifier()
model_2.fit(X_train,y_train_encoded)
print('Training accuracy: ',model_2.score(X_train,y_train_encoded))


# In[ ]:


y_prediction = model_2.predict(X_test)
y_prediction


# In[ ]:


print("Plot confusion matrix",plot_confusion_matrix(model_2,X_test,y_test_encoded))


# In[ ]:


print("Classification Report",classification_report(y_test_encoded,y_prediction))


# In[ ]:


## model 3
from sklearn.ensemble import RandomForestClassifier
model_3=RandomForestClassifier()
model_3.fit(X_train,y_train_encoded)
print('Training accuracy: ',model_3.score(X_train,y_train_encoded))


# In[ ]:


y_predict = model_3.predict(X_test)
y_predict


# In[ ]:


print("Plot confusion matrix",plot_confusion_matrix(model_3,X_test,y_test_encoded))
    


# In[ ]:


print("Classification Report",classification_report(y_test_encoded,y_predict))


# In[ ]:


## model 4

from sklearn.ensemble import GradientBoostingClassifier
model_4 = GradientBoostingClassifier()
model_4.fit(X_train,y_train_encoded)
print('Training accuracy: ',model_4.score(X_train,y_train_encoded))


# In[ ]:


y_predic = model_4.predict(X_test)
y_predic


# In[ ]:


print("Plot confusion matrix",plot_confusion_matrix(model_4,X_test,y_test_encoded))


# In[ ]:


print("Classification Report",classification_report(y_test_encoded,y_predic))

