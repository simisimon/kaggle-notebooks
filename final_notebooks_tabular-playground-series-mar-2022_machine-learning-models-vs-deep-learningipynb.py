#!/usr/bin/env python
# coding: utf-8

# #                                                              CONTENT
# 1. [UNDERSTANDING CASE AND DATA](#1) <a id=15></a>
# 2. [DATA VISUALIZATION](#2)
# 3. [MACHINE LEARNING](#3)
#    - 3.1[Support Vector Machine](#4)
#    - 3.2[RandomForest](#5)
#    - 3.3[GradientBoosting](#6)
#    - 3.4[ XGBoosT](#7)
#    - 3.4[CATBoosT](#8)
# 4. [DEEP LEARNING](#9)
#    - 4.1[MODEL](#10)
#    - 4.1[OVER FITTING](#11)
# 5. [CLASSIFICATION CONCLUSIONS(MACHINE LEARNING MODELS VS DEEP LEARNING MODEL) ](#12)
# 
# 

# # 1.UNDERSTANDING CASE AND DATA <a id=1></a>

# - About this dataset
# - Age : Age of the patient
# - Sex : Sex of the patient
# - exang: exercise induced angina (1 = yes; 0 = no)
# - ca: number of major vessels (0-3)
# - cp : Chest Pain type chest pain type
# - Value 1: typical angina
# - Value 2: atypical angina
# - Value 3: non-anginal pain
# - Value 4: asymptomatic
# - trtbps : resting blood pressure (in mm Hg)
# - chol : cholestoral in mg/dl fetched via BMI sensor
# - fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# - rest_ecg : resting electrocardiographic results
# - Value 0: normal
# - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
# - thalach : maximum heart rate achieved
# - target : 0= less chance of heart attack 1= more chance of heart attack

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data=pd.read_csv("../input/heart-attack-analysis-prediction-dataset/heart.csv")



# In[ ]:


data.head(10)


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# -THERE IS NO MISSING VALUE

# In[ ]:


data.info()


# In[ ]:


data.describe().transpose()


# # 2.DATA VISUALIZATION <a id=2></a>

# In[ ]:


data["number"]=np.ones(len(data)).astype("int64",copy=True) #I ADDED A COLUMN WHICH CONSIST OF ONES 
data1=data.groupby(["age"]).sum().reset_index()
plt.figure(figsize=(15,15))
sns.barplot(x=data1.age,y=data1.number,data=data1)
plt.xlabel("AGE")
plt.ylabel("DEATH PEOPLE")
plt.title("NUMBER OF HEART ATTACKED BY AGE ")
plt.show()


# -THIS TABLE SHOW US HEART ATTACK BY AGE, WE CAN SEE BETWEEN 50 AND 60 AGE OF PEOPLE HAD HEART ATTACKS MORE

# In[ ]:


data1=data[data["output"]==1].groupby(["age"]).sum().reset_index()
plt.figure(figsize=(15,15))
sns.barplot(x=data1.age,y=data1.number,data=data1)
plt.xlabel("AGE")
plt.ylabel("DEATH PEOPLE")
plt.title("NUMBER OF DEATH ATTACKED BY AGE ")
plt.show()


# -WE CAN SEE AGAIN BETWEEN 50 AND 60 AGE OF PEOPLE DIED DUE TO HEART ATTACK.

# In[ ]:


data1=data[data["output"]==0].groupby(["sex"]).sum().reset_index()[["sex","number"]]
data2=data[data["output"]==1].groupby(["sex"]).sum().reset_index()[["sex","number"]]


# In[ ]:


plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.pie(data1["number"],autopct = '%0.0f%%',labels=("WOMAN","MAN"), shadow=True,explode=[0.1,0])
plt.title("NUMBER OF NON-DEADS BY GENDER")
plt.subplot(1, 2, 2)
plt.pie(data2["number"],autopct = '%0.0f%%',labels=("WOMAN","MAN"), shadow=True,explode=[0.1,0],startangle=-50)
plt.title("NUMBER OF DEATH BY GENDER")
plt.show()


# -THE PEOPLE WHO HAVE BOTH HEART ATTACK AND DIE OF HEART ATTACK ARE MOSTLY MEN.

# In[ ]:


data4=data.groupby(["cp"]).sum().reset_index()[["cp","number"]]
data5=data[data["output"]==1].groupby(["cp"]).sum().reset_index()[["cp","number"]]
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
sns.barplot(x=["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"],y=data4.number,data=data4)
plt.xlabel("CP TYPE")
plt.title("NUMBER OF HEART ATTACK BY CP TYPE")
plt.subplot(1,2,2)
sns.barplot(x=["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"],y=data5.number,data=data5)
plt.xlabel("CP TYPE")
plt.title("NUMBER OF DEATH BY CP TYPE")
plt.show()


# -THOSE GRAPHICS SHOW US WHICH CHEST PAIN TYPE PEOPLE HEART ATTACK OR DIED

# In[ ]:


data6=round(data5.number/data4.number*100,0)
data6=data6.reset_index()
plt.figure(figsize=(10,10))
sns.barplot(x=["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"],y=data6["number"])
plt.title("NUMBER OF DEATH BY PERCENT")
plt.show()


# -ATYPICAL ANGINA WHICH IS CHEST PAIN TYPE IS MORE SERIOUSLY THAN OTHERS.

# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(data.corr(),annot=True)
plt.show()


# -THIS CORRELATION MATRIX SHOWS THAT HEART ATTACK DEPENDS ON cp(CHEST PAIN),thalachh(MAXIMUM HEART RATE ACHIEVED),slp(Slope)

# # 3.MACHINE LEARNING <a id=3></a>

# In[ ]:


from sklearn.model_selection import train_test_split 
x=data.iloc[:,0:-2]
y=data.iloc[:,-2]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.25)


# In[ ]:


from sklearn.preprocessing import StandardScaler
x_train=StandardScaler().fit_transform(x_train)
x_test=StandardScaler().fit_transform(x_test)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# # 3.1 Support Vector Machine <a id=4></a>

# In[ ]:


from sklearn.svm import SVC
svc=SVC(C=1,gamma=0.05,kernel="rbf",random_state=42)
svc.fit(x_train,y_train)
print("Accuracy score before the GridSearch method",accuracy_score(y_test,svc.predict(x_test)))
SVC=SVC()
parameters = {"kernel":('rbf','linear'),"C":np.arange(1,5,1),'gamma':[0.00001,0.005,0.01,0.05,0.1,0.5]}
GSCV=GridSearchCV(SVC,parameters)
GSCV.fit(x_train,y_train)
y_pred = GSCV.predict(x_test)
print("Accuracy score after the GridSearch method",accuracy_score(y_test,y_pred))
SVC_SCORE=(accuracy_score(y_test,y_pred))
print("best parameters",GSCV.best_params_)


 


# # 3.2 RandomForest <a id=5></a>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(random_state=42)
RFC.fit(x_train,y_train)
print("Accuracy score before the GridSearch method",accuracy_score(y_test,RFC.predict(x_test)))

rfc=RandomForestClassifier(random_state=42)
parameters = {"criterion":["entropy","gini"],"n_estimators":np.arange(1,50,1)}
GSCV=GridSearchCV(rfc,parameters)
GSCV.fit(x_train,y_train)
y_pred=GSCV.predict(x_test)
print("Accuracy score after the GridSearch method",accuracy_score(y_test,y_pred))
RandomForest_SCORE=(accuracy_score(y_test,y_pred))
print("best parameters",GSCV.best_params_)


 


# # 3.3 GradientBoosting <a id=6></a>

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
GBC=GradientBoostingClassifier(random_state=42)
GBC.fit(x_train,y_train)
print("Accuracy score before the GridSearch method",accuracy_score(y_test,GBC.predict(x_test)))

rfc=GradientBoostingClassifier(random_state=42)
parameters = {"criterion":["squared_error","mse"],"n_estimators":np.arange(1,50,1)}
GSCV=GridSearchCV(rfc,parameters)
GSCV.fit(x_train,y_train)
y_pred=GSCV.predict(x_test)
print("Accuracy score after the GridSearch method",accuracy_score(y_test,y_pred))
GradientBoosting_SCORE=(accuracy_score(y_test,y_pred))
print("best parameters",GSCV.best_params_)


# # 3.4 XGBoosT <a id=7></a> 

# In[ ]:


from xgboost import XGBClassifier
XGB=XGBClassifier(random_state=42)
XGB.fit(x_train,y_train)
y_pred=XGB.predict(x_test)
print("Accuracy score",accuracy_score(y_test,y_pred))
XGBClassifier_SCORE=(accuracy_score(y_test,y_pred))


# # 3.5 CATBoosT <a id=8></a> 

# In[ ]:


from catboost import CatBoostClassifier
CB=XGBClassifier(random_state=40)
CB.fit(x_train,y_train)
y_pred=CB.predict(x_test)
print("Accuracy score",accuracy_score(y_test,y_pred))
CatBoost_SCORE=(accuracy_score(y_test,y_pred))


# # 4.DEEP LEARNING <a id=9></a>

# # 4.1 MODEL <a id=10></a>

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[ ]:


tf.random.set_seed(0)
model=Sequential()
model.add(Dense(13,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy']) 
#WE USED sigmoid and binary_crossentropy BECAUSE OF OUR OUTPUT CONSISTED OF 0-1


# In[ ]:


history=model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),epochs=100,verbose=0)


# In[ ]:


y_pred=model.predict(x_test)
y_pred=(y_pred>0.5).astype(int)
from sklearn.metrics import  confusion_matrix
print(confusion_matrix(y_pred,y_test),"ysdaaadddddd")
print("Classification on Depp Learning Accuracy score",accuracy_score(y_test,y_pred))
Deep_Learning_Score=accuracy_score(y_test,y_pred)




# # 4.2 OVER FITTING PROBLEM <a id=11></a>

# ### Is there overfitting on our deep learning model?

# In[ ]:


graf=pd.DataFrame(history.history)
graf=graf.drop(["val_accuracy","accuracy"],axis=1)
graf.plot()



# -THIS CURVE OF VALUE LOSS SHOWS THAT THIS MODEL IS UNBALANCED.SO WE CAN SAY THIS MODEL HAS OVER FITTING SO WE HAVE TO FIX THIS PROBLEM WITH EARLY STOPPING METHOD

# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping
earlystopping=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)


# -WE USED EARLY STOPPING METHOD IN ORDER TO MAKE MINIMUM val_loss 

# In[ ]:


tf.random.set_seed(0)
model=Sequential()
model.add(Dense(13,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss="binary_crossentropy",metrics=['mse'])


# In[ ]:


history1=model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),epochs=100,verbose=1,callbacks=[earlystopping])


# In[ ]:


epoch=len(history1.epoch)


# -FOR OUR FIT VALUE OF EPOCH IS NOT 100 SO WE WILL START OUR MODEL AGAIN WITH NEW EPOCH

# In[ ]:


tf.random.set_seed(0)
model=Sequential()
model.add(Dense(13,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss="binary_crossentropy",metrics=['accuracy'])


# In[ ]:


history2=model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),epochs=epoch,verbose=0)
y_pred=model.predict(x_test)
y_pred=(y_pred>0.5).astype(int)
from sklearn.metrics import  confusion_matrix
print(confusion_matrix(y_pred,y_test),"ysdaaadddddd")
print("Classification on Depp Learning Accuracy score",accuracy_score(y_test,y_pred))
Deep_Learning_Score=accuracy_score(y_test,y_pred)


# In[ ]:


graf2=pd.DataFrame(history2.history)
graf2=graf2.drop(["val_accuracy","accuracy"],axis=1)
graf2.plot()


# -OUR NEW OUTPUT IS MORE BALANCED THAN OUR OLD OUTPUT.

# # 5.CLASSIFICATION CONCLUSIONS(MACHINE LEARNING MODELS VS DEEP LEARNING MODEL)  <a id=12></a>

# In[ ]:


df = pd.DataFrame(data =(SVC_SCORE,RandomForest_SCORE,GradientBoosting_SCORE,XGBClassifier_SCORE,CatBoost_SCORE,Deep_Learning_Score),index = ['Support Vector Machine','Random Forest','Gradient Boosting','XGBoost','CatBoost',"Deep Learning"], columns = ['Score'])
df


# -WE CAN SAY AS CONCLUSION THAT  WE CAN USE FOR THIS DATA CSV MACHINE LEARNING METHOD OR DEEP LEARNING
