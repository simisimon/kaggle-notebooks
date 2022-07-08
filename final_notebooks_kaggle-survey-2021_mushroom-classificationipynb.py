#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


# In[ ]:


data = pd.read_csv("../input/mushroom-classification/mushrooms.csv")


# In[ ]:


data.head()


# In[ ]:


values = {"b":"bell","c":"conical","x":"convex","f":"flat","k":"knobbed","s":"sunken"}
data["cap-shape"]=data["cap-shape"].replace(values)
values2 = {"f": "fibrous", "g": "grooves","y":"scaly","s": "smooth"}
data["cap-surface"]=data["cap-surface"].replace(values2)
values3 = {"n":"brown","b":"buff","c":"cinnamon","g":"gray","r":"green","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"}
data["cap-color"]=data["cap-color"].replace(values3)
values4 = {"a":"almond","l":"anise","c":"creosote","y":"fishy","f":"foul","m":"musty","n":"none","p":"pungent","s":"spicy"}
data["odor"]=data["odor"].replace(values4)
values5 = {"a":"attached","f":"free"}
data["gill-attachment"]=data["gill-attachment"].replace(values5)
values6 = {"c":"close","w":"crowded"}
data["gill-spacing"]=data["gill-spacing"].replace(values6)
values7 = {"b":"broad","n":"narrow"}
data["gill-size"]=data["gill-size"].replace(values7)
values8 = {"k":"black","b":"buff","n":"brown","h":"chocolate","g":"gray","r":"green","o":"orange","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"}
data["gill-color"]=data["gill-color"].replace(values8)
values9 = {"t":"tapering","e":"enlarging"}
data["stalk-shape"]=data["stalk-shape"].replace(values9)
values10 = {"b":"bulbous","c":"club","e":"equal","z":"rhizomorphs","r":"rooted","?":"?"}
data["stalk-root"]=data["stalk-root"].replace(values10)
values11 = {"s":"smooth","k":"silky","f":"fibrous","y":"scaly"}
data["stalk-surface-above-ring"]=data["stalk-surface-above-ring"].replace(values11)
data["stalk-surface-below-ring"]=data["stalk-surface-below-ring"].replace(values11)
values12 = {"n":"brown","b":"buff","c":"cinnamon","g":"gray","p":"pink","e":"red","w":"white","y":"yellow","o":"orange"}
data["stalk-color-above-ring"]=data["stalk-color-above-ring"].replace(values12)
data["stalk-color-below-ring"]=data["stalk-color-below-ring"].replace(values12)
veil_type = {"p":"partial","u":"universal"} 
data["veil-type"]=data["veil-type"].replace(veil_type)
veil_color={"n":"brown","o":"orange","w":"white","y":"yellow"} 
data["veil-color"]=data["veil-color"].replace(veil_color)
ring_number= {"n":"none","o":"one","t":"two"}
data["ring-number"]=data["ring-number"].replace(ring_number)
ring_type={"c":"cobwebby","e":"evanescent","f":"flaring","l":"large","n":"none","p":"pendant","s":"sheathing","z":"zone"}
data["ring-type"]=data["ring-type"].replace(ring_type)
spore_print_color= {"k":"black","n":"brown","b":"buff","h":"chocolate","r":"green","o":"orange","u":"purple","w":"white","y":"yellow"}
data["spore-print-color"]=data["spore-print-color"].replace(spore_print_color)
population={"a":"abundant","c":"clustered","n":"numerous","s":"scattered","v":"several","y":"solitary"}
data["population"]=data["population"].replace(population)
habitat={"g":"grasses","l":"leaves","m":"meadows","p":"paths","u":"urban","w":"waste","d":"woods"}
data["habitat"]=data["habitat"].replace(habitat)
bruises={"t":"bruises","f":"no"}
data["bruises"]=data["bruises"].replace(bruises)


# In[ ]:


df = data
df


# # Exploring Data
# ---

# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# #### There is no Null values

# In[ ]:


# I noticed that in the satlk_root feature there is an unknown value ("?") keeps showing
df['stalk-root'].value_counts()


# #### since a lot of records take this value it's better that we keep it (there exists some situations when we don't know the root of the plant) 

# In[ ]:


#it seems that the column "veil-type" has only one value so dropping it won't affect our work
print(df['veil-type'].value_counts())
df.drop('veil-type',axis = 1, inplace=True)


# ### In order to get acurate results and acurate measures for acuurcy we have to check for imbalance in the dataset

# In[ ]:


# to get the percentage of each label of the class
classLabelsCounts = df["class"].value_counts()
print(classLabelsCounts)

colors = sns.color_palette('mako')
plt.figure(figsize=(8, 8))
plt.pie(classLabelsCounts , labels = ["Poisonous", "Edible"], colors = colors, autopct = "%.0f%%")
plt.show()


# #### We don't need to use oversampling to handle imbalance , the data is well balanced

# # EDA
# ---

# In[ ]:


sns.set()
featuresCount = len(df.columns) - 1
f, axes = plt.subplots(featuresCount ,2, figsize=(20,150), sharey = False) 
k = 1

axes[0][0].set_title('Distribution of Mushroom by Feature and Class',fontsize=15,fontfamily='serif',fontweight='bold')    
axes[0][1].set_title('Distribution of Mushroom by Feature',fontsize=15,fontfamily='serif',fontweight='bold')

for i in range(0,featuresCount):
    s = sns.countplot(x = df.columns[k], data = df, hue = 'class', ax=axes[i][0], palette = ['#512b58','#fe346e'])
    axes[i][0].set_xlabel(df.columns[k], fontsize=20)
    axes[i][0].set_ylabel('Number of the Mushrooms', fontsize=15)
    axes[i][0].tick_params(labelsize=10)
    axes[i][0].legend(['Poisonous', 'Edible'], loc='upper right', prop={'size': 10})
    
    for p in s.patches:
        s.annotate(format(p.get_height()), 
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha = 'center', va = 'center', 
        xytext = (0, 9), 
        fontsize = 8,
        textcoords = 'offset points')
    
    
    sns.countplot(x=df.columns[k],
             data=df,
             ax=axes[i][1],
             palette="mako",
             order=df[df.columns[k]].value_counts().index
             )
    axes[i][1].set_ylabel('Number of the Mushrooms', fontsize=15)
    axes[i][1].set_xlabel(df.columns[k], fontsize=20)

    
    k = k+1


# # Data Preprocessing
# ---

# In[ ]:


# label encoding
mappings = dict()

le = LabelEncoder()

for column in df.columns:
    df[column] = le.fit_transform(df[column])
    mappings_dict = {index: label for index, label in enumerate(le.classes_)}
    mappings[column] = mappings_dict


# In[ ]:


for col in mappings.keys():
    print(f"{col} --> {mappings[col]}")
    print("--------------------------------------------------------")


# In[ ]:


mask = np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(28, 20))
sns.heatmap(df.corr(),annot=True, mask = mask, cmap = "mako")
plt.show()


# ### From the correlation heat map we can see that there is no dominanting feature and pretty much all the features correlate to the debendent variable "class"

# In[ ]:


# scaling the features and splitting the data
X = df.iloc[:,1:]
y = df.iloc[:,0]


scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_scaled, y, test_size=0.2, shuffle=True, random_state=77)


# In[ ]:


X_scaled.head()


# In[ ]:


y.head()


# # Training the models
# ---

# In[ ]:


log_model = LogisticRegression(max_iter=50000)
svm_model = SVC(kernel='rbf')
knn_model = KNeighborsClassifier()


# In[ ]:


# cross validation phase with 10 folds to select the best model
result_knn = cross_validate(knn_model,Xtrain,Ytrain,cv = 10)
result_log = cross_validate(log_model, Xtrain,Ytrain,cv = 10)
result_svm = cross_validate(svm_model, Xtrain,Ytrain,cv = 10)


# In[ ]:


model_results = [("Logistic Regression",result_log),("Support Vector Machine",result_svm),("KNN",result_knn)]

for name, res in model_results:
    print(f"{name} avg training time: {round(res['fit_time'].sum() / len(res['fit_time']) * 1000,1)} ms",)
    print(f"{name} avg testing time: {round(res['score_time'].sum() / len(res['score_time']) * 1000,1)} ms",)
    print(f"{name} avg teseting score: {res['test_score'].sum() / len(res['test_score']) * 100}%",)
    print (30*"-")
    


# In[ ]:


log_model.fit(Xtrain,Ytrain)
svm_model.fit(Xtrain, Ytrain)
knn_model.fit(Xtrain, Ytrain)


# In[ ]:


Ypred_log = log_model.predict(Xtest)
Ypred_svm = svm_model.predict(Xtest)
Ypred_knn = knn_model.predict(Xtest)


# In[ ]:


print("Classification report (Logistic Regression)")
print(classification_report(Ytest,Ypred_log))
print("-------------------------------------")
print("Classification report (SVM)")
print(classification_report(Ytest,Ypred_svm))
print("-------------------------------------")
print("Classification report (KNN)")
print(classification_report(Ytest,Ypred_knn))


# In[ ]:


f, axes = plt.subplots(1 ,3, figsize=(20,5))
cf_matrix = confusion_matrix(Ytest, Ypred_log)
sns.heatmap(cf_matrix,ax = axes[0], annot=True, fmt="1", cmap ="mako");
axes[0].set_title('Logistic Regression',fontsize=15,fontfamily='serif',fontweight='bold')  

cf_matrix = confusion_matrix(Ytest, Ypred_svm)
sns.heatmap(cf_matrix,ax = axes[1], annot=True, fmt="1", cmap ="mako");
axes[1].set_title('SVM',fontsize=15,fontfamily='serif',fontweight='bold') 

cf_matrix = confusion_matrix(Ytest, Ypred_knn)
sns.heatmap(cf_matrix,ax = axes[2], annot=True, fmt="1", cmap ="mako");
axes[2].set_title('KNN',fontsize=15,fontfamily='serif',fontweight='bold');


# #### We can see from the classification report and the training score that there is no overfitting in any of the three models 

# # Feature Selection
# ---

# In[ ]:


# i selected the top 10 features to use them in the Deployment
kbest = SelectKBest(score_func=chi2,k=5)
kbest.fit(df.drop('class',axis=1) ,df['class'])
scores = pd.DataFrame(kbest.scores_)
columns = pd.DataFrame(df.drop('class',axis=1).columns)
KbestScores = pd.concat([columns,scores],axis=1)
KbestScores.columns = ['Attribute','Score']
KbestScores.sort_values(by='Score',ascending=False)


# In[ ]:


columns = pd.DataFrame(df.drop('class',axis=1).columns)
KbestScores = pd.concat([columns,scores],axis=1)
KbestScores.columns = ['Attribute','Score']
KbestScores.sort_values(by='Score',ascending=False,inplace=True)


# In[ ]:


X_Kbest = df.loc[:,KbestScores["Attribute"][:10]]
X_trainK,X_testK,Y_trainK,Y_testK = train_test_split(X_Kbest, y)

svm_model_Kbest = SVC(kernel='rbf')


# In[ ]:


svm_model_Kbest.fit(X_trainK, Y_trainK)


# In[ ]:


Y_predK = svm_model_Kbest.predict(X_testK)
# svm_model_Kbest.n_features_
print(classification_report(Y_testK,Y_predK))


# In[ ]:


KbestScores["Attribute"][:10]
svm_model_Kbest.predict(pd.DataFrame(dict(X.loc[2,KbestScores["Attribute"][:10]]), index=[0]))


# In[ ]:


pickle.dump(svm_model_Kbest, open("model.pkl", 'wb'))


# In[ ]:


KbestScores["Attribute"][:10]


# In[ ]:


dict(X.loc[0,KbestScores["Attribute"][:10]])


# In[ ]:


dict(X.loc[2,KbestScores["Attribute"][:10]])

