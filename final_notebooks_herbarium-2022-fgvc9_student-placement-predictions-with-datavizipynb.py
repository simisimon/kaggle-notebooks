#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import os
os.listdir("../input/")


# In[ ]:


df = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")


# In[ ]:


print("The shape of the training dataset is : ", df.shape)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


dict = {}
for i in list(df.columns):
    dict[i] = df[i].value_counts().shape[0]

pd.DataFrame(dict,index=["unique count"]).transpose()


# In[ ]:


cat_cols = ['gender','ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
con_cols = ["ssc_p","hsc_p","degree_p","etest_p","mba_p","salary"]
target_col = ["status"]
print("The categorial cols are : ", cat_cols)
print("The continuous cols are : ", con_cols)
print("The target variable is :  ", target_col)


# In[ ]:


df[con_cols].describe().transpose()


# In[ ]:


df[cat_cols].describe().transpose()


# In[ ]:


df.isnull().sum()


# In[ ]:


df['salary'] = df['salary'].fillna(0)


# In[ ]:


df.isnull().sum()


# In[ ]:


df['SSC_Grade'] = pd.cut(x=df['ssc_p'], bins=[39.99,49.99,59.99,69.99,79.99,89.99,100],
                     labels=['E','D','C', 'B', 'A','S'])
df['HSC_Grade'] = pd.cut(x=df['hsc_p'], bins=[39.99,49.99,59.99,69.99,79.99,89.99,100],
                     labels=['E','D','C', 'B', 'A','S'])
df['Degree_Grade'] = pd.cut(x=df['degree_p'], bins=[39.99,49.99,59.99,69.99,79.99,89.99,100],
                     labels=['E','D','C', 'B', 'A','S'])      
df['ETEST_Grade'] = pd.cut(x=df['etest_p'], bins=[39.99,49.99,59.99,69.99,79.99,89.99,100],
                     labels=['E','D','C', 'B', 'A','S'])
df['MBA_Grade'] = pd.cut(x=df['mba_p'], bins=[39.99,49.99,54.99,59.99,64.99,69.99,80],
                     labels=['E','D','C', 'B', 'A','S'])        


# In[ ]:


df.head()


# In[ ]:


fig = plt.figure(figsize=(18,18))
gs = fig.add_gridspec(3,4)
gs.update(wspace=0.3, hspace=0.15)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,1])
ax4 = fig.add_subplot(gs[1,2])
ax5 = fig.add_subplot(gs[-2,0])
ax6 = fig.add_subplot(gs[1,3])
ax7 = fig.add_subplot(gs[0,3])

background_color = "#FFF8DC"

fig.patch.set_facecolor(background_color) 
ax0.set_facecolor(background_color) 
ax1.set_facecolor(background_color) 
ax2.set_facecolor(background_color) 
ax3.set_facecolor(background_color) 
ax4.set_facecolor(background_color) 
ax5.set_facecolor(background_color) 
ax6.set_facecolor(background_color) 
ax7.set_facecolor(background_color) 

# Title of the plot
ax0.text(0.5,0.5,"Counting Categorical features \nand checking status\n among various categories\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 16,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')

ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.tick_params(left=False, bottom=False)

# performance of each field Count

ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax1, data=df, x='gender',palette = 'Oranges', hue='status',edgecolor='black',**{'hatch':'/','linewidth':2})
ax1.set_xlabel("gender",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax1.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax2.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax2, data=df, x='ssc_b',palette = 'Oranges', hue='status',edgecolor='black',**{'hatch':'/','linewidth':2})
ax2.set_xlabel("ssc_b",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax2.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax3, data=df, x='hsc_b',palette = 'Oranges',hue='status',edgecolor='black',**{'hatch':'/','linewidth':2})
ax3.set_xlabel("hsc_b",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax3.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax4.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax4, data=df, x='hsc_s',palette = 'Oranges',hue='status',edgecolor='black',**{'hatch':'/','linewidth':2})
ax4.set_xlabel("hsc_s",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax4.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax5.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax5, data=df, x='degree_t',palette = 'Oranges',hue='status',edgecolor='black',**{'hatch':'/','linewidth':2})
ax5.set_xlabel("degree_t",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax5.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax6.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax6, data=df, x='workex',palette = 'Oranges',hue='status',edgecolor='black',**{'hatch':'/','linewidth':2})
ax6.set_xlabel("workex",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax6.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax7.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax7, data=df, x='specialisation',palette = 'Oranges',hue='status',edgecolor='black',**{'hatch':'/','linewidth':2})
ax7.set_xlabel("specialisation",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax7.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax0.spines["top"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax0.spines["bottom"].set_visible(False)
ax0.spines["right"].set_visible(False)

ax1.spines["top"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax3.spines["top"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.spines["right"].set_visible(False)

ax4.spines["top"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax4.spines["right"].set_visible(False)

ax5.spines["top"].set_visible(False)
ax5.spines["left"].set_visible(False)
ax5.spines["right"].set_visible(False)

ax6.spines["top"].set_visible(False)
ax6.spines["left"].set_visible(False)
ax6.spines["right"].set_visible(False)

ax7.spines["top"].set_visible(False)
ax7.spines["left"].set_visible(False)
ax7.spines["right"].set_visible(False)


# In[ ]:


fig = plt.figure(figsize=(25,25))
gs = fig.add_gridspec(3,4)
gs.update(wspace=0.3, hspace=0.15)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])

background_color = "#FFF8DC"

fig.patch.set_facecolor(background_color) 
ax0.set_facecolor(background_color) 
ax1.set_facecolor(background_color) 

# Title of the plot
ax0.text(0.5,0.5,"Understanding the\n target variable\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 16,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')

ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.tick_params(left=False, bottom=False)

ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax1, data=df, x='status',palette = 'Oranges', edgecolor='black',**{'hatch':'/','linewidth':2})
ax1.set_xlabel("status",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax1.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax0.spines["top"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax0.spines["bottom"].set_visible(False)
ax0.spines["right"].set_visible(False)

ax1.spines["top"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)


# In[ ]:


fig = plt.figure(figsize=(22,15))
gs = fig.add_gridspec(3,3)
gs.update(wspace=0.3, hspace=0.15)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,1])
ax4 = fig.add_subplot(gs[1,2])
ax5 = fig.add_subplot(gs[2,2])
ax6 = fig.add_subplot(gs[2,1])

background_color = "#ffe6e6"
fig.patch.set_facecolor(background_color) 
ax0.set_facecolor(background_color) 
ax1.set_facecolor(background_color) 
ax2.set_facecolor(background_color) 
ax3.set_facecolor(background_color) 
ax4.set_facecolor(background_color) 
ax5.set_facecolor(background_color) 
ax6.set_facecolor(background_color) 

# Title of the plot
ax0.text(0.6,-0.5,"Understanding Continuous \nfeatures\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')

ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.tick_params(left=False, bottom=False)

# performance of each field Count
#"ssc_p","hsc_p","degree_p","etest_p","mba_p","salary"
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.histplot(ax=ax1, data=df[df['salary']>0], x='salary',palette = 'RdPu', kde=True, edgecolor='black')
ax1.set_xlabel("salary",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax1.set_ylabel("Count",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax2.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.histplot(ax=ax2, data=df, x = 'hsc_p', hue="status",palette = 'RdPu', bins=20, kde=True, edgecolor='black')
ax2.set_xlabel("hsc_p",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax2.set_ylabel("Count",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.histplot(ax=ax3, data=df, x = 'degree_p', hue="status",palette = 'RdPu', bins=20, kde=True, edgecolor='black')
ax3.set_xlabel("degree_p",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax3.set_ylabel("Count",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax4.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.histplot(ax=ax4, data=df, x = 'etest_p', hue="status",palette = 'RdPu', bins=20, kde=True, edgecolor='black')
ax4.set_xlabel("etest_p",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax4.set_ylabel("Count",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax5.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.histplot(ax=ax5, data=df, x = 'mba_p', hue="status",palette = 'RdPu', bins=20, kde=True, edgecolor='black')
ax5.set_xlabel("mba_p",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax5.set_ylabel("Count",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")


ax6.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.histplot(ax=ax6, data=df, x = 'ssc_p', hue="status",palette = 'RdPu', bins=20, kde=True, edgecolor='black')
ax6.set_xlabel("ssc_p",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax6.set_ylabel("Count",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")



ax0.spines["top"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax0.spines["bottom"].set_visible(False)
ax0.spines["right"].set_visible(False)

ax1.spines["top"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax3.spines["top"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.spines["right"].set_visible(False)

ax4.spines["top"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax4.spines["right"].set_visible(False)

ax5.spines["top"].set_visible(False)
ax5.spines["left"].set_visible(False)
ax5.spines["right"].set_visible(False)

ax6.spines["top"].set_visible(False)
ax6.spines["left"].set_visible(False)
ax6.spines["right"].set_visible(False)


# In[ ]:


fig = plt.figure(figsize=(25,25))
gs = fig.add_gridspec(3,3)
gs.update(wspace=0.3, hspace=0.15)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2= fig.add_subplot(gs[0,2])

background_color = "#B0E0E6"
fig.patch.set_facecolor(background_color) 
ax0.set_facecolor(background_color) 
ax1.set_facecolor(background_color) 
ax2.set_facecolor(background_color) 

ax0.text(0.6,0.5,"Salary analysis of Placed Students\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')

ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.tick_params(left=False, bottom=False)

ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.histplot(ax=ax1,data=df[df['salary']>0], x='salary', kde=True, edgecolor='black')
ax1.set_xlabel("Salary",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax1.set_ylabel("Count",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax2.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.boxplot(ax=ax2,data=df[df['salary']>0], x='salary')
ax2.set_xlabel("Salary",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax2.set_ylabel("Count",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax0.spines["top"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax0.spines["bottom"].set_visible(False)
ax0.spines["right"].set_visible(False)

ax1.spines["top"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)



# In[ ]:


sns.set(rc={'figure.figsize':(12,8)})
sns.set(style="white", color_codes=True)
sns.jointplot(data=df,x="etest_p", y="salary", kind='kde',palette="magma")


# In[ ]:


fig = plt.figure(figsize=(25,25))
gs = fig.add_gridspec(3,3)
gs.update(wspace=0.3, hspace=0.15)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])


background_color = "#EEDFCC"
fig.patch.set_facecolor(background_color) 
ax0.set_facecolor(background_color) 
ax1.set_facecolor(background_color) 
ax2.set_facecolor(background_color) 

ax0.text(0.6,0.5,"Salary analysis of Placed Students\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')

ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.tick_params(left=False, bottom=False)

ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.swarmplot(ax=ax1,y = df['salary'], x = df['specialisation'])
ax1.set_xlabel("specialisation",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax1.set_ylabel("salary",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")


ax2.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.swarmplot(y = df['salary'], x = df['gender'],palette='CMRmap_r')
ax2.set_xlabel("gender",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax2.set_ylabel("salary",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")

ax0.spines["top"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax0.spines["bottom"].set_visible(False)
ax0.spines["right"].set_visible(False)

ax1.spines["top"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)



# In[ ]:


df.head()


# In[ ]:


df.drop(['mba_p'], axis = 1,inplace=True) 
df.head()


# Label Encoding

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
object_cols=['gender','workex','specialisation','status','hsc_b','ssc_b']

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    df[col] = label_encoder.fit_transform(df[col])
df.head()


# One hot encoding

# In[ ]:


dummy_hsc_s=pd.get_dummies(df['hsc_s'], prefix='dummy')
dummy_degree_t=pd.get_dummies(df['degree_t'], prefix='dummy')
dummy_SSC_Grade=pd.get_dummies(df['SSC_Grade'], prefix='dummyssc')
dummy_HSC_Grade=pd.get_dummies(df['HSC_Grade'], prefix='dummyhsc')
dummy_Degree_Grade=pd.get_dummies(df['Degree_Grade'], prefix='dummydegree')
dummy_ETEST_Grade=pd.get_dummies(df['ETEST_Grade'], prefix='dummyetest')
dummy_MBA_Grade=pd.get_dummies(df['MBA_Grade'], prefix='dummymba')

placement_coded = pd.concat([df,dummy_hsc_s,dummy_degree_t,dummy_SSC_Grade,dummy_HSC_Grade,dummy_Degree_Grade,dummy_ETEST_Grade,dummy_MBA_Grade],axis=1)
placement_coded.drop(['hsc_s','degree_t','salary','SSC_Grade','HSC_Grade','Degree_Grade','ETEST_Grade','MBA_Grade'],axis=1, inplace=True)
placement_coded.head()


# In[ ]:


#Assigning the target(y) and predictor variable(X)


# In[ ]:


X=placement_coded.drop(['status','ssc_b','hsc_b'],axis=1)
y=placement_coded.status


# In[ ]:


#Train and Test Split (80:20)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8,random_state=1)
print("Input Training:",X_train.shape)
print("Input Test:",X_test.shape)
print("Output Training:",y_train.shape)
print("Output Test:",y_test.shape)


# Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# Confusion matrix and Classification report

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",confusion_matrix)
from sklearn.metrics import classification_report
print("Classification Report:\n",classification_report(y_test, y_pred))


# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rt=RandomForestClassifier(n_estimators=100)
rt.fit(X_train,y_train)
y_pred=rt.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Naive Bayes Classifier with Cross Validation

# In[ ]:


#Importing and fitting
from sklearn.naive_bayes import BernoulliNB 
from sklearn.model_selection import cross_val_score
gnb = BernoulliNB() 
gnb.fit(X_train, y_train) 
  
#Applying and predicting 
y_pred = gnb.predict(X_test) 
cv_scores = cross_val_score(gnb, X, y, 
                            cv=10,
                            scoring='precision')
print("Cross-validation precision: %f" % cv_scores.mean())

