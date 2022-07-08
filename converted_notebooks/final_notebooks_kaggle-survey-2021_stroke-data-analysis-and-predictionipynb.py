#!/usr/bin/env python
# coding: utf-8

# ### Importing the libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Importing Data
raw_data=pd.read_csv('../input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv')


# In[ ]:


data=raw_data.copy()
data


# In[ ]:


# The id column is not relevant
data.drop(columns=['id'],inplace=True) 
data


# ### Questions to ask:
# 1) Male/Female who has more strokes.
# 
# 2) People of which age group are more likely to get a stroke.
# 
# 3) Is hypertension a cause? 
# 
# 4) A person with heart disease is more likely to get a stroke (need confirmation).
# 
# 5) Marriage may be a cause of strokes.
# 
# 6) People working in private jobs may be the majority of people with strokes(mostly cause of stress).
# 
# 7) People living in urban areas have more chances of getting stroke? (need to confirm)
# 
# 8) Glucose levels are important and must be observed closely with other things.
# 
# 9) BMI must be closely observed with age and gender.
# 
# 10) People who smoke are more likely to get a stroke (need confirmation).
# 
# 

# In[ ]:


data.info()


# ### Descriptive Analytics

# In[ ]:


data.describe()


# ##### Observations : 
# 1) BMI contains missing values.
# 
# 2) The average age is 43.
# 
# 3) The average bmi is 28 (will change after imputation).
# 
# 4) The minimum age is questionable.
# 
# 5) Average glucose level is 106 (can be useful later).

# ### Handling Missing Values

# In[ ]:


def draw_missing_data_table(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
draw_missing_data_table(data)


# In[ ]:


# Imputing the missing values with the mean
data=data.fillna(np.mean(data['bmi']))


# In[ ]:


# Classifying data into numerical and categorical variables.
data_numerical=data[['age','avg_glucose_level','bmi']]
data_categorical=data[['gender', 'hypertension', 'heart_disease', 'ever_married','work_type', 'Residence_type', 
                       'smoking_status', 'stroke']]


# ### Numerical Variable analysis
# 

# In[ ]:


fig, ax = plt.subplots(figsize = (12,6))
fig.patch.set_facecolor('#f6f5f7')
ax.set_facecolor('#f6f5f5')
sns.kdeplot(data=data_numerical,shade=True,palette='rocket') # Distributions
# removing spines
for i in ["top","right"]:
    ax.spines[i].set_visible(False)
plt.title('Kde Plots for age, avg_glucose_level and bmi',weight='bold')


# #### Observations :
# 1) The avg_glucose_level is heavily skewed towards right and most of the distribution is between 50-150.
# 
# 2) The bmi is between 0 and 50 for most of the people.
# 
# 3) The age is distributed between 0 and 100 almost evenly.
# 
# 4) The data is not normally distributed (have to normalize or standardize).

# In[ ]:


# Skewness and kurtosis
s_k=[]
for i in data_numerical.columns:
    s_k.append([i,data_numerical[i].skew(),data_numerical[i].kurt()])
skew_kurt=pd.DataFrame(s_k,columns=['Columns','Skewness','Kurtosis'])
skew_kurt


# As a general rule of thumb: If skewness is less than -1 or greater than 1, the distribution is highly skewed. If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed. If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.

# ### Analysis with Stroke

# In[ ]:


fig = plt.figure(figsize=(20,13))
gs = fig.add_gridspec(3,3)
gs.update(wspace=0.4, hspace=0.4)
# adding figures
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])
ax4 = fig.add_subplot(gs[2,0])
ax5 = fig.add_subplot(gs[2,1])
axes=[ax0,ax1,ax2,ax3,ax4,ax5]
background_color = '#f6f5f7'
for i in axes:
    i.set_facecolor(background_color)
fig.patch.set_facecolor(background_color) 
#https://www.geeksforgeeks.org/kde-plot-visualization-with-pandas-and-seaborn/
sns.kdeplot(ax=ax0,x=data.loc[data['stroke']==1]['age'],color='crimson',label='Stroke',shade=True)
sns.kdeplot(ax=ax0,x=data.loc[data['stroke']==0]['age'],color='coral',label='No Stroke',shade=True)
ax0.legend(loc = 'upper left')
ax0.grid(linestyle='--', axis='y')

ax1.text(0.5,0.5,'Distribution of Age wrt Stroke',horizontalalignment = 'center',verticalalignment = 'center',fontsize = 18,fontfamily='serif')

sns.kdeplot(ax=ax2,x=data.loc[data['stroke']==1]['avg_glucose_level'],color='crimson',label='Stroke',shade=True)
sns.kdeplot(ax=ax2,x=data.loc[data['stroke']==0]['avg_glucose_level'],color='coral',label='No Stroke',shade=True)
ax2.legend(loc = 'upper right')
ax2.grid(linestyle='--', axis='y')

ax3.text(0.5,0.5,'Distribution of Glucose level\n wrt Stroke',horizontalalignment = 'center',verticalalignment = 'center',fontsize = 18,fontfamily='serif')


sns.kdeplot(ax=ax4,x=data.loc[data['stroke']==1]['bmi'],color='crimson',label='Stroke',shade=True)
sns.kdeplot(ax=ax4,x=data.loc[data['stroke']==0]['bmi'],color='coral',label='No Stroke',shade=True)
ax4.legend(loc = 'upper right')
ax4.grid(linestyle='--', axis='y')

ax5.text(0.5,0.5,'Distribution of BMI\n wrt Stroke',horizontalalignment = 'center',verticalalignment = 'center',fontsize = 18,fontfamily='serif')
# removing labels

axes1=[ax1,ax3,ax5]
for i in axes1:
    i.spines["bottom"].set_visible(False)
    i.spines["left"].set_visible(False)
    i.set_xlabel("")
    i.set_ylabel("")
    i.set_xticklabels([])
    i.set_yticklabels([])
    i.tick_params(left=False, bottom=False)
# removing spines of figures
for i in ["top","left","right"]:
    ax0.spines[i].set_visible(False)
    ax1.spines[i].set_visible(False)
    ax2.spines[i].set_visible(False)
    ax3.spines[i].set_visible(False)
    ax4.spines[i].set_visible(False)
    ax5.spines[i].set_visible(False)


# ### Bivariate analysis of Numerical Variables
# 

# In[ ]:


fig = plt.figure(figsize=(17,12))
gs = fig.add_gridspec(2,2)
ax0=fig.add_subplot(gs[0,0])
ax1=fig.add_subplot(gs[0,1])
ax2=fig.add_subplot(gs[1,0])
axes=[ax0,ax1,ax2]
background_color = '#f6f5f7'
for i in axes:
    i.set_facecolor(background_color)
fig.patch.set_facecolor(background_color) 
# Age and bmi
sns.scatterplot(ax=ax0,x=data_numerical['age'],y=data_numerical['bmi'],hue=data_categorical['stroke'],palette="OrRd")
ax0.set_title('Age and BMI',fontweight='bold')
# Age and Glucose
sns.scatterplot(ax=ax1,x=data_numerical['age'],y=data_numerical['avg_glucose_level'],hue=data_categorical['stroke'],palette="OrRd")
ax1.set_title('Age and Average Glucose',fontweight='bold')
# BMI and Glucose
sns.scatterplot(ax=ax2,x=data_numerical['bmi'],y=data_numerical['avg_glucose_level'],hue=data_categorical['stroke'],palette="OrRd")
ax2.set_title('BMI and Average Glucose',fontweight='bold')
#removing spines
for i in ["top","right"]:
    ax0.spines[i].set_visible(False)
    ax1.spines[i].set_visible(False)
    ax2.spines[i].set_visible(False)


# #### Observations :
# 1) The data appears to be highly imbalanced ( only few points where stroke = 1)
# 
# 2) There are few outliers in Bmi & Age and BMI & Avg Glucose levels (keeping them as they are only few). 
# 
# 3) The Age group is heavily distributed between 0 and 60 (no particular range has significantly more people than others).
# 
# 4) Age and Avg glucose levels can be split into 2 clusters ( one less than 150 and other more than that). Only few cases of people with glucose levels less than 150 experienced stroke.
# 
# 5) BMI and Glucose levels confirm that people with less than 150 glucose levels are less prone to strokes than people with glucose levels more than 150 level. BMI >40 have low avg glucose

# ### Correlation plot for numerical variables
# 

# In[ ]:


fig=plt.figure(figsize=(10,5),dpi=100)
gs=fig.add_gridspec(1,2)
# adding subplots
ax0=fig.add_subplot(gs[0,0])
ax1=fig.add_subplot(gs[0,1])
axes=[ax0,ax1]
background_color = '#f6f5f7'
# changing background color of our plots
for i in axes:
    i.set_facecolor(background_color)
# changing the figure background color
fig.patch.set_facecolor(background_color) 
# heatmap of numerical data
corrmat = data_numerical.corr()
sns.heatmap(ax=ax0,data=corrmat,annot=True, cmap="OrRd",square=True)
ax1.text(0.5,0.5,'No strong correlation between\n age,glucose levels and bmi',horizontalalignment = 'center',verticalalignment = 'center',fontsize = 15,fontfamily='serif')
ax1.spines["bottom"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.tick_params(left=False, bottom=False)
for i in ["top","right","bottom","left"]:
    ax1.spines[i].set_visible(False)
plt.text(-1.7,1.1,'Heatmap of Numerical Variables',fontsize=18,fontweight='bold',fontfamily='serif')    


# As there is no strong correlation between the variables, we can ignore the chances of multicollinearity.

# ### Pairplot

# In[ ]:


fig=plt.figure(figsize=(20,15),dpi=100)
sns.pairplot(data=data,hue='stroke',size=2,palette='OrRd')
plt.show()


# ### Univariate Analysis of Categorical Variables

# In[ ]:


fig=plt.figure(figsize=(20,23))
background_color = '#f6f5f7'
fig.patch.set_facecolor(background_color) 
for indx,val in enumerate(data_categorical.columns):
    ax=plt.subplot(4,2,indx+1)
    ax.set_facecolor(background_color)
    ax.set_title(val,fontweight='bold',fontfamily='serif')
    for i in ['top','right']:
        ax.spines[i].set_visible(False)
    ax.grid(linestyle=':',axis='y')
    sns.countplot(data_categorical[val],palette='OrRd')


# #### Observations :
# 1) Females are more than male in our data. The Other category in gender is not visible as it contains only one value.
# 
# 2) The number of people without hypertension are way less than people who has it.
# 
# 3) The number of people with heart disease is extremely low.
#  
# 4) The number of people who are married are way more than unmarried people(makes sense as the distribution is between 0 and 60)
# 
# 5) People seem to prefer working in private companies while the number of self-emplyed/ govt_job and children seems to be equal in number (children can be ignored). Unemployed people are extremely less.
# 
# 6) Not a big difference between the population in urban and rural area.
# 
# 7) The Unknown category reprsents that we do not know if a person smoked or not. If the size of unknown is too large then we will remove it from our model. Non-smokers are way more than people who smoked/used to smoke which is a good thing.
# 
# 8) Number of people with strokes are less than 1000 in number.

# ### Analysing Categorical Variables with Stroke
# 

# In[ ]:


data_cat=data[['gender', 'hypertension', 'heart_disease', 'ever_married','work_type', 'Residence_type', 
                       'smoking_status']]
fig=plt.figure(figsize=(20,23))
background_color = '#f6f5f7'
fig.patch.set_facecolor(background_color) 
for indx,val in enumerate(data_cat.columns):
    ax=plt.subplot(4,2,indx+1)
    ax.set_facecolor(background_color)
    ax.set_title(val,fontweight='bold',fontfamily='serif')
    for i in ['top','right']:
        ax.spines[i].set_visible(False)
    ax.grid(linestyle=':',axis='y')
    sns.countplot(data_cat[val],palette='OrRd_r',hue=data['stroke'])


# #### Observations :
# 1) Seems that number of male and female who has stroke are equal in number.
# 
# 2) The number of people who do not have hypertension also shows signs of no stroke. And people with hypertension also do not show signs of more people with stroke (may be due to the fact that our data has so many negative(0) variables).
# 
# 3) The people with heart diesease show signs of stroke too(as expected).
# 
# 4) The people who got married show signs of stroke way more than people who are unmarried ( expected i guess).
# 
# 5) Private employees seems to experience stroke more than other work_types(may be due to work pressure). Self-employed people do show signs of stroke(may be due to reasons like heart disease,tension etc). Children can be ignored.
# 
# 6) Almost no difference between people living in urban and rural areas in terms of stroke occurence.
# 
# 7) People who formerly smoked and who smoke (combined) are showing signs of stroke way more than people who never smoked (considering the sample size of people who never smoked and people who used to smoke and smoke now).
# 

# ### Looking closely at stroke and some of the variables

# The observations for below graphs are the same as the figures above. 

# In[ ]:


# Smoking Type and Stroke
pd_stroke = pd.pivot_table(data=data[data['stroke']==1],index=data['smoking_status'],values='stroke',aggfunc='count').reset_index()
fig,ax=plt.subplots(1,1,figsize=(10,5))
background_color = '#f6f5f7'
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)
ax.bar(pd_stroke['smoking_status'],pd_stroke['stroke'],width=0.55,linewidth=0.7,color=sns.color_palette('OrRd'))
for idx,val in enumerate(pd_stroke['stroke']):
    ax.text(idx,val+1,round(val,1),horizontalalignment='center')
ax.grid(linestyle=':',axis='y',alpha=0.5)
for i in ['top','right']:
    ax.spines[i].set_visible(False)
plt.text(-0.7,100,'Smoking Status and Stroke',fontsize=18,fontweight='bold',fontfamily='serif')


# In[ ]:


# Marrital Status and Stroke
pd_stroke = pd.pivot_table(data=data[data['stroke']==1],index=data['ever_married'],values='stroke',aggfunc='count').reset_index()
fig,ax=plt.subplots(1,1,figsize=(10,5))
background_color = '#f6f5f7'
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)
ax.bar(pd_stroke['ever_married'],pd_stroke['stroke'],width=0.55,linewidth=0.7,color=sns.color_palette('OrRd'))
for an in pd_stroke.index:
    ax.annotate(pd_stroke['stroke'][an],xy=(pd_stroke['ever_married'][an],pd_stroke['stroke'][an]+5),va='center',ha='center')
ax.grid(linestyle=':',axis='y',alpha=0.5)
for i in ['top','right']:
    ax.spines[i].set_visible(False)
plt.text(-0.7,250,'Marrital Status and Stroke',fontsize=18,fontweight='bold',fontfamily='serif')


# In[ ]:


# Work Type and Stroke
pd_stroke = pd.pivot_table(data=data[data['stroke']==1],index=data['work_type'],values='stroke',aggfunc='count').reset_index()
fig,ax=plt.subplots(1,1,figsize=(10,5))
background_color = '#f6f5f7'
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)
ax.bar(pd_stroke['work_type'],pd_stroke['stroke'],width=0.55,linewidth=0.7,color=sns.color_palette('OrRd'))
for idx,val in enumerate(pd_stroke['stroke']):
    ax.text(idx,val+1,round(val,1),horizontalalignment='center')
ax.grid(linestyle=':',axis='y',alpha=0.5)
for i in ['top','right']:
    ax.spines[i].set_visible(False)
plt.text(-0.7,170,'Work Type and Stroke',fontsize=18,fontweight='bold',fontfamily='serif')


# In[ ]:


# Hypertension and Stroke
pd_stroke = pd.pivot_table(data=data[data['stroke']==1],index=data['hypertension'],values='stroke',aggfunc='count').reset_index()
fig,ax=plt.subplots(1,1,figsize=(10,5))
background_color = '#f6f5f7'
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)
sns.barplot(ax=ax,x=pd_stroke['hypertension'],y=pd_stroke['stroke'],palette='OrRd')
for idx,val in enumerate(pd_stroke['stroke']):
    ax.text(idx,val+1,round(val,1),horizontalalignment='center')
ax.grid(linestyle=':',axis='y',alpha=0.5)
for i in ['top','right']:
    ax.spines[i].set_visible(False)
plt.text(-0.7,210,'Hypertension and Stroke',fontsize=18,fontweight='bold',fontfamily='serif')


# In[ ]:


# Heart Disease and Stroke
pd_stroke = pd.pivot_table(data=data[data['stroke']==1],index=data['heart_disease'],values='stroke',aggfunc='count').reset_index()
fig,ax=plt.subplots(1,1,figsize=(10,5))
background_color = '#f6f5f7'
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)
sns.barplot(ax=ax,x=pd_stroke['heart_disease'],y=pd_stroke['stroke'],palette='OrRd')
for idx,val in enumerate(pd_stroke['stroke']):
    ax.text(idx,val+1,round(val,1),horizontalalignment='center')
ax.grid(linestyle=':',axis='y',alpha=0.5)
for i in ['top','right']:
    ax.spines[i].set_visible(False)
plt.text(-0.7,220,'Heart Disease and Stroke',fontsize=18,fontweight='bold',fontfamily='serif')


# ### Correlation Matrix 

# In[ ]:


fig=plt.figure(figsize=(10,5),dpi=100)
gs=fig.add_gridspec(1,2)
# adding subplots
ax0=fig.add_subplot(gs[0,0])
ax1=fig.add_subplot(gs[0,1])
axes=[ax0,ax1]
background_color = '#f6f5f7'
# changing background color of our plots
for i in axes:
    i.set_facecolor(background_color)
# changing the figure background color
fig.patch.set_facecolor(background_color) 
# heatmap of numerical data
corrmat = data.corr()
sns.heatmap(ax=ax0,data=corrmat,annot=True, cmap="OrRd",square=True)
ax1.text(0.5,0.5,'No strong correlation between\n any of the features',horizontalalignment = 'center',verticalalignment = 'center',fontsize = 15,fontfamily='serif')
ax1.spines["bottom"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.tick_params(left=False, bottom=False)
for i in ["top","right","bottom","left"]:
    ax1.spines[i].set_visible(False)
plt.text(-1.7,1.1,'Heatmap of Data',fontsize=18,fontweight='bold',fontfamily='serif')    


# #### Observations:
# 1) No strong correlation between our features.
# 
# 2) The highest correlation can be observed between body mass index(bmi) and age.
# 
# 3) The weakest correlation can be observed between heart_disease and hyper_tension (questionable).

# ### Data Preprocessing

# In[ ]:


# Convert Marrital Status, Residence and Gender into 0's and 1's
data['gender']=data['gender'].apply(lambda x : 1 if x=='Male' else 0) 
data["Residence_type"] = data["Residence_type"].apply(lambda x: 1 if x=="Urban" else 0)
data["ever_married"] = data["ever_married"].apply(lambda x: 1 if x=="Yes" else 0)
# Removing the observations that have smoking type unknown. 
data=data[data['smoking_status']!='Unknown']


# In[ ]:


# One Hot encoding smoking_status, work_type
data_dummies = data[['smoking_status','work_type']]
data_dummies=pd.get_dummies(data_dummies)
data.drop(columns=['smoking_status','work_type'],inplace=True)


# In[ ]:


data_stroke=data['stroke']
data.drop(columns=['stroke'],inplace=True)
data=data.merge(data_dummies,left_index=True, right_index=True,how='left')


# In[ ]:


data_stroke


# ### Splitting the data into training and testing sets.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,data_stroke,test_size=0.25,random_state=0)


# In[ ]:


# Standardizing our training and testing data.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# We perform feature scaling after splitting the data into training and testing sets in order to avoid data leakage.

# In[ ]:


from imblearn.over_sampling import SMOTE
sm = SMOTE()
x_train, y_train = sm.fit_resample(x_train, y_train.ravel())
print("Before OverSampling, counts of label '1': {}".format(sum(data_stroke==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(data_stroke==0)))

print('After OverSampling, the shape of train_X: {}'.format(x_train.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train==0)))


# ### Training the Models

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, roc_curve,precision_recall_curve, auc,confusion_matrix, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.impute import KNNImputer

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff


# In[ ]:


#https://towardsdatascience.com/model-evaluation-techniques-for-classification-models-eac30092c38b
def Model(model):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    model_train_score = model.score(x_train, y_train)
    model_test_score = model.score(x_test, y_test)
    prediction = model.predict(x_test)
    cm = confusion_matrix(y_test,prediction)
    print('Testing Score \n',score)
    plot_confusion_matrix(model,x_test,y_test,cmap="OrRd")
    metrics.plot_roc_curve(model, x_test, y_test)  


# In[ ]:


def plot_cm(cm,title):
    z = cm
    x = ['No stroke', 'stroke']
    y = x
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='deep')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix {}</b></i>'.format(title),
                      #xaxis = dict(title='x'),
                      #yaxis = dict(title='x')
                     )

    # add custom xaxis title
    fig.add_annotation({'font':{'color':"black",'size':14},
                            'x':0.5,
                            'y':-0.10,
                            'showarrow':False,
                            'text':"Predicted value",
                            'xref':"paper",
                            'yref':"paper"})
    
    fig.add_annotation({'font':{'color':"black",'size':14},
                            'x':-0.15,
                            'y':0.5,
                            'showarrow':False,
                            'text':"Real value",
                            'textangle':-90,
                            'xref':"paper",
                            'yref':"paper"})


    # adjust margins to make room for yaxis title
    fig.update_layout(margin={'t':50, 'l':20},width=750,height=750)
    


    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()



def hist_score(score):
    models_names = [
        'Logistic Regression',
    'KNearest Neighbor',
    'Decision Tree Classifier',
    'Random Forest Classifier',
    'Ada Boost',
    'SVM',
    'XG Boost',
    'Cat Boost']

    plt.rcParams['figure.figsize']=20,8
    sns.set_style('darkgrid')
    ax = sns.barplot(x=models_names, y=score, palette = "inferno", saturation =2.0)
    plt.xlabel('Classifier Models', fontsize = 20 )
    plt.ylabel('% of Accuracy', fontsize = 20)
    plt.title('Accuracy of different Classifier Models on test set', fontsize = 20)
    plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
    plt.yticks(fontsize = 12)
    for i in ax.patches:
        width, height = i.get_width(), i.get_height()
        x, y = i.get_xy() 
        ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
    plt.show()
    
def mse_plot(score):
    models_names = [
        'Logistic Regression',
    'KNearest Neighbor',
    'Decision Tree Classifier',
    'Random Forest Classifier',
    'Ada Boost',
    'SVM',
    'XG Boost',
    'Cat Boost']

    plt.rcParams['figure.figsize']=20,8
    sns.set_style('darkgrid')
    ax = sns.barplot(x=models_names, y=score, palette = "inferno", saturation =2.0)
    plt.xlabel('Classifier Models', fontsize = 20 )
    plt.ylabel('% of Accuracy', fontsize = 20)
    plt.title('MSE of different Classifier Models on test set', fontsize = 20)
    plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
    plt.yticks(fontsize = 12)
    for i in ax.patches:
        width, height = i.get_width(), i.get_height()
        x, y = i.get_xy() 
        ax.annotate(height, (x + width, y + height), ha='center', fontsize = 'x-large')
    plt.show()


# In[ ]:


def run_exp_on_feature(x_train,y_train,x_test,y_test):
    #x_train,x_test,y_train,y_test = train_test_split(features,labels, test_size=0.2, random_state=23)
    models= [['Logistic Regression ',LogisticRegression()],
            ['KNearest Neighbor ',KNeighborsClassifier()],
            ['Decision Tree Classifier ',DecisionTreeClassifier()],
            ['Random Forest Classifier ',RandomForestClassifier()],
            ['Ada Boost ',AdaBoostClassifier()],
            ['SVM ',SVC()],
            ['XG Boost',XGBClassifier()],
            ['Cat Boost',CatBoostClassifier(logging_level='Silent')]]

    models_score = []
    models_mse = []
    models_rmse = []
    for name,model in models:

        model = model
        model.fit(x_train,y_train)
        model_pred = model.predict(x_test)
        cm_model = confusion_matrix(y_test, model_pred)
        print(cm_model)
        models_score.append(accuracy_score(y_test,model.predict(x_test)))
        models_mse.append(mean_squared_error(y_train, model.predict(x_train), squared=True))
        models_rmse.append(mean_squared_error(y_train, model.predict(x_train), squared=False))

        print(name)
        print('Validation Accuracy: ',accuracy_score(y_test,model.predict(x_test)))
        print('Training Accuracy: ',accuracy_score(y_train,model.predict(x_train)))
        print('MSE: ', mean_squared_error(y_train, model.predict(x_train), squared=True))
        print('RMSE: ', mean_squared_error(y_train, model.predict(x_train), squared=False))
        print('F1 Score: ',classification_report(y_train, model.predict(x_train), labels=[1,0]))
        print('############################################')
        plot_cm(cm_model,title=name+"model")
        fpr, tpr, thresholds = roc_curve(y_test, model_pred)

        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels={'x':'False Positive Rate', 'y':'True Positive Rate'},
            width=700, height=500
        )
        fig.add_shape(
            type='line', line={'dash':'dash'},
            x0=0, x1=1, y0=0, y1=1
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        fig.show()
    
        
    return models_score, models_mse, models_rmse


# In[ ]:


models_score, models_mse, models_rmse = run_exp_on_feature(x_train,y_train,x_test,y_test)


# In[ ]:


hist_score(models_score)
mse_plot(models_mse)
mse_plot(models_rmse)


# In[ ]:


y_train.ravel()


# #### The confusion matrix interpretation:
# 1) The first element is True Negative([0,0]) - They are classified as 0 and our model correctly classified them as 0.
# 
# 2) The second element is False Positive([0,1]) - Their actual value is 0 but our model predicted them as 1.
# 
# 3) The third element is False Negative([1,0]) - Their actual value is 1 but our model predicted them as 0.
# 
# 4) The Fourth element is True Positive([1,1]) - Their actual value is 1 and our model predicted them as 1.
# 
# 

# ### Check out my other notebooks here:
# 1) https://www.kaggle.com/ruthvikpvs/heart-attack-eda-and-prediction
# 
# 2) https://www.kaggle.com/ruthvikpvs/students-performance-eda-and-prediction

# ### References :
# 1) https://www.kaggle.com/ahmedterry/stroke-prediction-eda-classification-models
# 
# 2) https://www.kaggle.com/namanmanchanda/heart-attack-eda-prediction-90-accuracy
# 
# 3) https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79
# 
# 4) https://www.kaggle.com/subinium/simple-matplotlib-visualization-tips

# ### Do upvote the kernel if you find it useful. Feedback is highly appreciated. Thank You.

# In[ ]:




