#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
pyo.init_notebook_mode()

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        df = pd.read_csv(os.path.join(dirname, filename))
        
RED = 'firebrick'
BLUE = 'darkblue'
GREEN = 'forestgreen'

y_col_color_info = {
    'Yes':[RED, 0.6], # color & opacity
    'No':[GREEN, 0.05],
}
TEMPLATE = 'simple_white'


# # 0. Intro
# I'm going to check the Heart Disease data. (https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)  
# The information included in the data is as follows, and I'm thinking of checking the factors that increase the **HeartDisease**.
# 
# * HeartDisease : Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI)
# > * BMI : Body Mass Index (BMI)
# > * Smoking : Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]
# > * AlcoholDrinking : Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week
# > * Stroke : (Ever told) (you had) a stroke?
# > * PhysicalHealth : Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical * health not good? (0-30 days)
# > * MentalHealth : Thinking about your mental health, for how many days during the past 30 days was your mental health not good? (0-30 days)
# > * DiffWalking : Do you have serious difficulty walking or climbing stairs?
# > * Sex : Are you male or female?
# > * AgeCategory : Fourteen-level age category
# > * Race : Imputed race/ethnicity value
# > * Diabetic : (Ever told) (you had) diabetes?
# > * PhysicalActivity : Adults who reported doing physical activity or exercise during the past 30 days other than their regular job
# > * GenHealth : Would you say that in general your health is...
# > * SleepTime : On average, how many hours of sleep do you get in a 24-hour period?
# > * Asthma : (Ever told) (you had) asthma?
# > * KidneyDisease : Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?
# > * SkinCancer : (Ever told) (you had) skin cancer?
#   
# I think we can divide these factors into four major categories.  
# > Demography : Sex, AgeCategory, Race  
# > Life Style : Smoking, AlcoholDrinking, PhysicalActivity, SleepTime  
# > Health Condition : BMI, PhysicalHealth, MentalHealth, GenHealth, DiffWalking  
# > Disease : Stroke, KidneyDisease, Diabetic, Asthma, SkinCancer  

# # 1. Data
# First, the data types are as follows.  
# When I checked *df.info()*, there is no null value, and the data is mixed with object and float type columns.  
# This is a mixture of numeric and categorical data.  

# In[ ]:


Y_COL = "HeartDisease"
display(df.head(3))
display(df.info())


# ## 1.1 Set Continuous Columns and Categorical Columns
# Each column was divided into continuous and categorical columns based on dtype.  

# In[ ]:


cont_cols = list(df.select_dtypes(include=['float', 'int']).columns)
print(f"Continuous\n : {', '.join(cont_cols)}\n")

display(pd.DataFrame(df.select_dtypes(include=['O']).nunique()).T)
temp_ = df.select_dtypes(include=['O']).nunique()
bina_cols = list(temp_[temp_ == 2].index)
cate_cols = list(temp_[temp_ > 2].index)

print(f"Categorical (2 unique)\n : {', '.join(bina_cols)}")
print(f"Categorical (over 2)\n : {', '.join(cate_cols)}")


# Among the categorical columns, Age Category has 13 unique values and is as follows.  
# It represents the range of age, each containing the order of age information, so we converted it to the starting value of each range.  
# In other words, I changed Age Category to Continuous column.

# In[ ]:


print(f"AgeCategory unique value\n : {df['AgeCategory'].unique()}")

df['AgeCategory'] = df['AgeCategory'].apply(lambda x : int(x[:2]))

cate_cols.remove('AgeCategory')
cont_cols.append('AgeCategory')

print(f"Continuous\n : {', '.join(cont_cols)}\n")


# ## 1.2 Split Data
# I'm going to test the prediction at the end, so I'll separate the data that I don't see to prevent data leakage.  
# I left 30% of the data for the test.  
# Based on the training data, 8.56% of people have heart disease.  

# In[ ]:


def split_df(df, y_col, random_state, test_size):
    x_cols = [c for c in df.columns if c != y_col]
    # print(f"Y : {y_col}")
    # print(f"X : {', '.join(x_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(df[x_cols], df[[y_col]],
                                                        stratify=df[[y_col]],
                                                        random_state=random_state, test_size=test_size)
    X_train[y_col] = y_train
    X_test[y_col] = y_test
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True)

train_df, test_df = split_df(df, Y_COL, 0, 0.3)
base_ratio = len(train_df[train_df[Y_COL]=='Yes']) / len(train_df)
print(f"{Y_COL} value : {', '.join(train_df[Y_COL].unique())}")
print(f"{Y_COL}=Yes ratio : {base_ratio : .4f}")


# # 2. Demography
# Let's look at it from a demographic perspective.  
# The factors involved are: Sex, AgeCategory, Race  

# ## 2.1 Age
# I guessed, but the older the person, the higher the number of heart disease.  
# The higher a person's age, the greater the number of heart disease, and the higher the number of heart disease at each age group. The cumulative sum of the total heart disease rate also shows a higher rate of increase in old age.  
# > *67.32% of the heart disease population is distributed over the age of 65.*

# In[ ]:


# prepare data
reverse_age_cate = {
    18:'18-24', 25:'25-29', 30:'30-34', 35:'35-39', 40:'40-44', 45:'45-49', 50:'50-54', 55:'55-59',
    60:'60-64', 65:'65-69', 70:'70-74', 75:'75-79', 80:'80 or older'
}

temp_ = train_df[[Y_COL, 'Sex', 'AgeCategory', 'Race']]
temp_["AgeCategory"] = temp_["AgeCategory"].apply(lambda x : reverse_age_cate[x])

temp_age_ = temp_[temp_[Y_COL]=='Yes'].groupby(['AgeCategory'])[[Y_COL]].count().rename(columns={Y_COL:f'{Y_COL}_count'})
temp_age_['total_count'] = temp_.groupby(['AgeCategory'])[[Y_COL]].count()
temp_age_[f'{Y_COL}_ratio'] = temp_age_[f'{Y_COL}_count'] / temp_age_['total_count']
temp_age_[f'{Y_COL}_cumratio_total'] = temp_age_[f'{Y_COL}_count'].cumsum() / temp_age_['total_count'].cumsum()
temp_age_[f'{Y_COL}_cumratio_yes'] = temp_age_[f'{Y_COL}_count'].cumsum() / len(temp_[temp_[Y_COL]=='Yes'])

# visualization
fig = make_subplots(specs=[[{"secondary_y":True}]])
fig.add_trace(go.Scatter(x=temp_age_.index, y=temp_age_[f'{Y_COL}_count'], line=dict(color='gray'), name='The number of heart disease by age'), secondary_y=False)
fig.add_trace(go.Scatter(x=temp_age_.index, y=temp_age_['total_count'], line=dict(color='lightgray'), name='The number of people by age'), secondary_y=False)
fig.update_layout(title='<b>The number of heart disease by age</b>', template=TEMPLATE)
fig.update_layout(legend=dict(x=0, y=1.4), margin=dict(l=20, r=20, t=200, b=70))
fig.update_yaxes(title="Count", rangemode="tozero", secondary_y=False)
fig.update_xaxes(title="AgeCategory")
fig.show()


fig = make_subplots(specs=[[{"secondary_y":True}]])
fig.add_trace(go.Bar(x=temp_age_.index, y=temp_age_[f'{Y_COL}_ratio'], marker=dict(color=['lightgray']*9+[RED]*4), name='The ratio of heart disease in age group'),
              secondary_y=False)
fig.add_trace(go.Scatter(x=temp_age_.index, y=temp_age_[f'{Y_COL}_cumratio_yes'], line=dict(color='gray'), name='The cumulative sum of ratio of heart diseas by age'),
              secondary_y=True)
fig.update_layout(title='<b>The percentage of heart disease by age</b>', template=TEMPLATE)
fig.update_layout(legend=dict(x=0, y=1.4), margin=dict(l=20, r=20, t=200, b=70))
fig.update_yaxes(title="Ratio", rangemode="tozero", secondary_y=False)
fig.update_yaxes(title="Cumulative Sum of Ratio", rangemode="tozero", secondary_y=True)
fig.update_xaxes(title="AgeCategory")
fig.show()


# ## 2.2 Sex and Race
# Basically, heart disease is more common in male.  
# (Personally, life habits and activities vary greatly depending on gender, so attention should be paid to interpretation.)  
# * Female : 0.0672
# * Male : 0.1059 (*)  
# 
# By race, American Indian/Alaskan Native and White were the highest, and the lowest was Asian.  
# (Two races were ahead of the total heart disease rate.)
# * Total Heart Disease Rate : 0.0856
# * American Indian/Alaskan Native : 0.1018 (*)
# * Asian : 0.0337
# * Black : 0.0750
# * Hispanic : 0.0526
# * Other : 0.0808
# * White : 0.0919 (*)
# 
# > *In general, men have a high rate of heart disease.*
# > *American Indian/Alaskan Native has a high rate of heart disease, and Asian has a low rate of heart disease.*

# In[ ]:


# prepare data
# # Sex
# temp_sr_ = pd.pivot_table(temp_[[Y_COL, 'Sex']], index=['Sex'], columns=[Y_COL], aggfunc='size')
# temp_sr_.columns = [c for c in temp_sr_.columns]
# temp_sr_[f"{Y_COL}_ratio"] = temp_sr_['Yes'] / (temp_sr_['Yes']+temp_sr_['No'])
# display(temp_sr_)

# # Race
# temp_sr_ = pd.pivot_table(temp_[[Y_COL, 'Race']], index=['Race'], columns=[Y_COL], aggfunc='size')
# temp_sr_.columns = [c for c in temp_sr_.columns]
# temp_sr_[f"{Y_COL}_ratio"] = temp_sr_['Yes'] / (temp_sr_['Yes']+temp_sr_['No'])
# temp_sr_['c'] = temp_sr_[f"{Y_COL}_ratio"] > base_ratio
# display(temp_sr_)

# Both
temp_sr_ = pd.pivot_table(temp_, index=['Sex', 'Race'], columns=[Y_COL], aggfunc='size')
temp_sr_[f"{Y_COL}_ratio"] = temp_sr_['Yes'] / (temp_sr_['Yes']+temp_sr_['No'])
temp_sr_ = temp_sr_.reset_index()
temp_sr_['Demo_RS'] = temp_sr_['Race'] + '_' + temp_sr_['Sex'].apply(lambda x : x[0])
temp_sr_ = temp_sr_[['Demo_RS', f"{Y_COL}_ratio"]]
# temp_sr_ = temp_sr_.append(pd.DataFrame({'Demo_RS':[f'Total_{Y_COL}_ratio'], f"{Y_COL}_ratio":[base_ratio]}), ignore_index=True)
temp_sr_ = temp_sr_.sort_values(by=[f"{Y_COL}_ratio"], ascending=True)

# visualization
fig = go.Figure(go.Bar(
            x=temp_sr_[f"{Y_COL}_ratio"],
            y=temp_sr_['Demo_RS'],
            marker=dict(color=['lightpink' if '_F' in r else 'cornflowerblue' for r in temp_sr_['Demo_RS']]),
            orientation='h'))

fig.add_vline(x=base_ratio, line_width=2, line_color='darkgreen',
              annotation_text=f"Total ratio of heart disease", 
              annotation_position="right"
             )

fig.update_layout(title='<b>The percentage of heart disease by race & sex</b>', template=TEMPLATE)
fig.update_xaxes(title="Ratio", rangemode="tozero")
fig.show()


# # 3. Life Style
# The factors involved are: Smoking, AlcoholDrinking, PhysicalActivity, SleepTime

# ## 3.1 Smoking and Alcohol Drinking
# In the case of these factors, excessive cases are considered harmful to the body. However, since the question was asked with certain criteria, information on smoking and drinking is limited. I think it would have been nice if the weekly smoking and drinking were recorded.
# - Smoking : Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]
# - AlcoholDrinking : Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week  
#   
# Looking at the rate of heart disease among people in each group, the rate of heart disease was unusually high in the group that smoked and drank moderately.
# > *For people who drank (not heavy) and smoked, the rate of heart disease is 12.77%, somewhat higher than the overall rate of heart disease.*

# In[ ]:


# prepare data
temp_ = train_df[[Y_COL, 'Smoking', 'AlcoholDrinking']]

def yn_pivot(df, cols):
    temp_ = pd.pivot_table(df[cols], index=cols[1:], columns=cols[0], aggfunc='size')
    temp_['Total'] = temp_.sum(axis=1)
    temp_['No_ratio'] = temp_['No'] / temp_['Total']
    temp_['Yes_ratio'] = temp_['Yes'] / temp_['Total']
    return temp_

temp_sm_ad_ = yn_pivot(temp_, [Y_COL, 'Smoking', 'AlcoholDrinking'])
temp_sm_ad_ = temp_sm_ad_.reset_index()
temp_sm_ad_['LS_SA'] = temp_sm_ad_['Smoking'].apply(lambda x : 'Smoking' if x == 'Yes' else 'Non Smoking') + ' &<br>' +\
                        temp_sm_ad_['AlcoholDrinking'].apply(lambda x : 'Heavy Drink' if x == 'Yes' else 'Non Heavy')

temp_ad_ = yn_pivot(temp_, [Y_COL, 'AlcoholDrinking'])
temp_ad_ = temp_ad_.reset_index()
temp_ad_['LS_SA'] = 'All &<br>' + temp_sm_ad_['AlcoholDrinking'].apply(lambda x : 'Heavy Drink' if x == 'Yes' else 'Non Heavy')

temp_sm_ = yn_pivot(temp_, [Y_COL, 'Smoking'])
temp_sm_ = temp_sm_.reset_index()
temp_sm_['LS_SA'] = temp_sm_['Smoking'].apply(lambda x : 'Smoking' if x == 'Yes' else 'Non Smoking') + ' &<br>All'

temp_sm_ad_ = temp_sm_ad_.append(temp_ad_)
temp_sm_ad_ = temp_sm_ad_.append(temp_sm_)

# visualization
fig = go.Figure(go.Bar(
            x=temp_sm_ad_['Yes_ratio'],
            y=temp_sm_ad_['LS_SA'],
            marker=dict(color=['lightgray', 'lightgray', RED, 'lightgray', RED, 'lightgray', 'lightgray', RED]),
            orientation='h'))

fig.add_vline(x=base_ratio, line_width=2, line_color='darkgreen',
              annotation_text=f"Total ratio of heart disease", 
              annotation_position="bottom right"
             )

fig.update_layout(title='<b>The percentage of heart disease by smoking & heavy drinking</b>', template=TEMPLATE)
fig.update_xaxes(title="Ratio", rangemode="tozero")
fig.show()


# ## 3.2 Physical Activity and Sleep Time
# In general, moderate physical activity and moderate sleep time help prevent diseases. For groups with physical activity and above average sleep time, the rate of heart disease is expected to be relatively low.  
# For sleep time, the 25th percentile value is 6, and if it is above this value, I will say it is appropriate.  
# If you look at the graph below, the rate of heart disease in each group seems to be as expected.  
# > *The group with insufficient physical activity and lack of sleep has a 17.86% rate of heart disease.*

# In[ ]:


# prepare data
temp_ = train_df[[Y_COL, 'PhysicalActivity', 'SleepTime']]
temp_['SleepTime_6'] = temp_['SleepTime'].apply(lambda x : 'Sleep over6' if x >= 6 else 'Sleep under6')
temp_pa_st_ = yn_pivot(temp_, [Y_COL, 'PhysicalActivity', 'SleepTime_6'])
temp_pa_st_ = temp_pa_st_.reset_index()
temp_pa_st_['PA_ST'] = temp_pa_st_['PhysicalActivity'].apply(lambda x : 'Activity' if x == 'Yes' else 'Non Activity') + ' &<br>' +\
                        temp_pa_st_['SleepTime_6'].apply(lambda x : 'Good Sleep' if x == 'Sleep over6' else 'Bad Sleep')

temp_st_ = yn_pivot(temp_, [Y_COL, 'SleepTime_6'])
temp_st_ = temp_st_.reset_index()
temp_st_['PA_ST'] = 'All &<br>' + temp_st_['SleepTime_6'].apply(lambda x : 'Good Sleep' if x == 'Sleep over6' else 'Bad Sleep')

temp_pa_ = yn_pivot(temp_, [Y_COL, 'PhysicalActivity'])
temp_pa_ = temp_pa_.reset_index()
temp_pa_['PA_ST'] = temp_pa_['PhysicalActivity'].apply(lambda x : 'Activity' if x == 'Yes' else 'Non Activity') + ' &<br>All'

temp_pa_st_ = temp_pa_st_.append(temp_st_)
temp_pa_st_ = temp_pa_st_.append(temp_pa_)

# visualization
fig = go.Figure(go.Bar(
            x=temp_pa_st_['Yes_ratio'],
            y=temp_pa_st_['PA_ST'],
            marker=dict(color=['lightgray', RED, GREEN, 'lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray']),
            orientation='h'))

fig.add_vline(x=base_ratio, line_width=2, line_color='darkgreen',
              annotation_text=f"Total ratio of heart disease", 
              annotation_position="top right"
             )

fig.update_layout(title='<b>The percentage of heart disease by physical activity & sleep time</b>', template=TEMPLATE)
fig.update_xaxes(title="Ratio", rangemode="tozero")
fig.show()


# ## 3.3 Best and Worst
# Finally, let's find out which group has the highest and lowest rates of heart disease for the total factor. As we confirmed earlier, it is consistent with the general view except for the amount of alcohol consumed. If all four factors are good and all are bad, they are not located at the extremes based on the rate of heart disease, but they are still relatively far from the overall average.

# In[ ]:


# prepare data
temp_ = train_df[[Y_COL, 'Smoking', 'AlcoholDrinking', 'PhysicalActivity', 'SleepTime']]
temp_['SleepTime_6'] = temp_['SleepTime'].apply(lambda x : 'Sleep over6' if x >= 6 else 'Sleep under6')
temp_ = yn_pivot(temp_, [Y_COL, 'Smoking', 'AlcoholDrinking', 'PhysicalActivity', 'SleepTime_6'])
temp_ = temp_.reset_index()
temp_ = temp_.sort_values(by=["Yes_ratio"], ascending=True)
temp_['Smoking'] = temp_['Smoking'].apply(lambda x : 'Smoking' if x == 'Yes' else 'Non Smoking')
temp_['AlcoholDrinking'] = temp_['AlcoholDrinking'].apply(lambda x : 'Heavy Drink' if x == 'Yes' else 'Non Heavy')
temp_['PhysicalActivity'] = temp_['PhysicalActivity'].apply(lambda x : 'Activity' if x == 'Yes' else 'Non Activity')
temp_['SleepTime_6'] = temp_['SleepTime_6'].apply(lambda x : 'Good Sleep' if x == 'Sleep over6' else 'Bad Sleep')
temp_['group'] = temp_['Smoking'] + ' &  ' + temp_['AlcoholDrinking'] + ' & ' + \
                temp_['PhysicalActivity'] + ' & ' + temp_['SleepTime_6']

temp_9 = temp_[['group', 'Yes_ratio']].head(4)
temp_9 = temp_9.append(pd.DataFrame({'group':['...'], 'Yes_ratio':[0]}))
temp_9 = temp_9.append(temp_[['group', 'Yes_ratio']].tail(5))

# visualization
fig = go.Figure(go.Bar(
            x=temp_9['Yes_ratio'],
            y=temp_9['group'],
            text=[list(temp_9['group'])[0]]+['']*2+[list(temp_9['group'])[3]]+['']+\
                    [list(temp_9['group'])[-5]]+['']*3+[list(temp_9['group'])[-1]],
    textposition='auto',
            marker=dict(color=[GREEN,'lightgray','lightgray','DarkSeaGreen','white',\
                               'IndianRed','lightgray','lightgray','lightgray',RED\
                              ]),
            orientation='h'))

fig.add_vline(x=base_ratio, line_width=2, line_color='darkgreen',
              annotation_text=f"Total ratio of heart disease ({base_ratio:.4f})", 
              annotation_position="top"
             )
fig.add_annotation(x=0.05, y=4, text="...", font=dict(size=20), showarrow=False, yshift=5)

fig.update_layout(title='<b>The percentage of heart disease by life style</b>', template=TEMPLATE)
fig.update_xaxes(title="Ratio", rangemode="tozero")
fig.update_yaxes(showticklabels=False)
fig.show()


# # 4. Health Condition
# Let's check about health conditions.  
# The factors involved are: BMI, PhysicalHealth, MentalHealth, GenHealth, DiffWalking  
# Referring to the data description, the factors may be divided into objective and subjective indicators. Except for BMI, it is the result of answering with subjective judgment.

# ## 4.1 BMI
# There is no noticeable difference in the BMI distribution. The median BMI of people with heart disease seems to be a little bit larger.

# In[ ]:


# prepare data
temp_ = train_df[[Y_COL, 'BMI']]

# visualization
fig = go.Figure()
for y, color in zip(['No','Yes'], [GREEN, RED]):
    fig.add_trace(go.Violin(x=temp_[temp_[Y_COL]==y]['BMI'], line_color=color, name=y))

fig.update_traces(orientation='h', side='positive', width=3, points=False)
fig.update_layout(title='BMI distribution with and without Heart Disease', template=TEMPLATE, height=250)
fig.update_xaxes(title="BMI")
fig.show()


# To further check, I checked the rate of heart disease above a certain BMI. The rate of heart disease was calculated based on the BMI of one unit from BMI median to BMI 80.  
# What's unusual is that it increases to about 42 BMI and then shows an unstable trend. Also, the 75 percent tile is 31.41, and the unstable section seems ineffective because it is larger than 95 percent tile.  
# > *The association between BMI and heart disease seems to be relatively inferior to other factors.*

# In[ ]:


# prepare data
c = temp_['BMI'].median()
y_ratio = []
cs = []
for _ in range(0, 100):
    if c > 80: break
    cs.append(c)
    y_ratio.append(len(temp_[(temp_['BMI']>=c)&(temp_[Y_COL]=='Yes')]) / len(temp_[(temp_['BMI']>=c)]))
    c += 1
    
# visualization
fig = go.Figure(data=go.Scatter(x=cs, y=y_ratio))
fig.update_layout(title='Rate of heart disease above a certain BMI',
                  xaxis_title='BMI Criteria', yaxis_title='Heart Disease Ratio ',
                  template=TEMPLATE, height=350
                 )
fig.add_annotation(x=59.32, y=np.max(y_ratio), text="MAX :0.1272 for BMI 59.32↑", showarrow=True, arrowhead=1)
# about 95 percentile point : np.percentile(temp_['BMI'], 95) = 40.18
fig.add_annotation(x=40.32, y=0.1121, text="95% of the total BMI", showarrow=True, arrowhead=1)

fig.add_hline(y=base_ratio, line_width=2, line_color='darkgreen',
              annotation_text=f"Total ratio of heart disease ({base_ratio:.4f})", 
              annotation_position="bottom left"
             )
fig.add_vrect(x0="42", x1="80", fillcolor="IndianRed", opacity=0.1, layer="below", line_width=0)
fig.add_annotation(x=44, y=0.15, text="UNSTABLE", font=dict(size=13, color=RED), showarrow=False)
fig.show()


# ## 4.2 PhysicalHealth, MentalHealth, GenHealth and DiffWalking
# Let's check for a relatively subjective response.  
# These four factors seem to be related to each other, so I converted the bad case to 1 and added it up to calculate the score. The graph below shows the rate of heart disease above a specific score. If the score is 4 points or higher, the heart disease rate rises to about 28.67%, but people with 4 points are about 3.5% of the total. As a result, the more bad responses during health checkups, the higher the incidence of heart disease seems to increase.
# > *If you think more than two of your self-health diagnoses are bad, the rate of heart disease increases to about 16.39%.*

# In[ ]:


# prepare data
temp_ = train_df[[Y_COL, 'PhysicalHealth', 'MentalHealth', 'GenHealth', 'DiffWalking']]

for c in ['PhysicalHealth', 'MentalHealth']:
    temp_[c] = temp_[c].apply(lambda x : 1 if x > 0 else 0)
temp_['GenHealth'] = temp_['GenHealth'].apply(lambda x : 1 if (x == 'Poor') | (x == 'Fair') else 0)
temp_['DiffWalking'] = temp_['DiffWalking'].apply(lambda x : 1 if x == 'Yes' else 0)
temp_['HealthScore'] = temp_[['PhysicalHealth', 'MentalHealth', 'GenHealth', 'DiffWalking']].sum(axis=1)

y_ratio = []
tot_ratio = []
for s in range(0, 5):
    cur_ = temp_[temp_['HealthScore']>=s]
    y_ratio.append(len(cur_[cur_[Y_COL]=='Yes']) / len(cur_))
    tot_ratio.append(len(cur_)/len(temp_))
    
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(x=[i for i in range(0,5)], y=y_ratio, name=f'{Y_COL}=yes ratio', line=dict(color=RED)), secondary_y=False)
fig.add_trace(go.Scatter(x=[i for i in range(0,5)], y=tot_ratio, name=f'{Y_COL}=yes ratio', line=dict(color='gray')), secondary_y=True)

fig.update_layout(title='Rate of heart disease according to score', xaxis_title='Health Score', template=TEMPLATE, height=350)
fig.update_layout(showlegend=False)

fig.update_yaxes(title_text="Rate of people above Score", secondary_y=True)
fig.update_layout(
    yaxis=dict(
        title="Rate of heart disease above Score",
        titlefont=dict(
            color=RED
        )
    )
)
    
fig.show()


# # 5. Disease
# The last factor to be checked is a factor related to the presence or absence of a disease.  
# The factors are as follows. : Stroke, KidneyDisease, Diabetic, Asthma, SkinCancer  
#   
# The charts below show the rate of heart disease according to the presence or absence of each disease. For all diseases, the rate of heart disease in people who had diseases was higher than the overall average. Among them, the rate of heart disease in the stroke and kidney disease group was high. In particular, in the case of people with strokes, the rate of heart disease was about 36%.
# > *About 0.47% of the total were people with both stroke and kidney disease, of which about 53.2% had heart disease.*

# In[ ]:


# prepare data
temp_ = train_df[[Y_COL, 'Stroke', 'KidneyDisease', 'Diabetic', 'Asthma', 'SkinCancer']]

fig = make_subplots(
    rows=2, cols=3, subplot_titles=('< Stroke >', '< KidneyDisease >', '< Diabetic >', '< Asthma >', '< SkinCancer >')
)

# Add traces
for i, c in enumerate(['Stroke', 'KidneyDisease', 'Diabetic', 'Asthma', 'SkinCancer']):
    cur_ =  yn_pivot(temp_, [Y_COL, c]).reset_index()
    if c in ['Stroke', 'KidneyDisease']:
        colors = ['lightgray', RED]
    else:
        colors = ['lightgray']*len(cur_[c])
    fig.add_trace(go.Bar(x=[f"{v} Group" for v in cur_[c]], y=cur_['Yes_ratio'], marker=dict(color=colors)), row=int(i/3)+1, col=i%3+1)
    fig.update_yaxes(range=[0, 0.4], row=int(i/3)+1, col=i%3+1)
    
    if c == 'Stroke':
        text_ = f"Total ratio of heart disease"
    else:
        text_ = ''
    fig.add_hline(y=base_ratio, line_width=2, line_dash='dash', line_color='darkgreen',
              annotation_text=text_, annotation_position="top left", annotation_font_color='darkgreen'
             )


fig.update_layout(title='The rate of heart disease according to the presence or absence of each disease', template=TEMPLATE, showlegend=False)
fig.show()


# # 6. Heart disease score
# For the four major categories, I looked at which group had a high rate of heart disease. Here I will check the overall score. It is assumed that the higher the score, the higher the risk of heart disease.  
# Points were given on a specific basis for each item. Each criterion is as follows. If each criterion is met, one point is added. Items that were unclear in the EDA did not apply the criteria. (Pass)      
# - Demography
#   - Sex : Male
#   - AgeCategory : over 65
#   - Race : *Pass*
# - Life Style
#   - Smoking : Yes
#   - AlcoholDrinking : *Pass*
#   - PhysicalActivity : Non Activity
#   - SleepTime : uner 6 hours
# - Health Condition
#   - BMI : *Pass*
#   - PhysicalHealth : over 0
#   - MentalHealth : over 0
#   - GenHealth : Poor or Fair
#   - DiffWalking : Yes
# - Disease
#   - Stroke : Yes
#   - KidneyDisease : Yes
#   - Diabetic : Yes
#   - Asthma : Yes
#   - SkinCancer : Yes
#   
#   
# The graph below a percentage of heart disease, according to the total score of each shows the rate of heart disease in the score. The higher the score group, the higher the rate of heart disease. Up to five points account for about 69.1% of the total number of heart disease.

# In[ ]:


def dm_score(df):
    score = 0
    if df['Sex']=='Male': score += 1
    if df['AgeCategory']>=65: score += 1
    return  score
def ls_score(df):
    score = 0
    if df['Smoking']=='Yes': score += 1
    if df['PhysicalActivity']=='No': score += 1
    if df['SleepTime']<6: score += 1
    return score
def hc_score(df):
    score = 0
    if df['PhysicalHealth']>0: score += 1
    if df['MentalHealth']>0: score += 1
    if df['GenHealth'] in ['Poor', 'Fair']: score += 1
    if df['DiffWalking']=='Yes': score += 1
    return score
def ds_score(df):
    score = 0
    if df['Stroke']=='Yes': score += 1
    if df['KidneyDisease']=='Yes': score += 1
    if df['Diabetic']=='Yes': score += 1
    if df['Asthma']=='Yes': score += 1
    if df['SkinCancer']=='Yes': score += 1
    return score

# prepare data
temp_ = train_df.copy()
temp_['DM_SCORE'] = temp_.apply(lambda x: dm_score(x), axis=1) # Max 2
temp_['LS_SCORE'] = temp_.apply(lambda x: ls_score(x), axis=1) # Max 3
temp_['HC_SCORE'] = temp_.apply(lambda x: hc_score(x), axis=1) # Max 4
temp_['DS_SCORE'] = temp_.apply(lambda x: ds_score(x), axis=1) # Max 5
temp_['SCORE'] = temp_['DM_SCORE'] + temp_['LS_SCORE'] + temp_['HC_SCORE'] + temp_['DS_SCORE'] # Max 15

show_df = yn_pivot(temp_, [Y_COL, 'SCORE'])
show_df['Total_ratio'] = show_df['Total'] / show_df['Total'].sum()
show_df['No_cumsum'] = show_df['No'].cumsum()
show_df['Yes_cumsum'] = show_df['Yes'].cumsum()
show_df['Yes_ratio_cumsum'] = show_df['Yes_cumsum'] / (show_df['No_cumsum']+show_df['Yes_cumsum'])
show_df['Yes_ratio_cumsum/base'] = show_df['Yes_ratio_cumsum'] / base_ratio

fig = make_subplots(specs=[[{"secondary_y":True}]])
fig.add_trace(go.Bar(x=show_df.index, y=show_df['Yes_ratio'], marker=dict(color=['lightgray']*5+[RED]+['lightgray']*8),
                     name='Rate of heart disease in each score'),
              secondary_y=False)
fig.add_trace(go.Scatter(x=show_df.index, y=show_df['Yes_ratio_cumsum'], line=dict(color='gray'),
                         name='Cumulative heart disease rate'),
              secondary_y=True)
fig.update_layout(title='<b>Trends in rates of heart disease according to scores</b>', template=TEMPLATE)
fig.update_layout(legend=dict(x=0, y=1.4), margin=dict(l=20, r=20, t=200, b=70))
fig.update_yaxes(title="Heart Disease Ratio", rangemode="tozero", secondary_y=False)
fig.update_yaxes(title="Cumulative Sum of Ratio", rangemode="tozero", secondary_y=True)
fig.update_xaxes(title="Risk Score")
fig.show()


# # 7. Heart disease prediction
# Finally, I will test the classification of heart disease. I will check 3 accuracy indicators.  
# **Accuracy** : Correctly correct percentage of total predictions  
# **F1 Score** : F1 Score for predicting heart disease  
# **Type II error** : It's an actual heart disease, but it's predicted to be normal (*It's very dangerous if you miss a heart disease.*)

# ## 7.1 Baseline Model
# First, the baseline prediction. Most of them are not heart disease, so I'll predict all of them as normal people. For baseline models, the accuracy is high, but all heart disease people are missed.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

temp_test = test_df.copy()
y_test = temp_test[Y_COL].apply(lambda x : 1 if x=='Yes' else 0)

dummy_pred = [0]*len(test_df) # all normal
print(classification_report(y_test, dummy_pred)) # all normal
print(f"Accuracy : {100*accuracy_score(y_test, dummy_pred):.2f}%")
print(f"F1 Score : {100*f1_score(y_test, dummy_pred, pos_label=1):.2f}%")
print(f"Type II error : {confusion_matrix(y_test, dummy_pred)[1][0]} ({100*confusion_matrix(y_test, dummy_pred)[1][0]/sum(y_test):.1f}%)")


# ## 7.2 Risk Score Model (RandomForestClassifier)
# Next, I will test the model that I learned with 5 score. (Demography, Life Style, Health Condition, Disease and Total Score) Here, I only handled the data imbalance with class weight.  
# Although the accuracy of the Risk model has decreased a lot, there have been many improvements in Type II errors.  We missed about 26.9% of the total number of heart disease people.

# In[ ]:


f_cols = ['SCORE', 'DM_SCORE', 'LS_SCORE', 'HC_SCORE', 'DS_SCORE']

temp_train = train_df.copy()
temp_train['DM_SCORE'] = temp_train.apply(lambda x: dm_score(x), axis=1) # Max 2
temp_train['LS_SCORE'] = temp_train.apply(lambda x: ls_score(x), axis=1) # Max 3
temp_train['HC_SCORE'] = temp_train.apply(lambda x: hc_score(x), axis=1) # Max 4
temp_train['DS_SCORE'] = temp_train.apply(lambda x: ds_score(x), axis=1) # Max 5
temp_train['SCORE'] = temp_train['DM_SCORE'] + temp_train['LS_SCORE'] + temp_train['HC_SCORE'] + temp_train['DS_SCORE'] # Max 15

temp_test = test_df.copy()
temp_test['DM_SCORE'] = temp_test.apply(lambda x: dm_score(x), axis=1) # Max 2
temp_test['LS_SCORE'] = temp_test.apply(lambda x: ls_score(x), axis=1) # Max 3
temp_test['HC_SCORE'] = temp_test.apply(lambda x: hc_score(x), axis=1) # Max 4
temp_test['DS_SCORE'] = temp_test.apply(lambda x: ds_score(x), axis=1) # Max 5
temp_test['SCORE'] = temp_test['DM_SCORE'] + temp_test['LS_SCORE'] + temp_test['HC_SCORE'] + temp_test['DS_SCORE'] # Max 15

# for c in []:
#     le = LabelEncoder()
#     temp_train[c]=le.fit_transform(temp_train[c])
#     temp_test[c]=le.transform(temp_test[c])
    
X_train = temp_train[f_cols]
y_train = temp_train[Y_COL].apply(lambda x : 1 if x=='Yes' else 0)

X_test = temp_test[f_cols]
y_test = temp_test[Y_COL].apply(lambda x : 1 if x=='Yes' else 0)

yn_m = len(train_df[train_df[Y_COL]=='No']) / len(train_df[train_df[Y_COL]=='Yes'])
model = RandomForestClassifier(class_weight={0:1, 1:yn_m})
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred)) # all normal
print(f"Accuracy : {100*accuracy_score(y_test, y_pred):.2f}%")
print(f"F1 Score : {100*f1_score(y_test, y_pred, pos_label=1):.2f}%")
print(f"Type II error : {confusion_matrix(y_test, y_pred)[1][0]} ({100*confusion_matrix(y_test, y_pred)[1][0]/sum(y_test):.1f}%)")


# # 8. Summary
# 1. We divided the heart disease data into 4 categories (Demography, Life Style, Health Condition, Disease) and identified criteria for high heart disease rates.  
# 2. The criteria identified earlier were used to calculate four scores, some of which were unclear and not used by personal judgment.  
#   - The larger the total of the four scores, the higher the rate of heart disease.
# 3. Testing the classification model using these scores showed improved performance over the baseline.  
#   - Because it was a simple classification test, there is still room for further performance improvement with tuning and feature selection.
