#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Reading-and-Understanding-the-Data" data-toc-modified-id="Reading-and-Understanding-the-Data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Reading and Understanding the Data</a></span><ul class="toc-item"><li><span><a href="#Importing-data" data-toc-modified-id="Importing-data-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Importing data</a></span></li><li><span><a href="#Inspecting-data" data-toc-modified-id="Inspecting-data-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Inspecting data</a></span></li></ul></li><li><span><a href="#Exploring-features" data-toc-modified-id="Exploring-features-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Exploring features</a></span><ul class="toc-item"><li><span><a href="#Survived" data-toc-modified-id="Survived-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Survived</a></span></li><li><span><a href="#Pclass" data-toc-modified-id="Pclass-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Pclass</a></span></li><li><span><a href="#Name" data-toc-modified-id="Name-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Name</a></span></li><li><span><a href="#Sex" data-toc-modified-id="Sex-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Sex</a></span></li><li><span><a href="#Age" data-toc-modified-id="Age-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Age</a></span></li><li><span><a href="#SibSp" data-toc-modified-id="SibSp-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>SibSp</a></span></li><li><span><a href="#Parch" data-toc-modified-id="Parch-2.7"><span class="toc-item-num">2.7&nbsp;&nbsp;</span>Parch</a></span></li><li><span><a href="#New-column-'Alone'" data-toc-modified-id="New-column-'Alone'-2.8"><span class="toc-item-num">2.8&nbsp;&nbsp;</span>New column 'Alone'</a></span></li><li><span><a href="#Ticket" data-toc-modified-id="Ticket-2.9"><span class="toc-item-num">2.9&nbsp;&nbsp;</span>Ticket</a></span></li><li><span><a href="#Fare" data-toc-modified-id="Fare-2.10"><span class="toc-item-num">2.10&nbsp;&nbsp;</span>Fare</a></span></li><li><span><a href="#Cabin" data-toc-modified-id="Cabin-2.11"><span class="toc-item-num">2.11&nbsp;&nbsp;</span>Cabin</a></span></li><li><span><a href="#Embarked" data-toc-modified-id="Embarked-2.12"><span class="toc-item-num">2.12&nbsp;&nbsp;</span>Embarked</a></span></li><li><span><a href="#PassengerID" data-toc-modified-id="PassengerID-2.13"><span class="toc-item-num">2.13&nbsp;&nbsp;</span>PassengerID</a></span></li></ul></li><li><span><a href="#Data-Cleaning" data-toc-modified-id="Data-Cleaning-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data Cleaning</a></span><ul class="toc-item"><li><span><a href="#Deal-with-Missing-values" data-toc-modified-id="Deal-with-Missing-values-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Deal with Missing values</a></span><ul class="toc-item"><li><span><a href="#Embarked" data-toc-modified-id="Embarked-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Embarked</a></span></li><li><span><a href="#Age-Range" data-toc-modified-id="Age-Range-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>Age Range</a></span></li><li><span><a href="#Fare-Range" data-toc-modified-id="Fare-Range-3.1.3"><span class="toc-item-num">3.1.3&nbsp;&nbsp;</span>Fare Range</a></span></li></ul></li></ul></li><li><span><a href="#Creating-dummy-features" data-toc-modified-id="Creating-dummy-features-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Creating dummy features</a></span></li><li><span><a href="#Test-Train-Split" data-toc-modified-id="Test-Train-Split-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Test-Train Split</a></span></li><li><span><a href="#Feature-Scaling" data-toc-modified-id="Feature-Scaling-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Feature Scaling</a></span></li><li><span><a href="#Looking-at-Correlations" data-toc-modified-id="Looking-at-Correlations-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Looking at Correlations</a></span></li><li><span><a href="#Feature-Selection" data-toc-modified-id="Feature-Selection-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Feature Selection</a></span></li><li><span><a href="#Assessing-the-model-with-StatsModels" data-toc-modified-id="Assessing-the-model-with-StatsModels-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Assessing the model with StatsModels</a></span></li><li><span><a href="#Checking-VIFs" data-toc-modified-id="Checking-VIFs-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Checking VIFs</a></span></li><li><span><a href="#Model-Evaluation" data-toc-modified-id="Model-Evaluation-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Model Evaluation</a></span></li><li><span><a href="#Finding-the-optimal-cut-off" data-toc-modified-id="Finding-the-optimal-cut-off-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Finding the optimal cut off</a></span></li><li><span><a href="#Precision-and-Recall" data-toc-modified-id="Precision-and-Recall-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Precision and Recall</a></span></li><li><span><a href="#Making-predictions-on-the-test-set" data-toc-modified-id="Making-predictions-on-the-test-set-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Making predictions on the test set</a></span><ul class="toc-item"><li><span><a href="#Making-prediction" data-toc-modified-id="Making-prediction-14.1"><span class="toc-item-num">14.1&nbsp;&nbsp;</span>Making prediction</a></span></li><li><span><a href="#Evaluation" data-toc-modified-id="Evaluation-14.2"><span class="toc-item-num">14.2&nbsp;&nbsp;</span>Evaluation</a></span></li></ul></li><li><span><a href="#Making-predictions-on-the-test-data" data-toc-modified-id="Making-predictions-on-the-test-data-15"><span class="toc-item-num">15&nbsp;&nbsp;</span>Making predictions on the test data</a></span></li></ul></div>

# ## Reading and Understanding the Data

# In[ ]:


# Not show the warnings
import warnings
warnings.filterwarnings('ignore')

# import the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import xticks

# Display all columns
pd.set_option('display.max_columns',200)


# ### Importing data

# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
#train_data = pd.read_csv("train.csv")
train_data.head()


# In[ ]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
#test_data = pd.read_csv("test.csv")
test_data.head()


# In[ ]:


# Combine train & test data into 1 set
df = pd.concat([train_data, test_data], ignore_index = True, sort = False)
df.head()


# In[ ]:


# from sklearn.ensemble import RandomForestClassifier

# y = train_data["Survived"]

# features = ["Pclass", "Sex", "SibSp", "Parch"]
# #features = ["Pclass", "Sex", "SibSp", "Parch","Age","Ticket","Fare","Cabin","Embarked"]
# #features = ["Pclass", "Sex", "SibSp", "Parch","Fare"]
# X = pd.get_dummies(train_data[features])
# X_test = pd.get_dummies(test_data[features])

# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(X, y)
# predictions = model.predict(X_test)

# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
# output.to_csv('my_submission.csv', index=False)
# print("Your submission was successfully saved!")


# In[ ]:


# # Percentage of null values in each column
# data_useful_columns = test_data[features] 
# null_percentage = data_useful_columns.isnull().sum() * 100 / len(data_useful_columns)

# # Top columns have highest percentages of null values
# null_percentage.sort_values(ascending = False).head(50)


# In[ ]:


# # Import helpful libraries
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split

# y = train_data["Survived"]

# #features = ["Pclass", "Sex", "SibSp", "Parch"]
# features = ["Pclass", "Sex", "SibSp", "Parch","Age","Ticket","Fare","Cabin","Embarked"]
# #features = ["Pclass", "Sex", "SibSp", "Parch","Fare"]
# X = train_data[features]
# X_test = test_data[features]

# model = RandomForestRegressor(random_state=1)
# model.fit(X, y)
# predictions = model.predict(X_test)

# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
# output.to_csv('my_submission.csv', index=False)
# print("Your submission was successfully saved!")


# In[ ]:


# test_data["Survived"] = predictions


# In[ ]:


# test_data.head(5)


# In[ ]:


# output.head(5)


# ### Inspecting data

# In[ ]:


# Types of all columns
df.info(verbose=True)


# In[ ]:


# Check the number of rows and columns in the dataframe
df.shape


# In[ ]:


# Check the summary for the numeric columns 
df.describe()


# In[ ]:


# Percentage of null values in each column
null_percentage = df.isnull().sum() * 100 / len(df)


# In[ ]:


# Top columns have highest percentages of null values
null_percentage.sort_values(ascending = False).head(50)


# In[ ]:


# Look at the number of unique values each column has
df.nunique()


# ## Exploring features

# ### Survived

# In[ ]:


survived = train_data.loc[train_data.Survived == 1]["Survived"]
rate_survived = sum(survived) * 100 / len(train_data)

print("% of people survived: ", rate_survived)


# In[ ]:


# # Divide data into 2 sets
# survived = train_data[train_data.Survived==1]
# died = train_data[train_data.Survived==0]


# ### Pclass

# In[ ]:


# # Function to plot bivariates on heatmaps
# def heatmaps_bivariate(list_values,index):

#     plt.figure(figsize=[20,5])

#     plt.subplot(1,2,1)
#     plt.title('Survived') 
#     res = pd.pivot_table(data=survived,values=list_values,index=index,aggfunc=np.mean)
#     sns.heatmap(res,annot=True,cmap="RdYlGn",center=0.117)

#     plt.subplot(1,2,2)
#     plt.title('Died') 
#     res = pd.pivot_table(data=died,values=list_values,index=index,aggfunc=np.mean)
#     sns.heatmap(res,annot=True,cmap="RdYlGn",center=0.117)

#     plt.show()


# In[ ]:


# # Function to plot distribution
# def plot_univariate_pie(variable):
#     # Plot on a pie chart
#     fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,20))

#     # Non Defaulters
#     data_0 = died[variable].value_counts()
#     labels = data_0.index
#     ax1.pie(data_0, autopct='%1.1f%%',
#             shadow=True, startangle=90)
#     ax1.set_title('Died')
#     ax1.legend(labels, loc="lower right")

#     # Defaulters
#     data_1 = survived[variable].value_counts()
#     labels = data_1.index
#     ax2.pie(data_1, autopct='%1.1f%%',
#             shadow=True, startangle=90)
#     ax2.set_title('Survived')

#     ax2.legend(labels, loc="lower right")

#     plt.show()


# In[ ]:


# find_patterns
# def find_patterns_survived_died(attribute,data):
    
#     # plot_percent_survived 
#     plt.figure(figsize=(20,5))
#     df1 = data.groupby(attribute)["Survived"].value_counts(normalize=True)

#     df1 = df1.rename('percent').reset_index()
#     sns.barplot(x=attribute,y='percent',hue="Survived",data=df1)
    
#     # Pattern of survived
#     pattern_survived = df1[attribute][(df1.percent == 1) & (df1.Survived == 1)]
    
#     # Pattern of died
#     pattern_died = df1[attribute][(df1.percent == 1) & (df1.Survived == 0)]
    
#     return pattern_survived, pattern_died

# # check that pattern in data
# def check_patterns(pattern, attribute, data):
#     return data[data[attribute].isin(pattern)]


# In[ ]:


# Is Pclass a pattern?
# pattern_survived, pattern_died = find_patterns_survived_died('Pclass',train_data)


# In[ ]:


# check_patterns(pattern_died, 'Pclass', train_data)


# In[ ]:


# # Is SibSp a pattern?
# pattern_survived, pattern_died = find_patterns_survived_died('SibSp',train_data)


# In[ ]:


# Function to Plot count values in barplot
def plot_count_values_barplot(attribute, data):
    
    # plot_percent_survived 
    plt.figure(figsize=(20,5))
    df1 = data.groupby(attribute)["Survived"].value_counts(normalize=True)

    df1 = df1.rename('percent').reset_index()
    sns.barplot(x=attribute,y='percent',hue="Survived",data=df1)


# In[ ]:


# Plot count values in barplot 
plot_count_values_barplot('Pclass', train_data)


# Class 1 (Upper) had more chance to survive

# In[ ]:


# Assign values to conver Pclass to categorical type
df['Pclass'].replace([1, 2, 3], ['Upper','Middle','Lower'], inplace = True)


# In[ ]:


df['Pclass'].value_counts()


# ### Name

# In[ ]:


# Check 'Name' column
df['Name'].value_counts()


# In[ ]:


# Add title (from Passengers' names) to data set
df["Title"] = df["Name"].str.split(",").str[1].str.split(".").str[0].str.replace(" ", "")
df.head()


# In[ ]:


# Drop Name
df.drop("Name", axis=1, inplace=True)


# ### Sex

# In[ ]:


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women) * 100 / len(women)

print("% of women who survived:", rate_women)


# In[ ]:


men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men) * 100 / len(men)

print("% of men who survived:", rate_men)


# In[ ]:


# Converting binary variable (Yes/No) to 0/1
df['Sex'].replace(['male', 'female'], [1,0], inplace = True)


# ### Age

# In[ ]:


# Plot Age on a boxplot
train_data.boxplot(column='Age', return_type='axes',vert=False);


# In[ ]:


# Check min, max
df.Age.describe()


# In[ ]:


# Binning YEARS_BIRTH to Group of age
df['Age_range']=pd.cut(df['Age'], bins=[0,10,20,30,40,50,60,70,80], labels=['(0_10]','(10_20]','(20_30]', '(30_40]', '(40_50]', '(50_60]', '(60_70]', '(70_80]'])


# In[ ]:


# Plot count values in barplot 
plot_count_values_barplot('Age_range', df[~df.Survived.isnull()])


# Children less or equal to 10 years old had more chance to survive.

# In[ ]:


# Drop
df.drop('Age', axis=1, inplace=True)


# ### SibSp

# In[ ]:


# Plot count values in barplot 
plot_count_values_barplot('SibSp', df[~df.Survived.isnull()])


# Passengers with more sibblings had less chance to survive.

# ### Parch

# In[ ]:


# Plot count values in barplot 
plot_count_values_barplot('Parch', df[~df.Survived.isnull()])


# Passengers with more parents and children had less chance to survive.

# ### New column 'Alone'

# In[ ]:


# Check the number of rows
df.shape[0]


# In[ ]:


# Create new columns
df['Alone'] = np.NaN


# In[ ]:


# If no siblings & no parents & children -> Alone
for i in range(0,df.shape[0]):
    if df['SibSp'][i] + df['Parch'][i] == 0:
        df['Alone'][i] = 1
    else:
        df['Alone'][i] = 0


# In[ ]:


# Check values
df.Alone.value_counts()


# In[ ]:


# Plot count values in barplot 
plot_count_values_barplot('Alone', df[~df.Survived.isnull()])


# ### Ticket

# In[ ]:


# Check values
df.Ticket.value_counts()


# -> Has no meaning -> Drop

# In[ ]:


# Drop
df.drop('Ticket', axis=1, inplace=True)


# In[ ]:


# check columns
df.columns


# ### Fare

# In[ ]:


# Check values
df.Fare.value_counts()


# In[ ]:


# Define the range of quantiles to use: q=[0, .2, .4, .6, .8, 1]. Binning Fare based on quantiles.
range_labels = ['Very Low', 'Low', "Medium", 'High', 'Very high']
df["Fare_range"] = pd.qcut(df.Fare,
                              q=[0, .2, .4, .6, .8, 1],
                              labels=range_labels)


# In[ ]:


# Plot count values in barplot 
plot_count_values_barplot('Fare_range', df[~df.Survived.isnull()])


# People who paid more had more chance to survive.

# In[ ]:


# Drop
df.drop('Fare', axis=1, inplace=True)


# ### Cabin

# In[ ]:


# Check values
df.Cabin.value_counts()


# -> Has no meaning -> Drop

# In[ ]:


# Drop
df.drop('Cabin', axis=1, inplace=True)


# In[ ]:


# check columns
df.columns


# ### Embarked

# In[ ]:


# Check values
df.Embarked.value_counts()


# In[ ]:


# Plot count values in barplot 
plot_count_values_barplot('Embarked', df[~df.Survived.isnull()])


# Passengers who boarded from C had higher survival chances than passengers who boarded from S or Q.

# ### PassengerID

# In[ ]:


# Check values
df.PassengerId.value_counts()


# In[ ]:


# Drop
df.drop('PassengerId', axis=1, inplace=True)


# ## Data Cleaning

# ### Deal with Missing values

# In[ ]:


# Percentage of null values in each column
null_percentage = df.isnull().sum() * 100 / len(df)


# In[ ]:


# Top columns have highest percentages of null values
null_percentage.sort_values(ascending = False).head(50)


# In[ ]:


# Check how many rows missing values
df.isnull().sum()


# #### Embarked

# Only 2 rows missing Embarked -> Will fill with the value most passengers had

# In[ ]:


# Find the value most passengers had
df['Embarked'].mode()


# In[ ]:


# Check mode
df['Embarked'].mode()[0]


# In[ ]:


# Fill with the value most passengers had
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace = True)


# In[ ]:


# Check how many rows missing values
df.isnull().sum()


# #### Age Range

# In[ ]:


# Create a column to represent ['Sex', 'Pclass', 'Title']
df['Background'] = df["Sex"].map(str) + df["Pclass"].map(str) + df["Title"].map(str)
df['Background']


# In[ ]:


# Group by column 'background'
grouped_df = df[['Background', 'Age_range']].groupby(by='Background')

# Retrieve group
for key,item in grouped_df:
    a_group = grouped_df.get_group(key)
    print(a_group, "\n")
    
# Source: https://www.kite.com/python/answers/how-to-group-a-pandas-dataframe-by-multiple-columns-in-python


# In[ ]:


def fast_mode(df, key_cols, value_col):
    """ 
    Calculate a column mode, by group, ignoring null values. 

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame over which to calcualate the mode. 
    key_cols : list of str
        Columns to groupby for calculation of mode.
    value_col : str
        Column for which to calculate the mode. 

    Return
    ------ 
    pandas.DataFrame
        One row for the mode of value_col per key_cols group. If ties, 
        returns the one which is sorted first. 
    """
    return (df.groupby(key_cols + [value_col]).size() 
              .to_frame('counts').reset_index() 
              .sort_values('counts', ascending=False) 
              .drop_duplicates(subset=key_cols)).drop(columns='counts')

# Source: https://stackoverflow.com/questions/55562696/how-to-replace-missing-values-with-group-mode-in-pandas


# In[ ]:


# Fill the missing values with modes of the suitable group
df.loc[df.Age_range.isnull(), 'Age_range'] = df['Background'].map(fast_mode(df, ['Background'], 'Age_range').set_index(['Background']).Age_range)


# In[ ]:


# Check how many rows missing values
df.isnull().sum()


# #### Fare Range

# In[ ]:


# Check the row having null fare_range
df[df.Fare_range.isnull()]


# In[ ]:


# Fill the missing values with modes of the suitable group
df.loc[df.Fare_range.isnull(), 'Fare_range'] = df['Background'].map(fast_mode(df, ['Background'], 'Fare_range').set_index(['Background']).Fare_range)


# In[ ]:


# Check how many rows missing values
df.isnull().sum()


# In[ ]:


# Drop column
df.drop("Background", axis=1, inplace=True)


# In[ ]:


# Check df
df.head()


# ## Creating dummy features

# In[ ]:


# Check columns types
df.info()


# In[ ]:


# Looking for categorical variables
df.select_dtypes(['object','category']).columns


# In[ ]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy = pd.get_dummies(df[['Pclass', 'Embarked', 'Title', 'Age_range', 'Fare_range']], drop_first=True)

# Adding the results to the master dataframe
df = pd.concat([df, dummy], axis=1)


# In[ ]:


# We have created dummies for the below variables, so we can drop them
df = df.drop(['Pclass', 'Embarked', 'Title', 'Age_range', 'Fare_range'], 1)


# ## Test-Train Split

# In[ ]:


# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:


# Putting feature variable to X
X = df[~df.Survived.isnull()].drop(['Survived'], 1)

# Putting response variable to y
y = df[~df.Survived.isnull()]['Survived']


# In[ ]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ## Feature Scaling

# In[ ]:


# Show all numerical columns
X_train.describe().columns


# In[ ]:


# Scale features of X train set
scaler = StandardScaler()

X_train[['Sex', 'SibSp', 'Parch', 'Alone', 'Pclass_Middle', 'Pclass_Upper',
       'Embarked_Q', 'Embarked_S', 'Title_Col', 'Title_Don', 'Title_Dona',
       'Title_Dr', 'Title_Jonkheer', 'Title_Lady', 'Title_Major',
       'Title_Master', 'Title_Miss', 'Title_Mlle', 'Title_Mme', 'Title_Mr',
       'Title_Mrs', 'Title_Ms', 'Title_Rev', 'Title_Sir', 'Title_theCountess',
       'Age_range_(10_20]', 'Age_range_(20_30]', 'Age_range_(30_40]',
       'Age_range_(40_50]', 'Age_range_(50_60]', 'Age_range_(60_70]',
       'Age_range_(70_80]', 'Fare_range_Low', 'Fare_range_Medium',
       'Fare_range_High', 'Fare_range_Very high']] = scaler.fit_transform(X_train[['Sex', 'SibSp', 'Parch', 'Alone', 'Pclass_Middle', 'Pclass_Upper',
       'Embarked_Q', 'Embarked_S', 'Title_Col', 'Title_Don', 'Title_Dona',
       'Title_Dr', 'Title_Jonkheer', 'Title_Lady', 'Title_Major',
       'Title_Master', 'Title_Miss', 'Title_Mlle', 'Title_Mme', 'Title_Mr',
       'Title_Mrs', 'Title_Ms', 'Title_Rev', 'Title_Sir', 'Title_theCountess',
       'Age_range_(10_20]', 'Age_range_(20_30]', 'Age_range_(30_40]',
       'Age_range_(40_50]', 'Age_range_(50_60]', 'Age_range_(60_70]',
       'Age_range_(70_80]', 'Fare_range_Low', 'Fare_range_Medium',
       'Fare_range_High', 'Fare_range_Very high']])

X_train.head()


# ## Looking at Correlations

# In[ ]:


# Correlation matrix
cor_matrix = df.corr()
cor_matrix


# In[ ]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
mask = np.array(cor_matrix)
mask[np.tril_indices_from(mask)] = False
sns.heatmap(cor_matrix, mask=mask, vmax=.8, square=True, annot = True)
plt.show()


# In[ ]:


# Selecting strong correlation pairs (magnitude greater than 0.5)
corr_pairs = cor_matrix.unstack()
sorted_pairs = corr_pairs.sort_values(kind="quicksort")
strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]

print("Strong correlation pairs (magnitude greater than 0.5):")
print(strong_pairs)

# (Source: https://likegeeks.com/python-correlation-matrix/)


# Try another method as below:

# In[ ]:


# Remove duplicated pairs in the correlation table
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

# Get top pairs
def get_top_abs_correlations(df, n=10):
    corr_list = df.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    corr_list = corr_list.drop(labels=labels_to_drop).sort_values(ascending=False)
    return corr_list[0:n]

#(Source: https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas)


# In[ ]:


# Get top 10 correlation pairs
print("Top 10 correlation pairs:")
get_top_abs_correlations(cor_matrix, 10)


# In[ ]:


# Check columns
df.columns


# ## Feature Selection

# In[ ]:


# Create object of Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[ ]:


# Import RFE and select 15 variables
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[ ]:


# See the RFE Ranking
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[ ]:


# All of the columns selected by RFE
col = X_train.columns[rfe.support_]
col


# In[ ]:


# See the columns not selected by RFE
X_train.columns[~rfe.support_]


# In[ ]:


# New X_train = columns selected by RFE
X_train = X_train[col]


# ## Assessing the model with StatsModels

# In[ ]:


# Import statsmodels
import statsmodels.api as sm

# Assess the model with StatsModels
X_train_sm = sm.add_constant(X_train)
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# Insights: p-values > 0.05:
#  - Alone
#  - Title_Don	
#  - Title_Jonkheer
#  - Title_Rev
#  - Title_Sir
#  - Age_range_(20_30]
#  - Age_range_(40_50]	
#  - Age_range_(70_80]
#  - Fare_range_Very high

# ## Checking VIFs

# In[ ]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Drop
X_train = X_train.drop(["Title_Don"], 1)


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Drop column
X_train = X_train.drop("Title_Jonkheer", 1)


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train)
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Drop "Embarked_S"
X_train = X_train.drop("Title_Sir", 1)


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train)
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Drop
X_train = X_train.drop("Title_Rev", 1)


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train)
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Drop
X_train = X_train.drop("Age_range_(70_80]", 1)


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train)
logm6 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm6.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Drop
X_train = X_train.drop("Alone", 1)


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train)
logm7 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm7.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Drop
X_train = X_train.drop("Age_range_(40_50]", 1)


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train)
logm8 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm8.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Drop
X_train = X_train.drop("Age_range_(20_30]", 1)


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train)
logm9 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm9.fit()
res.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ## Model Evaluation

# In[ ]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[ ]:


# Reshape it into an array
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[ ]:


# Create dataframe with conversion flag and predicted probabilities
y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'Survived_Prob':y_train_pred})
y_train_pred_final.head()


# In[ ]:


# Creating new column 'predicted' with 1 if Converted_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[ ]:


# import metrics
from sklearn import metrics


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted))


# In[ ]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )
print(confusion)


# In[ ]:


# Evaluate other metrics
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[ ]:


# Let us calculate specificity
TN / float(TN+FP)


# ## Finding the optimal cut off

# In[ ]:


# ROC function
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[ ]:


# Draw the ROC
draw_roc(y_train_pred_final.Survived, y_train_pred_final.Survived_Prob)


# In[ ]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[ ]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i, accuracy, sensi, speci]
print(cutoff_df)


# In[ ]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])
plt.show()


# In[ ]:


# Creating new column 'predicted' with 1 if Survived_Prob > 0.37 else 0
y_train_pred_final['final_predicted'] = y_train_pred_final.Survived_Prob.map( lambda x: 1 if x >= 0.45 else 0)

# View top rows of y_train_pred_final
y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)


# In[ ]:


# Confusion matrix 
confusion2 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.final_predicted)
print(confusion2)


# In[ ]:


# Evaluate other metrics

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[ ]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[ ]:


# Let us calculate specificity
TN / float(TN+FP)


# ## Precision and Recall

# In[ ]:


#Looking at the confusion matrix again with the original predicted value
confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted)
confusion


# In[ ]:


# Calculate Precision
confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[ ]:


# Calculate Recall
confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[ ]:


# import precision_recall_curve
from sklearn.metrics import precision_recall_curve


# In[ ]:


# Calculate precision & recall
p, r, thresholds = precision_recall_curve(y_train_pred_final.Survived, y_train_pred_final.Survived_Prob)


# In[ ]:


# Plot precision & recall
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ## Making predictions on the test set

# ### Making prediction

# In[ ]:


# Applying the scaling on the test sets
num_vars = X_train.columns
X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
X_test[num_vars] = scaler.transform(X_test[num_vars])


# In[ ]:


# Limit columns of X_test like X_train
X_test = X_test[num_vars]


# In[ ]:


# Adding constant variable to test dataframe
X_test_m = sm.add_constant(X_test)


# In[ ]:


# Make prediction on test set
y_test_pred = res.predict(X_test_m)


# In[ ]:


# View y_test_pred
y_test_pred[:10]


# ### Evaluation

# In[ ]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[ ]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[ ]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[ ]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[ ]:


# Check y_pred_final
y_pred_final.head()


# In[ ]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={0 : 'Survived_Prob'})


# In[ ]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[ ]:


# Calculate final_predicted with cutoff point 0.37
y_pred_final['final_predicted'] = y_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.45 else 0)


# In[ ]:


# View y_pred_final
y_pred_final


# In[ ]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Survived, y_pred_final.final_predicted)


# In[ ]:


# Calculate confusion matrix
confusion2 = metrics.confusion_matrix(y_pred_final.Survived, y_pred_final.final_predicted)
confusion2


# In[ ]:


# Calculate TP, TN, FP, FN
TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[ ]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[ ]:


# Let's calculate specificity
TN / float(TN+FP)


# ## Making predictions on the test data

# In[ ]:


# copy test_data
test_copy = test_data.copy()


# In[ ]:


test_data = df[df.Survived.isnull()]


# In[ ]:


test_data.columns


# In[ ]:


# Applying the scaling on the test sets
num_vars = X_train.columns
X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
test_data[num_vars] = scaler.transform(test_data[num_vars])


# In[ ]:


# Select the columns in train_data for test_data as well
test_data = test_data[num_vars]
test_data.shape


# In[ ]:


# Make prediction on test set
y_test_data_pred = res.predict(sm.add_constant(test_data))


# In[ ]:


# View y_test_pred
y_test_data_pred[:10]


# In[ ]:


# Converting y_pred to a dataframe which is an array
y_test_data_pred = pd.DataFrame(y_test_data_pred)


# In[ ]:


y_test_data_pred


# In[ ]:


# Renaming the column 
y_test_data_pred= y_test_data_pred.rename(columns={0 : 'Survived_Prob'})


# In[ ]:


# Calculate final_predicted with cutoff point 0.37
y_test_data_pred['final_predicted'] = y_test_data_pred.Survived_Prob.map(lambda x: 1 if x > 0.45 else 0)


# In[ ]:


# View y_pred_final
y_test_data_pred


# In[ ]:


# Let's see the head of y_pred_final
y_test_data_pred.head()


# In[ ]:


# Removing index for both dataframes to append them side by side 
y_test_data_pred.reset_index(drop=True, inplace=True)
test_copy.reset_index(drop=True, inplace=True)


# In[ ]:


# Appending y_test_data_pred and test_data
submit = pd.concat([y_test_data_pred, test_copy['PassengerId']],axis=1)


# In[ ]:


submit.head()


# In[ ]:


# Renaming the column 
submit= submit.rename(columns={'final_predicted' : 'Survived'})


# In[ ]:


submit_final = submit[['PassengerId','Survived']]


# In[ ]:


submit_final


# In[ ]:


# Renaming the column 
submit_final= submit_final.rename(columns={'final_predicted': 'Survived'})


# In[ ]:


submit_final.to_csv('submit.csv', index = False)

