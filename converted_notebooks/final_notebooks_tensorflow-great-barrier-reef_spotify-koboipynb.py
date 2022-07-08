#!/usr/bin/env python
# coding: utf-8

# # Why is this an interesting question?
# I’d like to explore how good the features that spotify has created to classify songs are. I’ve been interested in the discover weekly playlist/ and song radio feature in the app. It’s still quite a hit or miss. So it’ll be cool to explore this spotify dataset

# # Data description¶
# The data for every feature of every song is very complete, and description of each feature is clearly documented in (https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features).
# 
# The data type for every feature is numerical. Most features' method of measure seems quite straightforward other than 'danceability' which I suspect could be rather subjective. However, since the dataset is specific to one user, the subjectiveness of the metric can be ignored.
# 
# There are a total of 195 songs, of which 100 are liked, and the rest (95) disliked

# In[ ]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np 


# In[ ]:


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, roc_auc_score 
from sklearn.tree import DecisionTreeClassifier,plot_tree,export_text
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.impute import KNNImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# # Exploarory Data Analysis 

# In[ ]:


df = pd.read_csv("../input/spotify-recommendation/data.csv")


# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()## there is no missing data


# # Data pre-processing
# creating Train and test sets

# In[ ]:


y=df["liked"]#the column of like data 
X= df.drop(["liked"],axis=1)#all the data without the liked to enter it to the model 


# In[ ]:


df['liked'].value_counts().plot(kind='pie',autopct='%2.f')
plt.show


# In[ ]:


plt.figure(figsize=(20,8))
heatmap = sns.heatmap(df.corr() , vmin=-1 , vmax=1 ,annot=True , cmap='BrBG')
heatmap.set_title('Correlation HeatMap' , fontdict = {'fontsize' : 18} , pad=12)


# In[ ]:


sns.pairplot(df,hue='liked')
# this function will create a gird of axes such that each numeric variable in data will be shared across the y-axes
# across a single row and the x-axes across a single column.The diagonal plots are treated differently 
# the diagonal plots are traeted differently: a univariate distribution plot is drwan to show the marginal 
# marignal distribution of the data in each column 


# ## Danceability 
# - data is right skewed 
# - filtering for 'liked' the distribution of dancebility has less variance and there is a distinct range where liked songs occupy  

# # correlation between all features and target column'liked'
# just based off this correlation matrix,we can tell that this spotify user has a preference for song:
# 1.scoring high on--> danceability,loudness,speechiness
# 2.scoring low on -->  instrumentalness, duration

# In[ ]:


liked = df['liked']==1
disliked = df['liked']==0
liked_songs = df[liked]
disliked_songs = df[disliked]
plt.figure(figsize=(8,6))
plt.hist(liked_songs['speechiness'],alpha=0.5,label="liked songs")
plt.hist(disliked_songs['speechiness'],alpha=0.5,label="disliked songs")

plt.xlabel("Speechiness",size=14)
plt.ylabel("Count",size=14)
plt.title("Comparison of speechiness between liked and disliked songs")
plt.legend(loc='upper right')


# Here,we can see that almost all the user's disliked songs very low on Speechiness

# In[ ]:


plt.figure(figsize=(8,6))
plt.hist(liked_songs['danceability'], alpha=0.5, label="liked songs")
plt.hist(disliked_songs['danceability'], alpha=0.5, label="disliked songs")

plt.xlabel("Speechiness", size=14)
plt.ylabel("Count", size=14)
plt.title("Comparison of danceability between liked and disliked songs")
plt.legend(loc='upper right')


# And here,we can see that many of the user's liked songs score very high on Danceability 
# From the two cahrs above,we can see that even if a song scores high on Danceability,it still wouldn't 
# guarantee teh user's like.We can also see that even if a song scores low on speechiness,it doesn't guarantee the user's dislike 

# I wonder if we combine these two features, we might get a starker difference
# 

# In[ ]:


high_speechiness = df['speechiness'] > 0.3 
high_danceability = df['danceability'] > 0.6

songs_with_high_dance_and_speech = df[high_speechiness & high_danceability]
plt.title("Distribution of likes among songs with high speechiness and danceability")
songs_with_high_dance_and_speech['liked'].value_counts().sort_values().plot(kind = 'barh')


# In[ ]:


songs_with_low_dance_and_speech = df[~high_speechiness & ~high_danceability]
plt.title("Distribution of likes among songs with low speechiness and danceability")
songs_with_low_dance_and_speech['liked'].value_counts().plot(kind = 'barh')


# it looks like if a song scores highly on both speechiness and danceability,it's very likely to get the user's 
# like and the opposite is true: if a song scores low on both speechiness and danceability,it would also be more 
# likely to be disliked.
# 
# Another interesting thing I noted in the correlation matrix was how danceability have a very strong inverse 
# relationship to instrumentalness(which makes sens).And i'm curious how this relationship would like ?

# In[ ]:


plt.plot(df['danceability'], df['instrumentalness'], 'o', markersize = 4, alpha=0.6)
plt.title("Scatter plot of songs based on instrumentalness and danceability")
plt.xlabel("Danceability", size=14)
plt.ylabel("Instrumentalness", size=14)


# In[ ]:


from scipy.stats import linregress

xs = df['danceability']
ys = df['instrumentalness']

# Compute the linear regression
res = linregress(xs, ys)

# Plot the line of best fit
fx = np.array([xs.min(), xs.max()])
fy = res.slope * fx + res.intercept
plt.plot(fx, fy, '-', alpha=0.7)

plt.title("Best fit line of relationship between instrumentalness and danceability")
plt.xlabel("Danceability", size=14)
plt.ylabel("Instrumentalness", size=14)


# In[ ]:


c = df.corr()
c['liked'].sort_values(kind="quicksort")


# Again,jsut to reiterate,it seems like author likes speechiness,danceability,non-instrumental,short songs,and that
# makes sense he mentioned he likes mostly french rap

# ## key 
# - data is categorical and relatively uniformly distributed 
# ## loudness 
# - Data is left skewed 
# - loudness correlated with energy 
# - fitering for 'liked' the distribution of loudness has less variance and there is a distinct range where liked 
# songs occupy 
# ## made 
# - data is binary 
# - no difference between 'liked' and 'unliked' distributions
# ## speechiness
# - data is right skewed
# - fitering for'liked' the distribution of speechiness has less variance for unliked songs and there is a distinct 
# range wehre songs are not liked 
# ## Acousticness
# - Data is right skewed
# - Filtering for 'liked' the distribution of accousticness follows similar shapes,however a certian range are not liked
# ## instrumentalness
# - majority of data is zero or near zero;however if the zero records are igonred the data is left skewed 
# - fitering for 'liked' the distribution of instrumentalness is extermely tight for songs that are liked
# ## liveness
# - data is right skewed
# - Distribution between liked and unliked songs are similar from the dataset
# - Clustring can be seen through the various features wiht respect to the 'liked' feature
# ## valence
# - Data appears fairly uniform
# - spliting the data into 'liked' and 'unliked' there is a specific range of valence where songs are liked and unliked
# ## Tempo 
# - tempo is an important feature in terms of music analysis.it can be as significant as melody,harmony or rhythm 
# - due to the fact that it represents the speed of a song and the mood it evokes.for instance,the higher the BPM of 
# - a song,the faster the song is and consequently more inspiring and joyful it tends to be. on the other hand, 
# - a low BPM means the song is slower,which can indicate sadness,romance or drama 
# - data appears to follow a normal distribution 
# - Liked songs are typically slightly higher tempo
# ## Duration 
# - Data appears to be a normal distribution with a low volume right tail 
# - clear clustring seen on the duration metric between liked and unliked songs
# - Durations of liked songs tend to be shorter
# ## Time signature
# - Data is categorical with the large majority with a time signature of 4
# - Tighter distribution with less variance for liked songs on the time signature
# ## Liked
# - Data is binary with either feature evenly distributed 
# - Data is likely this way due to the dataset provided being aggregated on liked and unliked songs
# 

# # Skew

# In[ ]:


plt.figure(figsize=(15,5))
sns.scatterplot(df.skew().sort_values(ascending=False),df.skew().sort_values(ascending=False).index);


# # observation 

# Energy¶
# Data is left skewed
# Filtering for 'liked', the distribution of energy has less variance and there is a distinct clusters 
# where songs are very likely to be unliked

# # Modeling 
# ## Logistic Regression Modeling 
# 

# In[ ]:


# Splitting dataset into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=14)
# scaling the various features to deal with the skew
scaler = StandardScaler()
X_train_n = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
X_test_n = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns)

# Creating an instance of a logistic regression model
log = LogisticRegression()

# Fitting the model with the scaled data
log.fit(X_train_n, y_train)

# Create predictions
predictions = log.predict(X_test_n)
log.score(X_test,y_test)## it is very strange score!!!very low !!


# In[ ]:


# Running SVM with default hyperparamete  and try RBF
from sklearn.svm import SVC
svc = SVC()#default paramteres
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
print("Accuracy Score:)")
print(metrics.accuracy_score(y_test,y_pred))## it is best than logistic regression model


# In[ ]:


svc=SVC(kernel='rbf')## tryied poly .. and it is performing poorly maybe because it is overfitting the training 
## dataset
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))


# # Building ML Models
# Decision tree classifier

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver="liblinear")## i used liblinear because data is small and 
#it is algorithm for optimization problem 
cross_val_score(clf,X,y,scoring="accuracy",cv=5).mean()
#clf is the estimator to fit the data
# X the data to fit 
# Y the target variable 
# scoring: is function with signature scorer which should return only a single value
# cv: determines the corss-validation splitting strategy.


# In[ ]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=1)
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[ ]:


y_train.value_counts()


# In[ ]:


print("Accuracy Score of the Decision Tree Model" , accuracy_score(y_test , predictions))
print("ROC AUC score of the Decision Tree Model is : " , roc_auc_score(y_test ,predictions))


# # confusion Matrix

# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


metrics.confusion_matrix(y_test,predictions)


# In[ ]:


#confusion Matrix of Decision Tree Model 
cm  = confusion_matrix(y_test , predictions)

x_axis_labels = ["Yes" , "No"]
y_axis_labels = ["Yes" , "No"]

f , ax = plt.subplots(figsize=(10,7))
sns.heatmap(cm , annot=True, linewidths=0.2 , linecolor="black" , fmt=".0f" , ax=ax , cmap="Greens" , 
           xticklabels=x_axis_labels , yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title("Confusin Matrix Of Decision Tree Model")


# # conclusion
# initial pass of the  random forest modeling provides a 97% score.
# As we can tell from the extremely high accuracy score, spotify song features are impressively useful!
