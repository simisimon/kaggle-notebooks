#!/usr/bin/env python
# coding: utf-8

# # Chargement des librairies python

# In[ ]:


import numpy as np #manipulation
import pandas as pd #traitement et l'analyse
import seaborn as sns #la visualisation
import matplotlib.pyplot as plt
plt.style.use("ggplot")  #pour le style ggplot

from mpl_toolkits.mplot3d import Axes3D
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud, STOPWORDS

import warnings # eliminer les erreurs de version
warnings.filterwarnings("ignore")


# In[ ]:


df=pd.read_csv("../input/maroc-avito-car-dataset/avito_car_dataset_ALL.csv", encoding='latin-1')


# élimination des marque rare que nous n'avons pas assez d'informations sur
df.drop(df[ (df['Marque'] == "Suzuki") | (df['Marque'] == "mini") 
           | (df['Marque'] == "Alfa Romeo") | (df['Marque'] == "Chevrolet")
          | (df['Marque'] == "Jeep") ].index, inplace=True)

# affichage des 5 premières ligne
df.head()


# In[ ]:


df.shape


# In[ ]:


# affichage de la liste et format des différentes variables

df.info()


# In[ ]:


#checking les valeurs nulles
df.isna().sum()


# # Préparation des données

# ### Remplacer les valeurs manquantes

# In[ ]:


df['Boite de vitesses'].value_counts()


# In[ ]:


df['Origine'].value_counts()


# In[ ]:


df['État'].value_counts()


# In[ ]:


df['Première main'].value_counts()


# In[ ]:


df['Boite de vitesses'].replace(to_replace='--',value='Manuelle',inplace=True)


df['Origine'].replace(to_replace='',value='WW au Maroc',inplace=True)
df['Nombre de portes'].replace(to_replace='',value='5',inplace=True)
df['État'].replace(to_replace='',value='Bon',inplace=True)
df['Première main'].replace(to_replace='',value='Non',inplace=True)

# convertir les booléens en 0 et 1
df.replace(to_replace=True,value=1,inplace=True)
df.replace(to_replace=False,value=0,inplace=True)


# In[ ]:


df.drop(columns=["Unnamed: 0","Lien","Secteur","Ville"],inplace=True) #suppression

#drop any row with NaN values
df = df.dropna()

#checking null value 
df.isna().sum()


# In[ ]:


#voir le dataset
df.head()


# # Analyse et mise en forme des données

# In[ ]:


# let see the data describe
df.describe().round(2)


# In[ ]:


# converssion en entiers

df['Année-Modèle'] = df['Année-Modèle'].astype(int)
df['Nombre de portes'] = df['Nombre de portes'].astype(int)


# In[ ]:


splited = df['Kilométrage'].str.split("-", n = 1, expand = True)
splited[0] = splited[0].str.replace(' ','').astype(int)
splited[1] = splited[1].str.replace(' ','').astype(int)
df['Kilométrage'] = (splited[1] + splited[0])/2


# In[ ]:


df['Puissance fiscale'] = df['Puissance fiscale'].astype(int)


# In[ ]:


df.describe().round(2)


# # Visualisation de données 

# #### La marque

# In[ ]:


df["Marque"].unique()


# In[ ]:


df["Marque"].value_counts()


# In[ ]:


plt.figure(figsize=(30,8))
sns.countplot(df["Marque"])
plt.show()


# #### les marques les plus présentes sont :
# ###### Volkswagen       
# ###### Renault          
# ###### Dacia           
# ###### Mercedes-Benz    
# ###### Peugeot          
#     

# #### Type de carburant

# In[ ]:


df["Type de carburant"].unique()


# In[ ]:


df['Type de carburant'].value_counts()


# In[ ]:


plt.figure(figsize=(15,7))
sns.countplot(df["Type de carburant"])
plt.xticks(rotation=90)
plt.show()


# #### les types les plus utilisés 
# ###### Diesel        
# ###### Essence    
#    
#     

# ### la corrélation entre le prix des voitures et l'année du modèle - utilisant un graphe interactif

# In[ ]:


px.scatter(df,x="Année-Modèle",y="Prix",animation_frame="Marque",color="Type de carburant")


# on constate que le prix augmente lors le modele du voiture est recent

# ### la corrélation entre le prix  et la marque

# In[ ]:


group=df.groupby("Marque")["Prix"].mean()
group.sort_values()


# In[ ]:


# graph group by

group.plot(kind= "bar", figsize=(15,7))


# 1-Land Rover 2-Audi 3-BMW 4-Peugeot 5-Mercedes-Benz
#             
#             
#        
#           

# ### la corrélation entre le prix  et le Type de carburant

# In[ ]:


group2=df.groupby("Type de carburant")["Prix"].mean()
group2.sort_values()


# In[ ]:


group2.plot(kind= "bar", figsize=(15,7))


# les véhicules qui ont le type de carburant hybride sont les plus chères 

# # Création d'un wordcloud (le nuage de mots)

# ### la marque
# on calcule la fréquence des marques. le but de cette etape est de savoir les mots qui apparaissent souvent dans ce dataset

# In[ ]:


comment_words = ''
stopwords = set(STOPWORDS)
 

for val in df.Marque:
     
  
    val = str(val)
 
  
    tokens = val.split()
     
  
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens)+" "
 
wordcloud = WordCloud(width = 1200, height = 1200,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
 
                     
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# ### le Modèle

# In[ ]:


comment_words = ''
stopwords = set(STOPWORDS)
 
# iterate through the csv file
for val in df.Modèle:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens)+" "
 
wordcloud = WordCloud(width = 1200, height = 1200,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
 
# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# # CONSTRUCTION DU MODELE

# In[ ]:


from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


# In[ ]:


# voir le data set
df.head()


# ### Numériser des variables 

# In[ ]:


# convertir les variables catégoriques en variables numériques


LE=LabelEncoder()
LE.fit(df["Marque"])
df["Marque"]=LE.transform(df["Marque"])

LE1=LabelEncoder()
LE1.fit(df["Modèle"])
df["Modèle"]=LE1.transform(df["Modèle"])

LE2=LabelEncoder()
LE2.fit(df["Type de carburant"])
df["Type de carburant"]=LE2.transform(df["Type de carburant"])

LE3=LabelEncoder()
LE3.fit(df['Boite de vitesses'])
df['Boite de vitesses']=LE3.transform(df['Boite de vitesses'])

LE4=LabelEncoder()
LE4.fit(df['Origine'])
df['Origine']=LE4.transform(df['Origine'])

LE6=LabelEncoder()
LE6.fit(df['État'])
df['État']=LE6.transform(df['État'])

LE7=LabelEncoder()
LE7.fit(df['Première main'])
df['Première main']=LE7.transform(df['Première main'])

df.head() # show the data


# In[ ]:


#deviser le dataset en une base de donnees test et d'entrainement

X = df.drop(columns="Prix")           
y = df["Prix"]    # y = le target 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("X Train : ", X_train.shape)
print("X Test  : ", X_test.shape)
print("Y Train : ", y_train.shape)
print("Y Test  : ", y_test.shape)


# ## Model Linear Regression

# In[ ]:


#  model Linear Regression

LinearRegression_model=LinearRegression(fit_intercept=True,normalize=False,copy_X=True, n_jobs=None)

# fit model

LinearRegression_model.fit(X_train,y_train)

# Score X and Y - test and train

print("Score the X-train with Y-train is : ", LinearRegression_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", LinearRegression_model.score(X_test,y_test))

# Expected value Y using X test
y_pred_LR=LinearRegression_model.predict(X_test)

LinearRegression_model_score = r2_score(y_test,y_pred_LR)
print(" le Score de Linear Regression " , LinearRegression_model_score)


#  ### Linear Regression score : 1.18 %

# ## Decision Tree Regressor Model
# 

# In[ ]:


DecisionTreeRegressor_model=DecisionTreeRegressor()

# fit model

DecisionTreeRegressor_model.fit(X_train,y_train)

# Score X and Y - test and train

print("Score de X-train avec Y-train est : ", DecisionTreeRegressor_model.score(X_train,y_train))
print("Score de X-test  avec Y-test  est : ", DecisionTreeRegressor_model.score(X_test,y_test))

# Expected value Y using X test
y_predDTR=DecisionTreeRegressor_model.predict(X_test)

# Model Evaluation

DecisionTreeRegressor_model_score = r2_score(y_test,y_predDTR)
print("  le Score Decision Tree Regressor model  " , DecisionTreeRegressor_model_score)


# ### Decision Tree Regressor Score : 92.9%

# ## K Neighbors Regressor Model

# In[ ]:


KNeighborsRegressor_model=KNeighborsRegressor(n_neighbors=5,weights='uniform',algorithm='auto',leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=None)

# fit model

KNeighborsRegressor_model.fit(X_train,y_train)

# Score X and Y - test and train

print("Score de X-train avec Y-train est : ", DecisionTreeRegressor_model.score(X_train,y_train))
print("Score de X-test  aves Y-test  est : ", DecisionTreeRegressor_model.score(X_test,y_test))

# Expected value Y using X test
y_predKN=KNeighborsRegressor_model.predict(X_test)

# Model Evaluation
KNeighborsRegressor_model_score = r2_score(y_test,y_predKN)
print(" le Score K Neighbors Regressor Model " , KNeighborsRegressor_model_score) 


# ### K Neighbors Regressor Score : 35.9 %

# ##  Random Forest Regressor Model

# In[ ]:


RandomForestRegressor_model=RandomForestRegressor()

# fit model

RandomForestRegressor_model.fit(X_train,y_train)

# Score X and Y - test and train

print("Score de X-train avec Y-train est  : ", RandomForestRegressor_model.score(X_train,y_train))
print("Score de X-test  aves Y-test  est : ", RandomForestRegressor_model.score(X_test,y_test))

# Expected value Y using X test
y_predRFR=RandomForestRegressor_model.predict(X_test)

# Model Evaluation
RandomForestRegressor_model_score = r2_score(y_test,y_predRFR)
print(" le Score Random Forest Regressor Model " , RandomForestRegressor_model_score) 




#  ### Random Forest Regressor Score : 89.7 %

# In[ ]:


def predict(
    Marque,\
    Modele,\
    Annee_Modele,\
    Type_de_carburant,\
    Puissance_fiscale,\
    Kilometrage='30 000 - 34 999',\
    etat='Très bon',\
    Boite_de_vitesses='Manuelle',\
    Nombre_de_portes=5,\
    Origine='WW au Maroc',\
    Premiere_main='Non',\
    Jantes_aluminium=False,\
    Airbags=False,\
    Climatisation=False,\
    Systeme_de_navigation_GPS=False,\
    Toit_ouvrant=False,\
    Sieges_cuir=False,\
    Radar_de_recul=False,\
    Camera_de_recul=False,\
    Vitres_electriques=False,\
    ABS=False,\
    ESP=False,\
    Regulateur_de_vitesse=False,\
    Limiteur_de_vitesse=False,\
    CD_MP3_Bluetooth=False,\
    Ordinateur_de_bord=False,\
    Verrouillage_centralise_a_distance=False,\
):
    
    print('\n\n',Marque, Modele)
    Marque= LE.transform([Marque])[0]
    Modele= LE1.transform([Modele])[0]
    Type_de_carburant= LE2.transform([Type_de_carburant])
    Boite_de_vitesses= LE3.transform([Boite_de_vitesses])
    Origine= LE4.transform([Origine])
    Premiere_main= LE7.transform([Premiere_main])
    etat= LE6.transform([etat])
    Kilometrage= Kilometrage.split('-') 
    Kilometrage = int(Kilometrage[0].replace(' ','')) + int(Kilometrage[1].replace(' ',''))/2

    car = [Marque, \
    Modele, \
    Annee_Modele, \
    Kilometrage, \
    Type_de_carburant, \
    Puissance_fiscale, \
    Boite_de_vitesses, \
    Nombre_de_portes, \
    Origine, \
    Premiere_main, \
    etat, \
    Jantes_aluminium, \
    Airbags, \
    Climatisation, \
    Systeme_de_navigation_GPS, \
    Toit_ouvrant, \
    Sieges_cuir, \
    Radar_de_recul, \
    Camera_de_recul, \
    Vitres_electriques, \
    ABS, \
    ESP, \
    Regulateur_de_vitesse, \
    Limiteur_de_vitesse, \
    CD_MP3_Bluetooth, \
    Ordinateur_de_bord, \
    Verrouillage_centralise_a_distance]
    
    predictions = [ round(RandomForestRegressor_model.predict([car])[0],2),\
                   round(KNeighborsRegressor_model.predict([car])[0],2),\
                   round(DecisionTreeRegressor_model.predict([car])[0],2),\
                   np.round(LinearRegression_model.predict([car])[0],2)]
    
    
    print('The price of this car is between ',min(predictions),'dh and',max(predictions),'dh \n')
     
    print('\u001b[32m','RandomForestRegressor_model:', predictions[0],'Score: ',round(RandomForestRegressor_model_score,3)*100,'%','\x1b[0m' )
    print('\u001b[31m','KNeighborsRegressor_model:', predictions[1],'Score: ',round(KNeighborsRegressor_model_score,3)*100,'%','\x1b[0m' )
    print('\u001b[32m','DecisionTreeRegressor_model:',predictions[2],'Score: ',round(DecisionTreeRegressor_model_score,3)*100,'%','\x1b[0m' )
    print('\u001b[31m','LinearRegression_model:',predictions[3],'Score: ',round(LinearRegression_model_score,3)*100,'%','\x1b[0m' )
    
    
    print('\n>>> le prix du véhicule estimé', predictions[2])





# https://www.avito.ma/fr/souani/voitures/RENAULT_CLIO_4_1_5_DCI_6CV_BUSINESS_TOUTES_OPTIONS_45173707.htm
predict(
    Marque='Renault',\
    Modele='Clio',\
    Annee_Modele=2018,\
    Type_de_carburant='Diesel',\
    Puissance_fiscale=6,
    Kilometrage='130 000 - 139 999',\
    etat='Excellent',\
    Boite_de_vitesses='Manuelle',\
    Nombre_de_portes=5,\
    Origine='WW au Maroc',\
    Premiere_main='Oui',\
    Jantes_aluminium=True,\
    Airbags=True,\
    Climatisation=True,\
    Systeme_de_navigation_GPS=True,\
    Toit_ouvrant=False,\
    Sieges_cuir=False,\
    Radar_de_recul=False,\
    Camera_de_recul=False,\
    Vitres_electriques=True,\
    ABS=True,\
    ESP=False,\
    Regulateur_de_vitesse=True,\
    Limiteur_de_vitesse=True,\
    CD_MP3_Bluetooth=True,\
    Ordinateur_de_bord=True,\
    Verrouillage_centralise_a_distance=True,\

)


# https://www.avito.ma/fr/casablanca/voitures/Dacia_Sandero_Stepway__49981956.htm
predict(
    Marque='Dacia',\
    Modele='Sandero',\
    Annee_Modele=2019,\
    Type_de_carburant='Diesel',\
    Puissance_fiscale=7,
    Kilometrage='15 000 - 19 999',\
    etat='Excellent',\
    Boite_de_vitesses='Manuelle',\
    Nombre_de_portes=5,\
    Origine='WW au Maroc',\
    Premiere_main='Oui',\
    Jantes_aluminium=False,\
    Airbags=True,\
    Climatisation=True,\
    Systeme_de_navigation_GPS=True,\
    Toit_ouvrant=False,\
    Sieges_cuir=False,\
    Radar_de_recul=True,\
    Camera_de_recul=True,\
    Vitres_electriques=True,\
    ABS=True,\
    ESP=True,\
    Regulateur_de_vitesse=True,\
    Limiteur_de_vitesse=True,\
    CD_MP3_Bluetooth=True,\
    Ordinateur_de_bord=True,\
    Verrouillage_centralise_a_distance=True,\

)


