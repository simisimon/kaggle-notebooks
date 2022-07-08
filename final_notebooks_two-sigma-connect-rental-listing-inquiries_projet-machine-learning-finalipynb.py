#!/usr/bin/env python
# coding: utf-8

# **Etape 1 : Quel est le type de problème ?**
# 
# - Il s'agit de prédire le nombre de demandes que reçoit une annonce sur le site Renthop. (apprentissage supervisé)
# - Variable d'intérêt : "interest_level" qui a trois catégories 'high' 'medium' 'low'
# - C'est un problème de classification

# In[ ]:


#Le sous-dossier contenant nos données

import os
print(os.listdir("../input"))


# In[ ]:


#Librairies classiques de data science

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing 

import datashader as ds
from datashader import transfer_functions as tf


# In[ ]:


#Les données d'apprentissage (train)

train = pd.read_json("../input/two-sigma-connect-rental-listing-inquiries/train.json")
train = train.reset_index()
train.pop("index")
train.shape


# In[ ]:


train.head()


# In[ ]:


#Les données de validation (test)

test = pd.read_json("../input/two-sigma-connect-rental-listing-inquiries/test.json")
test = test.reset_index()
test.pop("index");

test.shape


# In[ ]:


#Le fichier de soumission (submission) à Kaggle sur lequel est calculé notre score.

submission = pd.read_csv("../input/two-sigma-connect-rental-listing-inquiries/sample_submission.csv")

submission.head()


# In[ ]:


submission.shape


# In[ ]:


#Réindexation du test selon l'ordre de submission

test = test.set_index('listing_id')
test = test.reindex(index=submission['listing_id'])
test = test.reset_index()

test.head()


# In[ ]:


#Regroupement des train et test avec indicatrice, pour avoir de nouvelles variables créées sur les deux
#sets en même temps

train["train_test"] = 'train'
test["train_test"] = 'test'
train = pd.concat([train, test],axis=0,sort=True)
train.shape


# In[ ]:


# Création de features

train["price_type"] = np.log(train["price"])/(train["bedrooms"]+1)
train["room_sum"] = train["bedrooms"]+train["bathrooms"] 
train["num_features"] = train["features"].apply(len)
train["num_description_words"] = train["description"].apply(lambda x: len(x.split(" ")))


# **Etape 2 : Quelles sont les données ? **
# 
# Regardons tout d'abord la distribution de la variable d'intérêt : 'interest_level' qui prend les valeurs 'high' 'medium' ou 'low'. 

# In[ ]:


color = sns.color_palette()
freq = train['interest_level'].value_counts()
sns.barplot(freq.index, freq.values, color=color[4])
plt.ylabel('Frequence')
plt.xlabel('Interest level')
plt.show()


# La majorité des appartements de la base train ont reçu un intérêt bas 'low'.

# In[ ]:


plt.hist(train["bathrooms"],normed=True,bins=range(7),alpha=0.5, color = color[3])
plt.xlabel('Number of bathrooms', fontsize=13)
plt.ylabel('freq', fontsize=13)
plt.show()


# 80% des annonces sont des appartements avec une salle de bain unique.

# In[ ]:


# Nombre moyen de salles de bain par niveau d'intérêt 
plt.figure(figsize=(10,6))
sns.barplot(x='interest_level', y='bathrooms', data=train, order=['low', 'medium', 'high'])
plt.xlabel('Interest Level')
plt.ylabel('Nb moyen de salles de bain');


# En moyenne, les appartements de niveau d'intérêt low ont relativement plus de salles de bains que les annonces qui ont été classées medium ou high. On peut supposer qu'en effet, plus standards, et surement moins chers, les appartements classiques avec une salle de bain ont plus de succès ? 

# **Bedrooms : **

# In[ ]:


plt.hist(train["bedrooms"],normed=True,bins=range(8), alpha=0.5, color = color[1])
plt.xlabel('Number of bedrooms', fontsize=13)
plt.ylabel('freq', fontsize=13)
plt.show()


# Environ 20% des appartements n'ont pas de chambre. On peut deviner que ce sont des studios? 

# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x='interest_level', y='bedrooms', data=train, order=['low', 'medium', 'high'])
plt.xlabel('Interest Level')
plt.ylabel('Nb moyen de chambres')


# Distribution bivariée bathrooms/bedrooms

# In[ ]:


train2 = train.query("bathrooms < 5")

fig, ax = plt.subplots()
h = ax.hist2d(train2["bathrooms"], train2["bedrooms"],normed=True)
plt.colorbar(h[3], ax=ax)

plt.xlabel('bathrooms')
plt.ylabel('bedrooms')

plt.show()


# **Exploration de la variable "features" - liste de caractéristiques de l'appartement**
# 
# On créée une liste des caractéristiques totales listées dans toutes les annonces

# In[ ]:


import itertools as it
#chain.from_iterable(['ABC', 'DEF']) --> A B C D E F

ft = []
train['features'].apply(lambda x: ft.append(x))
ft=list(it.chain.from_iterable(ft))
print("Nombre d'éléments: " + str(len(ft)))

uniq_ft = set(ft)
print("Nombre d'éléments uniques: " + str(len(uniq_ft)))


# **Feature recupéré dans les forums de Kaggle**
# 
# Correspond à la date de creation de chaque dossier contenant les images de l'annonce correspondante
# 
# 

# In[ ]:


image_timestamp = pd.read_csv('../input/hugoboum-renthop-image-timestamp/listing_image_time.csv')
image_timestamp.columns = ['listing_id', 'timestamp']
image_timestamp.head()


# In[ ]:


train = pd.merge(train,image_timestamp,on='listing_id',how='left')


# **# Vérification des NA et valeurs abberantes**

# In[ ]:


train = train.dropna(thresh=2) # car test a la colonne interest_level vide en toute logique
train.shape


# In[ ]:


print(train.query("latitude == 0")["latitude"].agg("count"))
print(train.query("longitude == 0")["longitude"].agg("count"))


# In[ ]:


#Inputation de l'emplacement median pour les points trop éloignés

latitude_median = np.median(train["latitude"])
longitude_median = np.median(train["longitude"])

print(latitude_median)
print(longitude_median)


# In[ ]:


train['latitude'].values[train['latitude'] < 39] = latitude_median
train['latitude'].values[train['latitude'] > 41] = latitude_median

train['longitude'].values[train['longitude'] <-80] = longitude_median
train['longitude'].values[train['longitude'] > -70] = longitude_median

print('Valeurs abberantes ')
print(train.query("latitude == 0")["latitude"].agg("count"))
print(train.query("longitude == 0")["longitude"].agg("count"))


# **Etudions les prix : **

# In[ ]:


#Quantiles de prix
train["price"].describe()


# In[ ]:


#Distribution log-prix
sns.distplot(np.log(train["price"]),hist=False)
plt.show()


# In[ ]:


#Test de (log) normalité, pour les prix
from scipy.stats import jarque_bera

jb = jarque_bera(np.log(train["price"]))
print("p-value Jarque-Bera normality test: " + str(jb[1]))


# In[ ]:


#Zoom sur le gros des prix

sns.distplot(train.query("price < 10000")["price"] ,hist=False)
plt.show()


# In[ ]:


#Catégorisation des prix

train["price_cat"] = pd.cut(train["price"],np.array([0,1500,5000,500000,5000000]))


# In[ ]:


train.groupby("price_cat").agg("count").iloc[:,1]


# In[ ]:


train["price_cat"] = train["price_cat"].astype("str")


# In[ ]:


#Reformatage des catégories de prix selon le format accepté dans les modèles

le = preprocessing.LabelEncoder()
train["price_cat"] = le.fit_transform(train["price_cat"])


# **rues les plus courantes**

# In[ ]:


streets = train.groupby(train["display_address"]).agg(["count"]).iloc[:,1]


# In[ ]:


streets.shape


# In[ ]:


k=100

topstreets = streets.sort_values(ascending=False).iloc[:k]
topstreets = pd.DataFrame(topstreets)
topstreets["street"] = topstreets.index
topstreets = topstreets.unstack().iloc[:k]
topstreets = topstreets.reset_index().drop(["level_0","level_1"],axis=1)
topstreets.columns = ["display_address","display_address_count"]
topstreets.head()


# In[ ]:


plt.figure(figsize=(40,200))
sns.barplot(y="display_address", x="display_address_count", data=topstreets)
plt.show()


# In[ ]:


#On remet cette information comme variable dans le train 

train  = pd.merge(train,topstreets,on='display_address',how='left')


# **Même procédure pour les managers**

# In[ ]:


managers = train.groupby(train["manager_id"]).agg(["count"]).iloc[:,1]


managers = pd.DataFrame(managers)
managers = managers.unstack()
managers["manager_id"] = managers.index

managers = managers.reset_index().drop(["level_0","level_1"],axis=1)
managers.columns = ["manager_id","manager_id_count"]
managers = managers.iloc[:-1,:]
managers.head()


# In[ ]:


train  = pd.merge(train,managers,on='manager_id',how='left')


# **On s'intéresse maintenant aux dates**

# In[ ]:


train["created"].iloc[2]


# In[ ]:


train["created"] = train["created"].astype("datetime64")


# In[ ]:


#Création de variables temporelles

train["year"] = train["created"].dt.year 
train["month"] = train["created"].dt.month
train["day"] = train["created"].dt.dayofweek
train["hour"] = train["created"].dt.hour 


# In[ ]:


#Distributions des variables temporelles
# Les heures de la journée

cnt_hour = train['hour'].value_counts()
sns.barplot(cnt_hour.index, cnt_hour.values, alpha=0.7, color=color[2])
plt.xlabel('Nombre annonces')
plt.ylabel('Heure de la journée')
plt.show()


# Nous pouvons observer que la majorité des annonces sont crées entre 1h et 6h du matin. On peut penser qu'il s'agit des heures auxquelles les mises à jour du site sont effectuées. 
# Cette variable n'est pas pertinente pour l'analyse.

# In[ ]:


# Les jours de la semaine : Lundi = 0

cnt_day = train['day'].value_counts()
sns.barplot(cnt_day.index, cnt_day.values, alpha=0.7, color=color[3])
plt.ylabel('Nombre annonces')
plt.xlabel('jour de la semaine')
plt.show()


# On remarque que plus d'annonces créées le mardi, mercredi et jeudi plutôt que le weekend.

# In[ ]:


cnt_month = train['month'].value_counts()
sns.barplot(cnt_month.index, cnt_month.values, alpha=0.7, color=color[3])
plt.ylabel('Nombre annonces')
plt.xlabel('mois de l annee')
plt.show()


# On observe les annonces par mois : on s'attend à ce qu'il y ait plus d'annonces sur les mois avant l'été, car en général, on quitte un appartement pour déménager pendant les vacances. 

# **Graphiques bivariés**
# 
# Coloris par valeur de la variable d'interêt

# In[ ]:


#Filtrage des données seulement numériques pour analyse de corrélations

train_numeric = train._get_numeric_data()
train_numeric = pd.concat([train_numeric,train["interest_level"],train["train_test"]],axis=1)
train_numeric.head()


# In[ ]:


#Fonction pratique pour sortir un grand nombre de graphs.

sns.pairplot(train_numeric, hue="interest_level")
plt.show()


# Ces graphs permettent de visualiser:
# * en diagonale: toutes les distributions colorées par interest_level
# * hors diagonale: les nuages de points croisés entre chaque variable deux à deux
# 
# On peut ainsi voir si des variables séparent bien les niveaux d'interest_level

# Coloris par valeur de la variable indicatrice de train ou test

# In[ ]:


train_numeric["train_test"] = train["train_test"]
sns.pairplot(train_numeric, hue="train_test")
plt.show()


# En refaisant la même chose avec la variable indicatrice train_test en coloris, on peut visualiser la différence de distribution des données du train et du test sets.

# In[ ]:


train_numeric = train_numeric.drop(['year','train_test'],axis=1);


# In[ ]:


le = preprocessing.LabelEncoder()
train_numeric["interest_level"] = le.fit_transform(train_numeric["interest_level"].fillna('0'))
np.unique(train_numeric["interest_level"])


# In[ ]:


train_numeric = train_numeric.query('interest_level>0')


# Corrélations

# In[ ]:


train_corrs = train_numeric.corr()


# In[ ]:


sns.heatmap(train_corrs,cmap="viridis")


# In[ ]:


#Corrélations avec la variable d'interêt

train_interestlevel_corrs = train_corrs['interest_level']
train_interestlevel_corrs = pd.DataFrame(train_interestlevel_corrs)
train_interestlevel_corrs['variable'] = train_interestlevel_corrs.index
train_interestlevel_corrs = train_interestlevel_corrs [train_interestlevel_corrs['variable'] != 'interest_level']

sns.barplot(y='variable', x='interest_level', data=train_interestlevel_corrs)
plt.show()


# **Création d'une carte**

# In[ ]:


plot_width  = int(750)
plot_height = int(plot_width//1.2)
x_range, y_range = ((-8242000,-8210000), (4965000,4990000))
train["longitude_mercator"],train["latitude_mercator"] = ds.utils.lnglat_to_meters(train["longitude"],train["latitude"])

cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range)
agg = cvs.points(train, 'longitude_mercator', 'latitude_mercator',ds.count("bedrooms"))
img = tf.shade(agg, cmap=ds.colors.inferno, how='eq_hist')
img


# **Création d'une variable par clustering**

# In[ ]:


#Seul l'algorithme KMeans est adapté à notre taille de dataset

from sklearn.cluster import KMeans

NYC_Cluster =  KMeans(n_clusters=15, random_state=42)

geodata = pd.concat([train["latitude"],train["longitude"]], axis=1)
train["nyc_cluster"] = NYC_Cluster.fit_predict(geodata)
                     
#Ajouter d'autres variables dans les données ne fait que brouiller les clusters

print(train.groupby("nyc_cluster").agg("count").iloc[:,1])

plot = sns.lmplot(data=train,x="longitude",y="latitude",hue="nyc_cluster",legend="full",palette="Set3",fit_reg=False)
plt.show()

#Zoom sur Manhattan

train2 = train.query("longitude > -74.03")
train2 = train2.query("longitude < -73.9")
train2 = train2.query("latitude > 40.65")
train2 = train2.query("latitude < 40.87")

train2 = train2.sample(n=1000)
print("Zoom sur Manhattan")
plot = sns.lmplot(data=train2,x="longitude",y="latitude",hue="nyc_cluster",legend="full",palette="Set3",fit_reg=False)
plt.show()

tmp = pd.concat([geodata,train ["nyc_cluster"]],axis=1)
print("Corrélations")
print(tmp.corr())


# **Latent Semantic Analysis (TF-IDF puis SVD tronquée)**

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()


# In[ ]:


tfidfeats = vectorizer.fit_transform(train["description"])


# In[ ]:


tfidfeats.shape


# In[ ]:


from sklearn.decomposition import TruncatedSVD


# In[ ]:


svd = TruncatedSVD(n_components=100)


# In[ ]:


lsa_feats = svd.fit_transform(tfidfeats)


# In[ ]:


plt.plot(svd.explained_variance_ratio_.cumsum())
plt.show()
print("Variance expliquée: " + str(svd.explained_variance_ratio_.sum()))


# In[ ]:


lsa_feats = pd.DataFrame(lsa_feats)
lsa_feats_colnames = ["lsa_" + str(i+1) for i in range(100)]
lsa_feats.columns = lsa_feats_colnames
train = pd.concat([train,lsa_feats],axis=1)


# In[ ]:


train.head()


# In[ ]:


train["features"] = train["features"].astype("str")


# In[ ]:


train.head()


# In[ ]:


tfidfeats2 = vectorizer.fit_transform(train["features"])


# In[ ]:


tfidfeats2.shape


# In[ ]:


svd2 = TruncatedSVD(n_components=20)


# In[ ]:


lsa_feats2 = svd2.fit_transform(tfidfeats2)


# In[ ]:


plt.plot(svd2.explained_variance_ratio_.cumsum())
plt.show()
print("Variance expliquée: " + str(svd2.explained_variance_ratio_.sum()))


# In[ ]:


lsa_feats2 = pd.DataFrame(lsa_feats2)
lsa_feats2_colnames = ["lsa2_" + str(i+1) for i in range(20)]
lsa_feats2.columns = lsa_feats2_colnames
train = pd.concat([train,lsa_feats2],axis=1)


# In[ ]:


train.head()


# **Analyse de sentiments**

# In[ ]:


from textblob import TextBlob


# In[ ]:


description_polarity, description_subjectivity = [],[]


for desc in train["description"]:
    desc_blob= TextBlob(desc)
    description_polarity.append(desc_blob.sentiment.polarity)
    description_subjectivity.append(desc_blob.sentiment.subjectivity)

train["description_polarity"] = description_polarity
train["description_subjectivity"] = description_subjectivity


# In[ ]:


train.head()


# **Photos**
# 
# Nous n'avons pas pu télécharger les photos (torrent mort) mais nous avons tout de même les liens

# In[ ]:


train["photos"][1]


# In[ ]:


train["photos_count"] =  train.photos.apply(len)


# In[ ]:


train["photos_count"][1]


# On crée une variable on l'on garde seulement la première photo de chaque annonce
# Cela reduit la taille à télécharger à environ 200ko* 50 000 = 10Go 
# 
# photos_2 = []
# 
# for item in train["photos"]:
#     try:
#         photos_2.append(item[0])
#     except IndexError:
#         photos_2.append(np.nan)
# train["photos_2"] = photos_2
# 
# 
# import requests
# from PIL import Image
# import io
# 
# images = []
# 
# for image_url in train["photos_2"]:
#     try:
#         img_data = requests.get(url= image_url).content
#         img = Image.open(io.BytesIO(img_data))
#         images.append(np.array(img))
#         
#     except requests.exceptions.MissingSchema:
#         images.append(np.nan)
#         
# images = pd.DataFrame(images)
# print(images.shape)
# 
# Exporter les images en csv pour les remettre en dataset Kaggle que l'on liera ensuite ("+Add Dataset")
# images.to_csv()
# 
# 
# Impossible d'effectuer cette étape;
# "failed. exited with code 137". Problème de mémoire.
# 

# **Modélisation et prédiction**
# 
# On utilise un Gradient Boosting

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import GridSearchCV


# In[ ]:


#Variables dont a extrait l'info et/ou inutilisables

train = train.drop(['building_id','created','description','display_address',
           'features','manager_id','photos','street_address'],axis=1)

train['display_address_count'] = train['display_address_count'].astype("float")
train['manager_id_count'] = train['manager_id_count'].astype("float")


# FeatureTools
# import featuretools as ft
# 
# es = ft.EntitySet(id = 'feature_engineering')
# 
# table = pd.DataFrame(train.drop(['interest_level','longitude_mercator','latitude_mercator'],axis=1).iloc[:,:15])
# 
# es = es.entity_from_dataframe(entity_id = 'main_table', 
#                               dataframe = table, 
#                               index = 'listing_id')
# 
# es

#  ft.primitives.list_primitives()

# features, feature_names = ft.dfs(entityset = es, target_entity = 'main_table', max_depth=2,
#                                  agg_primitives = ['mean', 'min', 'max'],
#                                  trans_primitives = ['subtract_numeric', 'divide_numeric'])
# 

# features.head()

# feature_names

# train = pd.merge(train,features,on='listing_id',how='left')

# In[ ]:


#Séparation train et test

test = train[train.train_test == "test"]
train = train[train.train_test == "train"]

train.pop("train_test")
train_target = train.pop("interest_level")

test = test.drop(['interest_level','train_test'],axis=1)

train.head()


# train_scaled = preprocessing.scale(train)
# 
# train_scaled = pd.DataFrame(train_scaled)
# 
# train_scaled.columns = train.columns
# 
# train = train_scaled

# test_scaled = preprocessing.scale(test)
# 
# test_scaled = pd.DataFrame(test_scaled)
# 
# test_scaled.columns = test.columns
# 
# test = test_scaled

# In[ ]:


gradient_booster = lgb.LGBMClassifier()


# On utilise une cross valisation et une grid search
# La loss function optimisée (logarithmic loss) est celle qui score les participants au leaderboard

# In[ ]:


grid_parameters = {'n_estimators': [100,500,1000],'num_leaves': [10,30,50],
                   'reg_alpha' : [0.2,0.4,0.6],'learning_rate' : [0.05,0.1]}


# In[ ]:


gs = GridSearchCV(estimator=gradient_booster, param_grid = grid_parameters, 
                  cv= 4, scoring= 'neg_log_loss')


# Estimation
# 

# In[ ]:


gs.fit(train,train_target)


# In[ ]:


gs.cv_results_['mean_test_score']


# In[ ]:


best_log_loss = -1 * np.max(gs.cv_results_['mean_test_score'])

print("La log-loss de validation croisée du meilleur parametrage est de " + str(best_log_loss))


# In[ ]:


#Hyperparametrage retenu
gs.best_estimator_


# In[ ]:


#Utilisation faite des variables
plt.figure(figsize=(40,200))

df = pd.concat([pd.DataFrame(train.columns),
                pd.DataFrame(gs.best_estimator_.feature_importances_)],
               axis=1)
df.columns = ['variable', 'importance']
sns.barplot(y="variable", x="importance", data= df)
plt.show()


# In[ ]:


gs.best_estimator_.classes_


# **Prédiction**

# In[ ]:


#Prédiction du test (proba associée à chaque classe pour chaque observation)

predictions = gs.predict_proba(test)
predictions = pd.DataFrame(predictions)

predictions.columns = ['high','low','medium']


# In[ ]:


#Remise dans le fichier à soumettre pour être classé

submission['high'] = predictions['high']
submission['medium'] = predictions['medium']
submission['low'] = predictions['low']

submission.head()


# **Soumission**

# In[ ]:


submission.to_csv('projet ML.csv', index=False);


# Pour aller plus loin dans la modélisation:
# * Une Random Forest 
# * Un ensembling (stacking) des deux modèles ?
