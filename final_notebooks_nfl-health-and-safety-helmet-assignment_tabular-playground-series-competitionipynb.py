#!/usr/bin/env python
# coding: utf-8

# <center><h1 style="color:#1a1a1a;
#                     font-size:3em">
#         Projet: 🤖 Machine learning 
#         </h1> 
#         <h2 style="color:#1a1a1a;
#                     font-size:2em">
#         Problème de classification des séries chronologiques 📉
#        </h2>
# </center>

# <div style="font-size:1.3em">    <span>
#     Réalisé par :¶
#     </span>
#       <ul>
#          <li>Lazrek Imane</li>
#          <li>Ali Salmi</li>
#       </ul>
#     <span>
#     Encadré par :¶
#     </span>
#       <ul>
#          <li>M. Lotfi elaachak</li>
#       </ul>
# </div>
# 

# <p style="font-size:1.5em">📜 Table des matières:</p>
# <div style="font-size:1.3em">
#     <ul>
#        <li>
#           <a href="#Intro-section">Introduction 📖</a>
#           <ul>
#              <li><a href="#overview">Aperçu</a></li>
#              <li><a href="#dataset">Jeu de données</a></li>
#           </ul>
#        </li>
#        <li>
#           <a href="#Analysis-section">Analyses et Transformations 🔎</a>
#           <ul>
#              <li><a href="#eda">Exploration de données</a></li>
#              <li><a href="#fc">Features correlation</a></li>
#              <li><a href="#vis">Visualisations</a></li>
#           </ul>
#        </li>
#        <li>
#           <a href="#pre-processing">Pré-traitement ⚙️</a>
#           <ul>
#              <li><a href="#fe">Feature Engineering</a></li>
#              <li><a href="#sfs">Sequential Feature selection</a></li>
#           </ul>
#        </li>
#        <li>
#           <a href="#model-building">Construction des modèles 🛠️</a>
#           <ul>
#              <li><a href="#tts">Test/Train Split</a></li>
#              <li><a href="#gb"> Gradient boosting</a></li>
#              <li><a href="#xgb"> XGBoost ( eXtreme Gradient Boosting )</a></li>
#           </ul>
#        </li>
#        <li>
#           <a href="#ps">Predictions and submission 🗃️</a>
#           <ul>
#              <li><a href="#fetd">Feature engineering on test data</a></li>
#              <li><a href="#pre"> Predictions</a></li>
#           </ul>
#        </li>
#        <li><a href="#Conclusion">Conclusion 📌</a></li>
#     </ul>
# </div>

# <center id="Intro-section">
#         <h1 style="color:#1a1a1a;
#                     font-size:2em">
#         Introduction 📖
#         </h1>
# </center>

# <div id="overview">
#         <h1 style="color:#1a1a1a">
#          ⮞  Aperçu
#         </h1>
# </div>

# Dans cette compétition, vous classerez des séquences de 60 secondes de données de capteur, indiquant si un sujet était dans l'un des deux états d'activité pendant la durée de la séquence.

# <div id="dataset">
#         <h1 style="color:#1a1a1a">
#          ⮞  Jeu de données
#         </h1>
# </div>

# - **train.csv** : le jeu de training, comprenant ~26 000 enregistrements de 60 secondes de treize capteurs biologiques pour près d'un millier de participants expérimentaux
#      - *séquence* - un identifiant unique pour chaque séquence
#      - *subject* - un identifiant unique pour le sujet de l'expérience
#      - *step* - pas de temps de l'enregistrement, par intervalles d'une seconde
#      - *sensor_00* - sensor_12 - la valeur de chacun des treize capteurs à ce pas de temps
# - **train_labels.csv** : l'étiquette de classe pour chaque séquence.
#      - *séquence* - l'identifiant unique pour chaque séquence.
#      - *state* - l'état associé à chaque séquence. C'est la cible que vous essayez de prédire.
# - **test.csv** : le jeu de test. Pour chacune des ~12 000 séquences, vous devez prédire une valeur pour l'état de cette séquence.
# - **sample_submission.csv** : un exemple de fichier de soumission au format correct.

# <center id="Analysis-section">
#         <h1 style="color:#1a1a1a;
#                     font-size:2em">
#         Analyses et Transformations 🔎
#         </h1>
# </center>

# <div id="eda">
#         <h1 style="color:#1a1a1a">
#          ⮞  Exploration de données
#         </h1>
# </div>

# <h4 style="color:grey"> Importation de bibliothèques </h4>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from itertools import chain
from sklearn import metrics
import scipy.stats
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score



# In[ ]:


#  ignorer les avertissements
import warnings

warnings.filterwarnings("ignore")


# <span style="color:grey; font-size:1.2em">Important nos jeu de données sous les fichier <b>train.csv</b> content les , <b>train_labels.csv</b> et <b>test.csv</b> content  les données.</span>

# In[ ]:


# Dataset train
train = pd.read_csv('../input/tabular-playground-series-apr-2022/train.csv')

# Dataset test
test = pd.read_csv('../input/tabular-playground-series-apr-2022/test.csv')

# les labels du dataset
train_labels = pd.read_csv('../input/tabular-playground-series-apr-2022/train_labels.csv')


# <h4 style="color:grey"> Explorant notre jeu de données </h4>

# In[ ]:


train


# In[ ]:


train_labels


# <span style="color:grey; font-size:1.2em">Un aperçu des données en utilisant les fonctions <b>info()</b> et <b>describe()</b> du pandas pour examiner les données. </span>

# In[ ]:


train.info()


# In[ ]:


train_labels.info()


# In[ ]:


test.info()


# In[ ]:


train.describe()


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# <span style="color:grey; font-size:1.2em">Les données n'ont pas de <b>valeurs manquantes</b>, nous n'effectuerons donc pas un <b>nettoyage des données.</b> </span>

# In[ ]:


print(f'les elements du train: from {train.subject.min()} to {train.subject.max()}')
print(f'les elements du test: from {test.subject.min()} to {test.subject.max()}')
print()


# In[ ]:


len(train['subject'].unique())


# In[ ]:


np.sort(train['subject'].unique())


# In[ ]:


len(test['subject'].unique())


# In[ ]:


len(train['sequence'].unique())


# - Il y a **25968 séquences** (étiquetées **de 0 à 25967**) dans le **train** avec 672 sujets.
# - Les données de train ont ***1558080*** lignes, ce qui est logique puisque nous avons que chaque séquence a **60 pas, un pas par seconde ** (25968*60=1558080).

# In[ ]:


len(train_labels['sequence'].unique())


# In[ ]:


train_labels['sequence'].unique()


# In[ ]:


len(test['sequence'].unique())


# In[ ]:


def plot_sequence_count_distribution(df, title):
    plt.figure(figsize=(10,6))
    temp = df.subject.value_counts().sort_values() // 60
    plt.bar(range(len(temp)), temp, width=1)
    plt.xlabel('subject')
    plt.ylabel('sequence count')
    plt.title(f'Sequence count distribution over {title} subjects')

plot_sequence_count_distribution(train, 'training')


# In[ ]:


plot_sequence_count_distribution(test, 'test')


# <span style="color:grey; font-size:1.2em">La distribution de dataset de training et de test est très similaire.</span>

# <h4 style="color:grey"> Création de la colonne de l'etat pour le label</h4>

# In[ ]:


train = train.merge(train_labels, how='left')
train.head(123)


# <h4 style="color:grey"> Suppression de séquences avec des valeurs bloquées</h4> 

# In[ ]:


train_unique_1 = train.drop(['subject', 'step', 'state', 'sensor_02'], axis=1).groupby(['sequence']).agg(lambda x: x.nunique() == 1).sum(axis=1).sort_values(ascending=False)


# In[ ]:


train_unique_1


# In[ ]:


at_least_8 = train_unique_1[train_unique_1>1]
'Sequences with at least 8 sensor stuck: ', len(at_least_8)
stuck = list(at_least_8.index)
stuck


# In[ ]:


len(stuck)


# In[ ]:


train = train.drop(train.loc[train['sequence'].isin(stuck)].index, axis = 0)
train_labels = train_labels.drop(train_labels.loc[train_labels['sequence'].isin(stuck)].index, axis = 0)


# In[ ]:


train.describe()


# <h4 style="color:grey">Cherchant la moyenne </h4> 

# Donc on va chercher la différence entre deux "state" en termes de moyennes.

# In[ ]:


means = train.groupby('state').mean()
display(means)
display(means.diff()) # difference between state 0 and 1


# In[ ]:


medians = train.groupby('state').median()
display(medians)
display(medians.diff()) 


# Il semble que nous ayons des différences entre les "state" en termes de moyennes.

# <h4 style="color:grey"> Compter les séquences par sujet </h4>

# Voyons maintenant combien de séquences il y a par sujet.

# In[ ]:


count_sub = pd.DataFrame(train.subject.value_counts().sort_values().reset_index() )
count_sub


# In[ ]:


count_sub['number of sequences'] = (count_sub['subject']/60).astype(int) #dividing by 60 seconds to obtain the right count
count_sub.drop(['subject'], axis = 1, inplace = True)


# In[ ]:


count_sub['subject'] = count_sub['index']
count_sub.drop(['index'], axis = 1, inplace = True)
count_sub


# De cette façon, en utilisant les train-labels, nous savons quel état était la séquence.
# Il semble que pour rassembler des informations pour la classification, il est utile de regrouper par séquence.
# Voir pour un graphique montrant que les sujets avec plus de séquences ont tendance à être à l'état 1.

# 
# <div id="fc">
#         <h1 style="color:#1a1a1a">
#          ⮞  Features correlation 
#         </h1>
# </div>

# In[ ]:


#features correlation

colormap = plt.cm.RdBu
plt.figure(figsize=(18,15));
plt.title('Features correlation', y=1.05, size=20);
features  = [col for col in train.columns if col not in ('sequence','step','subject', 'state')]
sns.heatmap(train[features].corr(),linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# 
# <div id="vis">
#         <h1 style="color:#1a1a1a">
#          ⮞  Visualisations
#         </h1>
# </div>

# In[ ]:


serie = pd.DataFrame(train.loc[(train['subject'] == 0) & (train['sequence'] == 207)].set_index('step')) 
serie.head()


# Tracer les capteurs de données du sujet 0, séquence 207 le long des étapes (état 0)

# In[ ]:


plt.figure(figsize=(30,8))
for i in list(serie.columns[2:15]):
    plt.plot(serie[i], label = i)
    plt.grid(True)
plt.legend()
plt.show()


# In[ ]:


row_1=serie.iloc[0]
row_1[2:15]


# On va tracer pour chaque seconde tous les capteurs (sujet 0, séquence 207, état 0)

# In[ ]:


plt.figure(figsize=(30,8))
for i in range(0,59):
    plt.plot((serie.iloc[i])[2:15], label = i)
    plt.grid(True)
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(30,8))
for i in (0,1,2,3):
    plt.plot((serie.iloc[i])[2:15], label = i)
    plt.grid(True)
plt.grid(True)
plt.legend()
plt.show()


# Médiane de tous les capteurs à chaque étape (sujet 0, séquence 207, état 0)

# In[ ]:


plt.figure(figsize=(30,8))
totmed = []
for i in range(0,59):
    median = serie.iloc[i][2:15].median()
    totmed.append(median) 
plt.plot(totmed)
plt.grid(True)
plt.ylim([-2, 2])
plt.show()


# médiane de tous les capteurs à chaque étape (sujet 0, séquence 207, état 0)

# In[ ]:


plt.figure(figsize=(30,8))
totmed = []
for i in serie.columns[2:15]:
    median = serie[i].median()
    totmed.append(median) 
plt.plot(totmed)
plt.grid(True)
plt.ylim([-2, 2])
plt.show()


# In[ ]:


serie1 = pd.DataFrame(train.loc[(train['subject'] == 327) & (train['sequence'] == 25967)].set_index('step')) 
serie1.head()


# Tracez pour chaque seconde tous les capteurs (sujet 0, séquence 207, état 0, bleu vs sujet 327, séquence 25967, état 0, rouge)

# In[ ]:


plt.figure(figsize=(30,8))
for i in range(0,59):
    plt.plot((serie.iloc[i])[2:15], label = i, color ='blue')
    plt.plot((serie1.iloc[i])[2:15], label = i, color ='red')
    plt.grid(True)
plt.grid(True)
#plt.legend()
plt.show()


# médiane de tous les capteurs à chaque étape (sujet 0, séquence 207, état 0, bleu vs sujet 327, séquence 25967, état 0, rouge)

# In[ ]:


plt.figure(figsize=(30,8))
totmed = []
totmed1 = []
for i in range(0,59):
    median = serie.iloc[i][2:15].median()
    totmed.append(median) 
    median1 = serie1.iloc[i][2:15].median()
    totmed1.append(median1)
plt.plot(totmed, label = 'median subject 0',color = 'blue')
plt.plot(totmed1,label = 'median subject 327', color = 'red')
plt.legend()
plt.grid(True)
plt.ylim([-2, 2])
plt.show()


# <div id="fe">
#         <h1 style="color:#1a1a1a">
#          ⮞  Features Engineering
#         </h1>
# </div>

# In[ ]:


def features(df):
    out_df = df.groupby('sequence').agg(['mean','max', 'min', 'std', scipy.stats.variation, scipy.stats.iqr,'median', 'skew', pd.DataFrame.kurt])
    #out_df2 = df.groupby('sequence').apply(pd.DataFrame.kurt)
    out_df.columns = ['_'.join(col).strip() for col in out_df.columns]

    return out_df


# In[ ]:


sensors = [col for col in train.columns if 'sensor_' in col]
sensors
def engineer(df):
    new_df = pd.DataFrame([], index=df.index)
    for sensor in sensors:
        new_df[sensor + '_mean'] = df[sensor].mean(axis=1)
        new_df[sensor + '_std'] = df[sensor].std(axis=1)
        new_df[sensor + '_sm'] = np.nan_to_num(new_df[sensor + '_std'] / 
                                               new_df[sensor + '_mean'].abs()).clip(-1e30, 1e30) # Compute the coefficient of variation, which is the standard deviation divided by the mean.
        new_df[sensor + '_iqr'] = scipy.stats.iqr(df[sensor], axis=1)
        new_df[sensor + '_median'] = df[sensor].median(axis=1)
        #new_df[sensor + '_skew'] = df[sensor].skew(axis=1)
        new_df[sensor + '_kurtosis'] = scipy.stats.kurtosis(df[sensor], axis=1)
        new_df['sensor_02_up'] = (df.sensor_02.diff(axis=1) > 0).sum(axis=1)
        new_df['sensor_02_down'] = (df.sensor_02.diff(axis=1) < 0).sum(axis=1)
        new_df['sensor_02_upsum'] = df.sensor_02.diff(axis=1).clip(0, None).sum(axis=1)
        new_df['sensor_02_downsum'] = df.sensor_02.diff(axis=1) .clip(None, 0).sum(axis=1)
        new_df['sensor_02_upmax'] = df.sensor_02.diff(axis=1).max(axis=1)
        new_df['sensor_02_downmax'] = df.sensor_02.diff(axis=1).min(axis=1)
        new_df['sensor_02_upmean'] = np.nan_to_num(new_df['sensor_02_upsum'] / new_df['sensor_02_up'], posinf=40)
        new_df['sensor_02_downmean'] = np.nan_to_num(new_df['sensor_02_downsum'] / new_df['sensor_02_down'], neginf=-40)
    return new_df


# In[ ]:


train_pivoted = train.pivot(index=['sequence','subject','state'], columns='step', values=[col for col in train.columns if 'sensor_' in col])

train_pivoted


# In[ ]:


train_pivoted_feat = engineer(train_pivoted)
train_pivoted_feat


# In[ ]:


count_sub


# In[ ]:


count_sub.set_index('subject', inplace=True)
count_sub


# Ajout d'une colonne de comptage

# In[ ]:


train_pivoted_feat = train_pivoted_feat.join(count_sub, how = 'inner') # create a column count by joining the 2 dataframe


# In[ ]:


train_pivoted_feat


# In[ ]:


train_pivoted_feat1 = train_pivoted_feat.droplevel(1)
train_pivoted_feat1
train_pivoted_feat2 = train_pivoted_feat1.droplevel(1)
train_pivoted_feat2


# In[ ]:


X = train_pivoted_feat2

y =train_labels['state']

X


# In[ ]:


y


# Fonctionnalités à retirer d'AMBROSM EDA

# In[ ]:


dropped_features = ['sensor_05_kurt', 'sensor_08_mean',
                    'sensor_05_std', 'sensor_06_kurt',
                    'sensor_06_std', 'sensor_03_std',
                    'sensor_02_kurt', 'sensor_03_kurt',
                    'sensor_09_kurt', 'sensor_03_mean',
                    'sensor_00_mean', 'sensor_02_iqr',
                    'sensor_05_mean', 'sensor_06_mean',
                    'sensor_07_std', 'sensor_10_iqr',
                    'sensor_11_iqr', 'sensor_12_iqr',
                    'sensor_09_mean',
                     'sensor_05_iqr', 
                     'sensor_09_iqr', 
                    'sensor_07_iqr', 'sensor_10_mean']


# In[ ]:


selected_columns = X.columns
selected_columns = [f for f in selected_columns if f not in dropped_features]
len(selected_columns)


# In[ ]:


X = X[selected_columns]
X


# In[ ]:


index = [i for i in X.index if i not in stuck]


# In[ ]:


len(index)


# In[ ]:


X = X.loc[X.index.isin(index)]
X


# <div id="sfs">
#         <h1 style="color:#1a1a1a">
#          ⮞  Sequential Feature selection
#         </h1>
# </div>

# Nous commençons par sélectionner les 50 "meilleures" caractéristiques de l'ensemble de données Iris via la sélection séquentielle vers l'avant (SFS). Ici, nous définissons forward=True et floating=False. En choisissant cv=0, nous n'effectuons aucune validation croisée, par conséquent, la performance (ici : 'précision') est entièrement calculée sur l'ensemble d'apprentissage.

# In[ ]:


from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier


# In[ ]:


estimator = HistGradientBoostingClassifier()


# In[ ]:


X = X[['sensor_00_std',
   'sensor_00_sm',
   'sensor_00_median',
   'sensor_00_kurtosis',
   'sensor_02_upsum',
   'sensor_02_downsum',
   'sensor_02_upmax',
   'sensor_02_downmax',
   'sensor_02_upmean',
   'sensor_01_std',
   'sensor_01_iqr',
   'sensor_02_mean',
   'sensor_02_std',
   'sensor_02_sm',
   'sensor_02_kurtosis',
   'sensor_03_sm',
   'sensor_03_iqr',
   'sensor_03_median',
   'sensor_03_kurtosis',
   'sensor_04_mean',
   'sensor_04_std',
   'sensor_04_sm',
   'sensor_04_iqr',
   'sensor_04_median',
   'sensor_04_kurtosis',
   'sensor_05_sm',
   'sensor_05_median',
   'sensor_06_sm',
   'sensor_06_iqr',
   'sensor_07_mean',
   'sensor_07_median',
   'sensor_08_iqr',
   'sensor_08_kurtosis',
   'sensor_09_std',
   'sensor_09_median',
   'sensor_09_kurtosis',
   'sensor_10_std',
   'sensor_10_sm',
   'sensor_10_kurtosis',
   'sensor_11_sm',
   'sensor_11_kurtosis',
   'sensor_12_std',
   'sensor_12_sm',
   'sensor_12_kurtosis',
   'number of sequences']]


# In[ ]:


X


# <center id="model-building">
#         <h1 style="color:#1a1a1a;
#                     font-size:2em">
#             Construction des modèles 🛠️
#         </h1>
# </center>

# <div id="tts">
#         <h1 style="color:#1a1a1a">
#          ⮞  Train and split
#         </h1>
# </div>

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)#, random_state=42, stratify = y) 
#stratify parameter will preserve the proportion of target as in original dataset, in the train and test datasets as well.


# <div id="gb">
#         <h1 style="color:#1a1a1a">
#          ⮞   Gradient boosting
#         </h1>
# </div>

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


from sklearn.model_selection import KFold


# In[ ]:


gradient_booster = GradientBoostingClassifier(learning_rate=0.1, n_estimators = 100)
gradient_booster.get_params()


# In[ ]:


gradient_booster.fit(X_train,y_train)


# In[ ]:


y_pred = gradient_booster.predict(X_test)
y_pred


# In[ ]:


y_pred_proba = gradient_booster.predict_proba(X_test)
y_pred_proba


# In[ ]:


print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


score = GradientBoostingClassifier.score(gradient_booster, X_test,y_test)
print('Test Accuracy Score',score)


# In[ ]:


from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(GradientBoostingClassifier(learning_rate=0.1), X_train, y_train,cv=3)
accuracy


# In[ ]:


print("La précision du modèle avec cross validation est:",accuracy.mean() * 100)


# In[ ]:


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)


# In[ ]:


fpr


# In[ ]:


def plot_roc_curve(y_va, y_va_pred):
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = metrics.roc_curve(y_va, y_va_pred)
    plt.plot(fpr, tpr, color='r', lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.gca().set_aspect('equal')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.show()



# In[ ]:


plot_roc_curve(y_test, y_pred_proba[:,1])
print(metrics.auc(fpr, tpr))


# <div id="xgb">
#         <h1 style="color:#1a1a1a">
#          ⮞    XGBoost ( eXtreme Gradient Boosting )
#         </h1>
# </div>

# In[ ]:


params = {'n_estimators': 1200,
          'max_depth': 7,
          'learning_rate': 0.15,
          'subsample': 0.95,
          'colsample_bytree': 0.60,
          'reg_lambda': 1.50,
          'reg_alpha': 6.10,
          'gamma': 1.40,
          'random_state': 42,
          'eval_metric' : 'logloss',
          #'tree_method': 'gpu_hist',
         }


# In[ ]:


from xgboost  import XGBClassifier
#xgb = XGBClassifier(random_state = 2)
xgb = XGBClassifier(n_estimators=500, n_jobs=-1,
                          eval_metric=['logloss'],
                          #max_depth=10,
                          colsample_bytree=0.8,
                          #gamma=1.4,
                          reg_alpha=6, reg_lambda=1.5,
                          tree_method='hist',
                          learning_rate=0.03,
                          verbosity=1,
                          use_label_encoder=False, random_state=3)


# In[ ]:


xgb.fit(X_train, y_train)


# faire des prédictions pour les données de test

# In[ ]:


y_pred_XGB = xgb.predict(X_test)
y_pred_XGB


# In[ ]:


y_pred_XGB_proba = xgb.predict_proba(X_test)
y_pred_XGB_proba


# In[ ]:


print(classification_report(y_test,y_pred_XGB))
accuracy = accuracy_score(y_test, y_pred_XGB)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(xgb, X_train, y_train,cv=3)
accuracy


# In[ ]:


print("La précision du modèle avec cross validation est:",accuracy.mean() * 100)


# In[ ]:


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_XGB)


# In[ ]:


fpr


# In[ ]:


def plot_roc_curve(y_va, y_va_pred):
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = metrics.roc_curve(y_va, y_va_pred)
    plt.plot(fpr, tpr, color='r', lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.gca().set_aspect('equal')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Caractéristique de fonctionnement du récepteur")
    plt.show()


# In[ ]:


plot_roc_curve(y_test, y_pred_XGB_proba[:,1])
print(metrics.auc(fpr, tpr))


# <center id="ps">
#         <h1 style="color:#1a1a1a;
#                     font-size:2em">
#             Predictions and submission 🗃️
#         </h1>
# </center>

# <div id="fetd">
#         <h1 style="color:#1a1a1a">
#          ⮞  Feature engineering on test data
#         </h1>
# </div>

# In[ ]:


test


# In[ ]:


# counting how many sequences per subject
count_sub = pd.DataFrame(test.subject.value_counts().sort_values().reset_index() )
count_sub


# In[ ]:


count_sub['number of sequences'] = (count_sub['subject']/60).astype(int) #diviser par 60 secondes pour obtenir le bon décompte
count_sub.drop(['subject'], axis = 1, inplace = True)


# In[ ]:


count_sub['subject'] = count_sub['index']
count_sub.drop(['index'], axis = 1, inplace = True)
count_sub


# In[ ]:


count_sub.set_index('subject', inplace=True)
count_sub


# In[ ]:


test_pivoted = test.pivot(index=[ 'sequence','subject'], columns='step', values=[col for col in test.columns if 'sensor_' in col])

test_pivoted


# In[ ]:


test_pivoted_feat = engineer(test_pivoted)
test_pivoted_feat


# In[ ]:


test_pivoted_feat = test_pivoted_feat.join(count_sub, how = 'inner') # créer un nombre de colonnes en joignant les 2 dataframes


# In[ ]:


test_pivoted_feat


# In[ ]:


test_pivoted_feat1 = test_pivoted_feat.droplevel(1)
test_pivoted_feat1


# In[ ]:


selected_columns = test_pivoted_feat1.columns
selected_columns = [f for f in selected_columns if f not in dropped_features]
len(selected_columns)


# 
# <div id="pre">
#         <h1 style="color:#1a1a1a">
#          ⮞  Predictions
#         </h1>
# </div>

# In[ ]:


test_pivoted_feat1 = test_pivoted_feat1[['sensor_00_std',
   'sensor_00_sm',
   'sensor_00_median',
   'sensor_00_kurtosis',
   'sensor_02_upsum',
   'sensor_02_downsum',
   'sensor_02_upmax',
   'sensor_02_downmax',
   'sensor_02_upmean',
   'sensor_01_std',
   'sensor_01_iqr',
   'sensor_02_mean',
   'sensor_02_std',
   'sensor_02_sm',
   'sensor_02_kurtosis',
   'sensor_03_sm',
   'sensor_03_iqr',
   'sensor_03_median',
   'sensor_03_kurtosis',
   'sensor_04_mean',
   'sensor_04_std',
   'sensor_04_sm',
   'sensor_04_iqr',
   'sensor_04_median',
   'sensor_04_kurtosis',
   'sensor_05_sm',
   'sensor_05_median',
   'sensor_06_sm',
   'sensor_06_iqr',
   'sensor_07_mean',
   'sensor_07_median',
   'sensor_08_iqr',
   'sensor_08_kurtosis',
   'sensor_09_std',
   'sensor_09_median',
   'sensor_09_kurtosis',
   'sensor_10_std',
   'sensor_10_sm',
   'sensor_10_kurtosis',
   'sensor_11_sm',
   'sensor_11_kurtosis',
   'sensor_12_std',
   'sensor_12_sm',
   'sensor_12_kurtosis',
   'number of sequences']]


# In[ ]:


test_pivoted_feat1


# In[ ]:


#retraining
xgb.fit(X, y)


# In[ ]:


# make predictions for test data
#sub_pred = xgb.predict(test)
sub_pred = xgb.predict(test_pivoted_feat1)
#sub_pred = xgb.predict(test_pivoted_feat1[selected_columns])
len(sub_pred)


# In[ ]:


#sub_pred_proba = xgb.predict_proba(test)
#sub_pred_proba = xgb.predict_proba(test_pivoted_feat1[selected_columns])
sub_pred_proba = xgb.predict_proba(test_pivoted_feat1)
len(sub_pred_proba)


# In[ ]:


submission = pd.read_csv('../input/tabular-playground-series-apr-2022/sample_submission.csv')
submission


# For each sequence in the test set, you must predict a **probability for the state** variable. 

# In[ ]:


#submission['state'] = sub_pred
submission['state'] = sub_pred_proba[:,1]
#submission['state'] = y_pred_XGB_proba[:,1]
submission.to_csv('submission.csv', index = False)
submission


# <center id="Conclusion">
#         <h1 style="color:#1a1a1a;
#                     font-size:2em">
#         Conclusion 📌
#         </h1>
# </center>

# <div style="color:grey; font-size:1.2em">Le travail que nous avons réalisé a consisté à explorer comment la plupart des modèles utilisant des données supplémentaires surperforment les statistiques traditionnelles sur des séries temporelles univariées. On a bien appris qu'une analyse approfondie de nos séries est necessaire pour classifier qui peut nous obliger à appliquer des transformations à la série et peut déterminer notre choix de modèle. 
# <br>
# <br>
# <span style="color:black; font-size:1.3em; background-color:#FFFFA6">Ce projet nous a permis d'acquérir les techniques d'analyse, de transformation et les méthodes de classification sur les séries chronologiques.</span>
# </div>
