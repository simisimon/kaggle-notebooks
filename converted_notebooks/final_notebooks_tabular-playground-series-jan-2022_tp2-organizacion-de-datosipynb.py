#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Seleccionamos los que corresponden y creamos una copia:
d=pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv")
copia_dataset_ecommerce = d.copy()

sns.set(rc = {'figure.figsize':(10,10)})


# # Parte 1
# 
# ## Exploración inicial:
# 
# ### Nombres y tipos de columnas:
# 
# Para tener una idea de que información encontraremos dentro del dataset, utilizamos el metodo dtypes, mediante el cual podemos obtener el nombre y tipo de cada una de las columnas que lo conforman.

# In[ ]:


copia_dataset_ecommerce.dtypes


# ## Resumen del dataset:
# 
# Una vez reconocidos los campos que forman nuestro dataset, podemos empezar a explorar los datos que se encuentran en cada fila utilizando los métodos head y tail para obtener las primeras y últimas filas, respectivamente:
# 
# * Conjunto de las primeras 5 filas del dataset:

# In[ ]:


copia_dataset_ecommerce.head()


# * Conjunto de las últimas 5 filas del dataset:

# In[ ]:


copia_dataset_ecommerce.tail()


# *Nota: ignoramos la variable "Unnamed: 0" ya que esta esta repitiendo los datos del inidice. Esta variable sera eliminada cuando realicemos las transformaciones en el set.*

# ## Descripción de las variables del dataset:

# |Columna|Descripción|Tipo de Variable|
# |----------|--------------|---------------|
# |Clothing ID|Id de la prenda de ropa|Cualitativa Nominal|
# |Age|Edad del reviewer|Cuantitativa Continua|
# |Title|Titulo de la reseña|Cualitativa Nominal|
# |Review Text|Cuerpo de la reseña|Cualitativa Nominal|
# |Rating|Valoración de la reseña|Cuantitativa Ordinal|
# |Recommended IND|Si el producto es recomendado o no por el reviewer|Cualitativa Binaria|
# |Positive Feedback Count|Número de feedback positivo en la review|Cuantitativa Continua|
# |Division Name|Nombre del grupo en el que está el producto|Cualitativa Nominal|
# |Department Name|Nombre del categoria en la que está el producto|Cualitativa Nominal|
# |Class Name|Nombre del tipo de prenda|Cualitativa Nominal|

# ## Variables Cualitativas:

# Las variables cualitativas que identificamos en el dataset son: **Clothing ID**, **Title**, **Review Text**, **Recommend IND**, **Division Name**, **Departament Name** y **Class Name**, para las cuales creamos un nuevo DataFrame para poder analizarlas...

# In[ ]:


variables_cualitativas = pd.DataFrame(data = copia_dataset_ecommerce, columns = ["Clothing ID", "Title", "Review Text", "Recommended IND", "Division Name", "Department Name", "Class Name"])
variables_cualitativas


# ### Analizamos la cantidad de apariciones de cada una de las variables cualitativas:
# 

# #### Variables Clothing ID, Title y Review Text:

# In[ ]:


ids = variables_cualitativas["Clothing ID"].value_counts()
titulos = variables_cualitativas["Title"].value_counts()
textos = variables_cualitativas["Review Text"].value_counts()
ids


# * Las Ids, corresponden a un identificador de la prenda, el cual es unico.

# In[ ]:


ids.hist()


# * Podemos apreciar que la gran mayoria de prendas, tiene una review unica y solo unas pocas tienen varias. Es decir, los diferentes productos tienden a tener una unica review.

# In[ ]:


titulos


# In[ ]:


textos


# * Analizando los contenidos de estas dos variables, podemos encontrar que hacen referencia al Titulo y Cuerpo de la review, en los cuales, si bien hay repeticiones son mayoritariamente unicos por lo que no tiene sentido compararlos con graficos de barras.

# #### Variable Recommended IND:

# Esta variable corresponde a un binario indicando si el escritor de la review recomienda el producto o no.

# In[ ]:


recomendado = variables_cualitativas["Recommended IND"].value_counts()
grafico_recomendaciones = sns.countplot(x = "Recommended IND", data = variables_cualitativas)
recomendado


# * Podemos observar que la gran mayorias de las reviews recomendan el producto y solo unas pocas no, lo cual nos da a entender que **el dataset no esta balanceado con respecto de las cantidades de recomendaciones de producto**s.

# #### Variable Class Name:

# Esta variable indica a que **tipo de prenda** pertenece la review, las cuales pueden ser:

# In[ ]:


clase = variables_cualitativas["Class Name"].value_counts()
grafico_clases = sns.countplot(x = "Class Name", data = variables_cualitativas)
grafico_clases.set_xticklabels(grafico_clases.get_xticklabels(), rotation = 90)
grafico_clases
clase


# #### Variable Department Name:

# Esta variable divide las reseñas por **categorias de prendas**, los cuales pueden ser:

# In[ ]:


departamento = variables_cualitativas["Department Name"].value_counts()
grafico_departamentos = sns.countplot(x = "Department Name", data = variables_cualitativas)
departamento


# #### Variable Division Name:

# Esta variable nos divide cada una de las prendas a las cuales se les estan haciendo reviews en 3 grades **grupos**:
# * **Intimates**
# * **General**
# * **General Petite**

# In[ ]:


division = variables_cualitativas["Division Name"].value_counts()
grafico_divisiones = sns.countplot(x = "Division Name", data = variables_cualitativas)
division


# ## Variables Cuantitativas:

# Las variables cuantitativas que existen en el dataset son: **Age**, **Rating**, **Positive Feedback Count**, para las cuales creamos un nuevo DataFrame para poder analizarlas...

# In[ ]:


variables_cuantitativas = pd.DataFrame(data = copia_dataset_ecommerce, columns = ["Age", "Rating", "Positive Feedback Count"])
variables_cuantitativas


# In[ ]:


histograma_edades = sns.histplot(variables_cuantitativas.loc[:, "Age"], kde = True).set(title = "Distribución de las edades", xlabel = "Edades", ylabel = "Frecuencia")


# * Las edades en el dataset estan bastante bien distribuidas entre 18 y 70

# In[ ]:


grafico_ratings = sns.countplot(x = "Rating", data = variables_cuantitativas)


# * Los **ratings de las reviews estan muy desbalanceados**, teniendo una cantidad enorme de reviews de 5* y menos de 2000 para las de 1* y 2*.

# ### Medidas de resumen de las variables cuantitativas (Media, Maximos, Minimos, Cuartiles y Mediana):

# In[ ]:


medidas = variables_cuantitativas.describe()
medidas


# ## Preprocesamiento y transformación de datos:

# Para empezar, eliminamos la variable **Unnamed: 0**, la cual repite los indices de las reviews, para no tener datos repetidos.

# In[ ]:


#copia_dataset_ecommerce.drop(columns = ["Unnamed: 0"], inplace = True)
#copia_dataset_ecommerce


# ### Cantidad de valores nulos por columna:
# 
# Ya que no podemos revisar la totalidad del dataset, analizaremos si es que en alguna de las filas se encuentra algún registro con valores nulos.

# In[ ]:


copia_dataset_ecommerce.isna().sum()


# ### Respecto a las variables cualitativas:

# Entre estos tipos de variables, tenemos 5 las cuales poseen valores nulos, las cuales son:
# 
# * **Title = 3810**
# * **Review Text = 845**
# * **Division Name = 14**
# * **Department Name = 14**
# * **Class Name = 14**
# 
# 
# Primero comenzamos revisando las filas que contienen estos valores nulos:
# * Para la variable Title:

# In[ ]:


copia_dataset_ecommerce[copia_dataset_ecommerce["Title"].isna()]


# In[ ]:


copia_dataset_ecommerce[copia_dataset_ecommerce["Review Text"].isna()]


# Para estos datos que representan el titulo y cuerpo de la review, podemos identificar algunos casos:
# 
# 1. Tanto la variable **Title** como **Review Text** son **NaN**
# 2. Unicamente la variable **Title** es **NaN**
# 3. Unicamente la variable **Review Text** es **NaN**
# 
# Como el analisis que estamos haciendo se realizara en base al campo **Review Text** , podriamos tomar como reviews validas a que tienen unicamente **Title** nulo, por lo que no las eliminaremos.
# Por otra parte, para el **primer caso** no nos queda otra opción ya que no tenemos ninguna otra columna que contenga informacion similar para copiarla a esta, por lo que la eliminamos del dataset.
# Con respecto al **ultimo caso** podemos copiar los contenidos de esta con los de su titulo, tomando como validas a las reviews que poseen titulo pero no cuerpo y de esta forma no perderiamos registros que puedan resultarnos utiles para el analisis.

# In[ ]:


#Referencias:
#https://stackoverflow.com/questions/39128856/python-drop-row-if-two-columns-are-nan
#https://www.edureka.co/community/51168/pandas-fillna-with-another-column
copia_dataset_ecommerce = copia_dataset_ecommerce.dropna(subset = ["Title", "Review Text"], thresh=1)
copia_dataset_ecommerce["Review Text"] = np.where(copia_dataset_ecommerce["Review Text"].isnull(), copia_dataset_ecommerce["Title"], copia_dataset_ecommerce["Review Text"])
copia_dataset_ecommerce


# Volvemos a revisar los valores nulos para ver si el problema queda solucionado:

# In[ ]:


copia_dataset_ecommerce.isna().sum()


# * Para la variable Title: **Division Name**

# In[ ]:


copia_dataset_ecommerce[copia_dataset_ecommerce["Division Name"].isna()]


# A simple vista, podemos identificar que esos 13 NaN se debian a reviews que **si bien tienen un ID de prenda asignado, estos no poseen ningún grupo, categoria o tipo asignados**.
# 
# * La primera solucion a este problema que se nos ocurrió es buscar el producto por **ID de prenda** para poder obtener las 3 columnas faltnates, pero nos encontramos con el problema de que un mismo producto puede pertenecer a mas de un grupo, como por ejemplo la prenda con ID = 1104:
# 

# In[ ]:


copia_dataset_ecommerce[copia_dataset_ecommerce["Clothing ID"] == 1104]


# *La prenda con Clothing ID = 1104, puede pertenecer a mas de un grupo, en este caso General y General Petite*
# 
# Si bien la **categoria y tipo de prenda coinciden en todas**, seguiriamos teniendo el mismo problema, **no podriamos saber a que grupo pertenece**.
# 
# * Como ultima solucion, podemos direcamente obviar estos datos, ya que al ser solo 14 representan una cantidad casi minima comparada a los 23.000+ filas del set, por lo que los removemos del dataset y los tomamos como no relevantes para esta investigación.

# In[ ]:


copia_dataset_ecommerce = copia_dataset_ecommerce.dropna(subset = ["Division Name"])


# Una vez hecho esto, ya tendriamos solucionado el problema de los valores nulos de estas variables.

# In[ ]:


copia_dataset_ecommerce.isna().sum()


# ### Respecto a las variables cuantitativas:
# 
# Para estas variables no tenemos valores faltantes en el dataset, por lo cual podemos directamente **analizar los boxplots** de las 3 variables cualitativas del set en **busca de outlayers**:

# In[ ]:


sns.set(rc = {'figure.figsize':(6,6)})
box_age = sns.boxplot(y="Age", data = variables_cuantitativas).set(title = "Boxplot Age")
box_age


# In[ ]:


box_rating = sns.boxplot(y="Rating", data = variables_cuantitativas).set(title = "Boxplot Rating")
box_rating


# In[ ]:


box_feedback = sns.boxplot(y="Positive Feedback Count", data = variables_cuantitativas).set(title = "Boxplot Positive Feedback Count")
box_feedback


# Viendo estos boxplots identificamos que existen outlayers para todas las variables cuantitativas.
# Aunque tengamos una gran cantidad de outlayers, todos estos valores estan dentro de lo esperado:
# 
# * Las edades no superan los 100 años.
# * Los outlayers que figuran dentro del boxplot del rating son debido al gran desbalance que hay dentro de los valores de esta variable.
# * Para los positive feedback, esta dentro de lo esperable que algunas reviews sean mejor valuadas que otras. En nuestro caso, tenemos muchas variables que tienen upvotes casi nulos y otras pocas que poseen mas de 100. Esto puede estar dado por como son mostrados los reviews en la pagina, mostrando primero los mas votados, haciendo que estos pocos se lleven todos los upvotes. Otra opcion es que solamente algunos usuarios se preocupen por realizar estos upvotes.
# 
# Por estas razones, tomaremos estos outlayers como informacion valida y no sera necesario revisar particularmente utilizando los metodos de z_score.

# ### Construcción la variable objetivo:

# Agregamos una nueva variable a nuestro dataset **"Tipo review"**, la cual define la positividad de una review con las siguientes medidas:
# * SEGUN EL ATRIBUTO RAITING:
#     + Positva = 4, 5
#     + Negativa = 1, 2, 3
#   
# Para esto le asignaremos un booleano a esta columna (El cual representara como TRUE a las reviews positivas y FALSE a las reviews negativas)

# In[ ]:


copia_dataset_ecommerce["Tipo Review"] = 0
copia_dataset_ecommerce.loc[copia_dataset_ecommerce["Rating"] >= 4, "Tipo Review"] = 1
copia_dataset_ecommerce


# ## Generación y evaluación de modelos:

# Para realizar los modelos, realizamos una copia del dataset que contiene todos las transformaciones realizadas.

# In[ ]:


dataset_modelos = copia_dataset_ecommerce.copy()


# ### Division del conjunto de datos en un 70%/30%:

# In[ ]:


modelo_x = dataset_modelos["Review Text"].copy()
modelo_y = dataset_modelos["Tipo Review"].copy()
x_train, x_test, y_train, y_test = train_test_split(modelo_x, modelo_y, test_size = 0.3, random_state = 7)


# ### Entrenamiento de modelos y guardado de medidas:

# En esta etapa, entrenaremos diferentes algoritmos tal que a partir del texto en el campo “Review Text”,
# pueda clasificar correctamente la crítica como positiva o negativa. A partir de esto, extraeremos las medidas de los mismos **(Precision, Recall y F1)** y las guardaremos dentro de un nuevo **dataframe**.

# In[ ]:


data_frame_medidas = pd.DataFrame(columns = ["Algoritmo", "Medida", "Porcentaje"])


# #### Naïve Bayes:
# // [Referencia](https://scikit-learn.org/stable/modules/naive_bayes.html) // 

# In[ ]:


#Genero y entreno el modelo
modelo = make_pipeline(TfidfVectorizer(), MultinomialNB())
modelo.fit(x_train, y_train)
prueba_modelo_naive = modelo.predict(x_test)
#Obtengo medidas
precision = {
    "Algoritmo":"Naïve Bayes (Variable Review Text)", 
    "Medida":"Precision",
    "Porcentaje":accuracy_score(y_test, prueba_modelo_naive)
}
recall = {
    "Algoritmo":"Naïve Bayes (Variable Review Text)", 
    "Medida":"Recall",
    "Porcentaje":recall_score(y_test, prueba_modelo_naive)
}
F1 = {
    "Algoritmo":"Naïve Bayes (Variable Review Text)", 
    "Medida":"F1",
    "Porcentaje":f1_score(y_test, prueba_modelo_naive)
}
data_frame_medidas = data_frame_medidas.append(precision, ignore_index = True)
data_frame_medidas = data_frame_medidas.append(recall, ignore_index = True)
data_frame_medidas = data_frame_medidas.append(F1, ignore_index = True)


# #### Regresion Logistica
# // [Referencia](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) //

# In[ ]:


#Genero y entreno el modelo
modelo = make_pipeline(TfidfVectorizer(), LogisticRegression())
modelo.fit(x_train, y_train)
prueba_modelo_reg_log = modelo.predict(x_test)
#Obtengo medidas
precision = {
    "Algoritmo":"Regresión Logística Binaria (Variable Review Text)", 
    "Medida":"Precision",
    "Porcentaje":accuracy_score(y_test, prueba_modelo_reg_log)
}
recall = {
    "Algoritmo":"Regresión Logística Binaria (Variable Review Text)", 
    "Medida":"Recall",
    "Porcentaje":recall_score(y_test, prueba_modelo_reg_log)
}
F1 = {
    "Algoritmo":"Regresión Logística Binaria (Variable Review Text)", 
    "Medida":"F1",
    "Porcentaje":f1_score(y_test, prueba_modelo_reg_log)
}
data_frame_medidas = data_frame_medidas.append(precision, ignore_index = True)
data_frame_medidas = data_frame_medidas.append(recall, ignore_index = True)
data_frame_medidas = data_frame_medidas.append(F1, ignore_index = True)


# #### Arboles de decision: 
# // [Referencia](https://scikit-learn.org/stable/modules/tree.html) //

# In[ ]:


#Genero y entreno el modelo
modelo = make_pipeline(TfidfVectorizer(), DecisionTreeClassifier())
modelo.fit(x_train, y_train)
prueba_modelo_arbol_dec = modelo.predict(x_test)
#Obtengo medidas
precision = {
    "Algoritmo":"Arboles de decision (Variable Review Text)", 
    "Medida":"Precision",
    "Porcentaje":accuracy_score(y_test, prueba_modelo_arbol_dec)
}
recall = {
    "Algoritmo":"Arboles de decision (Variable Review Text)", 
    "Medida":"Recall",
    "Porcentaje":recall_score(y_test, prueba_modelo_arbol_dec)
}
F1 = {
    "Algoritmo":"Arboles de decision (Variable Review Text)", 
    "Medida":"F1",
    "Porcentaje":f1_score(y_test, prueba_modelo_arbol_dec)
}
data_frame_medidas = data_frame_medidas.append(precision, ignore_index = True)
data_frame_medidas = data_frame_medidas.append(recall, ignore_index = True)
data_frame_medidas = data_frame_medidas.append(F1, ignore_index = True)


# #### Random Forest:
# // [Referencia](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) //

# In[ ]:


#Genero y entreno el modelo
modelo = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
modelo.fit(x_train, y_train)
prueba_modelo_random_forest = modelo.predict(x_test)
#Obtengo medidas
precision = {
    "Algoritmo":"Random Forest (Variable Review Text)", 
    "Medida":"Precision",
    "Porcentaje":accuracy_score(y_test, prueba_modelo_random_forest)
}
recall = {
    "Algoritmo":"Random Forest (Variable Review Text)", 
    "Medida":"Recall",
    "Porcentaje":recall_score(y_test, prueba_modelo_random_forest)
}
F1 = {
    "Algoritmo":"Random Forest (Variable Review Text)", 
    "Medida":"F1",
    "Porcentaje":f1_score(y_test, prueba_modelo_random_forest)
}
data_frame_medidas = data_frame_medidas.append(precision, ignore_index = True)
data_frame_medidas = data_frame_medidas.append(recall, ignore_index = True)
data_frame_medidas = data_frame_medidas.append(F1, ignore_index = True)


# #### Evaluacion de clasificadores:
# 
# Una vez entrenados los 4 algoritmos de clasificacion, podemos pasar a revisar sus medidas de **recall**, **precision** y su **F1 score**.

# In[ ]:


data_frame_medidas


# In[ ]:


grafico_medidas = sns.catplot(data=data_frame_medidas, kind="bar", x="Algoritmo", y="Porcentaje", hue="Medida", alpha = 0.7)
for axes in grafico_medidas.axes.flat:
    _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
grafico_medidas


# Evaluamos las matrices de confusion de los modelos:

# In[ ]:


matriz_naive = pd.crosstab(y_test, prueba_modelo_naive)
#https://stackoverflow.com/questions/29647749/seaborn-showing-scientific-notation-in-heatmap-for-3-digit-numbers
matriz_naive = sns.heatmap(matriz_naive, annot = True, fmt='g')
matriz_naive.set(title = "Matriz de confusion del modelo Naive Bayes",xlabel  = "Resultado")


# In[ ]:


matriz_reg_log = pd.crosstab(y_test, prueba_modelo_reg_log)
matriz_reg_log = sns.heatmap(matriz_reg_log, annot = True, fmt='g')
matriz_reg_log.set(title = "Matriz de confusion del modelo Regresion Logistica",xlabel  = "Resultado")


# In[ ]:


matriz_arbol_dec = pd.crosstab(y_test, prueba_modelo_arbol_dec)
matriz_arbol_dec = sns.heatmap(matriz_arbol_dec, annot = True, fmt='g')
matriz_arbol_dec.set(title = "Matriz de confusion del modelo Arboles de Decision",xlabel  = "Resultado")


# In[ ]:


matriz_random_forest = pd.crosstab(y_test, prueba_modelo_random_forest)
matriz_random_forest = sns.heatmap(matriz_random_forest, annot = True, fmt='g')
matriz_random_forest.set(title = "Matriz de confusion del modelo Random Forest",xlabel  = "Resultado")


# Revisando las matrices de confusion y las medidas de todos los modelos, podemos observar que todos los modelos tienen **mala prediccion por las reviews negativas**.

# ### Generacion modelo en base a 5 clases de Rating:
# 
# Para generar este modelo elegiremos el modelo con mejor resultados del punto anterior, si bien todos tienen una tendencia a predecir mal  las reviews negativas, elegimos el mas balanceado de todos que seria **Regresion Logistica**. Una vez elegido el algoritmo, pasamos a la creacion del modelo:

# Al igual que con los modelos anteriores, comenzamos **generando un conjunto de pruebas 70%/30%**:

# In[ ]:


modelo_x = dataset_modelos["Review Text"].copy()
modelo_y = dataset_modelos["Rating"].copy()
x_train, x_test, y_train2, y_test2 = train_test_split(modelo_x, modelo_y, test_size = 0.3, random_state = 7)


# Luego, a partir de estos conjuntos, entrenamos el algoritmo y guardamos sus medidas en el mismo dataset de las anteriores:
# 
# *Nota: Al estar clasificando los distintos cuerpos de review en ratings, los cuales tienen 5 valores posibles, pasaremos a utilizar el algoritmo de regresion logista multinominal*

# In[ ]:


# Para el solver standar, no obtenemos resultados, ya que no llega a converger: /opt/conda/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
# Debido a esto, buscamos otro solver y buscamos el resultado con mejores resultados, que en nuestro caso fue solver = "saga"
modelo = make_pipeline(TfidfVectorizer(), LogisticRegression(solver = "saga", multi_class="multinomial")) 
modelo.fit(x_train, y_train2)
prueba_modelo_reg_log_multi = modelo.predict(x_test)
#Obtengo medidas
precision = {
    "Algoritmo":"Regresion Logistica Multinominal (Variable Rating)", 
    "Medida":"Precision",
    "Porcentaje":accuracy_score(y_test2, prueba_modelo_reg_log_multi)
}
#ValueError: Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
#https://stackoverflow.com/questions/52269187/facing-valueerror-target-is-multiclass-but-average-binary
recall = {
    "Algoritmo":"Regresion Logistica Multinominal (Variable Rating)", 
    "Medida":"Recall",
    "Porcentaje":recall_score(y_test2, prueba_modelo_reg_log_multi, average = "micro")
}
F1 = {
    "Algoritmo":"Regresion Logistica Multinominal (Variable Rating)", 
    "Medida":"F1",
    "Porcentaje":f1_score(y_test2, prueba_modelo_reg_log_multi, average = "micro")
}
data_frame_medidas = data_frame_medidas.append(precision, ignore_index = True)
data_frame_medidas = data_frame_medidas.append(recall, ignore_index = True) 
data_frame_medidas = data_frame_medidas.append(F1, ignore_index = True)


# In[ ]:


data_frame_medidas


# In[ ]:


grafico_medidas_2 = sns.catplot(data=data_frame_medidas, kind="bar", x="Algoritmo", y="Porcentaje", hue="Medida", alpha = 0.7)
for axes in grafico_medidas_2.axes.flat:
    _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
grafico_medidas


# In[ ]:


matriz_reg_logistica_rating = pd.crosstab(y_test2, prueba_modelo_reg_log_multi)
matriz_reg_logistica_rating = sns.heatmap(matriz_reg_logistica_rating, annot = True, fmt='g')
matriz_reg_logistica_rating.set(title = "Matriz de confusion del modelo Regresion Logistica Multinominal",xlabel  = "Resultado")


# A simple vista podemos observar que este nuevo modelo que clasifica a la **varibale rating** tiene una eficiencia mucho menor que con la variable **tipo review**

# ## Concluciones:

# Al momento de entrenar estos modelos, obtuvimos clasificadores que **no se desempeñaban muy bien con clasificar reviews negativas**. Esto esta directamente ligado al desbalance que vimos durante la exploracion inicial, donde teniamos una gran cantidad de reviews de 5 estrellas, las cuales llegaban a doblar o triplicar a los demas valores, generando asi que todos nuestros modelos entrenados para clasificar estas variables obtengan ese pobre desempeño.
# 
# * Esto se puede ver directamente en los graficos:
#     1. Modelo con mejores presiciones generales obtenido calificando la variable Tipo Review):
#     2. Modelo obtenido al clasificar mediante la variable Rating
#     3. Graficos de balance del dataset:

# In[ ]:


sns.set(rc = {'figure.figsize':(5,5)})
matriz_reg_log = pd.crosstab(y_test, prueba_modelo_reg_log)
matriz_reg_log = sns.heatmap(matriz_reg_log, annot = True, fmt='g')
matriz_reg_log.set(title = "Matriz de confusion del modelo Regresion Logistica",xlabel  = "Resultado")


# In[ ]:


matriz_reg_logistica_rating = pd.crosstab(y_test2, prueba_modelo_reg_log_multi)
matriz_reg_logistica_rating = sns.heatmap(matriz_reg_logistica_rating, annot = True, fmt='g')
matriz_reg_logistica_rating.set(title = "Matriz de confusion del modelo Regresion Logistica Multinominal",xlabel  = "Resultado")


# In[ ]:


grafico_ratings = sns.countplot(x = "Rating", data = copia_dataset_ecommerce)


# In[ ]:


grafico_ratings = sns.countplot(x = "Tipo Review", data = copia_dataset_ecommerce)


# En conclucion, si balancearamos estas variables, obtendriamos un modelo con mejores clasificaciones generales que los entrenados en base al dataset.

# # Parte 2

# In[ ]:


d=pd.read_csv("../input/hotelbookings/hotel_bookings.csv")
copia_dataset_hotel = d.copy()


# # Exploración inicial:
# 
# ### Nombres y tipos de columnas:

# In[ ]:


copia_dataset_hotel.dtypes


# ### Descripción del dataset:
# 
# La información presentada en la siguiente tabla fue extraída del paper *`Hotel booking demand datasets`* de *ELSEVIER*.
# 
# |Columna|Tipo de Variable|Descripción|
# |----------|--------------|---------------|
# |ADR|Numérica|Tarifa diaria promedio (Average Daily Rate) definida por American Hotel & Lodging Association|
# |Adults|Entero|Número de adultos|
# |Agent|Categórica|ID de la agencia de viaje que hizo la reserva|
# |ArrivalDateDayOfMonth|Entero|Día del mes de la fecha de llegada|
# |ArrivalDateMonth|Categórica|Mes de la fecha de llegada con 12 categorías: “January” a “December”|
# |ArrivalDateWeekNumber|Entero|Número de la semana de la fecha de llegada|
# |ArrivalDateYear|Entero|Año de la fecha de llegada|
# |AssignedRoomType|Categórica|Código para el tipo de habitación reservada|
# |Babies|Entero|Número de bebés|
# |BookingChanges|Entero|Número de cambios hechos en la reserva desde el día en que se hizo hasta el check-in o cancelación de la misma|
# |Children|Entero|Número de niños|
# |Company|Categórica|ID de la compañía o entidad que hizo o es responsable del pago de la reserva or responsible|
# |Country|Categórica|País de origen. Las categorías son representadas en el formato ISO 3155–3:2013|
# |CustomerType|Categórica|Tipo de reserva|
# |DaysInWaitingList|Entero|Número de días en que la reserva estuvo en la lista de espera|
# |DepositType|Categórica|Indicación de si un cliente hizo un depósito para garantizar la reserva|
# |DistributionChannel|Categórica|Canal de distribución de la reserva|
# |IsCanceled|Categórica binaria|Valor que indica si una reserva se canceló o no|
# |IsRepeatedGuest|Categórica binaria|Valor que indica si un cliente es repetido|
# |LeadTime|Entero|Número de días que pasaron entre la fecha de entrada de la reserva al PMS (Property Management System) y la fecha de llegada|
# |MarketSegment|Categórica|Designación de segmento de mercado|
# |Meal|Categórica|Tipo de comida reservada|
# |PreviousBookingsNotCanceled|Entero|Número de reservas previas sin cancelar|
# |PreviousCancellations|Entero|Número de reservas previas canceladas|
# |RequiredCardParkingSpaces|Entero|Número de estacionamientos requeridos por el cliente|
# |ReservationStatus|Categórica|Último estado de la reserva, asumiendo una de 3 categorías: Canceled, Check-Out, No-Show|

# ### Primeras filas del dataframe:

# In[ ]:


copia_dataset_hotel.head()


# ### Últimas filas del dataframe:

# In[ ]:


copia_dataset_hotel.tail()


# ### Valores nulos:

# In[ ]:


copia_dataset_hotel.isna().sum()


# Podemos apreciar que la columna **'company'** tiene 112593 filas vacias de 119390 campos, lo que equivale a **%94** del dataframe, por lo que simplemente dropeamos la columna. Adicionalmente llenamos los valores nulos, ya que nos van a generar errores en el árbol de decisión.

# In[ ]:


copia_dataset_hotel = copia_dataset_hotel.drop(['company'], axis = 1)
copia_dataset_hotel['children'] = copia_dataset_hotel['children'].fillna(0)
copia_dataset_hotel['agent'] = copia_dataset_hotel['agent'].fillna('?')
copia_dataset_hotel['country'] = copia_dataset_hotel['country'].fillna('?')


# Para las columnas de tipo 'object', las transformamos a entero para que sea un tipo válido para el árbol. Para esto usamos LabelEncoder.

# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for columna in copia_dataset_hotel.columns:
    if copia_dataset_hotel[columna].dtype == object:
        copia_dataset_hotel[columna] = le.fit_transform(copia_dataset_hotel[columna].astype(str))
    else:
        pass


# ### Distribución de la variable target

# In[ ]:


grafico_canceled = sns.countplot(x = "is_canceled", data = copia_dataset_hotel)


# In[ ]:


min_n = copia_dataset_hotel['is_canceled'].value_counts().min()
copia_dataset_hotel = copia_dataset_hotel.groupby('is_canceled', as_index=False, group_keys=False).apply(lambda x: x.sample(n=min_n))


# In[ ]:


grafico_canceled = sns.countplot(x = "is_canceled", data = copia_dataset_hotel)


# ## Correlación de atributos:
# 
# Graficamos las variables de a pares, aplicamos Correlación de Pearson y generamos un heatmap para poder evaluar las relaciones entre las variables:

# In[ ]:


sns.set(rc = {'figure.figsize':(30,30)})
copia_dataset_hotel_correlacion = copia_dataset_hotel.corr()
grafico_correlacion = sns.heatmap(copia_dataset_hotel_correlacion, annot =True)


# ### Variables con mayor correlación
# 
# * **reservation_status** (-0.92)

# In[ ]:


sns.set(rc = {'figure.figsize':(15,15)})
sns.scatterplot(data=copia_dataset_hotel, x='reservation_status', y='is_canceled', palette='deep')


# * **deposit_type** (0.47)

# In[ ]:


sns.scatterplot(data=copia_dataset_hotel, x='deposit_type', y='is_canceled', palette='deep')


# * **lead_time**(0.29).  A partir de lead_time > 500 es poco probable que haya un cancelamiento.

# In[ ]:


sns.scatterplot(data=copia_dataset_hotel, x='lead_time', y='is_canceled', palette='deep')


# * **country** (0.27)

# In[ ]:


sns.scatterplot(data=copia_dataset_hotel, x='country', y='is_canceled', palette='deep')


# ## Árbol de decisión
# 
# Importamos las librerías a utilizar.

# In[ ]:


import sklearn as sk
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold,RandomizedSearchCV
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, recall_score, accuracy_score,f1_score, make_scorer


# Separamos el dataset en conjuntos de entrenamiento y testeo con una proporción 80/20 respectivamente.

# In[ ]:


### dataset_hotel_x = copia_dataset_hotel.drop(['is_canceled'], axis='columns', inplace=False)
dataset_hotel_x = copia_dataset_hotel.drop(columns = ['is_canceled', 'reservation_status'], axis='columns', inplace=False)
dataset_hotel_y = copia_dataset_hotel['is_canceled'].copy()
x_train, x_test, y_train, y_test = train_test_split(dataset_hotel_x, dataset_hotel_y, test_size=0.2, random_state=12, stratify=copia_dataset_hotel['is_canceled'].values)


# In[ ]:


n=20

#Conjunto de parámetros que quiero usar
params_grid = {'criterion':['gini','entropy'],
               'min_samples_leaf':list(range(50,200)),
               'min_samples_split': list(range(50,200)),
               'ccp_alpha':np.linspace(0,0.05,n), 
               'max_depth':list(range(1,15))}
                

#Cantidad de splits para el Cross Validation
folds=5

#Kfold estratificado
kfoldcv = StratifiedKFold(n_splits=folds)

#Clasificador
base_tree = DecisionTreeClassifier() 

#Metrica que quiero optimizar F1 Score
scorer_fn = make_scorer(sk.metrics.f1_score)

#Random Search Cross Validation
randomcv = RandomizedSearchCV(estimator=base_tree,
                              param_distributions = params_grid,
                              scoring=scorer_fn,
                              cv=kfoldcv,
                              n_iter=n) 

#Busco los hiperparamtros que optimizan F1 Score
randomcv.fit(x_train,y_train);


# In[ ]:


#Mejores hiperparametros del arbol
print(randomcv.best_params_)
#Mejor métrica
print(randomcv.best_score_)


# In[ ]:


randomcv.cv_results_['mean_test_score']


# In[ ]:


#Atributos considerados y su importancia
best_tree = randomcv.best_estimator_
plt.barh(dataset_hotel_x.columns, best_tree.feature_importances_)


# In[ ]:


arbol = DecisionTreeClassifier().set_params(**randomcv.best_params_)
modelo = arbol.fit(x_train,y_train)
y_pred = modelo.predict(x_test)
ds_resultados = pd.DataFrame(zip(y_test,y_pred),columns=['test','pred'])

tabla=confusion_matrix(ds_resultados['test'], ds_resultados['pred'])
grf=sns.heatmap(tabla,annot=True)
plt.show()


# In[ ]:


accuracy=accuracy_score(ds_resultados['test'], ds_resultados['pred'], normalize=True)
recall=recall_score(ds_resultados['test'], ds_resultados['pred'])
f1=f1_score(ds_resultados['test'], ds_resultados['pred'])

print("Accuracy: "+str(accuracy))
print("Recall: "+str(recall))
print("f1 score: "+str(f1))


# In[ ]:


tree_plot_completo= plot_tree(modelo,feature_names=dataset_hotel_x.columns.to_list(),filled=True,rounded=True)
plt.show(tree_plot_completo)


# ### Random Forest

# In[ ]:


arbol = RandomForestClassifier(n_estimators=100)
modelo = arbol.fit(x_train,y_train)
y_pred = modelo.predict(x_test)
ds_resultados = pd.DataFrame(zip(y_test,y_pred),columns=['test','pred'])
tabla=confusion_matrix(ds_resultados['test'], ds_resultados['pred'])
grf=sns.heatmap(tabla,annot=True)
plt.show()


# In[ ]:


accuracy=accuracy_score(ds_resultados['test'], ds_resultados['pred'], normalize=True)
recall=recall_score(ds_resultados['test'], ds_resultados['pred'])
f1=f1_score(ds_resultados['test'], ds_resultados['pred'])

print("Accuracy: "+str(accuracy))
print("Recall: "+str(recall))
print("f1 score: "+str(f1))

