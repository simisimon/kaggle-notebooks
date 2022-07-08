#!/usr/bin/env python
# coding: utf-8

# # Trabajo Final Ciencia de Datos 2022
# 
# El siguiente trabajo tiene como objetivo presentar un caso de estudio sobre consumo de energía eléctrica basado en la continua medición de los artefactos eléctricos de un hogar en Estados Unidos, tomando mediciones en períodos de 10 minutos por el plazo de 3 meses continuos. 
# Con tal fin, hemos utilizado un conjunto de datos obtenido desde [Kaggle dataset](https://www.kaggle.com/datasets/sohommajumder21/appliances-energy-prediction-data-set).
# El estudio recabó amplia información del hogar, como la temperatura y presión de cada ambiente, el consumo producido por la iluminación (medida en forma conjunta), así como la presiñon atmosférica, la visibilidad, la velocidad del viento, la temperatura y la humedad externa. Cada medición fue realizada a intervalos de 10 minutos.
# 
# La presentación y estudio esta dividida en 2 fases. En la primer fase, se busca entender, relacionar y corroborar que todos los datos cargados fueran consistentes. Para luego dar paso a la 2da fase, en donde con la comprensión de los datos obtenidos de la fase anterior, se da paso a las pruebas y metodos de predicción.
# 
# A continuación, se explica el dataset utilizado:
# 
# | Nombre de la Columna   | Descripcion |
# | -----------------------|-------------|
# | Date | Fecha del muestreo |
# | Appliance | Energía consumida en Wh |
# | Lights | Energia consumida por las luces en Wh |
# | T1 | Temperatura de la 1° habitación |
# | RH_1 | Humedad de la 1° habitación |
# | T2 | Temperatura de la 2° habitación |
# | RH_2 | Humedad de la 2° habitación |
# | T3 | Temperatura de la 3° habitación |
# | RH_3 | Humedad de la 3° habitación |
# | T4 | Temperatura de la 4° habitación |
# | RH_4 | Humedad de la 4° habitación |
# | T5 | Temperatura de la 5° habitación |
# | RH_5 | Humedad de la 5° habitación |
# | T6 | Temperatura de la 6° habitación |
# | RH_6 | Humedad de la 6° habitación |
# | T7 | Temperatura de la 7° habitación |
# | RH_7 | Humedad de la 7° habitación |
# | T8 | Temperatura de la 8° habitación |
# | RH_8 | Humedad de la 8° habitación |
# | T9 | Temperatura de la 9° habitación |
# | RH_9 | Humedad de la 8° habitación |
# | T_out | Temperatura externa |
# | Press_mm_hg | Presion externa |
# | RH_out | Humedad externa |
# | WindSpeed | Velocidad del viento |
# | Visibility | Visibilidad fuera de la casa |
# 
# 
# 
# 
# 
# 
# 
# 
# En este notebook, se utiliza el lenguaje <span style="color:green;"> Python </span>  que es uno de los lenguajes de programación que domina dentro del ámbito de la estadística y machine learning. Al tratarse de un software de codigo libre, un número muy alto de  usuarios han podido implementar sus códigos, dando lugar a un número muy elevado de nuevas librerías donde se pueden encontrar prácticamente todas las técnicas de machine learning existentes.
# 
# ## Librerias utilizadas
# 
# Nombre de la Librería  |  Descripción  
# -----------------------|---------------
# scikit-learn  | es una librería que unifica bajo un único marco los principales algoritmos y funciones de Machine Learning facilitando todas las etapas de preprocesado, entrenamiento y validación de modelos predictivos.
# Pandas  | Es una biblioteca de software escrita como extensión de NumPy para manipulación y análisis de datos para el lenguaje de programación Python.
# Seaborn  | Es una libreria de visualizacion de datos basda en matplotlib. Provee una interface de alto nivel para dibujar graficos estadisticos atractivos e informativos. Al igual que Pandas, no realiza ningun ningun grafico por si mismo, sino que es un wrapper de matplotlib.
# Plotly  | Es una libreria open source desarrollada en Python. Es utilizada para visualizar datos creando graficos interactivos y de gran calidad. A traves de su interactividad, el usuario podra habilitar y deshabilitar partes del mismo, facilitando la visualizacion de lo que realmente le interese. Tipos de graficos que utilizaremos: Scatter (XY) Plot & mapa de calor o heat map.
# 
# 
# Por lo anteriormente desarrollado, podemos definir que se trata de un caso de **aprendizaje supervisado**.
# 

# In[ ]:


# Comienzo de codigo Python
# Librerias utilizadas para el ejercicio

# Data frames ops 
import numpy as np
import pandas as pd

# Plots ops
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# For modeling
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[ ]:


# Creacion del dataframe y lectura del archivo de soporte usando Pandas
df = pd.read_csv("../input/energy-consumptionv1/energydata_complete.csv")


# In[ ]:


# Eliminamos columnas que no vamos a utilizar 
df = df.drop(['Tdewpoint', 'rv1', 'rv2'], axis=1)


# In[ ]:


# Chequeo de consistencia usando primeras filas
df.head(6)


# In[ ]:


# Chequeo de consistencia usando ultimas filas
df.tail(4)


# Las funciones **head** y **tail** buscan evidenciar la consistencia del dataset a lo largo de todas sus filas
# y columnas. Por lo observado, existe consistencia a lo largo del dataset.

# In[ ]:


# Chequeamos la composicion y los tipos de datos de los campos
df.info()


# Observamos que la columna date es del tipo string. No es de gran utilidad al momento de trabajar con los datos, por lo cual la convertiremos al tipo _date_.

# In[ ]:


# Convertimos la columan date (del tipo string) al tipo fecha
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')


# Para los campos del tipo numéricos, obtenemos datos característicos como la media, la desviación estandar, los cuartiles, etc

# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# Observamos que la cantidad de filas en cada columna es la misma. Ademas notamos que **no** existen valores nulos para ninguna fila.

# In[ ]:


# Ordenamos los datos por fecha
df = df.sort_values('date', ascending=True)


# Generaremos un data frame mas pequeño para posibilitar una visualización más clara de los datos en los gráficos que desarollaremos mas adelante

# In[ ]:


# Nos quedamos con 1/8 de los datos del dataframe
small_qty = round(len(df.index)/4)
df_small = df[0:small_qty].copy()


# ---
# 
# # Fase EDA (Exploratory Data Analysis)
# 
# El código anterior sirvio para subir a memoria las librerias que se utilizarán en el ejercicio y, también, para crear el dataframe a utilizar, con apoyo de la libreria Pandas.
# El output de las funciones head y tail sirven para ver que la información contenida en el dataset es válida.
# Para evitar cualquier problema con la manipulación del dataframe original, se trabajara con una copia del mismo.
# Esta fase de <span style="color:orange;"> EDA </span>  es relevante para el estudio, ya que permite descubrir patrones, anomalías y crear asumidos apoyados en las representaciones gráficas a continuación.

# In[ ]:


df.boxplot(column=['Appliances'], rot=45, backend='matplotlib', figsize=(12,6))


# In[ ]:


full_chart = px.line(df, x='date', y=df.columns[:], title='Consumo de dispositivos en el tiempo')
full_chart.show()


# Respecto a la primer gráfica, se pueden apreciar algunos picos de consumo como también algunos períodos breves de caida o bajo consumo.
# La presión se mantiene en un nivel muy parejo, sin destacarse picos o caidas.
# Para sacar las primeras conclusiones, a continuación se trabajará en un periodo random de 2 semanas completas para identificar patrones y tendencias.

# ## Relacion entre Dispositivos y Luces (2 semanas)

# In[ ]:


# Grafica de doble entrada
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=df_small['date'], y=df_small['Appliances'], 
               name="Dispositivos",
               mode='lines'),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df_small['date'], y=df_small['lights'], 
               name='Luces',
               mode='lines'),
    secondary_y=True,
)

fig.update_layout(
    title='Consumo de Dispositivos & luces en 2 semanas',
    xaxis_title="Date")
    
fig.update_yaxes(title_text="Consumo Dispositivos (Wh)", secondary_y=False, color="blue")
fig.update_yaxes(title_text="Consumo luces (Wh)", secondary_y=True, color="red")
    
fig.show()


# Buscando relacionar datos, en la gráfica de 2 semanas, se observa como hay un patóon entre los dispositivos y las luces, que coinciden en la tendencia en alza como en baja. Este patrón podría sugerir una relación entre el consumo de día y de noche.

# ## Relacion entre Dispositivos y Temperatura (2 semanas)

# In[ ]:


# Grafica de doble entrada
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces

fig.add_trace(
    go.Scatter(x=df_small['date'], y=df_small['Appliances'], 
               name="Dispositivos",
               mode='lines'),
    secondary_y=False,
)
tempcolumns = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T_out']

for i in tempcolumns:
    fig.add_trace(
        go.Scatter(x=df_small['date'], y=df_small[i], 
                   name=i,
                   mode='lines'),
        secondary_y=True,
    )

fig.update_layout(
    title='Consumo de dispositivos y Temperatura en 2 semanas',
    xaxis_title="Date")
    
fig.update_yaxes(title_text="Dispositivos (Wh)", secondary_y=False, color="blue")
fig.update_yaxes(title_text="Temperatura (°Celsius)", secondary_y=True)    
    
fig.show()


# In[ ]:


# Grafica de doble entrada
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces

fig.add_trace(
    go.Scatter(x=df_small['date'], y=df_small['Appliances'], 
               name="Dispositivos",
               mode='lines'),
    secondary_y=False,
)
tempcolumns = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']

for i in tempcolumns:
    if i=='T6':
        continue
    else:
        fig.add_trace(
            go.Scatter(x=df_small['date'], y=df_small[i], 
                   name=i,
                   mode='lines'),
            secondary_y=True,
        )

fig.update_layout(
    title='Consumo de dispositivos y Temperatura en 2 semanas (sin temperatura externa)',
    xaxis_title="Date")
    
fig.update_yaxes(title_text="Dispositivos(Wh)", secondary_y=False, color="blue")
fig.update_yaxes(title_text="Temperatura(°Celsius)", secondary_y=True)    
    
fig.show()


# La conclusión que se puede obtener al relacionar las últimas dos gráficas, es que a medida que el consumo de los dispositivos hace pico, también lo hace la temperatura en todas las salas internas.

# ---
# 
# # Etapa FE (Feature Engineering)

# ## Mapa de Correlación General
# Utilizando un mapa de correlación de variables podremos apreciar rápidamente si existe algun _feature_ que este fuertemente relacionado directa o inversamente con nuestra variable objetivo, el consumo energético.

# In[ ]:


mcor = df.drop(['date'],axis=1).corr()
mcor.style.background_gradient(cmap='coolwarm')


# La correlación **no** ofrece ningún posible predictor, solo se observa una alta relación entre mismas variables, como el caso de la temperatura, ya que en todos los cuartos la diferencia es poca.
# El mismo fenómeno se observa con la variable Humedad.

# ## Consumo de energia entre el día y la noche
# Es plausible esperar que veamos picos de consumo durante el día, mientras que durante la noche debería disminuir el consumo en gran medida. Para poder gráficar y observar si nuestra hipótesis es correcta, adicionaremos una nueva columna donde indicaremos si el registro sucede durante el día o durante la noche. <span style="color:red;"> Elegiremos una franja horaria desde las 7am hasta las 11pm.EDA </span>

# In[ ]:


# Creamos una nueva columna donde indicamos si el registro corresponde al dia o a la noche
feature_df = df_small.copy()
feature_df['daytime'] = [1100 if i.hour < 23 and i.hour > 7 else 0 for i in feature_df['date']]

# Corroboramos que la columna se haya creado correctamente
feature_df[feature_df['daytime'] != 0]


# In[ ]:


# Creamos una nueva figura
fig = make_subplots()

# Add traces

# Agregamos el grafico de consumo
fig.add_trace(
    go.Scatter(x=feature_df['date'], y=feature_df['Appliances'], 
               name="Consumo",
               mode='lines')
)

# Agregamos las barras que delimitan la franja horaria durante el dia
fig.add_trace(
    go.Scatter(x=feature_df['date'], y=feature_df['daytime'],
               name='Daytime (7am - 11pm)',
               mode='none',
               fill='tozeroy'))

fig.update_layout(
    title='Consumo electrico en el dia durante aprox. dos semanas',
    xaxis_title="Fecha",
    yaxis_range=(0, 1100)
)
    
fig.update_yaxes(title_text="Consumo electrico (en Wh)", color="blue")
fig.show()


# Se puede concluir que el mayor consumo eléctrico se dá durante las horas del dáa (7am a 11pm). Podemos entonces analizar una tabla de correlación de predictores en estas dos franjas horarias.

# ## Mapa de correlación durante el día

# In[ ]:


df_daytime = feature_df[ feature_df['daytime'] == 1100 ]
daytime_mcor = df_daytime.drop(['date', 'daytime'],axis=1).corr()
daytime_mcor.style.background_gradient(cmap='coolwarm')


# En esta ocasión podemos observar con mayor claridad las correlaciones entre los predictores dado que agrupamos los datos durante el día. A las claras se observa que el consumo de las luces esta directamente correlacionado con el consumo general del hogar

# ## Mapa de correlación durante la noche

# In[ ]:


df_nighttime = feature_df[ feature_df['daytime'] != 1100 ]
nighttime_mcor = df_nighttime.drop(['date', 'daytime'],axis=1).corr()
nighttime_mcor.style.background_gradient(cmap='coolwarm')


# En esta ocasión podemos observar con mayor claridad las correlaciones entre los predictores dado que agrupamos los datos durante la noche. A las claras se observa que el consumo de las luces esta directamente correlacionado con el consumo general del hogar

# ## Consumo energético durante los días de semana
# Aquí buscaremos determinar si existe algún tipo de patrón en el consumo energético del hogar basados en el día de la semana. Para ello, graficaremos el consumo en un plano coordenado, y superpondremos barras verticales diferenciadas por un color para cada día de la semana.

# In[ ]:


# Columnas a agregar para indicar el dia de la semana
weekdays = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']

# Indice de los dias se la semana, siendo Lunes igual a 0 y domingo igual a 6.
# Utilizado para identificar el dia de la semana desde la propiedad dayofweek
weekdaysnum = [0,1,2,3,4,5,6] 
for a,b in zip(weekdays, weekdaysnum):  
    feature_df[a] = [1100 if i.dayofweek == b else 0 for i in feature_df['date']]


# In[ ]:


# Observamos las filas para el lunes
feature_df[feature_df['Lunes'] == 1100]


# In[ ]:


# Observamos las filas para el martes
feature_df[feature_df['Martes'] == 1100]


# In[ ]:


# Creamos una nuevo grafico base
fig = make_subplots()

# Agregamos el grafico del consumo a lo largo del tiempo
fig.add_trace(
    go.Scatter(x=feature_df['date'], y=feature_df['Appliances'], 
               name="Consumo",
               mode='lines')
)

# Por cada uno de los dias de la semana, agregamos una franza de un color
for i in weekdays:
    fig.add_trace(
        go.Scatter(x=feature_df['date'], y=feature_df[i],
                   name=i,
                   mode='none',
                   fill='tonexty'))

# Actualizamos el titulo y la referencia en los ejes
fig.update_layout(
    title='Consumo por dia de la semana',
    xaxis_title="Fecha",
    yaxis_range=(0, 1100)
)
fig.update_yaxes(title_text="Consumo (in Wh)", color="blue")

fig.show()


# In[ ]:


# Creamos una columna nueva para indicar el dia de la semana, siendo 0 el lunes, y 6 el domingo
feature_df['weekday'] = [i.dayofweek for i in feature_df['date']]

for a,b in zip(weekdays, weekdaysnum):
    feature_df['weekday'] = feature_df['weekday'].replace(b, a)

# Ahora creamos una nueva columna para indicar el numero de la semana correspondiente en el a#o calendario
feature_df['week'] = feature_df['date'].dt.isocalendar().week

# Finalmente creamos un nuevo data frame, en el cual agrupamos el consumo por semana calendario y dias de la semana
heatdf = pd.DataFrame(feature_df.groupby(['week', 'weekday'])['Appliances'].sum()).reset_index()

heatdf.info()


# In[ ]:


# Creamos un nuevo grafico
fig = make_subplots()

# Agregamos el mapa de calor, donde el eje x sera la semana calendario, el eje y los dias de la semana
# y el eje z (el calor) el consumo energetico
fig.add_trace(
    go.Heatmap(x=heatdf['week'], y=heatdf['weekday'], z=heatdf['Appliances'],
               colorbar=dict(title='Consumo (en Wh)'))
)

# Actualizamos el layout del grafico
fig.update_layout(
    title='Consumo energetico total por dia de la semana',
    xaxis_title="Semana numero x del año",
    yaxis={'categoryarray': weekdays}
)

fig.update_yaxes(title_text="Dia de la semana")

fig.show()


# <span style="color:red;"> Como observación de las gráficas anteriores, confirmamos entonces que no se aprecia ningún patrón de consumo claro relacionado con el día de la semana. </span> 

# ### Consumo energético durante los fines de semana
# Intentaremos corroborar si existe algún tipo de patrón relacionado al consumo energético y los días laborales (lunes a viernes) o de fin de semana (sábado y domingo). Para ello recurriremos al mismo método utilizado en el paso anterior: un gráfico donde distingimos las franjas en cuestión, y un mapa de calor.

# ### Gráfico por franjas

# In[ ]:


# Creamos dos nuevas columnas: dia de semana y fin de semana.
weekorweekend = ['DiaDeSemana', 'FinDeSemana']
weeknums = [[0,1,2,3,4], [5,6]]

for a,b in zip(weekorweekend, weeknums):
    feature_df[a] = [1100 if i.dayofweek in b else 0 for i in feature_df['date']]


# In[ ]:


# Comprobacion visual de la columna dia de semana
feature_df[feature_df['DiaDeSemana'] == 1100]


# In[ ]:


# Comprobacion visual de la columna fin de semana
feature_df[feature_df['FinDeSemana'] == 1100]


# In[ ]:


# Creamos un nuevo grafico
fig = make_subplots()

# Agregamos el trazo para el consumo 
fig.add_trace(
    go.Scatter(x=feature_df['date'], y=feature_df['Appliances'], 
               name="Consumo",
               mode='lines')
)

# Por cada nueva columna, graficamos una franja vertical para indicar que nos encontramos en ese lugar
for i in weekorweekend:
    fig.add_trace(
        go.Scatter(x=feature_df['date'], y=feature_df[i],
                   name=i,
                   mode='none',
                   fill='tonexty'))

# Actualizamos la referencia de los ejes y el titulo
fig.update_layout(
    title='Consumo energetico por dia de semana o fin de semana',
    xaxis_title="Date",
    yaxis_range=(0, 1100)
)
fig.update_yaxes(title_text="Consumo (en Wh)", color="blue")
fig.show()


# Sobre el gráfico "Consumo energético por día de semana o fin de semana", se puede concluir que prácticamente no hay diferencia entre los días de la semana con los fines de semana.

# In[ ]:


feature_df['es_diadesemana'] = ['DiaDeSemana' if i.dayofweek in weeknums[0] else 'FinDeSemana' for i in feature_df['date']]

# Agregamos una columna para indicar el dia de la semana en el ano calendario
feature_df['week'] = feature_df['date'].dt.isocalendar().week

# Antes de poder agrupar el consumo por semana calendario y grupo (fin de semana o dia de semana), debemos estandarizar los valores
# De lo contrario, estaremos comparando 5 dias de consumo de la semana con 2 dias del fin de semana dentro de la misma semana calendario

# Creamos una copia de la columand de consumo, para luego estandarizar sus valores
feature_df['stdappliances'] = feature_df['Appliances']

# Utilizamos dos mascaras para distinguir filas correspondientes a dias de semana o fin de semana
mask1 = feature_df['es_diadesemana'] == 'DiaDeSemana'
mask2 = feature_df['es_diadesemana'] == 'FinDeSemana'

feature_df.loc[mask1, 'stdappliances'] = feature_df['Appliances'].mask(mask1, feature_df['Appliances'] * (2/7))
feature_df.loc[mask2, 'stdappliances'] = feature_df['Appliances'].mask(mask2, feature_df['Appliances'] * (5/7))

# Creamos un nuevo dataframe, agrupando por semana calendario y grupo
heatdf = pd.DataFrame(feature_df.groupby(['week', 'es_diadesemana'])['stdappliances'].sum()).reset_index()

heatdf.info()


# In[ ]:


# Creamos un nuevo grafico
fig = make_subplots()


# Agregamos el mapa de calor, donde el eje x sera la semana calendario, el eje y si es dia de semana o fin de semana
# y el eje z (el calor) el consumo energetico
fig.add_trace(
    go.Heatmap(x=heatdf['week'], y=heatdf['es_diadesemana'], z=heatdf['stdappliances'],
               colorbar=dict(title='Consumo (en Wh)'
               )))

# Actualizamos la referencias en los ejes y el titulo
fig.update_layout(
    title='Total de consumo por dia de semana o fin de semana',
    xaxis_title="Semana del año"
)

fig.show()


# En este mapa de calor tampoco apreciamos ningún tipo de patrón significativo. Debemos aclarar que los datos de consumo fueron estandarizados para una correcta comparación visual en magnitudes. La estandarización fue realizada en proporciones de 2/7 para los días de semana y 5/7 para los fines de semana.

# ### Distribución de consumo energetico
# Aqui analizaremos como se encuentra distribuido el consumo energético utilizando un gráfico lineal y uno logarítmico.

# In[ ]:


# Creamos una columna donde aplicamos una funcion logaritmica a los consumos
df_dist = df.copy()
df_dist['logappliances'] = df['Appliances'].apply(lambda x: np.log2(x+1))

# Creamos dos graficos
fig, ax = plt.subplots(1,2, figsize=(16,6))


sns.histplot(x='Appliances', data=df_dist, binwidth=20, ax=ax[0])
sns.histplot(x='logappliances', data=df_dist, binwidth=0.5, ax=ax[1])

ax[0].set_title('Distribucion de consumo')
ax[1].set_title('Distribucion logaritmica de consumo')

plt.show()


# ---
# 
# # Modelizado (Modeling)

# 
# ## Data frame completo
# ### Regresión lineal con luces
# 

# In[ ]:


X = df[['lights']].copy()
y = df['Appliances']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

print("Coeficiente de determinacion (R^2) entrenamiento: ", model.score(X_train, y_train))
print('Coeficiente de determinación (R^2): ', model.score(X_test, y_test))
print('Pendiente: ', model.coef_)


# In[ ]:


# Generar predicciones
y_pred = model.predict(X_test)

results = {
    'Prediccion': y_pred[:10],
    'Realidad': y_test[:10]
}

diff = [x-y for x,y in zip(results['Prediccion'], results['Realidad'])]
results['Resultado'] = diff

compara = pd.DataFrame(results)

compara.head(10)


# In[ ]:


mae = mean_absolute_error(y_pred,y_test)
print('El MAE para este regresion lineal es: ', round(mae, 3))


# In[ ]:


plt.figure(figsize=(12,8))
plt.plot(X_test, y_test, 'o', label='Realidad')
plt.plot(X_test, y_pred, 'r-', label='Predicción')
plt.xlabel('Cantidad de luces encendidas')
plt.ylabel('Consumo energetico')
plt.legend(loc='upper left')
plt.show()


# Se puede observar que utilizando el modelo de regresión lineal utilizando el consumo de las luces (Wh), la predicción no es precisa, utilizando el dataset original.

# ### Regresión lineal con todos los predictores

# In[ ]:


X = df.drop(['Appliances', 'date'], axis=1).copy()
y = df['Appliances'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

print("Coeficiente de determinacion (R^2) entrenamiento: ", model.score(X_train, y_train))
print('Coeficiente de determinación (R^2): ', model.score(X_test, y_test))
print('Pendiente: ', model.coef_)


# In[ ]:


# Generar predicciones
y_pred = model.predict(X_test)

results = {
    'Prediccion': y_pred[:10],
    'Realidad': y_test[:10]
}

diff = [abs(x-float(y)) for x,y in zip(results['Prediccion'], results['Realidad'])]
results['Resultado'] = diff

compara = pd.DataFrame(results)

compara.head(10)


# In[ ]:


mae = mean_absolute_error(y_pred,y_test)
print('El MAE para este regresion lineal es: ', round(mae, 3))


# 
# ## Data frame diurno
# 
# 
# ### Regresión lineal con luces
# En este caso, se utiliza un dataset con los valores que corresponden al **horario diurno**, de esta manera separando claramente el dataset en dos partes, diurno y nocturno.

# In[ ]:


X = df_daytime[['lights']].copy()
y = df_daytime['Appliances']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

print("Coeficiente de determinacion (R^2) entrenamiento: ", model.score(X_train, y_train))
print('Coeficiente de determinación (R^2): ', model.score(X_test, y_test))
print('Pendiente: ', model.coef_)


# In[ ]:


# Generar predicciones
y_pred = model.predict(X_test)

results = {
    'Prediccion': y_pred[:10],
    'Realidad': y_test[:10]
}

diff = [x-y for x,y in zip(results['Prediccion'], results['Realidad'])]
results['Resultado'] = diff

compara = pd.DataFrame(results)

compara.head(10)


# In[ ]:


mae = mean_absolute_error(y_pred,y_test)
print('El MAE para este regresion lineal es: ', round(mae, 3))


# ### Regresión lineal con todos los predictores
# 

# In[ ]:


X = df_daytime.drop(['Appliances', 'date'], axis=1).copy()
y = df_daytime['Appliances'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

print("Coeficiente de determinacion (R^2) entrenamiento: ", model.score(X_train, y_train))
print('Coeficiente de determinación (R^2): ', model.score(X_test, y_test))
print('Pendiente: ', model.coef_)


# In[ ]:


# Generar predicciones
y_pred = model.predict(X_test)

results = {
    'Prediccion': y_pred[:10],
    'Realidad': y_test[:10]
}

diff = [abs(x-float(y)) for x,y in zip(results['Prediccion'], results['Realidad'])]
results['Resultado'] = diff

compara = pd.DataFrame(results)

compara.head(10)


# In[ ]:


mae = mean_absolute_error(y_pred,y_test)
print('El MAE para este regresion lineal es: ', round(mae, 3))


# Del procedimiento anterior se desprende que el modelo de regresión lineal múltiple permite predecir en mayor medida el consumo energético futuro. Sin embargo, su performance aun sigue siendo baja.

# 
# ### Regresión lineal con predictores de temperatura y luces
# 
# Vamos a crear un modelo usando todos solo los predictores de consumo energético de las luces y la temperatura de las habitaciones de disponibles para el horario que definimos como diurno.

# In[ ]:


x = df_daytime[['lights', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']].copy()
y = df_daytime['Appliances'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

print("Coeficiente de determinacion (R^2) entrenamiento: ", model.score(X_train, y_train))
print('Coeficiente de determinación (R^2): ', model.score(X_test, y_test))
print('Pendiente: ', model.coef_)


# In[ ]:


# Generar predicciones
y_pred = model.predict(X_test)

results = {
    'Prediccion': y_pred[:10],
    'Realidad': y_test[:10]
}

diff = [abs(x-float(y)) for x,y in zip(results['Prediccion'], results['Realidad'])]
results['Resultado'] = diff

compara = pd.DataFrame(results)

compara.head(10)


# In[ ]:


mae = mean_absolute_error(y_pred,y_test)
print('El MAE para este regresion lineal es: ', round(mae, 3))


# ## Random Forest
# 
# Ya hemos observado que los resultados utilizando los metodos de regresión no se acercan a valores utiles, por lo que se harán pruebas utilizando un algoritmo de clasificación, particularmente el algortimo de **random forest**.
# Un modelo Random Forest está formado por un conjunto de árboles de decisión individuales, cada uno entrenado con una muestra aleatoria extraída de los datos de entrenamiento originales.
# Esto implica que cada árbol se entrena con unos datos ligeramente distintos. En cada árbol individual, las observaciones se van distribuyendo por bifurcaciones o nodos, generando la estructura del árbol hasta alcanzar un nodo terminal. La predicción de una nueva observación se obtiene agregando las predicciones de todos los árboles individuales que forman el modelo.
# A continuación, una [Definicion de Random Forest](https://www.cienciadedatos.net/documentos/py08_random_forest_python.html) que ilustra su funcionamiento y casos de uso.
# 
# 

# In[ ]:


#y_axe = y.copy()
#X_axe = x.copy()
x_axe = df_daytime[['lights', 'T1', 'T2', 'T3']].copy()
y_axe = df_daytime['Appliances'].copy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_axe, y_axe, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200, random_state=0)

classifier.fit(X_train, y_train)

RandomForestClassifier(n_estimators=200, random_state=0)

predictions = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))


# In[ ]:


from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = predictions,
        squared = False
       )
print(f"El error (rmse) de test es: {rmse}")


# Finalmente, en la seccion de <span style="color:red;"> Modelizado </span> , se probaron varios modelos de predicción que nos permiten concluir que con el dataset trabajado, ni el modelo de regresión ni tampoco el de árbol de decisiones, lograron predecir con exito un resultado .

# ---
# # Cierre y conclusiones

# El trabajo realizado sobre este dataset consistió en tres etapas princiaples: exploración, selección de predictores y modelado. Cada una de ellas enfocada en trabajar en un aspecto en particular.
# 
# Durante la primera etapa, nos enfocamos en explorar los datos, intentando determinar patrones, agrupamiento de datos, franjas horarias donde se observe un patrón definido. Además, nos dedicamos a eliminar posibles valores que contaminen nuestro dataset, dismueyendo la efectividad de los modelos de predicción que utilizaríamos en etapas posteriores. 
# 
# Durante la segunda etapa, demostramos que existe una marcada correlación entre el consumo energético y la utilización de la iluminación del hogar. Además, quedó plasmada la importancia de separar nuestro dataset en dos grandes grupos de datos: los diurnos y los nocturnos. Las gráficas permiten observar con contundencia que existen dos patrones diferentes cuando se divide el dataset en los dos grupos mencionados anteriormente. Todo este análisis es de suma importancia a la hora de aplicar los modelos de machine learning, dado que permiten que los algoritmos obtengan una mejor performance a la hora de hacer predicciones. 
# 
# Durante la tercera y ultima etapa, dos tipos de algoritmos fueron elegidos entres los varios existentes en el mundo del data science: una regresión lineal, en su variantes simple y múltiple, y un random forest. El motivo por el cual elejimos estos dos se debe a la simplicidad de su ejecución y la aceptable performance que han demostrado a lo largo de las clases de la materia. Los distintos tipos de dataset (completo, diurno, nocturno) fueron utilizados para entrenar los modelos. Los mejores resultados fueron obtenidos empleando los datasets diurno y nocturno. A pesar de ello, en ningún caso el resultado es suficiente para poder predecir consumos energéticos con suficiente certeza.
# 
# En las siguientes lineas intentaremos esbozar algunas hipótesis para explicar los motivos por los que creemos que los resultados no fueron exitosos. Como premisa mas factible, pensamos que el dataset posee datos que no forman un patrón de consumo. Observamos que existen dias durante la semana donde no hay consumo, o el valor es extremadamente bajo. Podriamos suponer que algunos de esos días son feriados, vacaciones o días festivos. El dataset en sí mismo no etiqueta los días según estas categorías, por lo que la tarea de limpiar posibles valores no representativos se dificulta. Por otro lado, aunque no menos importante, existe la posibilidad de fallas en el análisis previamente realizado a causa de la inexperiencia de los integrantes del grupo en el area. Creemos que aún existe margen para mejorar la performance de nuestro estudio, ya sea a través de algoritmos mas sofisticados, como redes neuronales, o a través de una análisis mas exahustivo de los datos, que permita encontrar ciertos patrones no observados por el equipo.
# 
# Dejamos asi planteado nuestro trabajo, dando lugar futuras revisiones del mismo, las cuales permitan mejorar los resultados obtenidos.
