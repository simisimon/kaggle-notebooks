#!/usr/bin/env python
# coding: utf-8

# <center><img src="https://github.com/hse-ds/iad-applied-ds/blob/master/2021/hw/hw1/img/logo_hse.png?raw=1" width="1000"></center>
# 
# <h1><center>Прикладные задачи анализа данных</center></h1>
# <h2><center>Домашнее задание 4: рекомендательные системы</center></h2>

# # Введение
# 
# В этом задании Вы продолжите работать с данными из семинара [Articles Sharing and Reading from CI&T Deskdrop](https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop).

# # Загрузка и предобработка данных

# In[ ]:


import pandas as pd
import numpy as np
import math


# Загрузим данные и проведем предобраотку данных как на семинаре.

# In[ ]:


#!kaggle datasets download -d gspmoreira/articles-sharing-reading-from-cit-deskdrop
#!unzip articles-sharing-reading-from-cit-deskdrop.zip -d articles


# In[ ]:


articles_df = pd.read_csv("../input/articles-sharing-reading-from-cit-deskdrop/shared_articles.csv")
articles_df = articles_df[articles_df["eventType"] == "CONTENT SHARED"]
articles_df.head(2)


# In[ ]:


interactions_df = pd.read_csv("../input/articles-sharing-reading-from-cit-deskdrop/users_interactions.csv")
interactions_df.head(2)


# In[ ]:


interactions_df.personId = interactions_df.personId.astype(str)
interactions_df.contentId = interactions_df.contentId.astype(str)
articles_df.contentId = articles_df.contentId.astype(str)


# In[ ]:


# зададим словарь определяющий силу взаимодействия
event_type_strength = {
   "VIEW": 1.0,
   "LIKE": 2.0, 
   "BOOKMARK": 2.5, 
   "FOLLOW": 3.0,
   "COMMENT CREATED": 4.0,  
}

interactions_df["eventStrength"] = interactions_df.eventType.apply(lambda x: event_type_strength[x])


# Оставляем только тех пользователей, которые произамодействовали более чем с пятью статьями.

# In[ ]:


users_interactions_count_df = (
    interactions_df
    .groupby(["personId", "contentId"])
    .first()
    .reset_index()
    .groupby("personId").size())
print("# users:", len(users_interactions_count_df))

users_with_enough_interactions_df = \
    users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[["personId"]]
print("# users with at least 5 interactions:",len(users_with_enough_interactions_df))


# Оставляем только те взаимодействия, которые относятся к отфильтрованным пользователям.

# In[ ]:


interactions_from_selected_users_df = interactions_df.loc[np.in1d(interactions_df.personId,
            users_with_enough_interactions_df)]


# In[ ]:


print(f"# interactions before: {interactions_df.shape}")
print(f"# interactions after: {interactions_from_selected_users_df.shape}")


# Объединяем все взаимодействия пользователя по каждой статье и сглажиываем полученный результат, взяв от него логарифм.

# In[ ]:


def smooth_user_preference(x):
    return math.log(1+x, 2)
    
interactions_full_df = (
    interactions_from_selected_users_df
    .groupby(["personId", "contentId"]).eventStrength.sum()
    .apply(smooth_user_preference)
    .reset_index().set_index(["personId", "contentId"])
)
interactions_full_df["last_timestamp"] = (
    interactions_from_selected_users_df
    .groupby(["personId", "contentId"])["timestamp"].last()
)
        
interactions_full_df = interactions_full_df.reset_index()
interactions_full_df.head(5)


# Разобьём выборку на обучение и контроль по времени.

# In[ ]:


from sklearn.model_selection import train_test_split

split_ts = 1475519530
interactions_train_df = interactions_full_df.loc[interactions_full_df.last_timestamp < split_ts].copy()
interactions_test_df = interactions_full_df.loc[interactions_full_df.last_timestamp >= split_ts].copy()

print(f"# interactions on Train set: {len(interactions_train_df)}")
print(f"# interactions on Test set: {len(interactions_test_df)}")

interactions_train_df


# Для удобства подсчёта качества запишем данные в формате, где строка соответствует пользователю, а столбцы будут истинными метками и предсказаниями в виде списков.

# In[ ]:


interactions = (
    interactions_train_df
    .groupby("personId")["contentId"].agg(lambda x: list(x))
    .reset_index()
    .rename(columns={"contentId": "true_train"})
    .set_index("personId")
)

interactions["true_test"] = (
    interactions_test_df
    .groupby("personId")["contentId"].agg(lambda x: list(x))
)

# заполнение пропусков пустыми списками
interactions.loc[pd.isnull(interactions.true_test), "true_test"] = [
    "" for x in range(len(interactions.loc[pd.isnull(interactions.true_test), "true_test"]))]

interactions.head(1)


# # Библиотека LightFM

# Для рекомендации Вы будете пользоваться библиотекой [LightFM](https://making.lyst.com/lightfm/docs/home.html), в которой реализованы популярные алгоритмы. Для оценивания качества рекомендации, как и на семинаре, будем пользоваться метрикой *precision@10*.

# In[ ]:


#!pip install lightfm


# In[ ]:


from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset


# 

# ## Задание 1 (2 балла)

# Модели в LightFM работают с разреженными матрицами. Создайте разреженные матрицы `data_train` и `data_test` (размером количество пользователей на количество статей), такие что на пересечении строки пользователя и столбца статьи стоит сила их взаимодействия, если взаимодействие было, и стоит ноль, если взаимодействия не было.

# In[ ]:


import tensorflow
tensorflow.random.set_seed(10)


# In[ ]:


#https://making.lyst.com/lightfm/docs/lightfm.data.html взяла построение датасета из документации модели, которая потом предлагается для использования
dataset = Dataset()
dataset.fit(interactions_full_df.personId, interactions_full_df.contentId)

X = (interactions_train_df[["personId", "contentId", "eventStrength"]].apply(tuple,axis=1),interactions_test_df[["personId", "contentId", "eventStrength"]].apply(tuple,axis=1))


# In[ ]:


data_train = dataset.build_interactions(X[0])[1] #беру из массива выше первую часть, соответствующую трейну, и превращаю в матрицу
data_test = dataset.build_interactions(X[1])[1] #проделываю то же самое для теста


# ## Задание 2 (1 балл)

# Обучите модель LightFM с `loss="warp"` и посчитайте *precision@10* на тесте.

# In[ ]:


mlightFM = LightFM(k=10, loss = 'warp')
mlightFM.fit(data_train, epochs = 100) #вплоть до подбора параметров я старалась брать побольше эпох, чтобы наверняка...


# In[ ]:


precision_at_k(mlightFM, data_test, data_train, 10).mean()


# Качество откровенно грустное

# ## Задание 3 (3 балла)

# При вызове метода `fit` LightFM позволяет передавать в `item_features` признаковое описание объектов. Воспользуемся этим. Будем получать признаковое описание из текста статьи в виде [TF-IDF](https://ru.wikipedia.org/wiki/TF-IDF) (можно воспользоваться `TfidfVectorizer` из scikit-learn). Создайте матрицу `feat` размером количесвто статей на размер признакового описание и обучите LightFM с `loss="warp"` и посчитайте precision@10 на тесте.

# In[ ]:


df_new = pd.DataFrame(interactions_full_df.contentId.unique(), columns = ['contentId'])


# In[ ]:


adf = articles_df.copy()
ff = pd.merge(df_new, adf, on = 'contentId', how = 'left' ) #создаю датафрейм из статей, ранжированных через контентайд из датафрейма с интеракциями
ff['text'] = ff['text'].fillna('no text') #здесь просто пытаюсь заполнить пропуск хоть чем-то


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tr = TfidfVectorizer()

feat = tr.fit_transform(ff.text)


# In[ ]:


mlightFM2 = LightFM(k=10, loss = 'warp')
mlightFM2.fit(data_train, epochs = 100, item_features = feat)


# In[ ]:


precision_at_k(mlightFM2, data_test, data_train, 10, item_features = feat).mean()


# Веселее качество пока что не стало

# ## Задание 4 (2 балла)

# В задании 3 мы использовали сырой текст статей. В этом задании необходимо сначала сделать предобработку текста (привести к нижнему регистру, убрать стоп слова, привести слова к номральной форме и т.д.), после чего обучите модель и оценить качество на тестовых данных.

# In[ ]:


from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from string import punctuation
nltk.download("stopwords")
nltk.download('punkt')


# In[ ]:


#создаю новый датафрейм, включаю в него данные по айди, языку и, собственно, тексту статьи (заголовок решила опустить), заполняю пропуски в столбцах
adf_new = articles_df.copy()
adf_new['text'] = articles_df['text']
adf_new = adf_new[['contentId','lang','text']]
adf_new = pd.merge(df_new, adf_new, on = 'contentId', how = 'left' )
adf_new['text'] = adf_new['text'].fillna('unknown')
adf_new['lang'] = adf_new['lang'].fillna('no text')
adf_new.head()


# In[ ]:


adf_new.lang.value_counts() 


# In[ ]:


print(stopwords.fileids())


# nltk предлагает возможность взять готовый набор стоп-слов для английского, португальского и испанского
# испанский было принято решение не включать в набор стоп слов ввиду малого количества айтемов на испанском языке и потенциального пересечения испанских стоп-слов с португальскими качественными словами (хотя я пыталась, но это только ухудшало качество)
# латынь и японский за неимением готового набора стоп-слов at hand и малым количеством наблюдений остаются без стоп слов и очищаются только от пунктуации

# In[ ]:


en = stopwords.words('english') + list(punctuation) 
pt = stopwords.words('portuguese') + list(punctuation) 
#es = stopwords.words('spanish') + list(punctuation)


# In[ ]:


#!pip install langdetect


# In[ ]:


nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from langdetect import detect

lmtzr = WordNetLemmatizer()
stemmer = nltk.stem.RSLPStemmer()


# In[ ]:


#я решила попробовать по-разному работать с данными английского и португальского языка
#поэтому португальский язык подвергаю стеммингу при помощи стеммера, который, как мне сказали на stackoverflow, лучше подходит для испанского, а английский - леммизирую
def my_tokenizer(x):
    wt = word_tokenize(x)  
    if detect(x) == 'pt':  
        preprocessed = [stemmer.stem(word) for word in wt if word not in pt and word.isalpha()]  
    #elif detect(x) == 'es':
        #preprocessed = [SnowballStemmer('spanish').stem(word) for word in wt if word not in es and word.isalpha()]  
        #тут была жалкая попытка процессить испанский тоже, но она ухудшала качество... модель выдавала не больше 0.006 precision@k
    else:    
        preprocessed = [lmtzr.lemmatize(word) for word in wt if word not in en and word.isalpha()]
        #preprocessed = [SnowballStemmer('english').stem(word) for word in wt if word not in noise_en and word.isalpha()]
        #тут тоже запечатлеваю разные пробы пера в стемминге, но остановилась именно на лемматизации, а не стемминге с английским языком
    return preprocessed


# In[ ]:


stopwords_all = stopwords.words('portuguese') + list(punctuation) + stopwords.words('english') #+ stopwords.words('spanish')
tr2 = TfidfVectorizer(lowercase = True, tokenizer = my_tokenizer, stop_words = stopwords_all)
feat2 = tr2.fit_transform(adf_new['text'])


# In[ ]:


mlightFM3 = LightFM(k=10, loss = 'warp')
mlightFM3.fit(data_train, item_features = feat2, epochs = 100)


# In[ ]:


precision_at_k(mlightFM3, data_test, data_train, item_features = feat2, k = 10).mean()


# В сравнении с предыдущими моделями, качество улучшилось, однако незначительно - на доли процента. Предполагаю, что низкое значение precision@k может быть связано с отсутствием заголовка в тексте статьи, который мог бы позволить лучше обучаться. Рост качества может быть ограничен тем, что стеммингу не подвергаются японские и латинские айтемы, но, скорее всего, незначительно, поскольку в общей выборке их мало.

# ## Задание 5 (2 балла)

# Подберите гиперпараметры модели LightFM (`n_components` и др.) для улучшения качества модели.

# In[ ]:


# код беру целиком с https://stackoverflow.com/questions/49896816/how-do-i-optimize-the-hyperparameters-of-lightfm
import itertools



def sample_hyperparameters():
    """
    Yield possible hyperparameter choices.
    """

    while True:
        yield {
            "no_components": np.random.randint(10, 200), #чуть увеличила размах в сравнении с предложенным на сайте
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),  
            "loss": np.random.choice(["bpr", "warp", "warp-kos"]),
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-7),
            "user_alpha": np.random.exponential(1e-7),
            "max_sampled": np.random.randint(5, 50),
            "num_epochs": np.random.randint(5, 50),
        }

def random_search(train, test, num_samples=10, num_threads=1):
    """
    Sample random hyperparameters, fit a LightFM model, and evaluate it
    on the test set.

    Parameters
    ----------

    train: np.float32 coo_matrix of shape [n_users, n_items]
        Training data.
    test: np.float32 coo_matrix of shape [n_users, n_items]
        Test data.
    num_samples: int, optional
        Number of hyperparameter choices to evaluate.


    Returns
    -------

    generator of (precision, hyperparameter dict, fitted model)

    """

    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")

        model = LightFM(**hyperparams, k = 10, random_state = 0) 
        model.fit(train, epochs = num_epochs, num_threads=num_threads)

        score = precision_at_k(model, test, train, k = 10, num_threads=num_threads).mean()

        hyperparams["num_epochs"] = num_epochs

        yield (score, hyperparams, model)


# In[ ]:


(score, hyperparams, model) = max(random_search(data_train, data_test, num_threads = 2), key=lambda x: x[0])
print("Best score {} at {}".format(score, hyperparams))


# У меня были надежды на этот инструмент подбора, но тут все-таки не находится некоторый глобальный максимум по качеству, поэтому выводы подбора оказываются скорее наводкой, которую приходилось дорабатывать вручную. Скорее всего, я чего-то недопоняла и не доработала, но так или иначе, мне этот код в работе подсобил

# In[ ]:


m_final = LightFM(random_state = 0,
                   no_components = 100, 
                   learning_schedule = 'adagrad', 
                   loss = 'warp', 
                   learning_rate = 0.025, 
                   item_alpha =  1.4921059666702605e-09, 
                   user_alpha =  2.1366172295379337e-08,
                   max_sampled = 19)
m_final.fit(data_train, epochs = 26)

#немного видоизменяла параметры, предлагаемые механизмом подбора выше, в итоге остановилась на этих


# In[ ]:


precision_at_k(m_final, data_test, data_train, k = 10).mean()


# In[ ]:


m_final.fit(data_train, item_features = feat, epochs = 26)
precision_at_k(m_final, data_test, data_train, item_features = feat, k = 10).mean()


# In[ ]:


m_final.fit(data_train, item_features = feat2, epochs = 26)
precision_at_k(m_final, data_test, data_train, item_features = feat2, k = 10).mean()


# Лучшее качество получилось на модели с tf-idf но без серьезной предобработки
# около 0,0087

# ## Бонусное задание (3 балла)

# Выше мы использовали достаточно простое представление текста статьи в виде TF-IDF. В этом задании Вам нужно представить текст статьи (можно вместе с заголовком) в виде эмбеддинга полученного с помощью рекуррентной сети или трансформера (можно использовать любую предобученную модель, которая Вам нравится). Обучите модель с ипользованием этих эмеддингов и сравните результаты с предыдущими.

# In[ ]:


# Ваш код здесь

