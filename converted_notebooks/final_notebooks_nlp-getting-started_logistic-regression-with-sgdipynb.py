#!/usr/bin/env python
# coding: utf-8

# Twitter, acil durumlarda önemli bir iletişim kanalı haline geldi. Akıllı telefonların her yerde bulunması, insanların gözlemledikleri bir acil durumu gerçek zamanlı olarak duyurmalarını sağlar. Fakat dilimizdeki eş anlamlardan dolayı bazı tweetler herhangi bir acil durumu içermez. Problemimiz ise atılan tweetlerin gerçekten bir acil durumu içerip içermemediğini anlamaktır.
# ___
# Bu notebook'ta Twitter'da insanların attığı tweetler ve tweetlerin felaket olup olmadığı verisini içeren veriseti Stokastik Gradyan İnişli Lojistik Regresyon sınıflandırma modeli kullanılarak eğitilmiş daha sonra felaket verisi olmayan test veriseti ile bir tweetin felaket içerip içermediği tahmin edilmeye çalışılmıştır. 
# ___
# Benzer çalışmalar:
# - [SGD ile Destek Vektör Makineleri kullanımı](https://www.kaggle.com/code/naim99/tweet-classification-with-sgd)
# 
# - [Sade Lojistik Regresyon kullanımı](https://www.kaggle.com/code/chiragsidhdhapura/basic-eda-tf-idf-and-logistic-regression)
# 
# - [BERT kelime yerleştirme yönteminin kullanımı](https://www.kaggle.com/code/kirillklyukvin/fake-tweets-competition)

# # Kütüphane yükleme

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt # Grafikler için matplotlib modülü.
import nltk # Doğal dil işleme için kullanılan kütüphane
from collections import Counter # Saydırma işlemi için counter yüklendi.
from sklearn.feature_extraction.text import  TfidfVectorizer # Metinleri makinenin anlayacağı dile çevirmek için vektörlere çevirmeliyiz. Bu yüzden bir ham doküman koleksiyonunu TF-IDF özelliklerinin bir matrisine dönüştürmek için TfidfVectorizer kullanılır.
# TF: Term Frequency=terim frekansı: bir dokümanda belirli bir terimden kaç tane olduğunu bulur. / IDF: Inverse Document Frequency=Ters doküman frekansı: belirli terimin kaç farklı dokümanda geçtiğini hesaplar. 
# Tf-idf ise belirli terimin kaç farklı dokümanda kaç kere geçtiğine önem verir. Bu yüzden tf-idf kullanıldı.
from nltk.stem import WordNetLemmatizer # kelimelerin köküne indirgenmesi için kullanılır.
from nltk.tokenize import RegexpTokenizer # Cümleleri makine için daha anlaşılır yapmak için bölmek için kullanılır.


# # Veri içeri aktarma

# In[ ]:


data_train = pd.read_csv('../input/nlp-getting-started/train.csv') # Metin verilerinin olduğu kısım okutulur.
data_test = pd.read_csv('../input/nlp-getting-started/test.csv') # hangi metnin hangi kategoriye ait olduğu veriler okutulur.
data_submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
data_train


# In[ ]:


print("Train data\n", data_train.isna().sum(),"\n\nTest data\n",data_test.isna().sum(),"\n\nSubmission data\n",data_submission.isna().sum())


# Verilerimiz içerisinde:
#    - her tweetin eşsiz olduğunu belirten bir 'id' özelliği,
#    - tweetin anahtar kelimesini belirten bir 'keyword' özelliği,
#    - konumu belirten bir 'location' özelliği,
#    - tweet metnini içeren bir 'text' özelliği,
#    - tweetin felaket olup(1) olmadığını(0) belirten bir 'target' özelliği bulunmaktadır.

# In[ ]:


fig = plt.figure(figsize=(10,5))
data_train.groupby('target').text.count().plot.bar()

plt.xlabel("Hedef değerleri",fontsize=15)
plt.ylabel("Adet", fontsize=15)
plt.title("Felaket(1) ve Felaket Olmayan(0) Tweet Sayısı", fontsize=20)
plt.grid(True, color='b', axis='y')
plt.show()

# Veriler fazla dengesiz olmadığı için verileri yeniden örneklendirmeye gerek yoktur.


# In[ ]:


plt.figure(figsize = (10, 5))
ax = plt.axes()
ax = ((data_train.location.value_counts())[:10]).plot(kind = 'bar')
plt.title('Tweetlerin Geldiği Yerler', fontsize = 20)
plt.xlabel('Yer', fontsize = 15)
plt.ylabel('Adet', fontsize = 15)
plt.plot()


# In[ ]:


# kelime bulutu: külliyatta en çok geçen 500 kelime
from wordcloud import WordCloud # kelime bulutu için

long_string = ','.join(list(data_train.text.values))
wordcloud = WordCloud(background_color="white", max_words=500, width=700, height=500, contour_width=10, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# # Veri temizleme

# In[ ]:


data_train['text'] = data_train['text'].str.replace(r'w/e', 'whatever', regex=True)
data_train['text'] = data_train['text'].str.replace(r'w/', 'with', regex=True)
data_train['text'] = data_train['text'].str.replace(r'USAgov', 'USA government', regex=True)
data_train['text'] = data_train['text'].str.replace(r'recentlu', 'recently', regex=True)
data_train['text'] = data_train['text'].str.replace(r'Ph0tos', 'Photos', regex=True)
data_train['text'] = data_train['text'].str.replace(r'amirite', 'am I right', regex=True)
data_train['text'] = data_train['text'].str.replace(r'exp0sed', 'exposed', regex=True)
data_train['text'] = data_train['text'].str.replace(r'<3', 'love', regex=True)
data_train['text'] = data_train['text'].str.replace(r'Trfc', 'Traffic', regex=True)
data_train['text'] = data_train['text'].str.replace(r'TRAUMATISED', 'traumatized', regex=True)
data_train['text'] = data_train['text'].str.replace(r'won\'t', 'will not', regex=True)
data_train['text'] = data_train['text'].str.replace(r"won\'t've", 'will not have', regex=True)
data_train['text'] = data_train['text'].str.replace(r'can\'t', 'can not', regex=True)
data_train['text'] = data_train['text'].str.replace(r'don\'t', 'do not', regex=True)
data_train['text'] = data_train['text'].str.replace(r"can\'t've", 'can not have', regex=True)
data_train['text'] = data_train['text'].str.replace(r'\'re', 'are', regex=True)
data_train['text'] = data_train['text'].str.replace(r'\'m', 'am', regex=True)
data_train['text'] = data_train['text'].str.replace(r'\'ll', 'will', regex=True)
data_train['text'] = data_train['text'].str.replace(r'\'t', 'not', regex=True)
data_train['text'] = data_train['text'].str.replace(r'\'v', 'have', regex=True)
data_train['text'] = data_train['text'].str.replace(r'\'u', 'you', regex=True)
data_train['text'] = data_train['text'].str.replace(r'_', '', regex=True)


# In[ ]:


# gereksiz kelimeler atılır.
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer() # Kelimeleri köküne indirgemek için lemmatizer kullanılır. 

tokenizer = RegexpTokenizer(r'\w+') # Cümleleri küçük ifadelere böler değişkeni tanımlanır.
punct_re=lambda x :" ".join(tokenizer.tokenize(x.lower())) # Verileri küçük harflere dönüştür değişkeni tanımlanır.

nltk.download('stopwords') # nltk'den stopwords indirilir. Veri ne kadar sade olsa o kadar iyidir bundan dolayı cümleye anlam katmayan gereksiz kelimelerden(stopword) kurtulmak istenir.

stop_word_list = nltk.corpus.stopwords.words('english')

# stopword'leri çıkarmak için fonksiyon
def stopword_extraction(values):
    wordFilter = [word for word in values.split() if word not in stop_word_list]
    notStopword = " ".join(wordFilter)
    return notStopword

data_train["text"]=data_train["text"].apply(punct_re) # metin verileri küçük harflere çevir
data_train["text"]=data_train["text"].apply(lambda x : " ".join([lemmatizer.lemmatize(w) for w in x.split()])) # metin verilerini ayır köklerine indirge
data_train['text'] = data_train['text'].apply(lambda x: stopword_extraction(x)) # metin verilerinden stopword'leri çıkar

# data_train['text'] = data_train['text'].str.replace('\d+', '') # inputta bulunan sayıları sil.
data_train['text'] = data_train['text'].str.replace(r'\S*@\S*\s?', '', regex=True) # e-posta sil
data_train['text'] = data_train['text'].str.replace(r'[^\w\s]', '', regex=True) # noktalama işaretlerini sil
data_train['text'] = data_train['text'].str.replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', regex=True) # bağlantı sil


# modelin anlamsız sonuçlar doğurmasını engellemek için herhangi bir kelimeye bağlı olmayan kısa kelimeler çıkarılır.
remove_words =['co', 'ha', 'wa', 'û','amp', 'via','rt','v','r','n', 'u', 'i am']

sil = r'\b(?:{})\b'.format('|'.join(remove_words))
data_train['text'] = data_train['text'].str.replace(sil, '', regex=True)

data_train


# In[ ]:


long_string = ','.join(list(data_train.text.values))
wordcloud = WordCloud(background_color="white", max_words=500, width=700, height=500, contour_width=10, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# # Model

# In[ ]:


dataDoc = data_train['text'].values.tolist() # Metin verileri girdi olarak seçilir.
dataClass = data_train['target'].values.tolist() # kategori verileri çıktı olarak seçilir.

# Tfidf skorlama yöntemini kullanarak veriyi sayısallaştırmadan önce eğitim ve test olarak veriyi ayırıyoruz.
#x_train, x_valid, y_train, y_valid = train_test_split(dataDoc, dataClass, test_size = 0.2, random_state = 42)

#tfidf işlemi
tfidf_vectorizer = TfidfVectorizer(min_df=3) # min_df: Nadiren görünen terimleri göz ardı etmek için kullanılır. Şu anda bir terim 3 dokümandan az geçiyorsa göz ardı edilecek.

# metin verileri makinenin anlayacağı dile vektörlere dönüştürülür.
x_train_tfidf = tfidf_vectorizer.fit_transform(dataDoc)
x_test_tfidf = tfidf_vectorizer.transform(data_test.text)


# Sınıflandırma problemi olduğu için sınıflandırma yöntemi kullanıldı. 
# 
# Stokastik Gradyan Düşüşü (SGD), (doğrusal) Destek Vektör Makineleri ve Lojistik Regresyon gibi dışbükey kayıp fonksiyonları altında doğrusal sınıflandırıcıları ve regresörleri uydurmak için basit ama çok verimli bir yaklaşımdır. Açıkçası, SGD yalnızca bir optimizasyon tekniğidir ve belirli bir makine öğrenimi modeli ailesine karşılık gelmez. Bu sadece bir modeli eğitmenin bir yoludur.
# 
# Stokastik Gradyan İnişinin avantajları şunlardır:
# - Etkinlik.
# - Uygulama kolaylığı (kod ayarlama için birçok fırsat).
# 
# Stokastik Gradyan İnişinin dezavantajları şunları içerir:
# - SGD, düzenleme parametresi ve yineleme sayısı gibi bir dizi hiper parametre gerektirir.
# - SGD, özellik ölçeklendirmeye duyarlıdır.
# 
# [Kaynak](https://scikit-learn.org/stable/modules/sgd.html)
# 
# *Normalde StandardScaler kullanılması tavsiye ediliyor ama zaten Tf-idf kullandığım için ölçeklendirmeye gerek duymadım.*

# In[ ]:


from sklearn.linear_model import SGDClassifier # SGD'li Lojistik regresyon kullanmak için yüklenir.

lrsgd = SGDClassifier(loss="log", max_iter=1000, alpha=0.0001, random_state=42) 
lrsgd_clf = lrsgd.fit(x_train_tfidf, dataClass) # model eğitilir.
pred_test_lrsgd = lrsgd_clf.predict(x_test_tfidf) # tahmin yapılır


# In[ ]:


ids = data_test["id"]
submission_df = pd.DataFrame({"id": ids, "target": pred_test_lrsgd})
submission_df.reset_index(drop=True, inplace=True)


# In[ ]:


submission_df.to_csv("submission.csv", index=False)

