#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

##IMDB dataset
# imdb_path = '../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv'
# data = pd.read_csv(imdb_path)
# X = data['review']
# y = pd.Series(np.where(data['sentiment'].str.contains("positive"), 1, 0))

##Disaster tweets dataset
# dis_path = '../input/nlp-getting-started/train.csv'
# data  = pd.read_csv(dis_path)
# X = data['text']
# y = data['target']

##TripAdvisor reviews
# trip_path = '../input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv'
# data = pd.read_csv(trip_path)
# X = data['Review']
# y = pd.Series(np.where(data['Rating'] >3, 1, 0))

##News headlines sarcasm detection
news_path = '../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json'
data = pd.read_json(news_path, lines=True)
X = data['headline']
y = data['is_sarcastic']

print(data.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

num_words=2000
#print(train_data[0,])


# # TF-IDF
# To transform data I used [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
# The number of words to be used for the dictionary may be chosen with cross-validation, once built a model.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer( use_idf=True, max_features=num_words)
X_train_transformed = vectorizer.fit_transform(X_train).toarray()
X_test_transformed = vectorizer.transform(X_test).toarray()

print(type(X_test_transformed))
print("n_samples: %d, n_features: %d" % X_train_transformed.shape)
print("n_samples: %d, n_features: %d" % X_test_transformed.shape)


# # Logistic Regression
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logistic_regression_classifier = LogisticRegression()
logistic_regression_classifier.fit(X_train_transformed, y_train)

y_pred = logistic_regression_classifier.predict(X_test_transformed)

accuracy_score(y_pred,y_test)


# # Naive Bayes classifier

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_transformed, y_train)

y_pred = naive_bayes_classifier.predict(X_test_transformed)

accuracy_score(y_pred,y_test)


# # Dense classifier

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(input_dim = X_train_transformed.shape[1],units=16),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.summary()

history = model.fit(X_train_transformed, y_train, epochs=20,batch_size=1024,validation_data=(X_test_transformed,y_test))

results = model.evaluate(X_test_transformed, y_test)
print(results)

# model.save("IMDB_Sentiment_TFIDF.h5")


# In[ ]:


history_dict = history.history
history_dict.keys()


# In[ ]:


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

