#!/usr/bin/env python
# coding: utf-8

# ### Base-line соревнования (версия 3)
# ---
# ### **Open ML Course: Линейные модели** (весна - 2022)
# [ссылка на соревнование](https://ods.ai/tracks/linear-models-spring22/competitions/alice)
# 
# #### Цель третьей версии: реализовать пример кросс-валидации модели 

# In[ ]:


# Import libraries and set desired options
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


# Notebook by Yuri Kashnitsky, edited by Ivan Komarov. 
# 
# In this competition we are going to analyze a sequence of websites visited by a person to predict whether this person is Alice or not. The metric of evaluation is [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic). 

# ###  Data Downloading and Transformation
# First, read the training and test sets. 

# In[ ]:


times = ['time'+str(i) for i in range(1,11)]
times


# In[ ]:


### !!!внимание!!! эта ячека добавлена в бейзлайн для инициализации путей к датасетам и файлам
CURRENT_DIR = '../'  # имя текущей директории для каггл

PATH_TO_WORKDIR = CURRENT_DIR + 'working/'

PATH_TO_TRAIN = CURRENT_DIR + 'input/open-ml-course-linear-models-spring22/'


# In[ ]:


# Read the training and test data sets and parse dates
train_df = pd.read_csv(PATH_TO_TRAIN + 'train.csv',
                       index_col='session_id', parse_dates=times)

test_df = pd.read_csv(PATH_TO_TRAIN + 'test.csv',
                      index_col='session_id', parse_dates=['time1'])

# Sort the data by time
train_df = train_df.sort_values(by='time1')

# Look at the first rows of the training set
train_df.head()


# In[ ]:


train_df.info()


# The training data set contains the following features:
# 
# - **site1** – ID of the first visited website in the session
# - **time1** – visiting time for the first website in the session
# - ...
# - **site10** – ID of the tenth visited website in the session
# - **time10** – visiting time for the tenth website in the session
# - **target** – target variable, equals 1 for Alice's sessions, and 0 otherwise
#     
# **User sessions end either if a user has visited ten websites or if a session has lasted over thirty minutes.**
# 
# There are some empty values in the table, it means that some sessions contain less than ten websites. Replace empty values with 0 and change columns types to integer. Also load the websites dictionary and check how it looks:

# In[ ]:


# Change site1, ..., site10 columns type to integer and fill NA-values with zeros
sites = ['site'+str(i) for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

# Load websites dictionary
with open(PATH_TO_TRAIN + 'site_dic.pkl', 'rb') as input_file:
    site_dict = pickle.load(input_file)
    
# r before a string means "raw", i.e. take the string as it comes,
# e.g. as a file path without interpreting special symbols like \n

print('Websites total:', len(site_dict))



# In[ ]:


# See what's in the dict
list(site_dict.items())[:3]


# In[ ]:


# Size of the sets
print(test_df.shape, train_df.shape)


# In[ ]:


# What's inside the train
train_df.head()


# For the very basic model, we will use only the visited websites in the session (we will not take into account timestamp features). 
# 
# *Alice has her favorite sites, and the more often you see these sites in the session, the higher probability that this is an Alice session, and vice versa.*
# 
# Let us prepare the data, we will take only features `site1, site2, ... , site10` from the whole dataframe. Keep in mind that the missing values are replaced with zero. Here is what the first rows of the dataframe look like:

# In[ ]:


train_df[sites].head()


# Since IDs of sites have no meaning (does not matter if a site has an ID of 1 or 100), we need to think about how to encode the meaning of "this site in a session means higher probablity that it is an Alice session". 
# 
# We will use a technique called ["bag of words plus n-gram model"](https://en.wikipedia.org/wiki/Bag-of-words_model).
# 
# We will make a "site-session" matrix analogous to the term-document matrix.
# 
# We are not the first, and luckily there is a function CountVectorizer that will implement the above model. Type help(CountVectorizer) to learn about the function. 

# We will now initialize a "cv" (CountVectorizer's) instance which we need to train. 
# 
# We will use the following parameters:
# 
# _ngram range=(1, 3)_ - here we decide that we will use 
# 1) the name of the site, 
# 2) two consecutive site names, and 
# 3) three consecutive site names as features. 
# E.g. "google.com" or "google.com vk.com" or "google.com vk.com groups.live.com". 
# 
# CountVectorizer will create a large dictionary of 1, 2, and 3-gram strings of sites represented by their numerical IDs. However, this dictionary will be so so large that we may run into trouble with memory or we will just be inefficent chasing phantom combinations.
# 
# We will thus limit the dictionary to 50K of the most frequent n-grams:
# 
# _max features=50000_
# 
# Here is our empty instance:

# In[ ]:


cv = CountVectorizer(ngram_range=(1, 3), max_features=50000)


# CountVectorizer accepts "document strings", so let's prepare a string of our "documents" (i.e. sites), divided by space. Since the string will be huge, we will write this string in a text file using pandas:

# In[ ]:


train_df[sites].fillna(0).to_csv('train_sessions_text.txt', 
                                 sep=' ', index=None, header=None)
test_df[sites].fillna(0).to_csv('test_sessions_text.txt', 
                                sep=' ', index=None, header=None)


# Before we start using CountVectorizer, let's see how it works on a sub-set of 5 sessions:

# In[ ]:


five_sess = pd.read_csv('train_sessions_text.txt', sep=' ', nrows=5, header=None)


# In[ ]:


five_sess


# First of all, let's make an inverse dictionary which gives us a site name for ID.
# The direct dictionary came to us like this:

# In[ ]:


list(site_dict.items())[:3]


# In[ ]:


# The inverse dictionary:

new_dict = {}
for key in site_dict:
    new_dict[site_dict[key]] = key


# In[ ]:


# Let's check what's in it:

list(new_dict.items())[:3]


# In[ ]:


# Let's see site names in the five first sessions:

list_sites = []
for row in five_sess.values:
    row_sites = ' '.join([str(i) for i in row if i!=0])
    print(row_sites)
    list_sites.append(row_sites) 

print()
    
list_sites_names = []
for row in five_sess.values:
    row_sites = ' '.join([new_dict[i] for i in row if i!=0])
    print(row_sites)
    list_sites_names.append(row_sites)


# Here is what the fit and transform method -- i.e. learn the dictionary and make the matrix -- produces in our "cv":
# a sparse matrix. Why sparse? Because nrows * dict_size = usually will not fit in memory 
# (obviously, our 5 sessions will fit in memory so that we can look at them)

# In[ ]:


see_vect = cv.fit_transform(list_sites)

# Matrix dimensions: 5 sessions of 60 elements
see_vect


# In[ ]:


# Here is the dictionary of sites, 1 to 3-gram words. First 6 elements in the matrix:

cv.get_feature_names()[:6]


# In[ ]:


# A version with the site names. Note that security.debian.org has ID of 21.

for i, string in enumerate(cv.get_feature_names()):
    if i < 21:
        print (i+1, end=" ")
        for num in string.split():
            print(new_dict[int(num)], end=" ")
        print()


# In[ ]:


# Here is the session-site matrix, toarrray() helps us to see a sparse matrix since it is not large.

see_vect.toarray()


# In[ ]:


# The first session (row in the matrix) is this:

list_sites_names[0]


# Let's see how the first site of the first session, "security.debian.org", is recorded in the session-site matrix. 
# Its ID is 21 which corresponds to 3. It is the number of times this site was seen in the first session.
# Indeed, count for yourself in the cell above. 

# In[ ]:


first_row = see_vect.toarray()[0]

for one, two in zip(range(60),first_row):
    if one < 21:
        print (one+1, two)


# Let's go back to all sessions.

# Fit `CountVectorizer` to train data and transform the train and test data with it.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nwith open('train_sessions_text.txt') as inp_train_file:\n    X_train = cv.fit_transform(inp_train_file)\nwith open('test_sessions_text.txt') as inp_test_file:\n    X_test = cv.transform(inp_test_file)\n\nprint(X_train.shape, X_test.shape)\n\n# Note very big dimensions of matrices: 253561 * 50000 = 12678050000 elements in train! Only sparse matrices can take it.\n")


# **!!!Ахтунг!!!** во втором бейзлайне тут было добавление двух новых признаков.  
# Но давайте сначала обучим модель без новых признаков, замерим ее результаты на кросс-валидации. 
# А уже потом добавим новые признаки и сравним 

# ### Training the first model
# 
# So, we have an algorithm and data for it. Let us build our first model, using [logistic regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) implementation from ` Sklearn` with default parameters. We will use the first 90% of the data for training (the training data set is sorted by time) and the remaining 10% for validation. Let's write a simple function that returns the quality of the model and then train our first classifier:

# In[ ]:


def get_auc_lr_valid(X, y, C=1.0, seed=17, ratio = 0.9):
    
    # Split the data into the training and validation sets
    idx = int(round(X.shape[0] * ratio))
    
    # Classifier training
    lr = LogisticRegression(C=C, random_state=seed, solver='lbfgs', max_iter=500).fit(X[:idx, :], y[:idx])
    
    # Prediction for validation set
    y_pred = lr.predict_proba(X[idx:, :])[:, 1]
    
    # Calculate the quality
    score = roc_auc_score(y[idx:], y_pred)
    
    return score, lr


# In[ ]:


# Our target variable
y_train = train_df['target'].values


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Calculate metric on the validation set. 90% of train data for training. 10% for validation.\nscore, lr_1 = get_auc_lr_valid(X_train, y_train)\nprint(score)\n')


# In[ ]:


# 50% of train data for training:
score, _ = get_auc_lr_valid(X_train, y_train, ratio=0.5)
print(score)


# ### А теперь давайте проверим нашу модель на кросс-валидации  
# **!!!Ахтунг!!!** ниже добавлются несколько ячеек с кросс-валидацией (версия 3 бейзлайна)

# In[ ]:


from sklearn.model_selection import TimeSeriesSplit, cross_val_score

time_split = TimeSeriesSplit(n_splits=10)

def cv_get_auc_lr_valid(X, y, lr):
    '''
    функция проводит кросс-валидацию на заданном тренировочном фрейме X, таргете y и заранее обученной модели
    на выходе - массив ROC-AUC на каждом из 4-рех фолдах 
    '''
    cv_scores = cross_val_score(lr, X, y, cv=time_split, 
                            scoring='roc_auc', n_jobs=2)
    
    return cv_scores


# Представление о том как работает TimeSeriesSplit кросс-валидация на картинке ниже  
# ![image.png](attachment:41036910-0621-45ca-b185-777816d2c74e.png)  
# и по [ссылке](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html?highlight=timeseriessplit#sklearn.model_selection.TimeSeriesSplit)  
# почему лучше применять TimeSeriesSplit для этой задачи - для самостоятельного изучения

# In[ ]:


cv_scores_1 = cv_get_auc_lr_valid(X_train, y_train, lr_1)


# In[ ]:


np.round(cv_scores_1,4), f'Среднее значение по 10-ти фолдам: {round(cv_scores_1.mean(),4)}'


# Видно, что в среднем на 10-и фолдах мы получили метрику ROC-AUC = 0.8464. Видно что есть такие фолды, где точность нашей модели достаточно низкая.

# In[ ]:


# Function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = range(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# In[ ]:


# Make a prediction for test data set
y_test = lr_1.predict_proba(X_test)[:, 1]

# Write it to the file which could be submitted
write_to_submission_file(y_test, 'baseline_1_sokaa.csv')


# ### Add time features
# !!! Ахтунг!!! Ниже добавлены 4 ячейки в первый бейзлайн с кодом добавления новых признаков для примера (изменения реализованные во второй версии [бейзлайна](https://www.kaggle.com/code/sokolovaleks/alice-open-ml-course-baseline-2)

# In[ ]:


from scipy.sparse import hstack

def add_time_features(times, X_sparse):
    '''
    фукция добавления признака - принадлежность к вечеру
    '''
    hour = times['time1'].apply(lambda ts: ts.hour)
    
    evening = ((hour >= 19) & (hour <= 23)).astype('int').values.reshape(-1, 1)
    
    objects_to_hstack = [X_sparse, evening]
                
    X = hstack(objects_to_hstack)
    return X.tocsr()


# In[ ]:


train_times, test_times = train_df[times], test_df[times]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train_with_times = add_time_features(train_times, X_train)\nX_test_with_times = add_time_features(test_times, X_test)\n')


# In[ ]:


X_train_with_times.shape, X_test_with_times.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Calculate metric on the validation set. 90% of train data for training. 10% for validation.\nscore, lr_2_with_time = get_auc_lr_valid(X_train_with_times, y_train)\nprint(score)\n')


# In[ ]:


# в первой версии бейзлайна до добавления новых признаков ROC AUC был равен 0.9122
# после добавления двух новых признаков получили 0.9138 (+0.0016)


# In[ ]:


# 50% of train data for training:
score, _ = get_auc_lr_valid(X_train_with_times, y_train, ratio=0.5)
print(score)


# In[ ]:


# в первой версии бейзлайна до добавления  ОДНОГО нового признака ROC AUC на 50% train был равен 0.8225
# после добавления ОДНОГО нового признака получили 0.8244 (+0.0019)


# А теперь давайте проверим нашу модель2 с новым признаком на кросс-валидации  

# In[ ]:


cv_scores_2 = cv_get_auc_lr_valid(X_train_with_times, y_train, lr_2_with_time)


# In[ ]:


np.round(cv_scores_2,4), f'Среднее значение по 10-ти фолдам: {round(cv_scores_2.mean(),4)}'


# In[ ]:


cv_scores_2 > cv_scores_1


# Видно что на всех 10-ти фолдах наша метрика выросла. Таким образом мы можем оставить данный набор НОВЫХ признаков и подбирать следующие.

# In[ ]:


# Make a prediction for test data set
y_test = lr_2_with_time.predict_proba(X_test_with_times)[:, 1]

# Write it to the file which could be submitted
write_to_submission_file(y_test, 'baseline_2_with_evening_sokaa.csv')


# This model (add features) demonstrated the quality of 0,925894 on LB (leader bord)

# Давайте теперь добавим еще один признак

# In[ ]:


from scipy.sparse import hstack

def add_time_features(times, X_sparse):
    '''
    фукция добавления ДВУХ признаков - принадлежность к утру, вечеру
    '''
    hour = times['time1'].apply(lambda ts: ts.hour)
    
    morning = ((hour >= 7) & (hour <= 11)).astype('int').values.reshape(-1, 1)
    evening = ((hour >= 19) & (hour <= 23)).astype('int').values.reshape(-1, 1)
    
    objects_to_hstack = [X_sparse, morning, evening]
                
    X = hstack(objects_to_hstack)
    return X.tocsr()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train_with_times2 = add_time_features(train_times, X_train)\nX_test_with_times2 = add_time_features(test_times, X_test)\n')


# In[ ]:


X_train_with_times2.shape, X_test_with_times2.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Calculate metric on the validation set. 90% of train data for training. 10% for validation.\nscore, lr_3_with_time = get_auc_lr_valid(X_train_with_times2, y_train)\nprint(score)\n')


# А теперь давайте проверим нашу модель3 с новыми 2-мя признаками на кросс-валидации  

# In[ ]:


cv_scores_3 = cv_get_auc_lr_valid(X_train_with_times2, y_train, lr_3_with_time)


# In[ ]:


np.round(cv_scores_3,4), f'Среднее значение по 10-ти фолдам: {round(cv_scores_3.mean(),4)}'


# In[ ]:


cv_scores_3 > cv_scores_2


# In[ ]:


# Make a prediction for test data set
y_test = lr_3_with_time.predict_proba(X_test_with_times2)[:, 1]

# Write it to the file which could be submitted
write_to_submission_file(y_test, 'baseline_3_with_evening_morning_sokaa.csv')


# Таким образом с помощью кросс-валидации можно отвечать на вопрос на сколько полученная модель стабильна и на сколько новые фичи хороши. Но это конечно не едиственные способы проверки этих утверждений.
# Теперь вы можете подбирать новые признаки и растить скор. Удачи
# 
# Кстати вы можете добавить часы как было во втором кернеле и проверить почему они оверфитят (приводят к переобучению) модель 

# !!! Ахтунг!!! Я специально пытался выбрать признаки, которые не сильно увеличивают скор, чтобы тем ребятам которые самостоятельно строили решение было не обидно. 
# Но это решение уже около 0.95 на LB. Если немного доработать, то легко можно перелететь через следующую планку
# 

# I hope you like my kernel. Thanks for upvote  
# 
# Надеюсь ноутбук вам понравился.  
# Не стесняйтесь писать вопросы лично и в комментариях. Также буду рад любым замечаниям и предложениям по улучшению ноута.  
# Ну и если ноутбук показался вам интересным или полезным, то я буду благодарен вам за апвойт[1]  
# [1] Апвойт - кнопка со стрелкой вверх чуть левее от черной кнопки Edit в хедере (наверху с названием) этого кернела :)  
