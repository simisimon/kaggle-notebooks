#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import plotly.express as px
import seaborn as sns

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.ticker as ticker
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


credit = pd.read_csv('../input/disney-tv-shows-and-movies/credits.csv')
titles = pd.read_csv('../input/disney-tv-shows-and-movies/titles.csv')


# In[ ]:


titles.head(10)


# In[ ]:


credit.head(10)


# In[ ]:


pd.DataFrame(titles.count(), columns=["total"])


# In[ ]:


pd.DataFrame(credit.count(), columns=["total"])


# In[ ]:


import random

def get_random_color():
    r1 = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r1(),r1(),r1())


def get_histplot_central_tendency(df: dict, fields: list):
    for field in fields:
        f, (ax1) = plt.subplots(1, 1, figsize=(12, 6))
        v_dist_1 = df[field].values
        sns.histplot(v_dist_1, ax=ax1, color=get_random_color(), kde=True)

        mean=df[field].mean()
        median=df[field].median()
        mode=df[field].mode().values[0]

        ax1.axvline(mean, color='r', linestyle='--', label="Mean")
        ax1.axvline(median, color='g', linestyle='-', label="Mean")
        ax1.axvline(mode, color='b', linestyle='-', label="Mode")
        ax1.legend()
        #plt.grid()
        plt.plot(color="white", lw=3)
        plt.suptitle(f"{field} - Histogram analysis", fontsize=18)
        
def bar_plot_data(df: dict, field: str, subtitle: str, figsize=(5, 4), top_filter=15, df_filter=None):
    fig, ax1 = plt.subplots(figsize=figsize, dpi=100)
    
    for spline in ['top', 'right', 'left']:
        ax1.spines[spline].set_visible(False)
          
    if df_filter is None:
        df_filter = df[field].value_counts().rename_axis(field).reset_index(name='count')
    else:
        df_filter = df
        
    if top_filter:
        df_filter = df_filter.head(top_filter)
        
    sns.barplot(data=df_filter, palette='cool', x='count', y=field)
    ax1.tick_params(axis='both', which='both', labelsize=12, bottom=True, left=False)
    ax1.set_xlabel(f'count', fontsize=13, color = '#333F4B')
    ax1.set_ylabel(f'{field}', fontsize=13, color = '#333F4B')

    plt.plot(color="white", lw=3)
    fig.suptitle(subtitle, fontsize=18)
    plt.show()
    
def bar_count_plot(df: dict, title: str, color='Red'):
    fig, ax1 = plt.subplots(figsize=(25, 6), dpi=100)
    for spline in ['top', 'right', 'left']:
        ax1.spines[spline].set_visible(False)
        
    sns.barplot(data=df, palette='cool', x=df.index, y=df['release_year']) 
    ax1.tick_params(axis='both', which='both', labelsize=12, bottom=True, left=False)
    ax1.set_xlabel(f'Release year', fontsize=13, color = '#333F4B')
    ax1.set_ylabel(f'', fontsize=13, color = '#333F4B')
    
    plt.xticks(rotation=90)
    plt.gca().invert_xaxis()
    plt.plot(color="white", lw=3)
    fig.suptitle(title, fontsize=18)
    plt.show()


# In[ ]:


release_year = pd.DataFrame(titles['release_year'].value_counts())
bar_count_plot(release_year, 'Production years history')


# In[ ]:


titles['production_countries'] = titles['production_countries'].replace('[]', 'Unknown')


# In[ ]:


bar_plot_data(titles, "production_countries", "Production by country")


# In[ ]:


bar_plot_data(titles, "type", "Production by type")


# In[ ]:


bar_plot_data(titles, "title", "Production by title")


# In[ ]:


bar_plot_data(titles, "age_certification", "Production by age certification")


# In[ ]:


get_histplot_central_tendency(titles, ['release_year'])


# In[ ]:


get_histplot_central_tendency(titles, ['imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score'])


# In[ ]:


total_genre = dict()
for current in titles['genres']:
    current = current.replace("'", "").replace('"', "")
    current = list(current.replace('[', '').replace(']', '').split(","))
    for genre in current:
        if genre not in total_genre:
            total_genre[genre] = 0
        total_genre[genre] += 1
     
    
df_genre = {'genre': [], 'count': []}
total_genre = {k: v for k, v in sorted(total_genre.items(), reverse=True, key=lambda item: item[1])}
for i, j in total_genre.items():
    df_genre['genre'].append(i)
    df_genre['count'].append(j)
    
df_genre = pd.DataFrame.from_dict(df_genre)
df_genre.head(15)


# In[ ]:


def genre_plot(df_genre):
    fig, ax1 = plt.subplots(figsize=(5, 9), dpi=100)
    for spline in ['top', 'right', 'left']:
        ax1.spines[spline].set_visible(False)
    sns.barplot(data=df_genre, palette='cool', x='count', y='genre')
    ax1.tick_params(axis='both', which='both', labelsize=12, bottom=True, left=False)
    ax1.set_xlabel(f'count', fontsize=13, color = '#333F4B')
    ax1.set_ylabel(f'genre', fontsize=13, color = '#333F4B')

    plt.plot(color="white", lw=3)
    fig.suptitle("Production by genre", fontsize=18)
    plt.show()

genre_plot(df_genre)


# In[ ]:


movie_description = titles['title']
movie_imdb = titles['imdb_score']


# In[ ]:


movie_description['plot'] = movie_description.astype(str) + "\n" + movie_imdb.astype(str)
movies_df = movie_description


# In[ ]:


titles.head(1)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

features = ['title', 'type', 'genres', 'description', 'imdb_score', 'tmdb_popularity', 'tmdb_score']

def pre_processor_clean(titles):
    movies_title = titles.copy()
    for feature in features:
        movies_title[feature] = movies_title[feature].fillna('')
    return movies_title

def combine_features_string(current):
    rows = ""
    for feature in ['title', 'type', 'genres', 'description', 
                    'imdb_score', 'tmdb_popularity', 'tmdb_score']:
        rows += f'{current[feature]}\n '
    return rows
    

movies_title = pre_processor_clean(titles)
movies_title['features'] = movies_title.apply(combine_features_string, axis=1)
movies_title['features'].head(10)


# In[ ]:


vectorizer = CountVectorizer()
matrix_transform = vectorizer.fit_transform(movies_title["features"])
cosine_similiraty_rm = cosine_similarity(matrix_transform)


# In[ ]:


def get_index_using_title(title, movies):
    return movies[movies["title"] == title].index.values[0]

def select_movie(movies, movie, cosine_similiraty_rm, number_of_recommendations):
    similar_movies = list(enumerate(cosine_similiraty_rm[get_index_using_title(movies, movie)]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)[1:]
    sorted_similar_movies = sorted_similar_movies[0: number_of_recommendations]
    
    df_recommender = {"_id": [], "title": [], "description": [], "confidence": []}
    for i, similiraty_movie in enumerate(sorted_similar_movies):
        index_movie, confidence = similiraty_movie[0], similiraty_movie[1]
        filter_movie = movie[movie.index == index_movie].values[0]
        df_recommender["_id"].append(index_movie)
        df_recommender["title"].append(filter_movie[1])
        df_recommender["description"].append(filter_movie[3]) 
        df_recommender["confidence"].append(confidence)
    
    return pd.DataFrame(df_recommender)


# In[ ]:


cm = sns.light_palette("green", as_cmap=True)
movies_similiraty = select_movie("Cinderella", movies_title, cosine_similiraty_rm, 10)
movies_similiraty


# In[ ]:


def confidence_plot(movies_similirary):
    fig, ax1 = plt.subplots(figsize=(5, 9), dpi=100)
    for spline in ['top', 'right', 'left']:
        ax1.spines[spline].set_visible(False)
        
    sns.barplot(data=movies_similiraty, palette='cool', x='confidence', y='title')
    ax1.tick_params(axis='both', which='both', labelsize=12, bottom=True, left=False)
    #ax1.set_xlabel(f'confidence', fontsize=13, color = '#333F4B')
    ax1.set_ylabel(f'title', fontsize=13, color = '#333F4B')
    ax1.bar_label(ax1.containers[0], fontsize=13)
    plt.plot(color="white", lw=3)
    fig.suptitle("Recommendation movies", fontsize=18)
    plt.show()

confidence_plot(movies_similiraty)

