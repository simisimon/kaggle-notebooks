#!/usr/bin/env python
# coding: utf-8

# # Hello again, Kagglers! Hope you will like my work

# ## [Dataset](https://www.kaggle.com/datasets/ilyaryabov/fasttext-model-for-google-ai4code) with a pretrained models
# ## [Notebook](https://www.kaggle.com/ilyaryabov/fasttext-public-model-teaching) that describes how to create a fasttext model

# # The following notebook describes an algorithm for sorting cells using cosine distance in vector space

# In[ ]:


import numpy as np
import pandas as pd
import os
import re
import fasttext
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from pathlib import Path

pd.options.display.max_rows = 100


# In[ ]:


src = '/kaggle/input/AI4Code/'
data_dir = Path('../input/AI4Code')
fasttext_model = '/kaggle/input/fasttext-model-for-google-ai4code/model140000.bin'


# In[ ]:


src = '../input/AI4Code/'
train_orders_df = pd.read_csv(src + 'train_orders.csv')


# In[ ]:


stemmer = WordNetLemmatizer()

def preprocess_text(document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()
        #return document

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    
def preprocess_df(df):
    """
    This function is for processing sorce of notebook
    returns preprocessed dataframe
    """
    return [preprocess_text(message) for message in df.source]


# In[ ]:


model = fasttext.load_model(fasttext_model)


# In[ ]:


test_files = paths_test = list((data_dir / 'test').glob('*.json'))
#dfs = [ (pd.read_json(file)) for file in test_files]


# # Example of applying fasttext model:

# In[ ]:


df = pd.read_json(test_files[0])
df.source = df.source.apply(preprocess_text)
df


# # Some useful functions:

# In[ ]:


def check(result, file):
    """
    This function shows how notebook looks with predicted cell order
    returns nothing
    """
    notebook_df = pd.read_json(file,
                                dtype={'cell_type': 'category', 'source': 'str'}
                                ).rename_axis('cell_id')
    cells = result
    df = notebook_df.loc[cells]
    display(df)


# In[ ]:


def read_notebook_from_train_orders(file):
    """
    This function reads a notebook from i-th line of train_orders.csv with a correct cell order
    """
    id_ = file.split('/')[-1][:-5]
    _, cell_order = train_orders_df[train_orders_df.id == id_].values[0] #train_orders_df.iloc[i]
    path = src + 'train/' + id_ +'.json'
    cell_order = cell_order.split( )
    #print(cell_order)
    notebook_df = pd.read_json(
                            path,
                            dtype={'cell_type': 'category', 'source': 'str'}
                            ).rename_axis('cell_id')
    return notebook_df.loc[cell_order], cell_order # put cells in a correct cell order



def visualize_corr_matrix(true_table, axs, k):
    true_table.source = true_table.source.apply(preprocess_text)
    true_vectors = []
    for i in range(len(true_table)):
        sentence = true_table.source[i]
        sentence = preprocess_text(sentence)
        vector = model.get_sentence_vector(sentence)
        true_vectors.append(vector)
    matrix2 = cosine_similarity(true_vectors, true_vectors)
    axs[k].imshow(matrix2)


# In[ ]:


### Functions to avaluate the result: ###

def count_inversions_slowly(ranks):
    inversions = 0
    size = len(ranks)
    for i in range(size):
        for j in range(i+1, size):
            if ranks[i] > ranks[j]:
                total += 1
    return total

from bisect import bisect

# Actually O(N^2), but fast in practice for our data
def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):  # O(N)
        j = bisect(sorted_so_far, u)  # O(log N)
        inversions += i - j
        sorted_so_far.insert(j, u)  # O(N)
    return inversions

def kendall_tau(ground_truth, predictions):
    total_inversions = 0  # total inversions in predicted ranks across all instances
    total_2max = 0  # maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


# # The first mardown in the very first cell is often some greetings, desriptions of the following notebok or forks to another notebooks. It is not related to the following code cell and it lies far from it in a vector space, thus the model will not place it correctly.
# ## Let's hace a look at such markdowns:

# In[ ]:


train_orders = pd.read_csv(src + 'train_orders.csv')
first_markdowns = []

for i in tqdm(range(10000)):
    id_, cell_order = train_orders.iloc[i]
    cell = (cell_order.split(' ')[0])
    first_cell = (pd.read_json(src + 'train/' + id_ + '.json').loc[cell])
    if first_cell.cell_type == 'markdown':
        first_markdowns.append(first_cell.source)
        
print(len(first_markdowns))
first_markdowns[:5]


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=2,
        random_state=1).generate(str(data))
    fig = plt.figure(1, figsize=(18, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)
    plt.imshow(wordcloud)
    plt.show()


# In[ ]:


print("Word cloud for markdowns in the very beginning of notebooks:")
show_wordcloud(first_markdowns)


# In[ ]:


serch_list = ['hello', 'kagglers', 'this notebook', 'kaggle', 'welcome', 
              'competition', 'kernel', 'introduction', 'data analysis', 
              'https', 'data science', 'nbsp'] # These words almost sure stands in the markdowns at the very beginning of a notebook


# In[ ]:


def find_first_markdown(df):
    """
    This function finds a cell that should stand in the very beginning of notebooks
    retunrs cell index and a key word
    """
    table = df[df.cell_type == 'markdown']
    for word in serch_list:
        for i, line in enumerate(table.source):
            if re.search(word, line.lower()):
                return table.index[i], word
    return None, None


# # Example below shows how we can find such markdown from dataframe
# ## Markdown "This notebook illustrate how to speedup..." should definitely stay in the beginning

# In[ ]:


df = pd.read_json(test_files[2])
cell, word = find_first_markdown(df)
print(cell, word)
df.tail(1)


# # Main algorithm

# In[ ]:


def overall_algo(file, to_check=False, to_plot = False, train = False):
    """
    This is the main sorting algorithm
    It computes matrix of cosine distance between cells in a vector space, 
    and than it sorts them by finding the perfect matches between markdown and code cells
    returns the cell order
    """
    df = (pd.read_json(file))
    df.source = df.source.apply(preprocess_text)
    vectors = []

    for i in range(len(df)):
        sentence = df.source[i]
        sentence = preprocess_text(sentence)
        vector = model.get_sentence_vector(sentence)
        vectors.append(vector)

    matrix = cosine_similarity(vectors, vectors)
    if to_plot:
        f, axs = plt.subplots(1,3,figsize=(21,7)) # if you try the algo on a train dataset with true cell order
        #f, axs = plt.subplots(1,2,figsize=(20,10)) # if you try the algo on a test dataset you have just two graphs
        #axs[0].figure(figsize=(8,8))
        axs[0].imshow(matrix)
        axs[0].title.set_text('Initial Table')

    
    ### True notebook ###
    #id_ = file.split('/')[-1][:-5]
    if train:
        true_table, cell_order = read_notebook_from_train_orders(str(file))

    if to_plot:
        visualize_corr_matrix(true_table, axs, 2)
        axs[2].title.set_text('True Table')
     
    
    indexes_code = list(df[df.cell_type == 'code'].index)
    indexes_markdown = list(df[df.cell_type == 'markdown'].index)

    result = indexes_code.copy()

    #indexes_code , indexes_markdown
    N = len(indexes_code)
    K = len(indexes_markdown)
    #print(N, K)

    order = dict.fromkeys(np.arange(K))



    #### ALGORITHM WORKABLE #########

    markdowns_submatrix = matrix[N:,:-K].copy()
    order = dict.fromkeys(np.arange(K))
    indexes_to_order = list(np.arange(K))

    # find the markdown in the very beginning of the notebook and place it there
    cell, word = find_first_markdown(df)
    if cell != None:
        idx = indexes_markdown.index(cell)
        order[idx] = 0
        markdowns_submatrix[idx] = np.zeros(N)
        indexes_to_order.remove(idx)
        
        
    initial_markdowns_submatrix = markdowns_submatrix.copy()

    for i in range(K):
        for i in range(K):
            if i in indexes_to_order:
                most_similar_code = np.argmax(markdowns_submatrix[i])
                most_similar_markdown = np.argmax(markdowns_submatrix.T[most_similar_code])
                if most_similar_markdown == i:
                    #print(i, ' congrats')
                    order[i] = most_similar_code
                    indexes_to_order.remove(i)
                    markdowns_submatrix[most_similar_markdown] = np.zeros(N)
                    markdowns_submatrix[:, most_similar_code] = np.zeros(K)
        if (np.max(markdowns_submatrix) == 0): # stop creteria - all cells are sorted
            break
        markdowns_submatrix_old = markdowns_submatrix.copy()


    # if some mardowns left (number of mardown cells is greater than codes)
    for key, value in zip(indexes_to_order, np.argmax(initial_markdowns_submatrix[indexes_to_order], axis=1)):
        order[key] = value 

    indexes_code = list(df[df.cell_type == 'code'].index)
    indexes_markdown = list(df[df.cell_type == 'markdown'].index)

    result = indexes_code.copy()

    for i in range(K-1, -1, -1):
        markdown_cell = indexes_markdown[i]
        corresponding_code_cell = indexes_code[order[i]]
        new_index = result.index(corresponding_code_cell)
        result.insert(new_index, markdown_cell)

    if to_check:
        print("Algorithm: ")
        check(result, file)
        print("True notebook: ")
        #table, gt = read_notebook_from_train_orders(id_)
        display(true_table)
        
    if to_plot:
        visualize_corr_matrix(df.loc[result], axs, 1)
        axs[1].title.set_text('My Result')
    return result


# # An example of algorithm with graph:

# In[ ]:


file =  src + 'train/0001daf4c2c76d.json'
file =  src + 'train/000bbb79a2fe3c.json' # not bad
file =  src + 'train/000c0a9b2fef4d.json' # far not bad


predictions = overall_algo(file, to_check = True, to_plot = True, train = True)


# # Accuracy = 87% for table of 70 rows with 21 markdowns

# In[ ]:


true_table, ground_truth = read_notebook_from_train_orders(str(file))
score = kendall_tau([ground_truth], [predictions])
print(f"score = {score}")


# In[ ]:


df = (pd.read_json(file))
len(df), len(df[df.cell_type == 'markdown'])


# # Accuracy evaluation

# In[ ]:


def compare_algo_with_gt(file, to_check=True):
    id_ = file.split('/')[-1][:-5]
    if to_check:
        print(i, file, id_)
        print("Algorithm: ")
    algo_result = overall_algo(file, to_check=to_check)
    if to_check:
        print("True notebook: ")
    table, gt = read_notebook_from_train_orders(file)
    if to_check:
        display(table)
    return algo_result, gt


# In[ ]:


train_dir = '../input/AI4Code/train/'
train_files = os.listdir(train_dir)


# In[ ]:


predictions = []
ground_truth = []

N = 140

for i in tqdm(range(N)):
    file = train_dir + train_files[i]
    algo_result, cell_order = compare_algo_with_gt(file, to_check=False)
    ground_truth.append(cell_order)
    predictions.append(algo_result)

final_score = kendall_tau(ground_truth, predictions)
print(f"the accuracy on {N/1400}% of the data is {round(final_score, 3)}")


# ## 71.8% for now

# In[ ]:


df_result = pd.DataFrame(columns = ['id', 'cell_order'])
#df_example = pd.read_csv(src + 'sample_submission.csv')

for i, file in enumerate(test_files):
    print(i, file)
    id_ = str(test_files[i]).split('/')[-1][:-5]
    df = pd.read_json(file)
    example = df.index.tolist()
    result = overall_algo(file)
    try:
        result = overall_algo(file)
        if len(result) != len(df):
            result = ' '.join(example) #df_example.cell_order[i] # if result is weird (can't imagine how it's possible)
        elif (len(set(result) - set(example)) != 0) or (len( set(example) - set(result) ) != 0 ):
            result = ' '.join(example) #df_example.cell_order[i]
        else:
            result = ' '.join(result)
    except:
        result = ' '.join(example) # if algo failed put from example # never happened but just in case to avoid submission error 
      
    to_add = result
    df_result = df_result.append({'id':id_, 'cell_order':to_add}, ignore_index=True)

df_result


# In[ ]:


df_result.to_csv('submission.csv', index = False)


# In[ ]:


#cat submission.csv


# In[ ]:





# In[ ]:




