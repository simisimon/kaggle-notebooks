#!/usr/bin/env python
# coding: utf-8

# # Building a car price prediction model for the CarDekho website
# # Part 2: Data preparation and model building
# ![car_dekho.jpg](attachment:car_dekho.jpg)
# ## About the project
# [CarDekho](https://www.cardekho.com/) is India's leading car search venture that helps users buy cars. Its website and app carry rich automotive content such as expert reviews, detailed specs and prices, comparisons as well as videos and pictures of all car brands and models available in India. The company has tie-ups with many auto manufacturers, more than 4000 car dealers and numerous financial institutions to facilitate the purchase of vehicles.
# 
# In this project, we'll collect data about used cars from the CarDekho website, build a price prediction model and deploy it in a web app. The app may later be used by CarDekho users for evaluating the price of put up for sale vehicles or exploring collected data on their own.
# 
# The data collection and model building are described in 2 notebooks:
# * [Part 1: Data collection and splitting into train and validation datasets](https://nbviewer.org/github/ZaikaBohdan/ds_car_price_proj/blob/main/car_price_part_1.ipynb)
# * Part 2: Data preparation and model building (current)
# 
# **The goal of the current notebook** is to build a car price prediction model based on the train dataset and evaluate its success on the validation dataset. We will achieve it by following the next steps:
# 1. Clean train dataset.
# 2. Do exploratory data analysis and feature engineering on the train data.
# 3. Train regression models on the train dataset.
# 4. Choose the best one and test it on the validation dataset.
# 
# ## Links
# * [GitHub repository of the project](https://github.com/ZaikaBohdan/ds_car_price_proj)
# * [Web App](https://share.streamlit.io/zaikabohdan/ds_car_price_proj/main/app/app.py)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

pd.options.display.max_columns = None


# # Importing datasets

# In[ ]:


raw_train = pd.read_csv('https://raw.githubusercontent.com/ZaikaBohdan/ds_car_price_proj/main/data/train.csv')
raw_valid = pd.read_csv('https://raw.githubusercontent.com/ZaikaBohdan/ds_car_price_proj/main/data/valid.csv')

# fixing data type of seats column
raw_train = raw_train.astype({'seats':str})
raw_train['seats'] = raw_train['seats'].str.replace('.0','', regex=False).replace({'nan':np.nan})
clean_train = raw_train.copy()

raw_valid = raw_valid.astype({'seats':str})
raw_valid['seats'] = raw_valid['seats'].str.replace('.0','', regex=False).replace({'nan':np.nan})

clean_train.head()


# In[ ]:


clean_train.info()


# In[ ]:


raw_valid.head()


# # Data cleaning of train dataset
# Let's start by taking a look at the number of fields with null values.

# In[ ]:


def null_counter(df):
    display(
        pd.DataFrame(
            df.isnull().sum(axis=1).value_counts().sort_index(ascending=False)
        ).reset_index().rename(columns={'index':'n_nulls', 0:'n_rows'})
    )
    
def n_nulls_cols_counter(df, n_nulls):
    print(
        df[df.isnull().sum(axis=1)==n_nulls].apply(lambda row: frozenset(df.columns[row.isnull()]), axis=1).value_counts()
    )

def all_null_cols_counter(df):
    uniq_n_nulls = sorted(df.isnull().sum(axis=1).unique(), reverse = True)[:-1]
    for n in uniq_n_nulls:
        print('>>> List of columns with null values, where amount of nulls =', n,'<<<')
        n_nulls_cols_counter(df, n)
        print()


# In[ ]:


null_counter(clean_train)


# In[ ]:


all_null_cols_counter(clean_train)


# First, let's deal with the null value in *km_driven*, which previously contained an outlier in Part 1 of the project. We will fill it with the mean *km_driven* of cars with 'Fourth & Above Owner', since it was the reason why we dropped this value.

# In[ ]:


clean_train.loc[clean_train['km_driven'].isnull(), 'km_driven'] = clean_train.loc[clean_train['owner']=='Fourth & Above Owner', 'km_driven'].mean()
all_null_cols_counter(clean_train)


# Now let's get the brand of car from the *name* column and impute the remaining null values with the mean (*max_power_bhp, engine_cc*) and mode (*seats*) of the corresponding brand cars in the code cells below.

# In[ ]:


clean_train['brand'] = clean_train['name'].str.split().str[0]
clean_train['brand'].unique()


# Next values in new column *brand* should be changed:
# * Land -> Land Rover
# * Mini -> Mini Cooper
# * OpelCorsa -> Opel
# * Isuzu -> ISUZU
# * Ashok -> Ashok Leyland
# * Mercedes-AMG -> Mercedes-Benz

# In[ ]:


replace_dict = {
    'Land' : 'Land Rover', 
    'Mini' : 'Mini Cooper',
    'OpelCorsa' : 'Opel',
    'Isuzu' : 'ISUZU',
    'Ashok' : 'Ashok Leyland',
    'Mercedes-AMG' : 'Mercedes-Benz'
}
clean_train['brand'] = clean_train['brand'].replace(replace_dict)
clean_train['brand'].unique()


# In[ ]:


def fillna_stat_by_col(data, col, fill_method):
    df = data.copy()
    unique_vals = df.loc[df.isnull().any(axis=1), col].unique()
    for val in unique_vals:
        if fill_method=='mean':
            df[df[col]==val] = df[df[col]==val].fillna(df[df[col]==val].mean(numeric_only=True))
        elif fill_method=='mode':
            df[df[col]==val] = df[df[col]==val].fillna(df[df[col]==val].mode().iloc[0,:])
        else:
            print(f"Error: method '{fill_method}' is incorrect")
            return
    
    return df


# In[ ]:


print('>>> Before filling null values')
null_counter(clean_train)
clean_train.loc[:, clean_train.columns != 'seats'] = fillna_stat_by_col(
    clean_train.loc[:, clean_train.columns != 'seats'],
    'brand',
    'mean'
)
print('\n>>> After filling null values in engine_cc and max_power_bhp')
null_counter(clean_train)
clean_train = fillna_stat_by_col(
    clean_train,
    'brand',
    'mode'
)
print('\n>>> After filling null values in seats')
null_counter(clean_train)


# Let's take a look at the last row with null values.

# In[ ]:


clean_train.loc[clean_train.isnull().any(axis=1), :]


# We will fill these null values with information from [ultimatespecs.com](https://www.ultimatespecs.com/car-specs/Peugeot/1974/Peugeot-309-GLD---GRD.html).

# In[ ]:


clean_train.loc[clean_train.isnull().any(axis=1), ['seats', 'engine_cc', 'max_power_bhp']] = ['5', 1905, 64]
null_counter(clean_train)


# We don't have null values left. Let's drop the *name* column since we won't need it anymore and move to exploratory data analysis.

# In[ ]:


fe_train = clean_train.drop(columns='name')
fe_train.columns


# # Exploratory data analysis
# In this section, we will explore data and find patterns, which may be useful for feature engineering later. Since the goal of the project is to build the car price prediction model, the main purpose of this chapter is to find relations between features and the target column (*selling_price_inr*).
# 
# ## *year* and *transmission* features
# Let's start by plotting the scatter plot of *year* and *selling_price_inr* with *transmission* marked by color.

# In[ ]:


fig, scatter = plt.subplots(figsize = (15,10))
scatter = sns.scatterplot(data=fe_train, x="year", y="selling_price_inr", hue="transmission")


# In general, the mean and deviation of car prices increase with each year, and cars with automatic transmission tend to be more expensive.
# ## *km_driven* and *seller_type* features
# The visualization below shows the relation between *km_driven, seller_type* and *selling_price_inr* columns.

# In[ ]:


fig, scatter = plt.subplots(figsize = (15,10))
scatter = sns.scatterplot(data=fe_train, x="km_driven", y="selling_price_inr",hue="seller_type", palette="deep")


# As can be seen, selling prices tend to go down with the rise of *km_driven*. Also, cars with 200000-400000 km driven are usually sold by individuals, but any clear pattern between *seller_type* and *selling_price_inr* columns is not observed.
# 
# ## *engine_cc* and *max_power_bhp* features
# The scatter plots below represent dependency between *engine_cc, max_power_bhp* and *selling_price_inr*.

# In[ ]:


def plot_col_signif(df, feat_cols, targ_col, n_rows=1, n_cols=1):
    fig, axes = plt.subplots(nrows = n_rows, ncols = n_cols, figsize=(6*n_cols,4*n_rows), squeeze=False, constrained_layout = True)

    for i in range(n_rows):
        for j in range(n_cols):
            try:
                feat_col = feat_cols[i*n_cols+j]
            except:
                plt.show()
                return
            
            pears_corr = df.corr()[feat_col][targ_col]
            spear_corr = df.corr(method='spearman')[feat_col][targ_col]

            axes[i,j].scatter(df[feat_col], df[targ_col])
            axes[i,j].set_xlabel('feature')
            axes[i,j].set_ylabel('target')
            axes[i,j].set_title(f'Feature: {feat_col}\nPearson\'s r = {pears_corr:.2f}\n Spearman\'s ρ = {spear_corr:.2f}')
            
    plt.show()


# In[ ]:


plot_col_signif(fe_train, ['engine_cc', 'max_power_bhp'], 'selling_price_inr', 1, 2)


# As it is observed, both features have a moderate-high correlation with the selling price.
# 
# ## *brand* feature
# We'll visualize the mean selling price grouped by *brand* values below.

# In[ ]:


def mean_per_col_barh(df, col):
    mean_price_per_col = df.groupby([col]).mean()['selling_price_inr'].sort_values(ascending=False)

    fig, barplot = plt.subplots(figsize = (15,10))
    barplot = sns.barplot(x=mean_price_per_col.values, y=mean_price_per_col.index, orient='h', palette="viridis")
    barplot.bar_label(barplot.containers[0])
    barplot.spines['top'].set_visible(False)
    barplot.spines['right'].set_visible(False)
    barplot.spines['bottom'].set_visible(False)
    barplot.get_xaxis().set_ticks([])
    barplot.set_title(f'Mean car price for each {col}')
    barplot.set_ylabel('')
    plt.show()


# In[ ]:


mean_per_col_barh(fe_train, 'brand')


# Brand influences the price, and the chart above confirms that it is true for our data.
# ## *seats, fuel, owner*  features
# The box plots below show the distribution of selling price values grouped by *seats, fuel* and *owner*.

# In[ ]:


fig, boxplot = plt.subplots(figsize = (15,10))
boxplot = sns.boxplot(
    data=fe_train,
    x='selling_price_inr', 
    y='seats',
    orient='h',
    palette="viridis"
)


# It can be seen that selling price values of cars with 2, 4, 5 and 7 seats have a big deviation, while others have price < 2,500,000 in most cases.

# In[ ]:


fig, boxplot = plt.subplots(figsize = (15,10))
boxplot = sns.boxplot(
    data=fe_train,
    x='selling_price_inr', 
    y='fuel',
    orient='h',
    palette="viridis"
)


# The graph above reveals that selling prices of cars with petrol and diesel fuel also have a big deviation, while others have price < 2,000,000.

# In[ ]:


fig, boxplot = plt.subplots(figsize = (15,10))
boxplot = sns.boxplot(
    data=fe_train,
    x='selling_price_inr', 
    y='owner',
    orient='h',
    palette="viridis"
)


# As was expected, the price of a car tends to decrease with each owner. Also, there is a huge difference in the deviation of the selling price between the first and second owners.
# 
# ## Conclusion
# As a result of exploratory data analysis, we find out that:
# * selling price mean and deviation have a positive correlation with the **year** column;
# * cars with automatic **transmission** tend to be more expensive;
# * car price mean and deviation tend to go down with the rise of the **km_driven** column;
# * any clear pattern between **seller_type** and *selling_price_inr* columns is not observed;
# * **engine_cc** has a moderate correlation with the selling price;
# * **max_power_bhp** has high correlation with car price;
# * prices of cars with 2, 4, 5 and 7 **seats** have big deviation, while others have price < 2,500,000 in most cases;
# * selling prices of cars with petrol and diesel **fuel** also have big deviation, while others have price < 2,000,000;
# * the price of a car tends to decrease with each **owner**, and there is a huge difference in the deviation of the selling price between the first and second owner.
# 
# # Feature Engineering
# Let's use the knowledge we gained during exploratory analysis to create new features.

# In[ ]:


fe_train.head()


# The *fuel, seats, owner* values have similar patterns with selling price, so we will analogically create new *..._flg* fields:
# * *fuel_flg* = 1 if fuel is diesel or petrol, 0 in other cases;
# * *seats_flg* = 1 if car has 2, 4, 5 or 7 seats, 0 in other cases;
# * *owner_flg* = 1 if car owner is first, 0 in other cases.
# 
# *seller_type* and *transmission* have only 2 unique values, so we will create new *..._flg* fields by replacing them with 0 and 1.

# In[ ]:


def col_to_col_flg(data, col, vals_1):
    df = data.copy()
    df[col+'_flg'] = df.apply(lambda row: int(row[col] in vals_1), axis=1)
    print(f'>>> Check correctness of {col}_flg <<<\n')
    print(df[col+'_flg'].value_counts(), '\n')
    print(df[col].value_counts(), '\n')
    return df


# In[ ]:


cols_vals_1_dict = {
    'fuel': ['Diesel', 'Petrol'],
    'seats': ['2', '4', '5', '7'],
    'owner': ['First Owner'],
    'seller_type': ['Individual'],
    'transmission': ['Manual']
}

for col, vals_1 in cols_vals_1_dict.items():
    fe_train = col_to_col_flg(fe_train, col, vals_1)


# The *brand* column has valuable information for our model but contains a lot of unique values, so using it without any data encoding may not be the best approach. We will create next new fields based on this column:
# * brand_top_half = 1 if a brand is in the top half means of selling price grouped by brand, 0 in other cases;
# * brand_top_third = 1 if a brand is in the top third means of selling price grouped by brand, 0 in other cases;
# * brand_bottom_third = 1 if a brand is in the bottom third means of selling price grouped by brand, 0 in other cases.

# In[ ]:


mean_price_per_brand = fe_train.groupby(['brand']).mean()['selling_price_inr'].sort_values(ascending=False)
n = len(mean_price_per_brand)

fe_train['brand_top_half'] = fe_train.apply(lambda row: int(row['brand'] in mean_price_per_brand.iloc[:n//2].index), axis=1)
fe_train['brand_top_third'] = fe_train.apply(lambda row: int(row['brand'] in mean_price_per_brand.iloc[:n//3].index), axis=1)
fe_train['brand_bottom_third'] = fe_train.apply(lambda row: int(row['brand'] in mean_price_per_brand.iloc[-n//3:].index), axis=1)

fe_train.head()


# As the last step of this section, let's drop categorical columns and split the train dataset into features (X) and target (y).

# In[ ]:


def fe_split(fe_df):
    return fe_df.select_dtypes(include='number').drop(columns='selling_price_inr'), fe_df['selling_price_inr']

X_train, y_train = fe_split(fe_train)
X_train.head()


# In[ ]:


y_train.head()


# # Model Building
# We will use Random Forest Regressor as the model for predicting car prices.To find the best hyperparameters of the model, we will use the grid search method with an adjusted R-squared as the main score.

# In[ ]:


# creating adjusted R-squared score, since it isn't present in sklearn library
# estimator required for dynamic change of p value through cross-val process
def adj_r2(estimator, X, y_true):
    n, p = X.shape
    y_pred = estimator.predict(X)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2)*(n - 1)/(n - p -1)

param_dict = {
    'n_estimators': [50, 100, 150, 200],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [None, 5, 7, 9],
    'random_state': [0]
}
scoring_dict = {
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "adj_R2": adj_r2
}

def gscv(X_train, y_train):
    grid = GridSearchCV(
        RandomForestRegressor(),
        param_grid=param_dict,
        scoring=scoring_dict,
        refit="adj_R2"
    ) 
    grid.fit(X_train, y_train)
    
    return grid

grid = gscv(X_train, y_train)


# In[ ]:


# checking scores of best model
def check_results(grid):
    results_df = pd.DataFrame(grid.cv_results_)
    cols = [f'{el1}_test_{el2}' for el1 in ['mean', 'std', 'rank'] for el2 in ['MAE', 'MSE', 'adj_R2']]
    return results_df.loc[results_df['params']==grid.best_params_, cols].abs().T


# In[ ]:


check_results(grid)


# The scores are good enough, but let's try to improve model performance by narrowing down the variety of cars. We will drop rows with rare categorical values (count of values in column < 10, which is ~0.05% of data) because such cases are probably exclusions, so there is little need for them.

# In[ ]:


brand_count = fe_train['brand'].value_counts()
usual_brands = list(brand_count[brand_count > 10].index)
mask = (fe_train['brand'].isin(usual_brands)) & (fe_train['fuel']!='Electric') & (fe_train['seats']!='14')
f'We are dropping {(1-mask.sum() / len(fe_train))*100 : .2f}% of data'


# In[ ]:


X_train2, y_train2 = fe_split(fe_train[mask])

grid2 = gscv(X_train2, y_train2)


# In[ ]:


check_results(grid2)


# The model with slightly reduced data shows better results. Since we are dropping only 0.24% of data this way, we will choose this model for deployment. Now let's build a data preparation and prediction workflow for the validation dataset with code from the cells above.

# In[ ]:


# <============================================== Data Preparation functions ==============================================>
# >>>>>>>>>> Data Cleanning functions <<<<<<<<<<
# 1. create brand column from name
def brand_col(data):
    df = data.copy()
    df['brand'] = df['name'].str.split().str[0]
    
    replace_dict = {'Land' : 'Land Rover', 'Mini' : 'Mini Cooper', 'Isuzu' : 'ISUZU'}
    df['brand'] = df['brand'].replace(replace_dict)
    
    return df


# 2. check if all values seem to be correct
def values_check(data):
    df = data.copy()
    
    allowed_vals = {
        'fuel': [np.nan, 'Petrol', 'Diesel', 'CNG', 'LPG'],
        'seller_type': [np.nan, 'Individual', 'Dealer'],
        'transmission': [np.nan, 'Manual', 'Automatic'],
        'owner': [np.nan, 'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'],
        'seats': [np.nan, '2', '4', '5', '6', '7', '8', '9', '10'],
        'brand': [np.nan, 'Hyundai', 'Mahindra', 'Chevrolet', 'Honda', 'Ford', 'Tata', 'Toyota', 'Maruti', 'BMW', 'Volkswagen', 'Audi', 'Nissan', 'Skoda', 'Mercedes-Benz', 'Datsun', 'Renault', 'Fiat', 'MG', 'Jeep', 'Volvo', 'Kia', 'Land Rover', 'Mitsubishi', 'Jaguar', 'Porsche', 'Mini Cooper', 'ISUZU']
    }
    drop_df = pd.DataFrame(columns = df.columns)
    for col, allowed_vals in allowed_vals.items():
        drop_df = drop_df.append(df[~df[col].isin(allowed_vals)])
        df = df[df[col].isin(allowed_vals)].reset_index(drop=True)
        
    return df, drop_df


# 3.1. update of func for unseen data (known_df added as source of known data)
def fillna_stat_by_col(data, known_df, g_col, f_colls, fill_method):
    df = data.copy()
    unique_vals = df.loc[df.isnull().any(axis=1), g_col].unique()
    for val in unique_vals:
        if fill_method=='mean':
            df.loc[df[g_col]==val, f_colls] = df.loc[df[g_col]==val, f_colls].fillna(known_df.loc[known_df[g_col]==val, f_colls].mean(numeric_only=True))
        elif fill_method=='mode':
            df.loc[df[g_col]==val, f_colls] = df.loc[df[g_col]==val, f_colls].fillna(known_df.loc[known_df[g_col]==val, f_colls].mode().iloc[0,:])
        else:
            print(f"Error: method '{fill_method}' is incorrect.")
    
    return df

# 3.2. for filling all nulls with the help of fillna_stat_by_col() func
def all_fillna_stat(data, known_df):
    df = data.copy()
    
    # name (brand), year and owner will be mandatory input fields
    mode_all_col = 'seller_type'
    mode_brand_cols = ['fuel', 'transmission', 'seats']
    mean_brand_cols = ['engine_cc', 'max_power_bhp']
    mean_owner_col = 'km_driven'
        
    na_cols = df.columns[df.isna().any()].tolist()
    
    if mode_all_col in na_cols:
        df[mode_all_col] = known_df[mode_all_col].mode()
    
    if mean_owner_col in na_cols:
        df = fillna_stat_by_col(df, known_df, 'owner', mean_owner_col, 'mean')
    
    inter = lambda lst1, lst2: [value for value in lst1 if value in lst2]
    mode_brand_cols = inter(mode_brand_cols, na_cols)
    df = fillna_stat_by_col(df, known_df, 'brand', mode_brand_cols, 'mode')
    
    mean_brand_cols = inter(mean_brand_cols, na_cols)
    df = fillna_stat_by_col(df, known_df, 'brand', mean_brand_cols, 'mean')
    
    return df


# 4. check and drop nulls
def na_check(data, drop_data):
    df, drop_df = data.copy(), drop_data.copy()
    if df.isna().any().any():
        drop_df = drop_df.append(df[df.isna().any(axis=1)])
        df.dropna(inplace=True).reset_index(drop=True)
        
    return df, drop_df



# >>>>>>>>>> Feature Engineering functions <<<<<<<<<<
# 1.1. create '..._flg' column
def col_to_col_flg(data, col, vals_1):
    df = data.copy()
    df[col+'_flg'] = df.apply(lambda row: int(row[col] in vals_1), axis=1)
    return df

# 1.2. create '..._flg' columns with col_to_col_flg() func
def all_col_to_col_flg(data):
    df = data.copy()
    cols_vals_1_dict = {
        'fuel': ['Diesel', 'Petrol'],
        'seats': ['2', '4', '5', '7'],
        'owner': ['First Owner'],
        'seller_type': ['Individual'],
        'transmission': ['Manual']
    }

    for col, vals_1 in cols_vals_1_dict.items():
        df = col_to_col_flg(df, col, vals_1)
        
    return df


# 2. create columns from brand
def brand_by_mean_price(data, known_df):
    df = data.copy()
    mean_price_per_brand = known_df.groupby(['brand']).mean()['selling_price_inr'].sort_values(ascending=False)
    n = len(mean_price_per_brand)

    df['brand_top_half'] = df.apply(lambda row: int(row['brand'] in mean_price_per_brand.iloc[:n//2].index), axis=1)
    df['brand_top_third'] = df.apply(lambda row: int(row['brand'] in mean_price_per_brand.iloc[:n//3].index), axis=1)
    df['brand_bottom_third'] = df.apply(lambda row: int(row['brand'] in mean_price_per_brand.iloc[-n//3:].index), axis=1)

    return df


# >>>>>>>>>> Features/target split <<<<<<<<<<
def xy_split(data):
    return data.select_dtypes(include='number').drop(columns='selling_price_inr'), data['selling_price_inr']


# >>>>>>>>>> Main function (unify all functions above) <<<<<<<<<<
def data_prep(data, train_df, y_true_flg=False):
    df = data.copy()
    
    # Data Cleaning
    df = brand_col(df)
    df, drop_df = values_check(df)
    df = all_fillna_stat(df, train_df)
    df, drop_df = na_check(df, drop_df)
    
    # Feature Engineering
    df = all_col_to_col_flg(df)
    df = brand_by_mean_price(df, train_df)
    
    # Features/target split
    if y_true_flg:
        X, y = xy_split(df)
        return (X, y), drop_df
    
    else:
        return df, drop_df




# <============================================== Prediction function ==============================================>
# predict with evaluation of model accuracy
def pred_with_scores(model, X, y_true=None):
    y_pred = model.predict(X)
    if y_true is not None:
        r2 = r2_score(y_true, y_pred)
        n, p = X.shape
        score_dict={
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'adj_R2': 1 - (1 - r2)*(n - 1)/(n - p -1)
        }
        scores = pd.Series(score_dict)
        display(scores)
        
    return y_pred




# <======================================== Data Preparation + Prediction function ========================================>
def data_prep_and_predict(data, train_df, model, y_true_flg=False, return_drop=False):
    df = data.copy()
    df, drop_df = data_prep(df, train_df, y_true_flg=y_true_flg)
    if y_true_flg:
        y_pred = pred_with_scores(model, df[0], df[1])
    else:
        y_pred = pred_with_scores(model, df)
    
    return (y_pred, drop_df) if return_drop else y_pred


# In[ ]:


rfr = grid2.best_estimator_
rfr


# In[ ]:


print('<===== Scores on validation dataset =====>')
y_pred, drop_df = data_prep_and_predict(raw_valid, clean_train, rfr, True, True)
print('<===== Scores of cross validation on train dataset  =====>')
print(check_results(grid2).iloc[:3])


# The output shows that the model performed almost equally well on unseen data, which is a good sign. As the last step, we will save the model in the *rfr_model.sav* file and the cleaned train dataset in the *clean_train.csv* file, so we can use them later.

# In[ ]:


pickle.dump(rfr, open('rfr_model.sav', 'wb'))
clean_train[mask].to_csv('clean_train.csv', index=False)


# That's the end of part 2 of the *'Building a car price prediction model for the CarDekho website'* project. You can check the web app with created model [here](https://share.streamlit.io/zaikabohdan/ds_car_price_proj/main/app/app.py).
