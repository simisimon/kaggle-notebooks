#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cudf
import cuml

target = "rating"


label_df = cudf.read_csv("data/train_ratings.csv")
print(label_df.shape)
label_df.head()


# In[ ]:


test_df = cudf.read_csv("data/test_ratings.csv")
print(test_df.shape)
test_df.head()


# In[ ]:


book_df = cudf.read_csv("data/books.csv")
print(book_df.shape)
book_df.head()


# In[ ]:


for col in book_df.columns:
    print(col, len(book_df[col].unique()))


# In[ ]:


ind = book_df.year.isin(['DK Publishing Inc', 'Gallimard'])
book_df.loc[ind]


# In[ ]:


book_df.loc[ind, "publisher"] = book_df.loc[ind, "year"]
book_df.loc[ind, "year"] = book_df.loc[ind, "author"]
book_df['year'] = book_df['year'].astype(int)
book_df.loc[ind, "author"] = ["Michael Teitelbaum", "Jean-Marie Gustave Le ClÃ?Â©zio","James Buckley"]
book_df.loc[ind, "title"] = ["DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)", "Peuple du ciel, suivi de \'Les Bergers","DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"]


# In[ ]:


user_df = cudf.read_csv("data/users.csv")
print(user_df.shape)
user_df.head()


# In[ ]:


features = []

for col in ["title", 'author', "publisher"]:
    book_df[f"{col}_count"] = book_df.groupby(col)["book_id"].transform("count")
    features.append(f"{col}_count")
    
for col in ["city", 'province', "country"]:
    user_df[f"{col}_count"] = user_df.groupby(col)["user_id"].transform("count")
    features.append(f"{col}_count")


# In[ ]:


df = label_df.append(test_df).merge(book_df, on="book_id").merge(user_df, on="user_id")
print(df.shape)
df.head()


# In[ ]:


df["is_test"] = df[target].isnull()
df["is_test"].mean()


# In[ ]:


df["book_id_city"] = df["book_id"].to_pandas() + "|" + df["city"].to_pandas()
df["user_id_author"] = df["user_id"].to_pandas() + "|" + df["author"].to_pandas()
df["user_id_publisher"] = df["user_id"].to_pandas() + "|" + df["publisher"].to_pandas()
df["age_book_id"] = (df["age"].to_pandas()//5).astype(str) + "|" + df["book_id"].to_pandas()
df["year_user_id"] = (df["year"].to_pandas()//10).astype(str) + "|" + df["user_id"].to_pandas()


# In[ ]:


for col in ["title", "author", "publisher", "city", "province", "country", "user_id", "book_id",
            "user_id_author", "book_id_city", 'user_id_publisher', 'age_book_id', 'year_user_id']:
    
    te = cuml.preprocessing.TargetEncoder(n_folds=20, smooth=100)
    df[f"{col}_te"] = 0
    
    df.loc[~df["is_test"], f"{col}_te"] = te.fit_transform(df[~df["is_test"]][col], df[~df["is_test"]][target])
    df.loc[df["is_test"], f"{col}_te"] = te.transform(df[df["is_test"]][col])
    df[f"{col}_te"] = df[f"{col}_te"].astype("float")
            
    df[f"{col}_fe"] = df.groupby(col)["id"].transform("count")
    
    features.extend([f"{col}_te", f"{col}_fe"])


# In[ ]:


features.extend(["age", "year"])


# In[ ]:


test_df = df[df["is_test"]].to_pandas().reset_index(drop=True)
df = df[~df["is_test"]].to_pandas().reset_index(drop=True)
df.shape


# In[ ]:


import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from sklearn.model_selection import KFold
import numpy as np

lgb_param = {"objective": "multiclass",
             #"metric": 'l2',
             "num_class": 10,
             "boosting": "gbdt", #dart
             "learning_rate": 0.05,
             "is_unbalance": False,
                    "min_data_in_leaf": 100,
                    "num_leaves": 31,
                    "feature_fraction": 0.8,
             "subsample": 0.5,
             "subsample_freq": 1,
             "boost_from_average": False,
             "max_bin": 1023
                    }





N_FOLDS = 4
N_ITER = 2


y_oof = np.zeros((df.shape[0], 10))
y_test = np.zeros((test_df.shape[0], 10))

for it in range(N_ITER):
    kfold = KFold(N_FOLDS, random_state=it, shuffle=True)

    for train_ind, val_ind in kfold.split(df):
        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]

        lgb_train = lgb.Dataset(train_df[features], train_df[target] - 1)
        lgb_val = lgb.Dataset(val_df[features], val_df[target] - 1)

        model = lgb.train(lgb_param, lgb_train, num_boost_round=400, valid_sets=[lgb_val],
                          verbose_eval=200)
        #model.save_model(f'models/lgb_{dtype}_{f}.txt')
        y_oof[val_ind] += model.predict(val_df[features])/N_ITER

        y_test += model.predict(test_df[features])/(N_FOLDS*N_ITER)
        print("...")


# In[ ]:


lgb.plot_importance(model, importance_type="gain", figsize=(12, 12))


# In[ ]:


df["pred_digit"] = np.argmax(y_oof, axis=1) + 1
df["pred_digit"].value_counts()


# In[ ]:


df[target].value_counts()


# In[ ]:


f1_score(df[target], df["pred_digit"], average="micro")


# In[ ]:


test_df["rating"] = np.argmax(y_test, axis=1) + 1
test_df["rating"].value_counts()


# In[ ]:


test_df.to_csv("sub_ahmet_cv364.csv", columns=["id", "rating"], index=False)


# In[ ]:




