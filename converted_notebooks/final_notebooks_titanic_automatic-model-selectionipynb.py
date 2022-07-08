#!/usr/bin/env python
# coding: utf-8

# # Auto model selection

# In[ ]:


import pandas as pd
import numpy as np
import re
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

class AutoModel():
    def __init__(self, objective = None):
        self.objective = objective
        self.best_model = None
        self.best_score = 0
        self.models = []
    def drop(self, s):
        if(s != np.nan):
            s = re.sub("[^0-9a-zA-Z]", "",  str(s))
            if(s == 'nan'):
                return np.nan
            return s
        else:
            return np.nan
    def drop_special_characters(self, arr):
        return pd.DataFrame([self.drop(s) for s in arr])
    
    def transform(self, X):
        
        X = pd.DataFrame(X)
        X.columns = map(str,range(X.shape[1]))
        for col in X.columns:
            X[col] = self.drop_special_characters(X[col])
            try:
                X[col] = X[col].map(int)
            except:
                pass
        return X
                
    def fit(self, X, y):
        X, y = pd.DataFrame(X), pd.DataFrame(y)
        y.columns = ['target']
        X.columns = map(str,range(X.shape[1]))
        df = pd.concat([X, y], axis = 1)
        
        if(df.dropna().shape[0] >= df.shape[0] * 0.95):
            df = df.dropna()
        X = df[X.columns]
        y = df[y.columns]
        
        X = self.transform(X)
        
        self.objective = 'regression' if y['target'].dtype == 'float64' or self.objective == 'regression' else 'classification'
        if(self.objective == 'regression'):
            estimators = [LGBMRegressor(), XGBRegressor(), LinearRegression(), MLPRegressor(), RandomForestRegressor()]
        else:
            estimators = [LGBMClassifier(), XGBClassifier(), LogisticRegression(), MLPClassifier(), RandomForestClassifier()]
            
        grids = [{
                  "n_estimators" : [10, 50, 100, 250, 500], 
                  "learning_rate" : [1e-4, 1e-3, 1e-2], 
                  "num_leaves" : [8, 16, 32, 64, 128],
                  "reg_alpha" : [0, 1e-3, 1e-2, 1e-1], 
                  "reg_lambda" : [0, 1e-3, 1e-2, 1e-1]
                },
                {
                  "n_estimators" : [10, 50, 100, 250, 500], 
                  "learning_rate" : [1e-4, 1e-3, 1e-2], 
                  "max_depth" : [2, 4, 6, 8, 10],
                  "reg_alpha" : [0, 1e-3, 1e-2, 1e-1], 
                  "reg_lambda" : [0, 1e-3, 1e-2, 1e-1]
                },
                {
                  "fit_intercept" : [True, False]  
                },
                {
                  "learning_rate" : ["constant", "invscaling", "adaptive"],
                  "learning_rate_init" : [1e-4, 1e-3, 1e-2],
                  "alpha" : [0, 1e-3, 1e-2, 1e-1]
                },
                {
                  "n_estimators" : [10, 50, 100, 250, 500], 
                  "max_depth" : [2, 4, 6, 8, 10],
                  "max_features" : ["sqrt", 1, 0.8, 0.6, 0.4, 0.2]
                }
            
        ]
        
        models = []
        for i in range(len(estimators)):
            numerical_transformer = Pipeline(steps = [
                ("imputer", SimpleImputer(strategy='constant')),
                ("polynomial", PolynomialFeatures(2)),
                ("Scaler", StandardScaler())
            ])

            categorical_transformer = Pipeline(steps = [
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
            ])
            
            categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
            numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
            l = y_train.shape[0]
            model = Pipeline([("Preprocessor", preprocessor) ,
                              ("Search", RandomizedSearchCV(n_iter = 50, estimator = estimators[i],
                              param_distributions = grids[i], cv = KFold(n_splits = 3, shuffle = True)))])
            
            model.fit(X_train, y_train)
            sc = model.score(X_test, y_test)
            models.append((model.fit(X, y), sc))
        models.sort(key = lambda x: x[1])
        self.models = models
        self.best_model = models[-1][0]
        self.best_score = models[-1][1]
    def predict(self, X):
        return pd.DataFrame(self.best_model.predict(self.transform(X)))
    def score(self, X, y):
        X = self.transform(X)
        return self.best_model.score(X, y)


# In[ ]:


import pandas as pd
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
train.head()


# In[ ]:


X = train.drop(["Survived"], axis = 1)
y = train.Survived


# In[ ]:


model = AutoModel()


# In[ ]:


model.fit(X, y)


# In[ ]:


print("Best score:", max([i[-1] for i in model.models]))


# In[ ]:


submission = pd.read_csv("../input/titanic/gender_submission.csv")


# In[ ]:


submission["Survived"] = model.predict(test)
submission.to_csv("submission.csv", index = False)


# In[ ]:




