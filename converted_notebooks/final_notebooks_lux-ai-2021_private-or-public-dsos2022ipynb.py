#!/usr/bin/env python
# coding: utf-8

# Public: ' GT86', ' RAV4', ' C-HR', ' Prius', ' Avensis', ' Hilux', ' PROACE VERSO', ' Land Cruiser', ' Supra', ' Camry'<br>
# Private: ' Corolla', ' Yaris', ' Auris', ' Aygo', ' Verso', ' Verso-S', ' IQ', ' Urban Cruiser'

# In[ ]:


#modelごとでprivate(小型車)/public(大型車)となるはずなので，最後の結果でモデルごとにprivate/publicとしてもいいかも


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


#追加データのあるものを読み込みます
df_test = pd.read_csv("../input/dsos2022/train_add_data_.csv")
df_train = pd.read_csv("../input/dsos2022/test_add_data_.csv")


# In[ ]:


#publicとprivateの分け方を考える


# In[ ]:


public_model = ['GT86','RAV4','C-HR','Prius','Avensis','Hilux','PROACE VERSO','Land Cruiser','Supra','Camry']
private_model = ['Corolla','Yaris','Auris','Aygo','Verso','Verso-S','IQ','Urban Cruiser']


# In[ ]:


df_train["pri_pub"] = 1#1:public 0:private
df_train.loc[df_train.model.isin(private_model),"pri_pub"] = 0


# In[ ]:


df = pd.concat([df_train, df_test])


# In[ ]:


use_col = ["engineSize","mpg","Class","Segment","pri_pub","Horsepower_max","FuelEconomy_max"]
use_col += ['Comprehensive evaluation',
       'Appearance design and body color', 'Driving performance',
       'Ride quality', 'Price evaluation', 'Interior_interior design_texture',
       'Fuel economy_economy', 'Equipment']
# for i in use_col:
#     plt.figure()
#     sns.countplot(y=i, data=df_test[use_col], hue='pri_pub')
#     plt.show()


# In[ ]:


df = df[use_col]


# In[ ]:


for f in ["Segment","Class"]:
    summary = df[f].value_counts()
    df['%s_count'%f] = df[f].map(summary)


# In[ ]:


cats = []
for col in df.columns:
    if df[col].dtype == 'object':
        cats.append(col)
        try:
            print(col, df[col].nunique())
        except:
            print(col,type(col))


# In[ ]:


# 処理が終わったので再度分割する
df_train = df.iloc[:len(df_train)]
df_test = df.iloc[len(df_train):]


# In[ ]:


# 不要なカラムを除く
drop_col = cats

df.drop(drop_col, axis=1, inplace=True)
df_train = df.iloc[:len(df_train)]
df_test = df.iloc[len(df_train):]


# In[ ]:


# featureとtargetを分離する
y_train = df_train.pri_pub
X_train = df_train.drop(['pri_pub'], axis=1)
X_test = df_test.drop(['pri_pub'], axis=1)


# In[ ]:


# 学習データの一部を検定（精度評価）用に切り出します
X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=71, stratify=y_train)


# In[ ]:


# 手元のデータで精度を見積もります。正解率95%の結果が得られましたね！
# 平均値だけでこの精度ですから、これは楽勝でしょうか。
from sklearn.metrics import accuracy_score
model = LGBMClassifier(learning_rate=0.05, n_estimators=500)
model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], eval_metric='mse', early_stopping_rounds=50)

y_pred = model.predict(X_val)
print(accuracy_score(y_val, y_pred))


# In[ ]:


# 全データで再学習して提出用ファイルを作成します
best_iter = model.best_iteration_
model = LGBMClassifier(learning_rate=0.05, n_estimators=best_iter)
model.fit(X_train, y_train)


# In[ ]:


df_test


# In[ ]:


y_pred = model.predict_proba(X_test)
y_df = pd.DataFrame(y_pred,columns = ["private","public"])
df_test = pd.read_csv("../input/dsos2022/train_add_data_.csv")
y_df["ID"] = df_test["ID"]
y_df["model"] = df_test["model"]
y_df["pri_pub"] = 0
y_df.loc[y_df.public >= 0.5,"pri_pub"] = 1
y_df
y_df.to_csv("y_pred.csv",index = False)


# In[ ]:


y_df.pri_pub.hist()


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# 変数重要度を確認しておきましょう
importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=['importance'])
importances.sort_values(['importance'], ascending=False, inplace=True)

plt.figure(figsize=[6,10])
plt.title('Feature Importance')
plt.barh(importances.index[::-1], importances.importance[::-1])
plt.xlabel('importance')
plt.show()


# In[ ]:




