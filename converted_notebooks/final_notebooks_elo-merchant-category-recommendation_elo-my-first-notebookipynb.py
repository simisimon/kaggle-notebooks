#!/usr/bin/env python
# coding: utf-8

# ### 1.对数据加载

# #### 加载需要的包

# In[ ]:


from numpy import *
import pandas as pd
from pandas import DataFrame,Series
import matplotlib.pyplot as plt   
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import os
from pandas.io.json import json_normalize
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import datetime
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)


# 查看数据文件

# In[ ]:


os.listdir('../input/')


# 'Data_Dictionary.xlsx', 关于数据集的信息
# 
# 'historical_transactions.csv', 每个card_id最3个月内的历史交易
# 
# 'merchants.csv', 有关数据集中所有商家的相关信息。
# 
# 'new_merchant_transactions.csv',新的交易数据集
# 
# 'sample_submission.csv', 样本提交文件包含你应该预测的所有card_ids
# 
# 'test.csv', 测试数据集
# 
# 'train.csv' 训练数据集
# 

# #### 加载数据集

# In[ ]:


train_df = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test_df = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




