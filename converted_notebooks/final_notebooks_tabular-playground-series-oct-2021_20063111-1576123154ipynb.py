#!/usr/bin/env python
# coding: utf-8

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


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import random 


# In[ ]:


#数据读取
train_data=pd.read_csv("/kaggle/input/ml2021-2022-2-kmeans/train.csv")
test_data=pd.read_csv("/kaggle/input/ml2021-2022-2-kmeans/test.csv")
train_data


# In[ ]:


data1=train_data.values

data_red=train_data[0:1139]
data_white=train_data[1140:]


# In[ ]:


data_train1=train_data.values
data_train=data_train1[:,:-1]
data_test=test_data.values
data_train


# In[ ]:


#数据处理
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scale = StandardScaler()
data_train = scale.fit_transform(data_train)
data_test = scale.fit_transform(data_test)


# In[ ]:


def distant(p1, p2):  # 计算距离
    return np.sqrt(np.sum((p1 - p2) ** 2))


def k_means(data_train, k, epoch):
    centers = {} # 初始聚类中心
    # 初始化，随机选k个样本作为初始聚类中心
    len1 = data_train.shape[0]  
    for id1, i in enumerate(random.sample(range(len1), k)):
        centers[id1] = data_train[i]  

    
    for i in range(epoch):  # 迭代次数
        kind = {}
        clusters = {}   
        for j in range(k):  # 初始化为空列表
            clusters[j] = []
            kind[j] = []
        for index,sample in enumerate(data_train):  # 遍历每个样本
            distances = []  # 计算该样本到每个聚类中心的距离
            for c in centers: 
                distances.append(distant(sample, centers[c])) 
            num = np.argmin(distances)  # 最小距离的索引
            clusters[num].append(sample)   # 将该样本添加到第idx个聚类中心
            kind[num].append(index)
        centers_pre = centers.copy()  # 记录之前的聚类中心点

        for c in clusters.keys():
            # 重新计算中心点（计算该聚类中心的所有样本的均值）
            centers[c] = np.mean(clusters[c], axis=0)
  
        unchange = True
        for c in centers:
            if distant(centers_pre[c], centers[c]) > 1e-8:  # 中心点变化是否大于允许误差
                unchange = False
                break
        if unchange == True:  
            print(f"迭代{i+1}次")
            break
    return centers, clusters, kind

def predict(test, centers):  # 预测新样本点所在的类
    # 计算p_data 到每个聚类中心的距离，然后返回距离最小所在的聚类。
    distances=[]
    for i in range(0,len(centers)):
        distances.append(distant(test, centers[i]))  
    return np.argmin(distances)


# In[ ]:


import operator
N=0
centers = {}
for t in range(0,30):
    centers1,clusters,kind = k_means(data_train,2,10000)
    color={}
    count={}
    for i in kind:
        count[i]=0
        for j in range(0,len(kind[i])):
            index=kind[i][j]
            if data_train1[index][12]=='red':
                count[i]=count[i]+1
    if count[0]>count[1]:
        m_index=0
    else:
        m_index=1
    max1=count[m_index]/len(kind[m_index])
    if(max1>N):
        N=max1
        centers=centers1
        print(count)
        print(f'{count[m_index]}   和    {len(kind[m_index])}')
        print()


# In[ ]:


color={}
if count[0]>count[1]:
    color[0]='red'
    color[1]='white'
else:
    color[1]='red'
    color[0]='white'

style=[]
num=0
for i in range(len(data_test)):
    num = predict(data_test[i],centers)
    style.append(color[num])



# In[ ]:


out_dict = {
    'id':list(np.arange(len(data_test))),
    'style':list(style)
}
out = pd.DataFrame(out_dict)
out.to_csv('submission.csv',index=False)

