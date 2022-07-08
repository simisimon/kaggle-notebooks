#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

# Any results you write to the current directory are saved as output.


# In[ ]:


app_events = pd.read_csv('../input/app_events.csv')
app_labels = pd.read_csv('../input/app_labels.csv')
events = pd.read_csv('../input/events.csv')
gatest = pd.read_csv('../input/gender_age_test.csv')
gatrain = pd.read_csv('../input/gender_age_train.csv')
label_categories = pd.read_csv('../input/label_categories.csv')
phone_brand_device_model = pd.read_csv('../input/phone_brand_device_model.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


gatrain.group.value_counts().sort_index(ascending=False).plot(kind='bar')


# In[ ]:


c = gender_age_train.groupby(['age','gender']).size().unstack().fillna(0)


# In[ ]:


c.plot(kind='bar',figsize=(12,6));


# In[ ]:


c


# In[ ]:


letarget = LabelEncoder().fit(gatrain.group.values)


# In[ ]:


gatrain.group.values


# In[ ]:


y = letarget.transform(gatrain.group.values)
n_classes = len(letarget.classes_)


# In[ ]:


letarget.classes_


# In[ ]:


pred = np.ones((gatrain.shape[0],n_classes))/n_classes


# In[ ]:


phone = pd.read_csv('../input/phone_brand_device_model.csv',encoding='utf-8')
phone.head(3)


# In[ ]:


print('{} rows'.format(phone.shape[0]))
print("unique values:")
for c in phone.columns:
    print('{}: {}'.format(c, phone[c].nunique()))


# In[ ]:


dup = phone.groupby('device_id').size()


# In[ ]:


dup = dup[dup>1]


# In[ ]:


dup.value_counts()



# In[ ]:




