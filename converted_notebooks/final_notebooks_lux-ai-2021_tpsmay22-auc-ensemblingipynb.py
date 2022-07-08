#!/usr/bin/env python
# coding: utf-8

# <div>
#     <h1 align="center" style="color:darkcyan;">AUC & Ensembling</h1>
#     <h1 align="center" style="color:darkcyan;">Tabular Playground Series - May 2022</h1>   
# </div>

# <div class="alert alert-success">  
# </div>

# #### Great public notebooks for this month. Thanks to everyone who published these notebooks. In particular: @ambrosm , @alexryzhkov , @dlaststark , @hiro5299834 , @kellibelcher , ...
# 
# #### For this reason, and given that the evaluation of this challenge is based on "AUC", I also decided to mention just a few important points regarding "Ensembling". I hope it is useful.

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

get_ipython().system('ls ../input/*')


# <div class="alert alert-success">  
# </div>

# <div class="alert alert-success">  
# </div>

# # <span style="color:darkcyan;">Evaluation: "ROC_AUC"</span>
# 
# #### Submissions are evaluated on area under the **ROC curve** between the predicted probability and the observed target.

# <img src="https://raw.githubusercontent.com/MehranKazeminia/fifa-worldcup-2018/master/roc600.png">

# In[ ]:


from sklearn.metrics import roc_auc_score, roc_curve, auc


# In[ ]:


def roc_auc(true_list, pred_list, a, b):
    
    fpr, tpr, _ = roc_curve(true_list, pred_list)    
    roc_auc = auc(fpr, tpr)

    print(f'FPR: {fpr}')
    print(f'TPR: {tpr}')
    print(f'{list(zip(fpr,tpr))}') 
    print(f'\n>>>>> ROC_AUC: %0.6f <<<<<\n' %roc_auc)
    
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(a, b), facecolor='lightgray')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'\nThe area under the ROC curve\n')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:


# Sample-1
true_list = np.array([1.0, 1.0, 1.0, 1.0, 1.0,      0.0, 0.0, 0.0, 0.0, 0.0])
pred_list = np.array([1.0, 1.0, 1.0, 0.5, 0.5,      0.0, 0.0, 0.0, 0.5, 0.5])

roc_auc(true_list, pred_list, 7, 7)


# In[ ]:


# Sample-2
true_list = np.array([1.0, 1.0, 1.0, 1.0, 1.0,      0.0, 0.0, 0.0, 0.0, 0.0])
pred_list = np.array([0.8, 120, 0.6, 0.5, 0.5,      0.4, -12, 0.1, 0.5, 0.5])

roc_auc(true_list, pred_list, 7, 7)


# ### **If you compare example one and example two, you will notice that:**
# #### In this evaluation; The values "ROC_AUC" for completely different predictions can be equal. 
# #### In the real world, you have to be careful. This should also be taken into account when "Ensembling".
# ### **What does this mean and why?**

# ![](https://miro.medium.com/max/1400/1*foMOQk2yPp745FTxhL8SCg.gif)
# - Image by: [Amine Aoullay](https://towardsdatascience.com/choosing-the-right-metric-is-a-huge-issue-99ccbe73de61) 

# #### You can see how to draw the ROC curve above. But let's see what happens when the overlap percentage increases or decreases:

# ![](https://miro.medium.com/max/1400/1*8F-fY3zanGzYeCX0NlN2MQ.gif)
# - Image by: [Amine Aoullay](https://towardsdatascience.com/choosing-the-right-metric-is-a-huge-issue-99ccbe73de61) 

# #### The less overlap, the fewer errors and the further the ROC curve moves up and left. Therefore, the better you separate your classes, the higher AUC will go. 
# 
# #### In other words, AUC doesn’t care about absolute values, it only cares about ranking. You just need to well seperate your classes the get a high AUC.
# 
# ### **So let's not forget that: The target metric in this competition is based on ranks rather than on actual values. That means that as long as the order of your values is fixed, the metric will stay the same.**

# <div class="alert alert-success">  
# </div>

# #### For more information, refer to the following address:
# 
# #### [Learning from imbalanced data.](http://www.jeremyjordan.me/imbalanced-data/)

# ![](https://www.jeremyjordan.me/content/images/2018/11/roc_cutoff-1.gif)

# <div class="alert alert-success">  
# </div>

# <div class="alert alert-success">  
# </div>

# # <span style="color:darkcyan;">Data-Sets</span>

# In[ ]:


import datatable as dt 

DF1 = dt.fread('../input/tabular-playground-series-may-2022/train.csv').to_pandas()
DF2 = dt.fread('../input/tabular-playground-series-may-2022/test.csv').to_pandas()
SAM = dt.fread('../input/tabular-playground-series-may-2022/sample_submission.csv').to_pandas()

display(DF1.shape, DF2.shape, SAM.shape)


# In[ ]:


hist_data = [DF1['target']]  
group_labels = ['y']
    
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False) 
fig.show()

DF1['target']


# <div class="alert alert-success">  
# </div>

# #### Thanks to: @dlaststark
# #### https://www.kaggle.com/code/dlaststark/tps-may22-what-tf-again

# In[ ]:


s99813 = pd.read_csv('../input/99813tps22may/99813TPS22MAY.csv')

hist_data = [s99813['target']]  
group_labels = ['s99813']
    
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False) 
fig.show()

s99813 , s99813['target'].min() , s99813['target'].max()


# <div class="alert alert-success">  
# </div>

# #### Thanks to: @alexryzhkov
# #### https://www.kaggle.com/code/alexryzhkov/tps-may-22-lightautoml-here-again

# In[ ]:


s99814 = pd.read_csv('../input/99814tps22may/99814TPS22MAY.csv')

hist_data = [s99814['target']]  
group_labels = ['s99814']
    
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False) 
fig.show()

s99814 , s99814['target'].min() , s99814['target'].max()


# <div class="alert alert-success">  
# </div>

# #### Thanks to: @ambrosm
# #### https://www.kaggle.com/code/ambrosm/tpsmay22-advanced-keras

# In[ ]:


s99822 = pd.read_csv('../input/99822tps22may/99822TPS22MAY.csv')

hist_data = [s99822['target']]  
group_labels = ['s99822']
    
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False) 
fig.show()

s99822 , s99822['target'].min() , s99822['target'].max()


# ### **<span style="color:darkred;">Note:</span>**
# 
# #### The result of this notebook is the ranking of actual values. So we should not use rankdata() again :)

# <div class="alert alert-success">  
# </div>

# <div class="alert alert-success">  
# </div>

# # <span style="color:darkcyan;">Ensembling with Rankdata</span>
# 
# #### For this step, we use the results of three excellent notebooks.

# In[ ]:


def gen_plt(main, support, generated):
    X  = main
    Y1 = support
    Y2 = generated
    
    plt.style.use('seaborn-whitegrid') 
    plt.figure(figsize=(9, 9), facecolor='lightgray')
    plt.title(f'\nE N S E M B L I N G\n')   
      
    plt.scatter(X, Y1, s=2.5, label='Support')    
    plt.scatter(X, Y2, s=2.5, label='Generated')
    plt.scatter(X, X , s=0.2, label='Main(X=Y)')
    
    plt.legend(fontsize=12, loc=2)
    # plt.savefig('Ensembling_1.png')
    plt.show()


# In[ ]:


Rs99813 = stats.rankdata(s99813['target'])
Rs99813 , len(Rs99813)


# In[ ]:


Rs99814 = stats.rankdata(s99814['target'])
Rs99814 , len(Rs99814)


# In[ ]:


Rs99822 = s99822['target'].values
Rs99822 , len(Rs99822)


# ### **<span style="color:darkred;">^^^^^ Do not use rankdata()</span>**
# 

# <div class="alert alert-success">  
# </div>

# ### **<span style="color:darkred;">Note:</span>**
# 
# #### Change the coefficients so that you may have better results. :)

# In[ ]:


sub1 = SAM.copy()
sub1['target'] = (Rs99814 * 0.50) + (Rs99813 * 0.50)

gen_plt(Rs99814, Rs99813, sub1['target'])
sub1 , sub1['target'].min() , sub1['target'].max()
# Public Score: 0.99821


# In[ ]:


sub2 = SAM.copy()
sub2['target'] = (Rs99822 * 0.50) + (sub1['target'] * 0.50)

gen_plt(Rs99822, sub1['target'], sub2['target'])
sub2 , sub2['target'].min() , sub2['target'].max()
# Public Score: 0.99827


# In[ ]:


sub3 = SAM.copy()
sub3['target'] = np.clip(sub2['target'], 100000, 600000)
sub3 , sub3['target'].min() , sub3['target'].max()


# ### **<span style="color:darkred;">Note:</span>**
# 
# #### It is not necessary to use np.clip ()

# In[ ]:


hist_data = [sub3['target'] ,sub2['target'] , sub1['target']]  
group_labels = ['sub3' , 'sub2' , 'sub1']
    
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False) 
fig.show()


# In[ ]:


sub1.to_csv("sub1.csv", index=False)
sub2.to_csv("sub2.csv", index=False)

sub3.to_csv("submission.csv", index=False)
get_ipython().system('ls')


# <div class="alert alert-success">  
# </div>

# <div class="alert alert-success">  
# </div>

# # <span style="color:darkcyan;">Ensembling with Power</span>
# 
# #### Another method is to use power for "Ensembling". We know that if we increase the numbers between zero and one to the power of two or four, etc., these numbers will shrink and will have certain distances from each other. The same thing (only in some cases) can lead to a good "Ensemble". Of course, this method is boring in my opinion and relies more on luck. For this reason, in the following, I will only describe how to do this and I will not do "Submit" and ....

# In[ ]:


def norm_list(main): 
    
    n   = main.copy()
    nv  = n.values
    min_nv = min(nv[:, 1])
    max_nv = max(nv[:, 1])

    norm  = main.copy()    
    normv = norm.values 
    
    for i in range (len(main)):
        nvn = (nv[i, 1] - min_nv) / (max_nv - min_nv)
        normv[i, 1] = nvn
        
    norm.iloc[:, 1] = normv[:, 1]
    return norm


# In[ ]:


# Determine the power:)
POWER = 2  # 2 OR 4 OR 8 OR ...


# In[ ]:


Ps99813 = s99813.target**POWER
Ps99813 


# In[ ]:


Ps99814 = s99814.target**POWER
Ps99814 


# In[ ]:


Ns99822 = norm_list(s99822)
Ns99822 , Ns99822['target'].min() , Ns99822['target'].max()


# ### **<span style="color:darkred;">Note:</span>**
# 
# #### We need to normalize the results of this particular notebook so that all values are between zero and one.

# In[ ]:


Ps99822 = Ns99822.target**POWER
Ps99822 


# <div class="alert alert-success">  
# </div>

# In[ ]:


subp1 = SAM.copy()
subp1['target'] = (Ps99814 * 0.50) + (Ps99813 * 0.50)

gen_plt(Ps99814, Ps99813, subp1['target'])
subp1 , subp1['target'].min() , subp1['target'].max()


# In[ ]:


subp2 = SAM.copy()
subp2['target'] = (Ps99822 * 0.50) + (subp1['target'] * 0.50)

gen_plt(Ps99822, subp1['target'], subp2['target'])
subp2 , subp2['target'].min() , subp2['target'].max()


# In[ ]:


hist_data = [subp2['target'] , subp1['target']]  
group_labels = ['subp2' , 'subp1']
    
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False) 
fig.show()


# In[ ]:


# subp1.to_csv("subp1.csv", index=False)
# subp2.to_csv("subp2.csv", index=False)
# !ls


# <div class="alert alert-success">  
# </div>

# <div class="alert alert-success">  
# </div>

# # <span style="color:darkcyan;">Good Luck.</span>
