#!/usr/bin/env python
# coding: utf-8

# <div>
#     <h1 align="center">Smart Ensembling</h1>    
#     <h1 align="center">G2Net Gravitational Wave Detection</h1> 
#     <h4 align="center">By: Somayyeh Gholami & Mehran Kazeminia</h4>
#     <h5 align="center">Modified by: kuroyuli</h4>
# </div>

# <div class="alert alert-success">
#     <h1 align="center">If you find this work useful, please don't forget upvoting :)</h1>
# </div>

# ## Below code is a modification of great notebook by [Somayyeh Gholami](https://www.kaggle.com/somayyehgholami) & [Mehran Kazeminia](https://www.kaggle.com/mehrankazeminia).<br> Check their [original work](https://www.kaggle.com/somayyehgholami/1-g2net-smart-ensembling).
# 
# I want more control over how to ensemble models.  Therefore, I modified the code so that I can change the coefficient for each area in the scatter plot of the main and support model prediction value.
# 
# Somayyeh Gholami & Mehran Kazeminiaさんが作ったアンサンブルのためのコードを一部変更しました。<br>
# 元コードはidの先頭文字によって係数を変えているのですが、ランダム性を加える以上の意味が分かりませんでした。<br>
# そこでアンサンブルに使うメインモデルとサポートモデルの予測値分布を表す散布図のエリアごとに係数を変えられるようにしました。<br>
# （メインモデルはそれまでのアンサンブル・モデル、サポートモデルは新しくアンサンブルに入れるモデルを表します。）

# ## Areas in the scatter plot

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')
im = Image.open("../input/coef-pic/coef.png")

plt.figure(figsize=(10, 10), dpi=50)
im_list = np.asarray(im)
plt.imshow(im_list)
plt.show()


# ## Import

# In[ ]:


import pandas as pd
import seaborn as sns

import plotly.figure_factory as ff
import plotly.express as px

from sklearn.metrics import roc_auc_score, roc_curve, auc


# <div class="alert alert-success">  
# </div>

# ## Functions

# In[ ]:


def get_var_name(var):
    for k,v in globals().items():
        if id(v) == id(var):
            name=k
    return name


# In[ ]:


def ensembling(main, support, coef1, coef2, coef3, coef4, coef5, coef6, coef7, coef8):
    
    main_name = get_var_name(locals().get('main'))
    supp_name = get_var_name(locals().get('support'))
    
    mod_a  = main.copy()
    mod_av = mod_a.values
    
    mod_b  = support.copy()
    mod_bv = mod_b.values    
              
    ense  = main.copy()
    ense_v = ense.values
 
    for i in range (len(main)):       
        diff = mod_bv[i, 1] - mod_av[i, 1]
        pred_a = mod_av[i, 1]       
        pred_b = mod_bv[i, 1] 
        
        if ((pred_a < 0.25) & (diff >= 0)):        
            pred = (pred_a * coef1) + (pred_b * (1.0 - coef1))

        elif ((pred_a < 0.25) & (diff < 0)):        
            pred = (pred_a * coef2) + (pred_b * (1.0 - coef2))
            
        elif ((pred_a >= 0.25) & (pred_a < 0.5) & (diff >= 0)):        
            pred = (pred_a * coef3) + (pred_b * (1.0 - coef3))

        elif ((pred_a >= 0.25) & (pred_a < 0.5) & (diff < 0)):        
            pred = (pred_a * coef4) + (pred_b * (1.0 - coef4))   
                      
        elif ((pred_a >= 0.5) & (pred_a < 0.75) & (diff >= 0)):        
            pred = (pred_a * coef5) + (pred_b * (1.0 - coef5))

        elif ((pred_a >= 0.5) & (pred_a < 0.75) & (diff < 0)):        
            pred = (pred_a * coef6) + (pred_b * (1.0 - coef6))
            
        elif ((pred_a >= 0.75) & (diff >= 0)):        
            pred = (pred_a * coef7) + (pred_b * (1.0 - coef7))

        elif ((pred_a >= 0.75) & (diff < 0)):        
            pred = (pred_a * coef8) + (pred_b * (1.0 - coef8))
        
        else:
            raise ValueError("if sentence error")
           
        ense_v[i, 1] = pred
        
    ense.iloc[:, 1] = ense_v[:, 1]

    ###############################    
    X  = mod_a.iloc[:, 1]
    Y1 = mod_b.iloc[:, 1]
    Y2 = ense.iloc[:, 1]
    
    plt.style.use('seaborn-whitegrid') 
    plt.figure(figsize=(9, 9), facecolor='lightgray')
    plt.title(f'\nE N S E M B L I N G\n')   
      
    plt.scatter(X, Y1, s=1.5, label=f'Support: {supp_name} {support.score}')    
    plt.scatter(X, Y2, s=1.5, label='Generated')
    plt.scatter(X, X, s=0.1, label=f'Main: {main_name}')
    
    plt.legend(fontsize=12, loc=2)
    #plt.savefig('Ensembling_1.png')
    plt.show()     
    ###############################   
    ense.iloc[:, 1] = ense.iloc[:, 1].astype(float)
    hist_data = [mod_b.iloc[:, 1], ense.iloc[:, 1], mod_a.iloc[:, 1]] 
    group_labels = [f'Support: {supp_name} {support.score}', 'Ensembling', f'Main: {main_name}']
    
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False)
    fig.show()   
    ###############################   
    
    return ense


# <div class="alert alert-success">  
# </div>

# ## Data Set

# Revise the path and its Public Score when you replace the model.  Put '860b' and so forth, if you have the same score.<br>
# モデルを入れ替える場合、パスとスコアを修正して下さい。同じスコアのモデルがある場合、'860b'などとして区別して下さい。

# Thanks to: @miklgr500 https://www.kaggle.com/miklgr500/g2net-efficientnetb1-tpu-evaluate/output

# In[ ]:


path0 = '../input/g2net-834/submission.csv'

model0 = pd.read_csv(path0).sort_values('id')
model0.score = '834a'


# Thanks to: @mrigendraagrawal https://www.kaggle.com/mrigendraagrawal/tf-g2net-eda-and-starter

# In[ ]:


path1 = '../input/g2net-855a/submission.csv'

model1 = pd.read_csv(path1).sort_values('id')
model1.score = '855a'


# Thanks to: @yasufuminakama https://www.kaggle.com/yasufuminakama/g2net-efficientnet-b7-baseline-inference

# In[ ]:


path2 = '../input/g2net-860/submission.csv'

model2 = pd.read_csv(path2).sort_values('id')
model2.score = '860a'


# Thanks to: @wabinab https://www.kaggle.com/wabinab/submission-baseline

# In[ ]:


path3 = '../input/g2net-861/submission.csv'

model3 = pd.read_csv(path3).sort_values('id')
model3.score = '861a'


# Thanks to: @ihelon https://www.kaggle.com/ihelon/g2net-eda-and-modeling/output?select=model_submission.csv

# In[ ]:


path4 = '../input/g2net-864/model_submission.csv'

model4 = pd.read_csv(path4).sort_values('id')
model4.score = '864a'


# Thanks to: @miklgr500 https://www.kaggle.com/miklgr500/cqt-g2net-efficientnetb1-tpu-inference/output?select=submission.csv
# 
# Thanks to: @xuxu1234 https://www.kaggle.com/xuxu1234/lb-0-866-g2net-efficientnetb7-tpu-inference

# In[ ]:


path5 = '../input/g2net-866/submission.csv'

model5 = pd.read_csv(path5).sort_values('id')
model5.score = '866a'


# Thanks to: @hidehisaarai1213 https://www.kaggle.com/hidehisaarai1213/g2net-tf-on-the-fly-cqt-tpu-inference

# In[ ]:


path6 = '../input/g2net-869/submission.csv'

model6 = pd.read_csv(path6).sort_values('id')
model6.score = '869a'


# In[ ]:


model_names = [f'm_{model0.score}', f'm_{model1.score}', f'm_{model2.score}', f'm_{model3.score}', f'm_{model4.score}', f'm_{model5.score}', f'm_{model6.score}']

hist_data = [model0.target, model1.target, model2.target, model3.target, model4.target, model5.target, model6.target]  
   
fig = ff.create_distplot(hist_data, model_names, bin_size=.2, show_hist=False, show_rug=False) 

fig.show()


# <div class="alert alert-success">  
# </div>

# ## Model prediction average and correlation table

# In[ ]:


mean_data=np.mean(hist_data, axis=1)
corr_data=np.corrcoef([model0.target, model1.target, model2.target, model3.target, model4.target, model5.target, model6.target])
corr_data = pd.DataFrame(corr_data)
corr = corr_data.applymap(lambda x : 0 if x >= 0.9999 else x)
corr_ = corr.copy()
corr.columns = model_names
corr['model'] = model_names
corr['pred_avg'] = mean_data
corr['cor_avg'] = (corr_.mean(axis='columns'))*len(corr_)/(len(corr_)-1)
corr['cor_max'] = corr_.max(axis='columns')
corr


# I prefer this table rather than a heatmap.<br>
# My strategy is to ensemble the most correlated models first and leave the less correlated and low scoring models for the latter ensemble.<br>In this way, I can judge the model is good for ensemble or not, and remove unnecessary models easily.<br>
# In this version, I didn't use model0(834a) and model1 (855a).
# 
# 各モデル同士の相関表です。Heatmapでもいいんですが、細かい違いを知るため表を使いました。<br>
# 私の戦略は「最初に相関が高いモデル同士でアンサンブルして、他との相関が低く、スコアも低いモデルは後回しにする。」というものです。<br>他との相関が低く、精度の低いモデルは「精度は低いものの、他のモデルの弱い部分を補ってアンサンブルとして良いモデルを作れるもの」か「精度が低くアンサンブルに悪影響を与えるもの」の可能性があり、それを見極めるのは後の方が良いのではという考えからです。（最初から入れてしまうと、外しにくくなる。）<br>このバージョンでは、model0(834a)とmodel1 (855a)をアンサンブルに加えていません。<br>この戦略が正しいかどうかは分かりません。みなさん試行錯誤してみて下さい。

# <div class="alert alert-success">
#     <h1 align="center">Ensembling</h1>
# </div>

# Large blue area and low correlation reflect big differences between main and support models. They may be an excellent combo for ensemble or one of which is an inferior model.<br>A smaller coefficient widens the orange area, thus dragging the ensemble close to the support model.
# 
# 青いエリアの広さはmainとsupportモデルの違いの大きさを示す傾向があります。correlationの方がより正確に違いが分かると思います。<br>小さなCoefficientはオレンジのエリアを広くし、アンサンブル・モデルをsupportモデルに近づけます。

# In[ ]:


sub1 = ensembling(model2, model3, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7)
print('sub1_average', np.mean(sub1.target))


# In[ ]:


sub2 = ensembling(sub1, model4, 0.3, 0.8, 0.3, 0.8, 0.2, 0.6, 0.2, 0.6)
print('sub2_average', np.mean(sub2.target))


# In[ ]:


sub3 = ensembling(sub2, model6, 0.3, 0.5, 0.3, 0.5, 0.3, 0.7, 0.6, 0.4)
print('sub3_average', np.mean(sub3.target))


# In[ ]:


sub4 = ensembling(sub3, model5, 0.4, 0.5, 0.4, 0.5, 0.5, 0.7, 0.65, 0.55)
print('sub4_average', np.mean(sub4.target))


# In[ ]:


sub5 = ensembling(sub4, model0, 1, 1, 1, 1, 1, 1, 1, 1)
print('sub5_average', np.mean(sub5.target))


# In[ ]:


sub6 = ensembling(sub5, model1, 1, 1, 1, 1, 1, 1, 1, 1)
print('sub6_average', np.mean(sub6.target))


# <div class="alert alert-success">  
# </div>

# ## Submission

# **It's important to make sure the Public Score is getting better for every step (sub1→sub2→...).**<br>
# Submitting all csv files leverage the value of this ensembling approach, even though it consumes daily submission allowance. If you find the step which worsen the score, you should change the coefficients or remove the model from ensemble.<br>
# I advise you to add "-sub1" and so forth to the Submission Description, for your record.<br>
# Out of Fold data may help you not to waste daily submission allowance, but that may be time consuming.
# 
# 大事なのは、sub1→sub2→...と進んでいくにつれて常にスコアが良くなること。なので、日毎のsubmit制限数を食ってしまうものの、sub1から順番にsubmitしていくことで、このアンサンブル法の真価が発揮できると思われます。スコアが悪くなるステップがあれば、そこで入れたsupportモデルの質・相性が悪いか Coefficientが不適切なので、Coefficientを調整するか、そのモデルをアンサンブルから抜く必要があります。<br>
# submitした後、どのcsvを出したのか分からなくなるので、Submission Description欄に"-sub1"などと加えておくと便利です。<br>
# Out of Foldデータがあれば、submitせずに確認できますが、全てのモデルに対して用意するのは大変そうですね。

# In[ ]:


sub1.to_csv("submission1.csv",index=False)
sub2.to_csv("submission2.csv",index=False)
sub3.to_csv("submission3.csv",index=False)
sub4.to_csv("submission4.csv",index=False)
sub5.to_csv("submission5.csv",index=False)

sub6.to_csv("submission_final.csv",index=False)
get_ipython().system('ls')


# <div class="alert alert-success">  
# </div>

# ## Thank you very much for reading.

# <div class="alert alert-success">  
# </div>
