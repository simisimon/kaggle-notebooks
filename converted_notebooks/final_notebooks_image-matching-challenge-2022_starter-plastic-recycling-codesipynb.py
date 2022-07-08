#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Click the blue "Edit Notebook" or "Fork Notebook" button at the top of this kernel to begin editing. Enjoy my dataset and code. <3

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


from fastai.vision import *
from fastai.metrics import error_rate, fbeta
import pandas as pd
import numpy as np


# In[ ]:


folder= 'working_dir'


# In[ ]:


path = Path('../input')
origin= Path('..')
dest = origin/folder

dest.mkdir(parents=True, exist_ok=True)
get_ipython().system('cp -r ../input/* {dest}/')
path.ls()


# In[ ]:


dest.ls()


# In[ ]:


path = Path('../input')


# In[ ]:


bs=32
tfms=ds_tfms=get_transforms(do_flip=False, max_rotate=0, max_zoom=1, max_lighting=0, max_warp=0)


# In[ ]:


np.random.seed(42)
data = (ImageList.from_folder(path)
        .split_by_rand_pct(0.2)
        .label_from_folder()
        .transform(tfms, size=128)
        .databunch())


# In[ ]:


data.show_batch(rows=3, figsize=(9,7))


# In[ ]:


classes = data.classes
print(classes)


# In[ ]:


from fastai.vision.learner import create_cnn,models
from fastai.vision import error_rate


# In[ ]:


learn = cnn_learner(data, models.resnet50, model_dir = '/tmp/models',  metrics=error_rate)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, slice(1e-03,4e-3))
learn.save('plastics_save_1', return_path=True)


# In[ ]:


learn.unfreeze()
learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(6, slice(1e-03,3e-4))
learn.save('plastics_save_2', return_path=True)


# In[ ]:


learn.recorder.plot_losses()


# ## Interpreting Results

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.most_confused()


# In[ ]:


interp.plot_top_losses(9)


# In[ ]:


learn.show_results(rows=3, figsize=(10,10))


# ## Exporting the model

# In[ ]:


learn.export('/kaggle/dest')


# ## Conclusion
# This is a starter package for you. To go forward from here, click the blue "Edit Notebook" button at the top of the kernel. This will create a copy of the code and environment for you to edit. If you want to contribute to the dataset, please contact me.
