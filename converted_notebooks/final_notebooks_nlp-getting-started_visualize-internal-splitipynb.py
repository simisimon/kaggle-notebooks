#!/usr/bin/env python
# coding: utf-8

# While fitting ALD distributions to the FC layer weights for each class $\theta_k$, I observed a recursive pattern in the isolated weights. Let $\theta_k^+$ denote the weights for the positive sub-class [i.e. the possibility that the sample belong to class k] \& $\theta_k^-$ denote the weights for the negative class [i.e. the possibility that the sample is outside training set]. I hypothesize that $\theta_k^+$ and $\theta_k^-$ can be further sub-divided and fitted with ALD. I hypothesize that the network recursively builds a tree like internal representation for each class.
# 
# **I have written a more formal summary of my findings in [this document](https://arxiv.org/pdf/2205.11908.pdf).**
# 
# The results of fitting ALD to FC weights are [here](https://drive.google.com/file/d/1ce90RTQKhYIoxhJrqvhxw6VdjLMEGHkm/view?usp=sharing) and visualization of the most discriminative neurons using Smooth GramCam++ is [here](https://drive.google.com/drive/folders/1aWOlXt20iZJGgaXFMusCTYmJkOyLImLj?usp=sharing).
# 
# 
# ![internal_split](https://raw.githubusercontent.com/sidml/interpret-fc-layer/main/images/internal_split.png)
# 
# 
# 

# In[ ]:


get_ipython().system('pip install -q timm')
get_ipython().system('pip install -q pyvis')
get_ipython().system('pip install -q torchcam')
get_ipython().system('wget https://raw.githubusercontent.com/rwightman/pytorch-image-models/master/results/results-imagenet.csv')


# In[ ]:


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm.auto import tqdm
import pdb
import torch.nn as nn
import os
from glob import glob
import pdb, json, timm
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import *
import torchvision.models as models
from torchcam.utils import overlay_mask
from pyvis.network import Network
from copy import deepcopy
import random


# In[ ]:


def seed_everything(seed=42):
    print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
# smoothgradcam relies on random noise which may
# lead to variable results if seed is not set
seed_everything()


# In[ ]:


def setup_imagenet_classes(data_dir = "../input/imagenetval"):
        
    # Read the categories
    with open(f"{data_dir}/imagenet_classes.txt", "r") as f:
        categories = [s.strip().lower() for s in f.readlines()]
    
    # https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
    cls_map = pd.read_csv(f"{data_dir}/map_clsloc.txt", sep=' ', header=None)
    cls_map.columns = ['imagenet_label', 'imagenet_clsnum', 'string_label']

    with open(f"{data_dir}/imagenet_class_index.json") as data_file:    
        data = json.load(data_file)              
            
    cls_stats = pd.DataFrame(data.values(), index=data.keys(),
                      columns=['imagenet_label', 'string_label'])
    cls_stats.index.name = 'pytorch_clsnum'
    cls_stats = cls_stats.reset_index()
    cls_stats = cls_stats.set_index('imagenet_label')
#     doesn't work for some reason!!!
#     cls_stats.loc[cls_map.imagenet_label, "imagenet_clsnum"] = cls_map['imagenet_clsnum']
    cls_stats.loc[cls_map.imagenet_label, "imagenet_clsnum"] = np.arange(1, 1001)
    cls_stats['imagenet_clsnum'] = cls_stats['imagenet_clsnum'].astype(int)
    
    val_sol = pd.read_csv(f"{data_dir}/LOC_train_solution.csv")
    all_rows = val_sol.PredictionString.apply(lambda x: x.strip().split(' '))

    class_counts = val_sol.ImageId.apply(lambda x: x.split('_')[0]).value_counts()
    class_counts.name = "cls_count"
    class_counts.index.name = 'imagenet_label'
    cls_stats = cls_stats.merge(class_counts, on='imagenet_label')

    cls_stats['pytorch_clsnum'] = cls_stats['pytorch_clsnum'].astype(int)


    if cls_stats.index.name!='pytorch_clsnum':
        cls_stats = cls_stats.reset_index()
        cls_stats = cls_stats.set_index('pytorch_clsnum')

    return cls_stats


def preprocess_img(img):    
    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img, (224, 224)) / 255.,
                             [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)  
    return input_tensor


# In[ ]:


class MyClass:  # Add Node feature
    def __init__(self, name, map2orig_idx, score, pos, parent):
        super(MyClass, self).__init__()
        self.name = name
        self.map2orig_idx = map2orig_idx
        self.parent = parent
        self.pos = pos
        self.score = score


# ## We recursively split the FC layer weights and apply Smooth GradCam++ using only a subset of weights

# In[ ]:


def save_image(model, orig_model, pil_im, input_tensor, 
               cam_extractor, out, map2orig_idx, img_path, cls_num):
    # Retrieve the CAM by passing the selected index and the model layer output

    model.zero_grad()
#     pdb.set_trace()
    activation_map = cam_extractor(map2orig_idx, out)

    with torch.no_grad():
        amap = activation_map[0].unsqueeze(0).unsqueeze(0)
        amap = torch.nn.functional.interpolate(amap,  size=(224, 224), mode='bicubic',
                                  align_corners=False)
        # normalize to 0-1
        amap = (amap - amap.min()) / (amap.max() - amap.min())
        new_tensor = input_tensor * amap
        score = orig_model(new_tensor)
        score = torch.nn.functional.softmax(score, -1)[0, cls_num]
        
    # Resize the CAM and overlay it
    result = overlay_mask(pil_im, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.7)
    # Display it
    plt.imshow(result); plt.axis('off'); plt.tight_layout()
    plt.savefig(img_path); plt.close()
    return score
        
def split_fit(model, orig_model, pil_im, input_tensor, net, out, y_after_split, yorig, node, depth, cam_extractor,
              fig_folder,cls_num, spacing=100):
    input_tensor.requires_grad = True
    y = y_after_split.copy()
    if len(y) < 20:
        return
    else:
        depth += 1
        mean = np.mean(y)
        y = y - mean
        pos_idx, neg_idx = np.where(y >= 0)[0], np.where(y < 0)[0]
        y_pos, y_neg = y[pos_idx], y[neg_idx]
    
        pos_node_name, neg_node_name = f'{node.name}_p{depth}', f'{node.name}_m{depth}'
        pos_img_path = f"{fig_folder}/{pos_node_name}.jpg"
        neg_img_path = f"{fig_folder}/{neg_node_name}.jpg"

        def process_pos_node():
            map2orig_idx = {}
            for i, idx in enumerate(pos_idx):
                map2orig_idx[i] = node.map2orig_idx[idx]
            score = save_image(model, orig_model, pil_im, input_tensor, 
                               cam_extractor, out, list(map2orig_idx.values()), pos_img_path, 
                              cls_num=cls_num)
            pos_node = MyClass(pos_node_name, map2orig_idx=map2orig_idx, score=score,
                               parent=node, pos=(node.pos[0]-spacing//2,node.pos[1]+spacing))
            net.add_node(pos_node.name, group=depth, x=pos_node.pos[0], y=pos_node.pos[1],
                     label=f'p{depth}_score:{score:.3f}', image=pos_img_path,  shape='image')
            net.add_edge(node.name, pos_node.name)        
            return pos_node

        def process_neg_node():
            map2orig_idx = {}
            for i, idx in enumerate(neg_idx):
                map2orig_idx[i] = node.map2orig_idx[idx]
            score = save_image(model, orig_model, pil_im, input_tensor, 
                               cam_extractor, out, list(map2orig_idx.values()), neg_img_path,
                              cls_num=cls_num)
            neg_node = MyClass(neg_node_name, map2orig_idx=map2orig_idx, score=score,
                               parent=node, pos=(node.pos[0]+spacing//2,node.pos[1]+spacing))
            net.add_node(neg_node.name, group=depth, x=neg_node.pos[0], y=neg_node.pos[1],
                     label=f'm{depth}_score:{score:.3f}', image=neg_img_path,  shape='image')
            net.add_edge(node.name, neg_node.name)
            return neg_node
            
        if node.name[-2]=='p':
            pos_node =  process_pos_node()
            split_fit(model, orig_model, pil_im,input_tensor, net, out, y[pos_idx], yorig, pos_node, depth, cam_extractor, 
                      fig_folder=fig_folder, spacing=spacing, cls_num=cls_num)
        elif node.name[-2]=='m':
            neg_node = process_neg_node()    
            split_fit(model, orig_model, pil_im, input_tensor, net, out, y[neg_idx], yorig, neg_node, depth, cam_extractor,
                      fig_folder=fig_folder, spacing=spacing, cls_num=cls_num)  
        else:
            pos_node = process_pos_node()
            neg_node = process_neg_node()   
            split_fit(model, orig_model, pil_im, input_tensor, net, out, y[pos_idx], yorig, pos_node, depth, cam_extractor, 
                      fig_folder=fig_folder, spacing=spacing, cls_num=cls_num)
            split_fit(model, orig_model, pil_im,input_tensor, net, out, y[neg_idx], yorig, neg_node, depth, cam_extractor,
                      fig_folder=fig_folder, spacing=spacing, cls_num=cls_num) 


# In[ ]:


def save_net_overlay(model, orig_model, pil_im, input_tensor, 
                     cam_extractor, y, model_name, cls_name, cls_num, 
                     out, orig_score):
    depth = 0
    pos = (0, 0)
    net = Network(height='1080px',
                  width='1920px',
                  directed=True)
    net.heading = f"{cls_name} Score:{orig_score:.3f}"
    net.add_node('base', value=0, shape='image',
                image=f"./generated/{model_name}/{cls_name}.jpg")
    map2orig_idx = {i:i for i in range(len(y))}
    base = MyClass('base', map2orig_idx=map2orig_idx, pos=pos,
                  score=orig_score, parent=None)

    fig_folder = f"./generated/{model_name}/{cls_name}/"
    os.makedirs(fig_folder, exist_ok=True)
    split_fit(model, orig_model, pil_im=pil_im, input_tensor=input_tensor, 
              net=net, out=out, y_after_split=y, yorig=y, node=base, depth=depth,
              cam_extractor=cam_extractor, fig_folder=fig_folder,
              spacing=100, cls_num=cls_num)
    net.toggle_physics(False)
    net.save_graph(f"./{model_name}_{cls_name}.html")

    
        
def viz_model(all_img_fn, cam_extractor, model_name, model, orig_model, fc_w, cls_stats_df, val_sol):
    os.makedirs(f"./generated/{model_name}/", exist_ok=True)
    softmax = torch.nn.Softmax(dim=-1)
    for i, fn in enumerate(all_img_fn[:200]):
        #  extract the class id for the image
        extracted_img_id = fn.split('/')[-1][:-5]
        actual_cls =  val_sol[val_sol.ImageId==extracted_img_id]["PredictionString"].values[0]
        actual_cls = actual_cls.split(' ')[0]
        sel_row = cls_stats_df.imagenet_label==actual_cls
        imagenet_clsnum = cls_stats_df.loc[sel_row,"imagenet_clsnum"].values[0]

        pytorch_clsnum = cls_stats_df[sel_row].index[0]
        
        img = read_image(fn)
        if img.shape[0]==1: continue
        input_tensor = preprocess_img(img)
        input_tensor.requires_grad = True
        pil_img = to_pil_image(img)
        
        # Preprocess your data and feed it to the model
        model.zero_grad()
        out = model(input_tensor)
        
        with torch.no_grad():
            #  get the probability scores for the original model
            logit = orig_model(input_tensor)
            score = softmax(logit)[:, pytorch_clsnum].squeeze()
        
        y = fc_w[pytorch_clsnum]
        cls_name = cls_stats_df.loc[pytorch_clsnum, "string_label"]
        cls_name = f"{i}{cls_name}"
        # save original image for reference
        plt.imshow(pil_img); plt.axis('off'); plt.tight_layout()
        plt.savefig(f"./generated/{model_name}/{cls_name}.jpg"); plt.close()
        
        save_net_overlay(model, orig_model, pil_img, input_tensor,
                         cam_extractor, y, model_name, cls_name,
                         cls_num=pytorch_clsnum, out=out,
                         orig_score=score)
        

        print(model_name, fn, cls_name, 'done')


# ## We will use models available on timm for our experiment

# In[ ]:


timm_results = pd.read_csv('results-imagenet.csv')
model_names = timm_results.model.values


# Extract the FC layer. We also need the original model to generate score predictions.

# In[ ]:


@torch.no_grad()
def setup_weights(model_name):
    model = timm.create_model(model_name, pretrained=True)
    state = model.state_dict()        
    # last two layers should be fc
    layer_name = list(state.keys())[-2:]
    try:
        fc_w = state[layer_name[0]].cpu().numpy()
        if len(fc_w.shape)!=2:
            print(layer_name, fc_w.shape)
            print('unable to extract fc layer')
            return None, None
        if len(layer_name)==2:
            fc_b = state[layer_name[1]].cpu().numpy()
        else:
            fc_b = None
    except Exception as e:
        print(e)
        return None, None
    orig_model = deepcopy(model)
    orig_model.eval()
    model.fc = nn.Identity()
    model.global_pool = nn.Identity()
    model.classifier = nn.Identity()
    return fc_w, fc_b, model, orig_model


# ### Kaggle kernel time limit will be exceeded if I plot for all the architectures and all the classes.
# ### So I visualize results only for 200 imagenet classes for resnet34.

# In[ ]:


weight_dir = "/root/.cache/torch/hub/checkpoints/"
data_dir = "../input/imagenetval"
cls_stats_df = setup_imagenet_classes(data_dir=data_dir)
all_fn = glob("../input/imagenet/imagenet/train/*.JPEG")
val_sol = pd.read_csv(f"{data_dir}/LOC_val_solution.csv")
os.makedirs("./generated", exist_ok=True)
# comment this if you want to generate results for every image
# kaggle kernel time limit will be exceeded so I visualize results only for resnet34.
model_names = ['resnet34']  


# In[ ]:


for model_name in model_names:
    print("\nprocessing:", model_name)
    try:
        fc_w, fc_b, model, orig_model = setup_weights(model_name)
        cam_extractor = SmoothGradCAMpp(model)
        cam_extractor._precheck = lambda *args: None
        viz_model(all_fn, cam_extractor, model_name, model, orig_model, fc_w, cls_stats_df, val_sol)
    except Exception as e:
        print(e)
        for f in glob(f"{weight_dir}/*.pth"):
            os.remove(f)   
        continue
    
    for f in glob(f"{weight_dir}/*.pth"):
        os.remove(f)   


# ## Zip and save the output

# In[ ]:


get_ipython().system('zip -rq generated.zip ./')
get_ipython().system('rm -rf ./generated/')
get_ipython().system('rm -rf *.html')
get_ipython().system('rm *.csv')

