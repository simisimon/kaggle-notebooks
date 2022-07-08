#!/usr/bin/env python
# coding: utf-8

# # <h1><center>⭐️YOLOV5 - OBJECT RECOGNITION⭐️</center></h1>
# 
# CNN-based Object Detectors are primarily applicable for recommendation systems. YOLO (You Only Look Once) models are used for Object detection with high performance. YOLO divides an image into a grid system, and each grid detects objects within itself. They can be used for real-time object detection based on the data streams. They require very few computational resources.
# 
# **YOLOV5 Architecture:**
# 
# ## **Block diagram of YOLOV5**
# 
# <img src='https://miro.medium.com/max/1400/1*e17LeKXUsSxdNTlSm_Cz8w.png'>
# 
# ## **Flow chart of YOLOv5 Architecture**
# 
# <img src='https://miro.medium.com/max/1400/1*mAxsoIEeLNM_2TP-n6vTew.png'>
# 
# 
# Fullcredit: https://medium.com/analytics-vidhya/object-detection-algorithm-yolo-v5-architecture-89e0a35472ef

# # **Steps:**
# 
# ## **Part1: Object detection**
# 
# ## **Part2: Team recognition**
# 
# 1. Create Custom data
# 2. Annotation the custom data --> Makesense.ai
# 3. Build the prediciton using predtrained model YOLOV5S and YOLOV5M
# 4. First process object detection coco 80dataset --> Then changed my custom data labels (custom.yaml)
# 5. Train the custom data
# 6. Inference the model to predict output
# 7. Display the output

# # **Part1: Object recognition - YOLOV5**

# ## **Download YOLO V5**
# 

# In[ ]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5  # clone')
get_ipython().run_line_magic('cd', 'yolov5')
get_ipython().run_line_magic('pip', 'install -qr requirements.txt  # install')

import torch
import utils
display = utils.notebook_init()  # checks


# ## **Mount Drive**

# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


# #unzip the data
# !unzip -q /content/drive/MyDrive/Football_analysis/inputdata/train_data.zip -d /content


# # **Build YOLOv5s Model**
# 
# ## **Train**

# In[ ]:


# Train YOLOv5s on COCO128 for 3 epochs
get_ipython().system('python train.py --img 640 --batch 16 --epochs 2 --data coco128.yaml --weights yolov5s.pt --cache')


# ## **Inference of training model**

# In[ ]:


get_ipython().system('python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.1 --source data/images/zidane.jpg')


# In[ ]:


display.Image(filename='runs/detect/exp2/zidane.jpg', width=600)


# In[ ]:


get_ipython().system('python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.1 --source data/images/bus.jpg')


# In[ ]:


display.Image(filename='runs/detect/exp7/bus.jpg', width=600)


# ## ***⭐⭐THANKS FOR VISITING GUYS⭐⭐***
# 
# ## ⭐⭐**PLAYER RECOGINITION----COMING SOON----PLAYER RECOGNITION**⭐⭐
