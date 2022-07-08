#!/usr/bin/env python
# coding: utf-8

# # <h1><center>⭐️YOLOV5 - FOOTBALL_PLAYER_RECOGNITION⭐️</center></h1>
# 
# ## ***Hello folks! This notebook only guide biases***
# 
# **Only focus on How to recognition,count,team players and explain the yolov5 _ full resource are given in to below link! try out your own**
# 
# **My github link: https://github.com/VK-Ant/Football_TeamDetection_PersonCount**
# 
# ## ***I'm trying this task in colab!***
# 
# ![f1 (1).jpg](attachment:76df6582-4e92-48f4-9d80-8d8eca502ef3.jpg)
# 
# ### **Only three changes in your work**
# 
# 1. first ---> change the create your own 'custom_data.yaml' file to 'yolov5/data/' paste this path
# 2. change the detect.py file(download my file and change)
# 3. Give input your own data
# 
# ## **I was upload my output data also! If you are any doubts ask me**

# # <h1><center>⭐️Yolo-Object recognition⭐️</center></h1>
# 
# CNN-based Object Detectors are primarily applicable for recommendation systems. YOLO (You Only Look Once) models are used for Object detection with high performance. YOLO divides an image into a grid system, and each grid detects objects within itself. They can be used for real-time object detection based on the data streams. They require very few computational resources.
# 
# ### ***If you want basics in object detection! How to use yolov5***
# 
# Refer this link: https://www.kaggle.com/code/venkatkumar001/object-recognition-yolov5-coco-dataset
# 

# # **Steps:**
# 
# 1. Create Custom data
# 2. Annotation the custom data --> Makesense.ai
# 3. Build the prediciton using predtrained model YOLOV5S and YOLOV5M
# 4. First process object detection coco 80dataset --> Then changed my custom data labels (custom.yaml)
# 5. Train the custom data
# 6. Inference the model to predict output
# 7. Display the output

# # **First step download in YOLOV5 all resources**

# ## **Download YOLO V5**
# 

# In[ ]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5  # clone')
get_ipython().run_line_magic('cd', 'yolov5')
get_ipython().run_line_magic('pip', 'install -qr requirements.txt  # install')

import torch
import utils
display = utils.notebook_init()  # checks


# # **2. Mount your google drive and add custom data**

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


#unzip the data
get_ipython().system('unzip -q /content/drive/MyDrive/Football_analysis/inputdata/train_data.zip -d /content')


# # **3. Train your custom data and yaml files**

# In[ ]:


# Train YOLOv5s on COCO128 for 3 epochs
get_ipython().system('python train.py --img 640 --batch 16 --epochs 60 --data custom_data.yaml --weights yolov5s.pt --cache')


# ## **This kind of output shows**
# 
# ## **This below represent path copy and paste in inference code for next step(model weights)**
# ![Screenshot from 2022-05-17 13-41-42.png](attachment:49c7173a-d02b-4796-92d0-ce3df4c69677.png)

# # **4. Inference detect the output**
# 
# ## **Change the detect.py file, Already i was changed so you used this (../input/football-analysis/yolov5_custom-data)**
# 
# ## **Important step is you are train the model output shows "runs/train/exp/weights/best.pt " change and run**
# 
# 

# In[ ]:


get_ipython().system('python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.1 --source /content/1.png')


# ![Screenshot from 2022-05-17 13-42-07.png](attachment:017523e7-96d5-4502-99e4-f280cf3e3520.png)

# ## **Display the output**

# In[ ]:


display.Image(filename='runs/detect/exp/1.png', width=600)


# # **Recognition and count output**
# 
# 
# ![7.jpg](attachment:478584cb-948f-4276-8316-d57583e58beb.jpg)
# 

# ## **Video input**

# In[ ]:


get_ipython().system('python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.1 --source /content/f1.png')


# ## **Display video**

# In[ ]:


from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = "../input/football-analysis/clear output_yolov5s/sample.mp4"

# Compressed video path
compressed_path = "../input/football-analysis/clear output_yolov5s/sample_compressed.mp4"

#os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)


# ## **⭐⭐THANKS FOR VISITING GUYS⭐⭐**
# 
# ### ***I believe this notebook usefull for you! Any doubts feel free to ask me!***
