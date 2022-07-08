#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2 as cv
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


csv = pd.read_csv('../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')
csv2 = pd.read_csv('../input/skin-cancer-mnist-ham10000/hmnist_28_28_RGB.csv')

print(csv.head())
print(csv2.head())


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.DataFrame(csv)
df2 = pd.DataFrame(csv2)
print(df2.label.value_counts())
df


# In[ ]:


describe = print(df.describe())
info = print(df.info())


# In[ ]:


np.mean(df.age)


# In[ ]:


df.shape


# In[ ]:


X = df.image_id
y = df.dx_type
X_train, X_val, y_train, y_val = train_test_split(X, y)
print(len(X_train))
print(len(X_val))
print(len(y_train))
print(len(y_val))


# In[ ]:


y = df2['label'].values
X = df2.drop(['label'] , axis=1).values
unique, counts = np.unique(y, return_counts=True)
result = np.column_stack((unique, counts)) 
result




# Data is not balanced. I want to detect melanocytic nevi (nv) so we will have 2 target classes. 

# In[ ]:





# In[ ]:


print(len(X[5]))
print(len(X))


# 

# In[ ]:


df2.loc[10000].at["label"] 


# Let's save label of melanocytic nevi (nv) (label 4) as 1 and rest as 0.   

# In[ ]:


for x in range(10014):
    if df2.loc[x].at["label"] == 4:
        df2.loc[x].at["label"] = 1
    else:
        df2.loc[x].at["label"] = 0
y = df2['label'].values
X = df2.drop(['label'] , axis=1).values
unique, counts = np.unique(y, return_counts=True)
result = np.column_stack((unique, counts)) 
result


# 

# In[ ]:


X.mean()
X.var()
X.max()


# In[ ]:


plt.imshow(X[10014].reshape(28,28,3))


# In[ ]:


for x in range(10014):
    if df2.loc[x].at["label"] == 1:
        df2.loc[x].at["label"] = 1
    else:
        df2.loc[x].at["label"] = 0


# işe yaramadı medianBlur

# In[ ]:


img = cv.imread('/kaggle/input/skin-cancer-mnist-ham10000/ham10000_images_part_2/ISIC_0031869.jpg')
median = cv.medianBlur(img, 1)
compare = np.concatenate((img, median), axis=1) #side by side comparison

plt.imshow(median)



# In[ ]:


img15 = cv.imread('/kaggle/input/skin-cancer-mnist-ham10000/ham10000_images_part_2/ISIC_0031869.jpg')

kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img15,-1,kernel)

plt.subplot(121),plt.imshow(img15),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()


# In[ ]:


img1 = cv.imread('/kaggle/input/skin-cancer-mnist-ham10000/ham10000_images_part_2/ISIC_0031869.jpg')
img_blur = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
sobelx = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
plt.imshow(sobelxy)


# In[ ]:


img = plt.imread('/kaggle/input/skin-cancer-mnist-ham10000/ham10000_images_part_2/ISIC_0031869.jpg')
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# In[ ]:


kernel = np.array([[0, -1, 0],
                   [-1, 7,-1],
                   [0, -1, 0]])
image_sharp = cv.filter2D(src=img, ddepth=-1, kernel=kernel)
plt.imshow(image_sharp)
#cv2.waitKey(1)


# In[ ]:


# load the original input image and display it to our screen
image = cv.imread('/kaggle/input/skin-cancer-mnist-ham10000/ham10000_images_part_2/ISIC_0031749.jpg')
plt.imshow(image)
# a mask is the same size as our image, but has only two pixel
# values, 0 and 255 -- pixels with a value of 0 (background) are
# ignored in the original image while mask pixels with a value of
# 255 (foreground) are allowed to be kept
mask = np.zeros(image.shape[:2], dtype="uint8")
cv.rectangle(mask, (0, 90), (290, 450), 255, -1)
plt.imshow(mask)
# apply our mask -- notice how only the person in the image is
# cropped out
masked = cv.bitwise_and(image, image, mask=mask)
plt.imshow(masked)


# 
