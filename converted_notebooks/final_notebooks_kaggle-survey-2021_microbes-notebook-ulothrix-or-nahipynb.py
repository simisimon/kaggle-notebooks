#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# #### This note book is still progress. Always view the latest version.

# # Objective: To identify Ulothrix from other microorganisms

# # Section 1: EDA

# > **This section prepares the data and checks some of its properties such as datatypes of each feature and its cardinality. Also, this section checks the correlation of each feature to the target and checks if the target is balance.**

# # Importing Libraries needed

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')


# The initial libraries imported are splitted into groups. The first group is for loading and manipulating data, second group is for visualization, third is for dealing with categorical features, and the last is for ignoring warnings.

# # Dataset

# In[ ]:


data = pd.read_csv('../input/microbes-dataset/microbes.csv')
print(f'Dataset shape: {data.shape}')
print()
print('Dataset head:')
data.head()


# Dataset currently has 26 features but there is one that is not needed which is the `Unnamed: 0`. So dataset actually has 25 columns. Remove `Unnamed: 0`.

# In[ ]:


data = data.drop(['Unnamed: 0'], axis = 1)


# *Feature Descriptions:*
# 
# 1. Solidity - It is the ratio of area of an object to the area of a convex hull of the object. Computed as Area/ConvexArea.
# 2. Eccentricity - The eccentricity is the ratio of length of major to minor axis of an object.
# 3. EquivDiameter - Diameter of a circle with the same area as the region.
# 4. Extrema - Extrema points in the region. The format of the vector is: top-left top-right right-top right-bottom bottom-right
# 5. FilledArea - Number of on pixels in FilledImage, returned as a scalar.
# 6. Extent - Ratio of the pixel area of a region with respect to the bounding box area of an object.
# 7. Orientation - The overall direction of the shape. The value ranges from -90 degrees to 90 degrees.
# 8. EulerNumber - Number of objects in the region minus the number of holes in those objects.
# 9. BoundingBox1 - Position and size of the smallest box (rectangle) which bounds the object.
# 10. BoundingBox2 - *no information for now*
# 11. BoundingBox3 - *no information for now*
# 12. BoundingBox4 - *no information for now*
# 13. ConvexHull1 - *no information for now*
# 14. ConvexHull2 - *no information for now*
# 15. ConvexHull3 - *no information for now*
# 16. ConvexHull4 - *no information for now*
# 17. MajorAxisLength - *no information for now*
# 18. MinorAxisLength - *no information for now*
# 19. Perimeter - *no information for now*
# 20. ConvexArea - *no information for now*
# 21. Centroid1 - *no information for now*
# 22. Centroid2 - *no information for now*
# 23. Area - *no information for now*
# 24. raddi - *no information for now*
# 25. microorganisms - *no information for now*

# # Missing Values

# In[ ]:


print('NUMBER OF MISSING VALUES FOR EACH FEATURE:')
print(data.isna().sum())


# No missing values for this dataset

# # Duplicates

# In[ ]:


print(f'Duplicates in the dataset: {data.duplicated().sum()}')
print(f'Percentage of duplicates: {data.duplicated().sum()/len(data)*100}%')


# There are so many duplicates, they need to be removed. I also expect that the shape of the dataset will be greatly reduced.

# In[ ]:


data = data.drop_duplicates()
print(f'Duplicates in the dataset: {data.duplicated().sum()}')
print(f'Percentage of duplicates: {data.duplicated().sum()/len(data)}%')
print(f'Dataset shape: {data.shape}')


# # Cardinality

# In[ ]:


data.nunique()


# Almost all features are continuous except for the microorganism which is categorical because it has 10 values. And upon checking from the head, the microorganisms has a dtype of object.

# # Data Types

# In[ ]:


data.dtypes


# The microorgranisms feature really has an object dtype. The others have continuous values.

# # Target Distribution

# In[ ]:


# Figure size
plt.figure(figsize=(8,8))

# Pie plot
data['microorganisms'].value_counts().plot.pie(autopct='%1.1f%%', textprops={'fontsize':12}).set_title("Target distribution")


# In[ ]:


data['microorganisms'].value_counts()


# The targets are not evenly distributed. The strategy that I will be using to this unbalanced dataset is undersampling.
# 
# Since my target is to identify if a microorganism is Ulothrix or not, my strategy is to get a sample from other microorganisms equal to the number of the lowest one which is Spirogyra. So what will happen is I will get 135 samples from other microorganism except Ulothrix, then combine all of them. That part of the dataset is the 'Not Ulothrix' part and will contain 135x9=1215 rows. The 'Ulothrix' part of the dataset will be reduced from 1631 rows to 1215 rows to make the dataset balanced between 'Not Ulothrix' and 'Ulothrix'.

# Encoding First the 'Not Ulothrix' part of the dataset.

# In[ ]:


data.loc[data['microorganisms'] != 'Ulothrix', 'microorganisms'] = 'Not Ulothrix'


# Applying undersampling

# In[ ]:


# Function for balancing
def sampling_k_elements(microCount, k=1215):
    if len(microCount) < k:
        return microCount
    return microCount.sample(k)

data = data.groupby('microorganisms').apply(sampling_k_elements).reset_index(drop=True)
data['microorganisms'].value_counts()


# Now that the dataset is reduced, the shape should be smaller.

# In[ ]:


print(f'Dataset shape: {data.shape}')
print()
print('Dataset head:')
data.head()


# # Correlations

# Encoding microorganisms feature first so that it can easily be included in correlation maps. 0 will be the value for 'Not Ulothrix' and 1 is for 'Ulothrix'

# In[ ]:


encoder = LabelEncoder()
data['microorganisms'] = encoder.fit_transform(data['microorganisms'])


# # Solidity

# Function for finding correlation.

# In[ ]:


def corr_map(feature, size=((7, 3.25))):  
  # Figure size
  plt.figure(figsize=size)

  # Histogram
  sns.histplot(data=data, x=feature, hue='microorganisms', binwidth=1, kde=True)

  # Aesthetics
  plt.title(f'{feature} distribution')
  plt.xlabel(f'{feature} Value')


# In[ ]:


corr_map('Solidity')


# - Microogranisms with Solidity value of 0 to 5 and 12 to 17 is most likely to be Ulothrix.
# - Solidity value of around 23 is Ulothrix.
# - Solidity values of 5 to 12 and 17 to 21 is msot likely to be Not Ulothrix

# # Eccentricity

# In[ ]:


corr_map('Eccentricity', (7, 9))


# I changed the size of the graph to see the bars on 0 to 5.
# 
# - Eccentricity of 0 to 19 is most likely to be Ulothrix.
# - 19 to 23 is most likely to be Not Ulothrix.

# # EquivDiameter

# In[ ]:


corr_map('EquivDiameter', (7, 7))


# - EquivDiameter of 2 to 5 is most likely to be Ulothrix.
# - EquivDiameter of 0 to 2 and 5 to 20 is most likely to be Not Ulothrix.

# **Note: Other features will be correlated soon**

# # Section 2: Baseline Models

# > **Baseline models will be made here before making models for submission.**

# Using a simple model for having a baseline accuracy without removing any features.

# In[ ]:


# Importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Copying dataset for testing baseline
baseline_data = data

# Separating into training and testing set
target = 'microorganisms'
y = baseline_data[target]
X = baseline_data.drop([target], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[ ]:


# Making the model
baseline_model = DecisionTreeClassifier(criterion='entropy', random_state=1)
baseline_model.fit(X_train, y_train)

# Accuracy of the model
baseline_model.score(X_test, y_test)


# The accuracy above is bad. Need to do better.
