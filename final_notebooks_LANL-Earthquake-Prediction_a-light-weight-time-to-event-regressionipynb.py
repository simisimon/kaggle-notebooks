#!/usr/bin/env python
# coding: utf-8

# A light weight time to event regression using differintegrated features.

# This is an approximation of the workflow that I followed to develop the final submission for LANL Earthquake prediction challenge. Sadly the best submission that I have was not the one that I selected for the final submission, however, the performance was really good. Hope it helps to develop new and more accurate models.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import differint.differint as df

from sklearn import preprocessing as pr
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

import csv
from os import listdir
from os.path import isfile, join


# Several functions used along with the kernel

# In[ ]:


#General plot style
def PlotStyle(Axes,Title,x_label,y_label):
    
    Axes.spines['top'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.xaxis.set_tick_params(labelsize=12)
    Axes.yaxis.set_tick_params(labelsize=12)
    Axes.set_ylabel(y_label,fontsize=14)
    Axes.set_xlabel(x_label,fontsize=14)
    Axes.set_title(Title)

def MinimalLoader(filename, delimiter=',', dtype=float):
  
  """
  modified from SO answer by Joe Kington
  """
  def IterFunc():
    with open(filename, 'r') as infile:
      for line in infile:
        line = line.rstrip().split(delimiter)
        for item in line:
          yield dtype(item)
    MinimalLoader.rowlength = len(line)

  data = np.fromiter(IterFunc(), dtype=dtype)
  data = data.reshape((-1, MinimalLoader.rowlength))

  return data

#Mean and standard deviation of a time series 
def GetScalerParameters(TimeSeries):
  return np.mean(TimeSeries),np.std(TimeSeries)

#Generates a Zero mean and unit variance signal 
def MakeScaledSeries(Signal,MeanValue,StdValue):
  StandardSignal=[(val-MeanValue)/StdValue for val in Signal]
  return StandardSignal

#Makes a matrix of time series samples 
def MakeSamplesMatrix(TimeSeries,TimeToFailure,FragmentSize,delay):
    
  """
  TimeSeries -> Data to be sampled
  TimeToFailure-> Time to failure 
  FragmentSize-> Size of the time series sample
  delay-> Number of steps to wait to get a new sample, manages the amount of overlay between samples
  """
  
  cData=TimeSeries
  cTim=TimeToFailure
  cFrag=FragmentSize
  container=[]
  time=[]
  nData=len(cData)
  counter=0
  
  for k in range(nData-cFrag):
    
    if counter==delay:
      
      cSample=list(cData[k:k+cFrag])
      container.append(cSample)
      time.append(cTim[k+cFrag])
      counter=0
      
    else:
      counter=counter+1

  return np.array(container),np.array(time)

#Data features
def MakeFeaturesRF(DataMatrix):
  
  cont=[]
  featuresList=[np.mean,np.std,np.ptp,np.min,np.sum]
  
  for sample in DataMatrix:
    sampCont=[]
    for feature in featuresList:
      sampCont.append(feature(sample))
      sampCont.append(np.abs(feature(sample)))
    
    sampCont.append(np.mean(np.diff(sample)))
    sampCont.append(np.mean(np.abs(np.diff(sample))))
    cont.append(sampCont)
    
  return np.array(cont)


# In[ ]:


Data=MinimalLoader(r'../input/lanl15/train15.csv',delimiter=',')

AcData=Data[:,0]
TimeData=Data[:,1]

GlobalMean,GlobalStd=GetScalerParameters(AcData)
ScaledData=MakeScaledSeries(AcData,GlobalMean,GlobalStd)

del Data,AcData


# Consider the following random forest regression 

# In[ ]:


SamplesData,TimeTE=MakeSamplesMatrix(ScaledData,TimeData,10000,10000)
FeaturesData=MakeFeaturesRF(SamplesData)
RFScaler=pr.MinMaxScaler()
RFScaler.fit(FeaturesData)
FeaturesData=RFScaler.transform(FeaturesData)

RFR=RandomForestRegressor(n_estimators=100)
RFR.fit(FeaturesData,TimeTE)


# With the following feature importances

# In[ ]:


plt.figure(1)
plt.plot(RFR.feature_importances_)
ax=plt.gca()
PlotStyle(ax,'','Features','Importance')


# We can see that the first and the third most important features have what it looks like a linear correlation with the time to event, and the second one returns a central tendency of the samples. 

# In[ ]:


plt.figure(2,figsize=(15,5))
plt.subplot(131)
plt.plot(FeaturesData[:,-1])
ax=plt.gca()
PlotStyle(ax,'','Samples','Abs Sum Of Changes')
plt.subplot(132)
plt.plot(FeaturesData[:,2])
ax=plt.gca()
PlotStyle(ax,'','Samples','Standard Deviation')
plt.subplot(133)
plt.plot(FeaturesData[:,-2])
ax=plt.gca()
PlotStyle(ax,'','Samples','Mean')


# Both features, standard deviation and the absolute sum of changes, measure the dissimilarity between consecutive points or to the sample mean. That difference could be seen as an approximation of the first derivative of the sample. However, a full integer differentiation could lead to an excessive loss of information.
# 
# We can see that by plotting the autocorrelation as a measure of information/memory, a full integer differentiation removes almost all the correlation inside the data. However, we can perform a differintegration between [-0.1,0.25] and retain most of the information.

# In[ ]:


Corr=[]
order=[]
df1=ScaledData[0:1500000]

for d in np.linspace(-1,1,20): 
  df2=df.GL(d,df1,num_points=len(df1)) 
  df2=MakeScaledSeries(df2,df2.mean(),df2.std())
  corr=np.corrcoef(df1,df2)[0,1] 
  Corr.append(corr)
  order.append(d)

plt.figure(3)
plt.plot(order,Corr)
ax=plt.gca()
PlotStyle(ax,'','Differintegration order','Correlation')


# A visual inspection of the absolute sum of changes of the differintegrated features shows a slight change in the slope. 

# In[ ]:


plt.figure(4,figsize=(15,5))

for order in [-0.1,0,0.1,0.2]:
    Der0=[np.sum(np.abs(df.GL(order,df1[k:k+10000],num_points=10000))) for k in range(300000,len(df1),10000)]
    Der1=[np.mean(df.GL(order,df1[k:k+10000],num_points=10000)) for k in range(300000,len(df1),10000)]
    
    plt.subplot(121)
    plt.plot(Der0,label='Order ='+str(order))
    plt.legend(loc=3)
    ax=plt.gca()
    PlotStyle(ax,'','Time','Abs Sum Of Changes')
    
    plt.subplot(122)
    plt.plot(Der1,label='Order ='+str(order))
    plt.legend(loc=3)
    ax=plt.gca()
    PlotStyle(ax,'','Time','Mean')
    
del df1,Der0,FeaturesData,SamplesData


# The final feature engineering performed to the selected features was to take the logarithm of the absolute sum of changes, trying to get a more steep slope between earthquakes. Box-Cox and Yeo-Jhonson also were considered, but at least I was not able to get any performance from those transformations. 

# In[ ]:


SeriesFragment=10000
Delay=int(0.1*SeriesFragment)
DerOrders=np.linspace(-0.1, 0.25, 6, endpoint=True)


# With the features already selected the next problem to tackle is the highly unbalanced dataset, to equalize the samples, first, the original signal is resampled using an overlapping scheme, allowing a 90% overlap between samples. Then all the samples are shuffled and the time to failure is segmented in 500 bins and only 20 samples are taken for each bin. 

# In[ ]:


#Location function 
def GetSampleLoc(SampleTime,boundaries):
  
  """
  
  Returns the bin index of a time to the next eartquake sample 
  
  SampleTime: Time To the next eartquake sample
  boundaries: list of the boundaries of the bined time to the next earquake distribution 
  
  """
  
  for k in range(len(boundaries)-1):
      
    if SampleTime>=boundaries[k] and SampleTime<=boundaries[k+1]:
        
      cLoc=k
      break
      
  return cLoc

#Equalizes the samples over the range of time to the next earthquake
def MakeEqualizedSamples(DataSamples,TimeSamples):
  
  """
  
  DataSamples:  Matrix of size (SampleSize,NumberOfSamples), contains the time 
                series samples
  Time Samples: Array of size (NumberOfSamples), contains the time to the next 
                earthquake
  
  """
  
  cData=DataSamples
  cTime=TimeSamples
  nData=len(cTime)
  nBins=500
  
  cMin,cMax=np.min(cTime),np.max(cTime)
  bins=np.linspace(cMin,cMax,num=nBins+1)
  
  SamplesCount=[0 for k in range(nBins)]
  
  Xcont=[]
  Ycont=[]
  
  index=[k for k in range(len(cTime))]
  np.random.shuffle(index)
  
  for k in range(nData):
    
    cXval=cData[index[k]]
    cYval=cTime[index[k]]
    
    cLoc=GetSampleLoc(cYval,bins)
    
    if SamplesCount[cLoc]<=20:
      
      Xcont.append(list(cXval))
      Ycont.append(cYval)
      SamplesCount[cLoc]=SamplesCount[cLoc]+1
      
  return np.array(Xcont),np.array(Ycont)


# In[ ]:


Samples,Times=MakeSamplesMatrix(ScaledData,TimeData,SeriesFragment,Delay)
SamplesE,TimesE=MakeEqualizedSamples(Samples,Times)

del Samples,Times


# In[ ]:


plt.figure(5,figsize=(15,5))
plt.subplot(121)
n, bins, patches=plt.hist(TimeTE,bins=1000)
ax=plt.gca()
PlotStyle(ax,'Normal Sampling','','')
plt.subplot(122)
n, bins, patches=plt.hist(TimesE,bins=1000)
ax=plt.gca()
PlotStyle(ax,'Random Sampling','','')


# With the equalized samples the logarithm of the mean absolute sum of changes and the mean of the differintegrated features is calculated over the selected interval. And such interval is divided into six values, resulting in twelve features per sample. 

# In[ ]:


#Calculate the features for each sample 
def CalculateFeatures(Sample,Orders):
  
  """
  Sample: Time series fragment
  Orders: Array of non integer differentiation orders 
  """

  container=[]
  nSample=len(Sample)
  
  for order in Orders:
      
    derSample=df.GL(order,Sample,num_points=nSample)
    absSample=np.abs(derSample)

    container.append(np.log(1+np.mean(absSample)))
    container.append(np.mean(derSample))

  return container

#A brief description 
def MakeDataMatrix(Samples,Orders):
  
  """
  Samples: Matrix of time series samples 
  Orders: Array of non integer differentiation orders
  """
  
  container=[]
  
  for samp in Samples:
    
    container.append(CalculateFeatures(samp,Orders))
    
  return np.array(container)


# All the features are scaled and the data is divided into two sets, one for grid search cross-validation (90%) and the rest to visualize the performance of the best solution 

# In[ ]:


Xtrain0=MakeDataMatrix(SamplesE,DerOrders)
ToMinMax=pr.MinMaxScaler()
ToMinMax.fit(Xtrain0)
MMData=ToMinMax.transform(Xtrain0)

Xtrain,Xtest,Ytrain,Ytest=train_test_split(MMData,TimesE, train_size = 0.9,test_size=0.1,shuffle=True)

del Xtrain0,MMData


# For the hyperparameter optimization a grid search is performed with 2 folds. 

# In[ ]:


params={'n_estimators':[10,100,150,200],
        'max_depth':[2,4,8,16,32,None],
        'min_samples_split':[0.1,0.5,1.0],
        'min_samples_leaf':[1,2,4],
        'bootstrap':[True,False]}


RFR=RandomForestRegressor() 
FinalModel=GridSearchCV(RFR,params,cv=2,verbose=1,n_jobs=2)
FinalModel.fit(Xtrain,Ytrain)
preds4 = FinalModel.predict(Xtest)


# Performance of the best model

# In[ ]:


MAE='Mean Absolute Error = ' +str(sum(np.abs(preds4-Ytest))/len(Ytest))

plt.figure(3)
plt.plot(preds4,Ytest,'bo',alpha=0.15)
plt.plot([0,17],[0,17],'r')
plt.xlim([0,17])
plt.ylim([0,17])
ax=plt.gca()
PlotStyle(ax,MAE,'Predicted','Real')


# Loading the test data 

# In[ ]:


TestSamples=np.genfromtxt(r'../input/tested/test.csv',delimiter=',')
TestIds=np.genfromtxt(r'../input/tested/ord.csv',delimiter=',')

SamplesFeatures=MakeDataMatrix(TestSamples,DerOrders)
ScaledTest=ToMinMax.transform(SamplesFeatures)
final=FinalModel.predict(ScaledTest)


# Saving the predictions

# In[ ]:


PredictionDir=r'../predictions.csv'
firstRow=['seg_id','time_to_failure']

with open(PredictionDir,'w',newline='') as output:
        
    writer=csv.writer(output)
    nData=len(final)
    writer.writerow(firstRow)
            
    for k in range(nData):
      cRow=[TestIds[k],final[k]]
      writer.writerow(cRow)
        


# 
