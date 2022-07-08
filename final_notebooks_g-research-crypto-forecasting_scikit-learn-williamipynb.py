#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#=========================================================================
# 資料集：手寫辨識資料集(digit-recognizer)
# 功能　：預測辨識手寫數字0~9
# 作者　：顏瑋良 (WeiLiang,Yan)
# 用意　：因中文範例教學較少，故撰寫此範本希望可以幫助剛學習的中文使用者，加速理解。
# 　　　　也在此作紀錄，給予lab未來學弟妹學習教學使用。
# 備註　：此題目使用CNN效果會比較好，但Sciikit-Learn尚未提供，故此篇先不測試CNN
#=========================================================================


# In[ ]:


#=========================================================================
# 以個人經驗做出簡易的作業標準流程(SOP)如下圖:
# 大致步驟分為三部分：(模型與資料處理皆很彈性沒有一定，所以參考即可)
# Step1.資料的御處理(Preprocessing data)
# Step2.模型選擇與建立(Data choose and build)
# STep3.模型驗證(Model validation)
#=========================================================================


# In[ ]:


print("\n簡易作業流程圖： \n\n")
from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://i.imgur.com/6FQ3BZA.png")


# In[ ]:


#==================================================
# 載入需要的套件，做資料的預處理(在後續執行時會一一講解)。
#==================================================
import pandas as pd
import numpy as np
from sklearn import preprocessing


# In[ ]:


#================================================================================
# Step1：資料觀察與預處理
# Step1-1 開啟檔案
# Step1-2 觀察資料
# Step1-3 處理資料的NAN
# Step1-4 特徵挑選及資料正規化與編碼
#================================================================================


# In[ ]:


#===============================================================================================
# Step1：資料觀察與預處理
#　｜
#　｜
#　－－－Step1-1 開啟檔案
#　　　　　｜
#　　     －－－ Step1-1-1　絕對位址
#　　　　　｜　　　　｜
#　　　　　｜　　　　－－－Step1-1-1-1 因電腦會將\與"跳脫字元"混淆，故將\改成/也可以順利讀取。
#　　　　　｜　　　　｜
#　　　　　｜　　　　－－－Step1-1-1-2 為了讓電腦不會將\與"跳脫字元"混淆，故將\改成\\也可以順利讀取。
#　　　　　｜　　　　｜
#　　　　　｜　　　　－－－Step1-1-1-3 因電腦會將\與"跳脫字元"混淆在""前加上r(表示使用原始字元顯示)
#　　　　　｜
#　　     －－－ Step1-1-2　相對位址
#　　　　　　　　　　｜
#　　　　　　　　　　－－－Step1-1-1-1 讀取資料相對位置讀取方法 (使用讀取Kaggle上的資料位置為範例)
#　　　　　　　　　　｜
#　　　　　　　　　　－－－Step1-1-1-2 因電腦讀取相對位址會是使用\，故無法使用絕對位置後面兩種方法會讀取失敗。
#===============================================================================================

# 絕對位置
#train_data = pd.read_csv("C:/Users/user/Desktop/機器學習教學/titanic/train.csv")
#train_data = pd.read_csv("C:\\Users\\user\\Desktop\\機器學習教學\\titanic\\train.csv")
#test_data = pd.read_csv(r"C:\Users\user\Desktop\機器學習教學\titanic\test.csv")

# 相對位置
train_data = pd.read_csv("../input/digit-recognizer/train.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")
    
print("\n\n訓練集資料：\n\n",train_data)
print("\n\n測試集資料：\n\n",test_data)


# In[ ]:


train_data.info()
#==============================================================================================
# 觀察知此資料無缺值(NAN)
#==============================================================================================


# In[ ]:


#=============================================================================================================
# Step1：資料觀察與預處理
#　｜
# 　－－－Step1-2 觀察資料 (介紹筆者喜愛使用指令)
#　　　　　｜　　　　
#　　　　　－－－ Step1-2-1　Pandas.DataFrame 類型
#　　　　　｜　　　　｜
#　　　　　｜　　　　－－－Step1-2-1-1 .columns      :判斷資料的行數及名稱
#　　　　　｜　　　　｜
#　　　　　｜　　　　－－－Step1-2-1-2 .info()　　　　：主因方便確認每行(column)是否有缺值(NAN)，和資料型態大小與行列數。
#　　　　　｜　　　　｜
#　　　　　｜　　　　－－－Step1-2-1-3 .shape        :這也可也用來單純確認，資料行列數(形狀)
#　　　　　｜　　　　｜
#　　　　　｜　　　　－－－Step1-2-1-4 .describe()　 ：描述資料基本狀態(如:平均值、標準差、最大值...等)
#　　　　　｜
#　　　　　｜　　　　
#　　　　　－－－Step1-2-2　Numpy.narray 類型
#　　　　　　　　　｜
#　　　　　　　　　－－－1-2-1 .unique()　　　：可以顯現出所有不重複的元素，可以判別標籤(label)內有幾類等。 
#　　　　　　　　　｜
#　　　　　　　　　－－－1-2-2 np.sort()　　　：蠻常需要將數值比大小等，可以將資料排序方便觀察。
#=============================================================================================================


# In[ ]:


#=============================================================================================================
# Step1：資料觀察與預處理
#　｜
# 　－－－Step1-2 觀察資料 (介紹筆者喜愛使用指令)
#　　　　　｜　　　　
#　　　　　－－－ Step1-2-1　Pandas.DataFrame 類型
#　　　　　 　　　　｜
#　　　　　 　　　　－－－Step1-2-1-1 .columns      :判斷資料的行數及名稱
#
#=============================================================================================================
# 觀察知有784個特徵及1個標籤
#=============================================================================================================
train_data.columns


# In[ ]:


#=============================================================================================================
# Step1：資料觀察與預處理
#　｜
# 　－－－Step1-2 觀察資料 (介紹筆者喜愛使用指令)
#　　　　　｜　　　　
#　　　　　－－－ Step1-2-1　Pandas.DataFrame 類型
#　　　　　 　　　　｜
#　　　　　　 　　　－－－Step1-2-1-2 .info()　　　　：主因方便確認每行(column)是否有缺值(NAN)，和資料型態大小與行列數。
#=============================================================================================================
# 觀察知資料無缺值
#=============================================================================================================
train_data.info()


# In[ ]:


#=============================================================================================================
# Step1：資料觀察與預處理
#　｜
# 　－－－Step1-2 觀察資料 (介紹筆者喜愛使用指令)
#　　　　　｜　　　　
#　　　　　－－－ Step1-2-1　Pandas.DataFrame 類型
#　　　　　 　　　　｜
#　　　　　 　　　　－－－Step1-2-1-3 .shape        :這也可也用來單純確認，資料行列數(形狀)
#
#=============================================================================================================
# 觀察到資料有42000列(rows)代表有42000筆資料，有785行(columns，但是其中一行(column)為標籤)，代表特徵只有784個。
#=============================================================================================================
train_data.shape 


# In[ ]:


#=============================================================================================================
# Step1：資料觀察與預處理
#　｜
# 　－－－Step1-2 觀察資料 (介紹筆者喜愛使用指令)
#　　　　　｜　　　　
#　　　　　－－－ Step1-2-1　Pandas.DataFrame 類型
#　　　　　 　　　　｜
#　　　　　 　　　　－－－Step1-2-1-4 .describe()　 ：描述資料基本狀態(如:平均值、標準差、最大值...等)
#
#=============================================================================================================
# 因資料為圖片以矩陣模式呈現，故數值較無法直覺看出端倪，不像現實生活的例子較好推測關聯性。
#=============================================================================================================

train_data.describe() 


# In[ ]:


#=============================================================================================================
# Step1：資料觀察與預處理
#　｜
# 　－－－Step1-2 觀察資料 (介紹筆者喜愛使用指令)
#　　　　　｜　　　　
#　　　　　－－－Step1-2-2　Numpy.narray 類型
#　　　　　　　　　｜
#　　　　　　　　　－－－1-2-1 .unique()　　　：可以顯現出所有不重複的元素，可以判別標籤(label)內有幾類等。 
#　　　　　　　　　｜
#　　　　　　　　　－－－1-2-2 np.sort()　　　：蠻常需要將數值比大小等，可以將資料排序方便觀察。
#
#=============================================================================================================
# 可轉型態觀察那在這就不特別轉型態，直接結合兩個指令一起操作，得知label標籤共有10個分別是0~9
#=============================================================================================================
np.sort(train_data['label'].unique())


# In[ ]:


#=============================================================================================================
# Step1：資料觀察與預處理
#　｜
# 　－－－Step1-3 處理資料的NAN (沒有一定硬性規定，只要能讓資料變得好訓練都可以)
#　　　　　｜　　　　
#　　　　　－－－Step1-3-1 遺失比例極小以至於不影響Data時
#　　　　　｜　　　　｜
#　　　　　｜　　　　－－－Step1-3-1 .columns      :直接捨棄
#　　　　　｜　　　  (例如：有1000筆資料只有1筆資料填寫不完整，此時只皆捨棄使用999筆資料去訓練，論時間成本會好些)
#　　　　　｜
#　　　　　｜
#　　　　　－－－Step1-3-2 遺失比例極高以至於不影響Data時
#　　　　　　　　　｜
#　　　　　　　　　－－－1-3-1 當遺失資料為數值型(連續)：可以使用平均數代替缺失值。
#　　　　　　　　　｜
#　　　　　　　　　－－－1-3-2 當遺失資料為分類型(離散)：可以使用眾數值代替缺失值。
#
#=============================================================================================================
# 因此資料集無缺失值故跳過此步驟。
#=============================================================================================================


# In[ ]:


#=============================================================================================================
# Step1：資料觀察與預處理
#　｜
# 　－－－Step1-4 挑選特徵並將資料正規化與編碼
#　　　　　｜－－－Step1-4-1 挑選特徵
#　　　　　｜　　　　(淘汰不重要的特徵，例如名字、學號...等具備唯一性值，通常沒有什麼參考性，因為大家都不一樣較難有共通點)
#　　　　　｜
#　　　　　｜－－－Step1-4-2 將資料正規化
#                 (將數值型資料正規化，映射至[0,1]之間，避免overflow也可以加快運算)
#　　　　　｜
#　　　　　｜－－－Step1-4-3 將資料編碼
#                 (將分類型資料編碼，因若是字串比較時間複雜度會比數值型高上許多，所以需要做編碼加快運算)
#
#=============================================================================================================
# Step1-4-1 挑選特徵
# 因圖片為矩陣呈現，故特徵行(columns)皆有相依的關係，所以無法捨去任何一個特徵。
#
# Step1-4-2 將資料正規化
# 因矩灰階圖片矩陣數值為0-255之間若要映射至0-1之間，可使用Z轉換等方法，但也可以直接/255，這樣可以達到將數值轉換成[0,1]之間
# (但這裡先不/255因為尚未將特徵跟標籤分離，所以直接/255會有影響)
#
# Step1-4-3 將資料編碼
# 可以使用onehot編碼，但label標籤本身也符合label編碼故這裡不再更改。
#=============================================================================================================


# In[ ]:


#================================================================================
# Step2.模型選擇與建立(Data choose and build)
#　｜
#　－－－模型選擇簡易分類
#　　　　　｜
#　　　　　－－－Step2-1 監督式學習(Data含有標籤)
#　　　　　|　　　　｜
#　　　　　|　　　　－－－Step2-1-1 線性回歸(Linear Regression)
#　　　　　|　　　　｜
#　　　　　|　　　　－－－Step2-1-2 分類(Classification)
#　　　　　|　　　　　　　　｜
#　　　　　|　　　　　　　　－－－Step2-1-1 支持向量機(Support Vector Machines)
#　　　　　|　　　　　　　　｜
#　　　　　|　　　　　　　　－－－Step2-1-2 最近的鄰居(Nearest Neighbors)
#　　　　　|　　　　　　　　｜
#　　　　　|　　　　　　　　－－－Step2-1-3 決策樹(Decision Trees)
#　　　　　|　　　　　　　　｜
#　　　　　|　　　　　　　　－－－Step2-1-4 隨機森林(Forests of randomized trees)
#　　　　　|　　　　　　　　｜
#　　　　　|　　　　　　　　－－－Step2-1-5 神經網路(Neural Network models)
#　　　　　|　　　　　　　　｜
#　　　　　|　　　　　　　　－－－Step2-1-6 高斯過程(GaussianProcess)｜
#　　　　　|
#　　　　　－－－Step2-2 無監督式學習(Data無標籤)
#　　　　　 　　　　｜
#　　　　　 　　　　－－－Step2-2-1 聚類(Clustering)
#　　　　　 　　　　　　　　｜
#　　　　　 　　　　　　　　－－－Step2-2-1 K均值(K-means)
#　　　　　 　　　　　　　　｜
#　　　　　 　　　　　　　　－－－Step2-2-2 神經網路(Neural Network models)
#================================================================================


# In[ ]:


#================================================================================　　　　　 　　　　　　　　｜
#　此類型問題為分類問題，筆者會在這使用每個模型，請大家可以跟著一起練習。
#　先將資料分成特徵與標籤（圖形矩陣呈現（特徵），為數字0~9（標籤））
#================================================================================
train_data_feature = train_data.drop('label',axis=1)  #捨去label特徵
train_data_label = train_data['label']


# In[ ]:


#================================================================================　　　　　 　　　　　　　　｜
#　特徵與標籤分開後就可以直接/255將值映射至[0,1]之間
#================================================================================
train_data_feature = train_data_feature/255


# In[ ]:


#==========================================================================================
# Step2.模型選擇與建立(Data choose and build)
#　　|
#　　－－－Step2-1-2 分類(Classification)
#　　　　　｜
#　　　　　－－－Step2-1-1 支持向量機(Support Vector Machines)
#
# train_test_split 套件功能，將資料集分成訓練及與測試集合(測試時可以看出是否過度擬合(overfitting))
#==========================================================================================
from sklearn.model_selection import train_test_split
from sklearn import svm
train_feature, test_feature, train_label, test_label = train_test_split(train_data_feature,train_data_label,test_size=0.2)

svm_model = svm.SVC()
svm_model.fit(train_feature, train_label)

#將原本train_data資料，分成訓練集與測試集合，並丟入訓練好的模型測試準確度
print ("支持向量機(Support Vector Machines)模型準確度(訓練集):",svm_model.score(train_feature, train_label))
print ("支持向量機(Support Vector Machines)模型準確度(測試集):",svm_model.score(test_feature, test_label))
svm_model_acc = svm_model.score(test_feature, test_label)

#==========================================================================================
# 觀察知訓練集與測試集的準確度無落差，故判斷支持向量機(Support Vector Machines)訓練無過度擬合的情形
#==========================================================================================



# In[ ]:


#===============================================================================================
# Step2.模型選擇與建立(Data choose and build)
#　　|
#　　－－－Step2-1-2 分類(Classification)
#　　　　　｜
#　　　　　－－－Step2-1-2 最近的鄰居(Nearest Neighbors)
#
# train_test_split 套件功能，將資料集分成訓練及與測試集合(測試時可以看出是否過度擬合(overfitting))
#===============================================================================================
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
train_feature, test_feature, train_label, test_label = train_test_split(train_data_feature,train_data_label,test_size=0.2)

KNeighbors_model = KNeighborsClassifier(n_neighbors=2)
KNeighbors_model.fit(train_feature, train_label)

#將原本train_data資料，分成訓練集與測試集合，並丟入訓練好的模型測試準確度
print ("最近的鄰居(Nearest Neighbors)模型準確度(訓練集)：",KNeighbors_model.score(train_feature, train_label))
print ("最近的鄰居(Nearest Neighbors)模型準確度(測試集)：",KNeighbors_model.score(test_feature, test_label))
KNeighbors_model_acc = KNeighbors_model.score(test_feature, test_label)

#==========================================================================================
# 觀察知訓練集與測試集的準確度有落差，故判斷最近的鄰居(Nearest Neighbors)訓練有過度擬合的情形
#==========================================================================================


# In[ ]:


#===============================================================================================
# Step2.模型選擇與建立(Data choose and build)
#　　|
#　　－－－Step2-1-2 分類(Classification)
#　　　　　｜
#　　　　　－－－Step2-1-3 決策樹(Decision Trees)
#
# train_test_split 套件功能，將資料集分成訓練及與測試集合(測試時可以看出是否過度擬合(overfitting))
#===============================================================================================
from sklearn.model_selection import train_test_split
from sklearn import tree
train_feature, test_feature, train_label, test_label = train_test_split(train_data_feature,train_data_label,test_size=0.2)

DecisionTree_model = tree.DecisionTreeClassifier()
DecisionTree_model.fit(train_feature, train_label)

#將原本train_data資料，分成訓練集與測試集合，並丟入訓練好的模型測試準確度
print ("決策樹(Decision Trees)模型準確度(訓練集)：",DecisionTree_model.score(train_feature, train_label))
print ("決策樹(Decision Trees)模型準確度(測試集)：",DecisionTree_model.score(test_feature, test_label))
DecisionTree_model_acc = DecisionTree_model.score(test_feature, test_label)

#==========================================================================================
# 觀察知訓練集與測試集的準確度有落差，故判斷決策樹(Decision Trees)訓練有過度擬合的情形
#==========================================================================================


# In[ ]:


#========================================================================================
# Step2.模型選擇與建立(Data choose and build)
#　　|
#　　－－－Step2-1-2 分類(Classification)
#　　　　　｜
#　　　　　－－－Step2-1-4 隨機森林(Forests of randomized trees)
#
# train_test_split 套件功能，將資料集分成訓練及與測試集合(測試時可以看出是否過度擬合(overfitting))
#========================================================================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
train_feature, test_feature, train_label, test_label = train_test_split(train_data_feature,train_data_label,test_size=0.2)

RandomForest_model = RandomForestClassifier(n_estimators=10)
RandomForest_model.fit(train_feature, train_label)

#將原本train_data資料，分成訓練集與測試集合，並丟入訓練好的模型測試準確度
print ("隨機森林(Forests of randomized trees)模型準確度(訓練集)：",RandomForest_model.score(train_feature, train_label))
print ("隨機森林(Forests of randomized trees)模型準確度(測試集)：",RandomForest_model.score(test_feature, test_label))
RandomForest_model_model_acc = RandomForest_model.score(test_feature, test_label)


#==========================================================================================
# 觀察知訓練集與測試集的準確度有落差，故判斷隨機森林(Forests of randomized trees)訓練有過度擬合的情形
#==========================================================================================


# In[ ]:


#=============================================================================================
# Step2.模型選擇與建立(Data choose and build)
#　　|
#　　－－－Step2-1-2 分類(Classification)
#　　　　　｜
#　　　　　－－－Step2-1-5 神經網路(Neural Network models)
#
# train_test_split 套件功能，將資料集分成訓練及與測試集合(測試時可以看出是否過度擬合(overfitting))
#=============================================================================================
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
train_feature, test_feature, train_label, test_label = train_test_split(train_data_feature,train_data_label,test_size=0.2)

MLP_model = MLPClassifier(solver='lbfgs', 
                                   alpha=1e-5,
                                   hidden_layer_sizes=(10, 10), 
                                   )
MLP_model.fit(train_feature, train_label)

#將原本train_data資料，分成訓練集與測試集合，並丟入訓練好的模型測試準確度
print ("神經網路(Neural Network models)模型準確度(訓練集)：",MLP_model.score(train_feature, train_label))
print ("神經網路(Neural Network models)模型準確度(測試集)：",MLP_model.score(test_feature, test_label))
MLP_model_acc = MLP_model.score(test_feature, test_label)

#==========================================================================================
# 觀察知訓練集與測試集的準確度無落差，故判斷神經網路(Neural Network models)訓練無過度擬合的情形
#==========================================================================================


# In[ ]:


"""
#=============================================================================================
# Step2.模型選擇與建立(Data choose and build)
#　　|
#　　－－－Step2-1-2 分類(Classification)
#　　　　　｜
#　　　　　－－－Step2-1-6 高斯過程(GaussianProcess)
#
# train_test_split 套件功能，將資料集分成訓練及與測試集合(測試時可以看出是否過度擬合(overfitting))
#=============================================================================================
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
train_feature, test_feature, train_label, test_label = train_test_split(train_data_feature,train_data_label,test_size=0.2)

GaussianProcess_model = GaussianProcessClassifier()
GaussianProcess_model.fit(train_feature, train_label)

#將原本train_data資料，分成訓練集與測試集合，並丟入訓練好的模型測試準確度
print ("高斯過程(GaussianProcess)模型準確度(訓練集)：",GaussianProcess_model.score(train_feature, train_label))
print ("高斯過程(GaussianProcess)模型準確度(測試集)：",GaussianProcess_model.score(test_feature, test_label))
GaussianProcess_model_acc = GaussianProcess_model.score(test_feature, test_label)


#==========================================================================================
# 記憶體不足無法順利訓練 ...  小小筆電這裡就不催它了  
#==========================================================================================
"""


# In[ ]:


models = pd.DataFrame({
    'Model': ['支持向量機(Support Vector Machines)', 
              '最近的鄰居(Nearest Neighbors)', 
              '決策樹(Decision Trees)',
              '隨機森林(Forests of randomized trees)', 
              '神經網路(Neural Network models)'
              #'高斯過程(GaussianProcess)' 
             ],
    'Score': [svm_model_acc,
              KNeighbors_model_acc,
              DecisionTree_model_acc,
              RandomForest_model_model_acc,
              MLP_model_acc
              #GaussianProcess_model_acc, 
              ]
                       })
models.sort_values(by='Score', ascending=False)



# In[ ]:


#==========================================================================================
# 將要預測的資料集仔載入(此資料不含標籤)，等於為測試集的特徵。
#==========================================================================================
test_data_feature = test_data


# In[ ]:


#==========================================================================================
# 選擇使用SVM模型，故先重新訓練一次，座使用test_data做預測
#==========================================================================================
from sklearn.model_selection import train_test_split
from sklearn import svm
train_feature, test_feature, train_label, test_label = train_test_split(train_data_feature,train_data_label,test_size=0.2)

svm_model = svm.SVC()
svm_model.fit(train_feature, train_label)

#將原本train_data資料，分成訓練集與測試集合，並丟入訓練好的模型測試準確度
print ("支持向量機(Support Vector Machines)模型準確度(訓練集):",svm_model.score(train_feature, train_label))
print ("支持向量機(Support Vector Machines)模型準確度(測試集):",svm_model.score(test_feature, test_label))
svm_model_acc = svm_model.score(test_feature, test_label)


test_data_label = svm_model.predict(test_data_feature)
test_data_label


#==========================================================================================
# 觀察知所有資料皆預測為5準確度有問題，故更改其他方式。
#==========================================================================================


# In[ ]:


#==========================================================================================
# 選擇使用隨機森林(Forests of randomized trees)模型，故先重新訓練一次，座使用test_data做預測
#==========================================================================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
train_feature, test_feature, train_label, test_label = train_test_split(train_data_feature,train_data_label,test_size=0.2)

RandomForest_model = RandomForestClassifier(n_estimators=10)
RandomForest_model.fit(train_feature, train_label)


test_data_label =RandomForest_model.predict(test_data_feature)
test_data_label

#==========================================================================================
# 觀察知預測不像svm一樣有異狀故採用。
#==========================================================================================


# In[ ]:


#==========================================================================================
# 讀取官網給予的繳交檔案，再將預測的結果寫入繳交預測。
#==========================================================================================
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")


# In[ ]:


sample_submission


# In[ ]:


#===============================================================================================
# 官網教學將你的預測資料輸出上傳，恭喜完成 =")
#===============================================================================================
output = pd.DataFrame({'ImageId': sample_submission["ImageId"], 'Label': test_data_label})
output


# In[ ]:


output.to_csv('Submission.csv', index=False)

