#!/usr/bin/env python
# coding: utf-8

# # Costa Rican Household Poverty Level Prediction 

# 지도학습이자 4중분류의 분류문제이며 예측하고자 하는 대상은 빈곤도를 나타내는 4개의 등급 (1단계가 최빈곤)

# 주어진 몇가지 column을 살펴보면  
#   
# idhogar : 각각의 가구를 구별해주는 unique한 요소이며 같은 가구내에 있는 구성원들은 같은 idhogar를 갖는다  
# parentesco1 : 가장인지의 여부를 알려준다  
#   
# Metric : 점수 평가방식은 Macro F1 score (Class가 여러개인 경우 각 Class별 Macro F1 score의 평균)  
#   
# from sklearn.metrics import f1_score  
# f1_score(y_true, y_predicted, average = 'macro`)  
#   
#   
# label(빈곤도 등급)이 균일하지않기 때문에 이를 감안하여 분석을 진행해야한다.
# 
# 

# # 분석 Roadmap
#   
# 1. 문제에 대한 이해 
# 2. 데이터 탐색
# 3. 머신러닝을 활용하기 위한 Feature engineering
# 4. 여러 기본적인 머신러닝 모델간의 비교
# 5. 보다 복잡한 머신러닝 모델의 비교 
# 6. 모델 최적화
# 7. 모델을 통한 예측과 분석
# 8. 결론 도출
#   

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns 


# In[ ]:


pd.options.display.max_columns  = None # 모든 column 확인


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()


# In[ ]:


#train.info()
train.info(verbose = 1, null_counts = True)


# column이 많은 경우에는 압축해서 해당 내용을 보여주는데 모두 출력하고 싶다면, verbose = 1 null_counts = True option을 주면 됩니다.  
# object type의 경우 모델에 바로 적용되기 힘들 수 있으므로 해당 column이 어떤 정보를 담고 있는지 확인해야 합니다.

# In[ ]:


test.info(verbose = 1, null_counts = True)


# 해당 내용이 int형인 column들의 unique 개수를 통해 2개인 것들은 0과 1로 무언가 True/False의 boolean형태를 표현하고 있다고 추론해볼 수 있겠습니다. 상당수의 column이 unique개수가 2개인것으로 보아 column은 boolean 형태의 질문들이 많은거 같습니다.

# In[ ]:


train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'blue', 
                                                                             figsize = (8, 6),
                                                                            edgecolor = 'k', linewidth = 2);
plt.xlabel('Number of Unique Values'); plt.ylabel('Count');
plt.title('Count of Unique Values in Integer Columns');


# 예를들면 refrig column은 집에 냉장고가 있는가? 에대한 값입니다. 있으면 1 없으면 0 이런식이겠죠?

# Float columns : float을 담고있는 column의 경우 연속적인 변수를 가지고 있습니다. 때문에 해당 column의 분포를 보는 것이 유의미하다고 판단됩니다.

# In[ ]:


from collections import OrderedDict

plt.figure(figsize = (20, 16))
plt.style.use('fivethirtyeight')

# Color mapping
colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})

# Iterate through the float columns
for i, col in enumerate(train.select_dtypes('float')):
    ax = plt.subplot(4, 2, i + 1)
    # Iterate through the poverty levels
    for poverty_level, color in colors.items():
        # Plot each poverty level as a separate line
        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 
                    ax = ax, color = color, label = poverty_mapping[poverty_level])
        
    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')

plt.subplots_adjust(top = 2)


# float 형태의 column에 대하여 해당 값들의 밀도를 찍어본 결과 class를 분류하는데에 용이하게 사용될 수 있음을 확인할 수 있다.  
# meaneduc 같은 경우 최종학력으로 볼 수 있는데 등급에 따라 그 평균값이 다르다.

# In[ ]:


# v2a1, Monthly rent payment
# v18q1, number of tablets household owns
# rez_esc, Years behind in school
# meaneduc,average years of education for adults (18+)
# overcrowding, # persons per room
# SQBovercrowding, overcrowding squared
# SQBdependency, dependency squared
# SQBmeaned, square of the mean years of education of adults (>=18) in the household


# object column은 다음과 같다  
# id와 idhogar는 개인과 가정을 구분짓는 column  
# dependency: Dependency rate, calculated = (19세이하 64세이상 명수)/(19세 초과 64세 미만 명수)  
# edjefe: years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0  # 학력 및 가장(남자)인지 yes / no로 표기되어있음  
# edjefa: years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0  # 학력 및 가장(여자)인지 yes / no로 표기되어있음  
# 

# In[ ]:


train.select_dtypes('object').head() #dtype이 섞여 있음 


# yes / no 값을 1 과 0으로 대응시켜주자 

# In[ ]:


mapping = {"yes": 1, "no": 0}

# Apply same operation to both train and test
for df in [train, test]:
    # Fill in the values with the correct mapping
    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)
    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)

train[['dependency', 'edjefa', 'edjefe']].describe()


# In[ ]:


plt.figure(figsize = (16, 12))

# Iterate through the float columns
for i, col in enumerate(['dependency', 'edjefa', 'edjefe']):
    ax = plt.subplot(3, 1, i + 1)
    # Iterate through the poverty levels
    for poverty_level, color in colors.items():
        # Plot each poverty level as a separate line
        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 
                    ax = ax, color = color, label = poverty_mapping[poverty_level])
        
    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')

plt.subplots_adjust(top = 2)


# 이제 어느정도 사용가능한 column이 되었고 실제로도 빈곤도와 어느정도 밀접한 관련이 있을것으로 보여집니다.  
# 추후에 예측값을 넣어주기 위해 test dataset에 Target column을 만들고 nan으로 채워주었습니다.

# In[ ]:


# Add null Target column to test
test['Target'] = np.nan
data = train.append(test, ignore_index = True)


# 이번에는 label을 중심으로 데이터를 살펴보겠습니다.  
# 그 가정의 가장인 사람의 등급을 보면 해당 가구별 빈곤도를 산출할 수 있습니다.

# In[ ]:


# Heads of household
heads = data.loc[data['parentesco1'] == 1].copy()

# Labels for training
train_labels = data.loc[(data['Target'].notnull()) & (data['parentesco1'] == 1), ['Target', 'idhogar']]

# Value counts of target
label_counts = train_labels['Target'].value_counts().sort_index()

# Bar plot of occurrences of each label
label_counts.plot.bar(figsize = (8, 6), 
                      color = colors.values(),
                      edgecolor = 'k', linewidth = 2)
# Formatting
plt.xlabel('Poverty Level'); plt.ylabel('Count'); 
plt.xticks([x - 1 for x in poverty_mapping.keys()], 
           list(poverty_mapping.values()), rotation = 60)
plt.title('Poverty Level Breakdown');

label_counts


# label값을 보니 굉장히 target 4에 편중되어있다는 것을 알 수 있습니다.  
# 때문에 평가방식이 weighted F1이 아닌 macro F1을 사용하는 것으로 보여집니다.  
# 또한 OverSampling 혹은 DownSampling을 통해서 score를 높일 수 있을것으로 보여집니다.  
# 해당 데이터의 경우에는 OverSampling이 적당해 보입니다. 

# 해당 빈곤도 데이터에는 오류가 있을 수 있는데 가정은 전체적으로 가난하지만, 특정 개인은 가난하지 않을 수 있습니다.  
# 따라서 가장의 빈곤도만을 가지고 해당 가정이 전부 같은 등급이라고 판단하는 것이 100% 정확한 추론은 아닙니다.  
# 실제로 그러한 데이터가 있는지 확인해보도록 하겠습니다.

# In[ ]:


# Groupby the household and figure out the number of unique values
all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))


# In[ ]:


train[train['idhogar'] == not_equal.index[0]][['idhogar', 'parentesco1', 'Target']]


# 가장이 없는 가구에 대한 분석  
# 가장이 없는 가구의 경우 각 개인의 빈곤도에 차이가 있는지 확인할 필요가 있음

# In[ ]:


households_leader = train.groupby('idhogar')['parentesco1'].sum()

# Find households without a head
households_no_head = train.loc[train['idhogar'].isin(households_leader[households_leader == 0].index), :]

print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))


# In[ ]:


# Find households without a head and where labels are different
households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
print('{} Households with no head have different labels.'.format(sum(households_no_head_equal == False)))


# 가구의 구성원 수에 따라 데이터가 편중될 수 있으니 가구의 대표인 가장의 값만 가지고 train을 진행합니다.  
# 가장이 없는 경우는 15개밖에 없으므로 이상치로 보고 제외하겠습니다.  
# 인구조사의 특성상 오류가 있을 수 있기 때문에 가장이 없는 데이터가 간혹 생기는거 같습니다.  

# In[ ]:


# Iterate through each household
for household in not_equal.index:
    # Find the correct label (for the head of household)
    true_target = int(train[(train['idhogar'] == household) & (train['parentesco1'] == 1.0)]['Target'])
    
    # Set the correct label for all members in the household
    train.loc[train['idhogar'] == household, 'Target'] = true_target
    
    
# Groupby the household and figure out the number of unique values
all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))


# 데이터 탐색과정에서 가장 중요하다고 할 수 있는 결측치에 관한 내용입니다.  
# 모델에 적용하기 전 결측치를 제외하거나 채워주는 형식으로 결측치를 제거해야합니다.  
# 보통 결측치의 경우 기존의 feature에 기반하여 채웁니다.  

# In[ ]:


# Number of missing in each column
missing = pd.DataFrame(data.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(data)

missing.sort_values('percent', ascending = False).head(10).drop('Target')


# v18q1: Number of tablet

# In[ ]:


def plot_value_counts(df, col, heads_only = False):
    """Plot value counts of a column, optionally with only the heads of a household"""
    # Select heads of household
    if heads_only:
        df = df.loc[df['parentesco1'] == 1].copy()
        
    plt.figure(figsize = (8, 6))
    df[col].value_counts().sort_index().plot.bar(color = 'blue',
                                                 edgecolor = 'k',
                                                 linewidth = 2)
    plt.xlabel(f'{col}') 
    plt.title(f'{col} Value Counts')
    plt.ylabel('Count')
    plt.show()


# In[ ]:


plot_value_counts(heads, 'v18q1')


# In[ ]:


heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())


# v18q1의 경우 테블릿이 없는 가구는 0으로 채워준다

# In[ ]:


data['v18q1'] = data['v18q1'].fillna(0)


# v2a1 : 월세

# In[ ]:


# Variables indicating home ownership
own_variables = [x for x in data if x.startswith('tipo')]


# Plot of the home ownership variables for home missing rent payments
data.loc[data['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (10, 8),
                                                                        color = 'green',
                                                              edgecolor = 'k', linewidth = 2);
plt.xticks([0, 1, 2, 3, 4],
           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],
          rotation = 60)
plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18)


# tipovivi1, =1 own and fully paid house  
# tipovivi2, =1 own, paying in installments  
# tipovivi3, =1 rented  
# tipovivi4, =1 precarious  
# tipovivi5, =1 other(assigned,  borrowed)  

# In[ ]:


# Fill in households that own the house with 0 rent payment
data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0

# Create missing rent payment column
data['v2a1-missing'] = data['v2a1'].isnull()

data['v2a1-missing'].value_counts()


# rez_esc: years behind in school
# 
# 

# In[ ]:


data.loc[data['rez_esc'].notnull()]['age'].describe()


# In[ ]:


data.loc[data['rez_esc'].isnull()]['age'].describe()


# 이 feature의 경우 교육관련된 특징이기 때문에 7 -19세 사이에만 데이터가 들어가있다.  
# 7세미만 19세 초과의 경우 모두 0을 넣어줘도 된다.

# In[ ]:


# If individual is over 19 or younger than 7 and missing years behind, set it to 0
data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0

# Add a flag for those between 7 and 19 with a missing value
data['rez_esc-missing'] = data['rez_esc'].isnull()


# rez_esc의 최댓값은 5이므로 5보다 큰 값은 모두 5로 바꿔준다.

# In[ ]:


data.loc[data['rez_esc'] > 5, 'rez_esc'] = 5


# Target과 변수간의 관계를 알 수 있는 plot을 그리는 함수

# In[ ]:


def plot_categoricals(x, y, data, annotate = True):
    """Plot counts of two categoricals.
    Size is raw count for each grouping.
    Percentages are for a given value of y."""
    
    # Raw counts 
    raw_counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = False))
    raw_counts = raw_counts.rename(columns = {x: 'raw_count'})
    
    # Calculate counts for each group of x and y
    counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = True))
    
    # Rename the column and reset the index
    counts = counts.rename(columns = {x: 'normalized_count'}).reset_index()
    counts['percent'] = 100 * counts['normalized_count']
    
    # Add the raw count
    counts['raw_count'] = list(raw_counts['raw_count'])
    
    plt.figure(figsize = (14, 10))
    # Scatter plot sized by percent
    plt.scatter(counts[x], counts[y], edgecolor = 'k', color = 'lightgreen',
                s = 100 * np.sqrt(counts['raw_count']), marker = 'o',
                alpha = 0.6, linewidth = 1.5)
    
    if annotate:
        # Annotate the plot with text
        for i, row in counts.iterrows():
            # Put text with appropriate offsets
            plt.annotate(xy = (row[x] - (1 / counts[x].nunique()), 
                               row[y] - (0.15 / counts[y].nunique())),
                         color = 'navy',
                         s = f"{round(row['percent'], 1)}%")
        
    # Set tick marks
    plt.yticks(counts[y].unique())
    plt.xticks(counts[x].unique())
    
    # Transform min and max to evenly space in square root domain
    sqr_min = int(np.sqrt(raw_counts['raw_count'].min()))
    sqr_max = int(np.sqrt(raw_counts['raw_count'].max()))
    
    # 5 sizes for legend
    msizes = list(range(sqr_min, sqr_max,
                        int(( sqr_max - sqr_min) / 5)))
    markers = []
    
    # Markers for legend
    for size in msizes:
        markers.append(plt.scatter([], [], s = 100 * size, 
                                   label = f'{int(round(np.square(size) / 100) * 100)}', 
                                   color = 'lightgreen',
                                   alpha = 0.6, edgecolor = 'k', linewidth = 1.5))
        
    # Legend and formatting
    plt.legend(handles = markers, title = 'Counts',
               labelspacing = 3, handletextpad = 2,
               fontsize = 16,
               loc = (1.10, 0.19))
    
    plt.annotate(f'* Size represents raw count while % is for a given y value.',
                 xy = (0, 1), xycoords = 'figure points', size = 10)
    
    # Adjust axes limits
    plt.xlim((counts[x].min() - (6 / counts[x].nunique()), 
              counts[x].max() + (6 / counts[x].nunique())))
    plt.ylim((counts[y].min() - (4 / counts[y].nunique()), 
              counts[y].max() + (4 / counts[y].nunique())))
                         
    plt.grid(None)
    plt.xlabel(f"{x}") 
    plt.ylabel(f"{y}")
    plt.title(f"{y} vs {x}")


# In[ ]:


plot_categoricals('rez_esc', 'Target', data)


# In[ ]:


plot_categoricals('escolari', 'Target', data, annotate = False)


# 나머지 결측치에 대해서는 중간값을 넣어주기로 한다. 중간값을 결측치에 넣어주는 것은 가장 흔한방법이다.  
#   
#   
# 다음은 어떤 비율만큼의 Target이 결측치로서 채워졌는지 확인하는 그래프이다. 

# In[ ]:


plot_value_counts(data[(data['rez_esc-missing'] == 1)], 
                  'Target')


# In[ ]:


plot_value_counts(data[(data['v2a1-missing'] == 1)], 
                  'Target')


# 전체 데이터에서는 Target4의 비중이 가장 높았는데 'v2a1'변수의 경우 Target2가 가장 많이 채워졌습니다.  
# 즉 결측치가 분석에 있어서 key가 되는 중요한 요소일 수 있습니다!

# # Feature Engineering

# column을 직접 확인하며 분류하는 것은 데이터분석에 있어 피할 수 없는 일입니다.  
# 해당 문제의 데이터가 140가지의 특징을 가지고 있습니다. 데이터 설명을 보며 묶어서 볼만한 요소들을 묶어줘야합니다.

# In[ ]:


# training 시키지 않을 변수들입니다. idhogar를 기준으로 데이터들을 묶어서 볼 것입니다. 
# idhogar = 가구당 id

id_ = ['Id', 'idhogar', 'Target']


# In[ ]:


ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone', 'rez_esc-missing']

ind_ordered = ['rez_esc', 'escolari', 'age']


# In[ ]:


hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1-missing']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']


# In[ ]:


sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 
        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']


# 모든 데이터 column을 포함하고 있는지 확인합니다. 
# 

# In[ ]:


x = ind_bool + ind_ordered + id_ + hh_bool + hh_ordered + hh_cont + sqr_

from collections import Counter

print('There are no repeats: ', np.all(np.array(list(Counter(x).values())) == 1))
print('We covered every variable: ', len(x) == data.shape[1])


# 상호간의 상관관계가 높은 변수들은 같이 가져가게되었을 때 제대로 된 데이터분석이 힘들어집니다. 

# In[ ]:


sns.lmplot('age', 'SQBage', data = data, fit_reg=False); 
plt.title('Squared Age versus Age'); # ; 를 붙여주면 깔끔하게 그래프만 보여주네요


# 두 변수간의 상관관계는 굉장히 높아보입니다. 둘 중의 하나만 가져가거나 두 변수를 통해 새로운 변수를 생성해내야합니다.

# In[ ]:


# Remove squared variables
data = data.drop(columns = sqr_)
data.shape


# Id Variables = 구분을 위한 column이므로 일단은 가져가겠습니다.  
# Household Level Variables = 가장이 있는 가구만 추출하고 가구를 기준으로 묶어주겠습니다. 

# In[ ]:


heads = data.loc[data['parentesco1'] == 1, :]
heads = heads[id_ + hh_bool + hh_cont + hh_ordered]
print(heads.shape)
heads.head()


# 변수간의 상관관계 분석 (corr_matrix)  
# 상관관계가 너무 높은 변수들은 확인 > 버려줘도 되는 변수들은 drop!

# In[ ]:


# Create correlation matrix
corr_matrix = heads.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop


# In[ ]:


corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9]


# In[ ]:


sns.heatmap(corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9],
            annot=True, cmap = plt.cm.autumn_r, fmt='.3f');


# r4t3, Total persons in the household  
# tamhog, size of the household  
# tamviv, number of persons living in the household  
# hhsize, household size  
# hogar_total, # of total individuals in the household  

# 상관관계가 0.9가 넘는 변수들은 보아하니 너무나 비슷한 내용의 변수들입니다. 

# In[ ]:


heads = heads.drop(columns = ['tamhog', 'hogar_total', 'r4t3'])


# In[ ]:


sns.lmplot('tamviv', 'hhsize', data, fit_reg=False, size = 8);
plt.title('Household size vs number of persons living in the household');


# 가족 구성원의 수보다 한 가정에 사는 사람의 수가 더 크다는 것을 알 수 있습니다.  
# 그렇다면 한 가정에 사는 사람의 수 - 가족 구성원의 수가 빈곤도에 따라 어떻게 다르게 분포되는지 보는 것이 도움이 될거같습니다. 

# In[ ]:


heads['hhsize-diff'] = heads['tamviv'] - heads['hhsize']
plot_categoricals('hhsize-diff', 'Target', heads)


# 확연한 차이는 볼 수 없었지만, 빈곤도에 따라 조금씩 그 차이가 다름을 알 수 있습니다.

# 전기사용에 관한 새로운 변수를 만들어보겠습니다.   
# 0: No electricity  
# 1: Electricity from cooperative  
# 2: Electricity from CNFL, ICA, ESPH/JASEC  
# 3: Electricity from private plant

# In[ ]:


elec = []

# Assign values
for i, row in heads.iterrows():
    if row['noelec'] == 1:
        elec.append(0)
    elif row['coopele'] == 1:
        elec.append(1)
    elif row['public'] == 1:
        elec.append(2)
    elif row['planpri'] == 1:
        elec.append(3)
    else:
        elec.append(np.nan)
        
# Record the new variable and missing flag
heads['elec'] = elec
heads['elec-missing'] = heads['elec'].isnull()

# Remove the electricity columns
# heads = heads.drop(columns = ['noelec', 'coopele', 'public', 'planpri'])


# In[ ]:


plot_categoricals('elec', 'Target', heads)


# area1은 집이 도시에 있는지 없는지를 알려주는 내용이므로 도시밖에 있는지 없는지를 알려주는 area2는 굳이 필요하지 않다고 판단했습니다. 

# In[ ]:


heads = heads.drop(columns = 'area2')
heads.groupby('area1')['Target'].value_counts(normalize = True)

# area1과 Target의 비율입니다. 


# 새로운 변수 생성

# In[ ]:


# Wall ordinal variable
heads['walls'] = np.argmax(np.array(heads[['epared1', 'epared2', 'epared3']]),axis = 1)

# heads = heads.drop(columns = ['epared1', 'epared2', 'epared3'])
plot_categoricals('walls', 'Target', heads)


# In[ ]:


# Roof ordinal variable
heads['roof'] = np.argmax(np.array(heads[['etecho1', 'etecho2', 'etecho3']]),
                           axis = 1)
# bad soso good 중에 제일 큰 값

heads = heads.drop(columns = ['etecho1', 'etecho2', 'etecho3'])

# Floor ordinal variable
heads['floor'] = np.argmax(np.array(heads[['eviv1', 'eviv2', 'eviv3']]),
                           axis = 1)
heads = heads.drop(columns = ['eviv1', 'eviv2', 'eviv3'])


# In[ ]:


# Create new feature
heads['walls+roof+floor'] = heads['walls'] + heads['roof'] + heads['floor']

plot_categoricals('walls+roof+floor', 'Target', heads, annotate=False)


# 집의 컨디션을 평가하기위해 화장실, 전기, 바닥, 수도, 천장이 결여되어있으면 penalty를 주도록 했다.

# In[ ]:


# No toilet, no electricity, no floor, no water service, no ceiling
heads['warning'] = 1 * (heads['sanitario1'] + 
                         (heads['elec'] == 0) + 
                         heads['pisonotiene'] + 
                         heads['abastaguano'] + 
                         (heads['cielorazo'] == 0))


# In[ ]:


plot_categoricals('warning', 'Target', data = heads)


# 어느정도 빈부격차가 보입니다.

# 이번에는 반대로 냉장고, 컴퓨터, 텔레비전, 테블릿이 있으면 가점을 주겠습니다.

# In[ ]:


# Owns a refrigerator, computer, tablet, and television
heads['bonus'] = 1 * (heads['refrig'] + 
                      heads['computer'] + 
                      (heads['v18q1'] > 0) + 
                      heads['television'])

sns.violinplot('bonus', 'Target', data = heads,
                figsize = (10, 6));
plt.title('Target vs Bonus Variable');


# 1 명당의 자료를 만들어 비율을 알 수 있습니다. 

# In[ ]:


heads['phones-per-capita'] = heads['qmobilephone'] / heads['tamviv']
heads['tablets-per-capita'] = heads['v18q1'] / heads['tamviv']
heads['rooms-per-capita'] = heads['rooms'] / heads['tamviv']
heads['rent-per-capita'] = heads['v2a1'] / heads['tamviv']


# In[ ]:


# Use only training data
train_heads = heads.loc[heads['Target'].notnull(), :].copy()

pcorrs = pd.DataFrame(train_heads.corr()['Target'].sort_values()).rename(columns = {'Target': 'pcorr'}).reset_index()
pcorrs = pcorrs.rename(columns = {'index': 'feature'})

print('Most negatively correlated variables:')
print(pcorrs.head())

print('\nMost positively correlated variables:')
print(pcorrs.dropna().tail())


# In[ ]:


variables = ['Target', 'dependency', 'warning', 'walls+roof+floor', 'meaneduc',
             'floor', 'r4m1', 'overcrowding']

# Calculate the correlations
corr_mat = train_heads[variables].corr().round(2)

# Draw a correlation heatmap
plt.rcParams['font.size'] = 18
plt.figure(figsize = (12, 12))
sns.heatmap(corr_mat, vmin = -0.5, vmax = 0.8, center = 0, 
            cmap = plt.cm.RdYlGn_r, annot = True);


# Individual Level Variables 

# In[ ]:


ind = data[id_ + ind_bool + ind_ordered]
ind.shape


# Redundant Individual Variables 

# In[ ]:


# Create correlation matrix
corr_matrix = ind.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop


# In[ ]:


ind = ind.drop(columns = 'male')


# In[ ]:


ind[[c for c in ind if c.startswith('instl')]].head()


# In[ ]:


ind['inst'] = np.argmax(np.array(ind[[c for c in ind if c.startswith('instl')]]), axis = 1)

plot_categoricals('inst', 'Target', ind, annotate = False);


# In[ ]:


plt.figure(figsize = (10, 8))
sns.violinplot(x = 'Target', y = 'inst', data = ind);
plt.title('Education Distribution by Target');


# In[ ]:


# Drop the education columns
# ind = ind.drop(columns = [c for c in ind if c.startswith('instlevel')])
ind.shape


# In[ ]:


# Define custom function
range_ = lambda x: x.max() - x.min()
range_.__name__ = 'range_'

# Group and aggregate
ind_agg = ind.drop(columns = 'Target').groupby('idhogar').agg(['min', 'max', 'sum', 'count', 'std', range_])
ind_agg.head()


# In[ ]:


# Rename the columns
new_col = []
for c in ind_agg.columns.levels[0]:
    for stat in ind_agg.columns.levels[1]:
        new_col.append(f'{c}-{stat}')
        
ind_agg.columns = new_col
ind_agg.head()


# # Feature Selection

# In[ ]:


# Create correlation matrix
corr_matrix = ind_agg.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

print(f'There are {len(to_drop)} correlated columns to remove.')


# In[ ]:


ind_agg = ind_agg.drop(columns = to_drop)
ind_feats = list(ind_agg.columns)

# Merge on the household id
final = heads.merge(ind_agg, on = 'idhogar', how = 'left')

print('Final features shape: ', final.shape)


# In[ ]:


final.head()


# # Modeling

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Custom scorer for cross validation
scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')


# In[ ]:


# Labels for training
train_labels = np.array(list(final[final['Target'].notnull()]['Target'].astype(np.uint8)))

# Extract the training data
train_set = final[final['Target'].notnull()].drop(columns = ['Id', 'idhogar', 'Target'])
test_set = final[final['Target'].isnull()].drop(columns = ['Id', 'idhogar', 'Target'])

# Submission base which is used for making submissions to the competition
submission_base = test[['Id', 'idhogar']].copy()


# In[ ]:


features = list(train_set.columns)

pipeline = Pipeline([('imputer', Imputer(strategy = 'median')), 
                      ('scaler', MinMaxScaler())])

# Fit and transform training data
train_set = pipeline.fit_transform(train_set)
test_set = pipeline.transform(test_set)


# In[ ]:


model = RandomForestClassifier(n_estimators=100, random_state=10, 
                               n_jobs = -1)
# 10 fold cross validation
cv_score = cross_val_score(model, train_set, train_labels, cv = 10, scoring = scorer)

print(f'10 Fold Cross Validation F1 Score = {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}')


# In[ ]:


model.fit(train_set, train_labels)

# Feature importances into a dataframe
feature_importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
feature_importances.head()


# In[ ]:


def plot_feature_importances(df, n = 10, threshold = None):
    """Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances. 
    
    Args:
        df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".
    
        n (int): Number of most important features to plot. Default is 15.
    
        threshold (float): Threshold for cumulative importance plot. If not provided, no plot is made. Default is None.
        
    Returns:
        df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1) 
                        and a cumulative importance column
    
    Note:
    
        * Normalization in this case means sums to 1. 
        * Cumulative importance is calculated by summing features from most to least important
        * A threshold of 0.9 will show the most important features needed to reach 90% of cumulative importance
    
    """
    plt.style.use('fivethirtyeight')
    
    # Sort features with most important at the head
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # Normalize the feature importances to add up to one and calculate cumulative importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    
    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'darkgreen', 
                            edgecolor = 'k', figsize = (12, 8),
                            legend = False, linewidth = 2)

    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title(f'{n} Most Important Features', size = 18)
    plt.gca().invert_yaxis()
    
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);
        
        # Number of features needed for threshold cumulative importance
        # This is the index (will need to add 1 for the actual number)
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')
        plt.show();
        
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 
                                                                                  100 * threshold))
    
    return df


# In[ ]:


norm_fi = plot_feature_importances(feature_importances, threshold=0.95)


# In[ ]:


# Model imports
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


import warnings 
from sklearn.exceptions import ConvergenceWarning

# Filter out warnings from models
warnings.filterwarnings('ignore', category = ConvergenceWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Dataframe to hold results
model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])

def cv_model(train, train_labels, model, name, model_results=None):
    """Perform 10 fold cross validation of a model"""
    
    cv_scores = cross_val_score(model, train, train_labels, cv = 10, scoring=scorer, n_jobs = -1)
    print(f'10 Fold CV Score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')
    
    if model_results is not None:
        model_results = model_results.append(pd.DataFrame({'model': name, 
                                                           'cv_mean': cv_scores.mean(), 
                                                            'cv_std': cv_scores.std()},
                                                           index = [0]),
                                             ignore_index = True)

        return model_results


# In[ ]:


model_results = cv_model(train_set, train_labels, LinearSVC(), 
                         'LSVC', model_results)


# In[ ]:


model_results = cv_model(train_set, train_labels, 
                         GaussianNB(), 'GNB', model_results)


# In[ ]:


model_results = cv_model(train_set, train_labels, 
                         MLPClassifier(hidden_layer_sizes=(32, 64, 128, 64, 32)),
                         'MLP', model_results)


# In[ ]:


model_results = cv_model(train_set, train_labels, 
                          LinearDiscriminantAnalysis(), 
                          'LDA', model_results)


# In[ ]:


model_results = cv_model(train_set, train_labels, 
                         RidgeClassifierCV(), 'RIDGE', model_results)


# In[ ]:


for n in [5, 10, 20]:
    print(f'\nKNN with {n} neighbors\n')
    model_results = cv_model(train_set, train_labels, 
                             KNeighborsClassifier(n_neighbors = n),
                             f'knn-{n}', model_results)


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier

model_results = cv_model(train_set, train_labels, 
                         ExtraTreesClassifier(n_estimators = 100, random_state = 10),
                         'EXT', model_results)


# # Comparing Model Performance

# In[ ]:


model_results = cv_model(train_set, train_labels,
                          RandomForestClassifier(100, random_state=10),
                              'RF', model_results)


# In[ ]:


model_results.set_index('model', inplace = True)
model_results['cv_mean'].plot.bar(color = 'orange', figsize = (8, 6),
                                  yerr = list(model_results['cv_std']),
                                  edgecolor = 'k', linewidth = 2)
plt.title('Model F1 Score Results');
plt.ylabel('Mean F1 Score (with error bar)');
model_results.reset_index(inplace = True)


# In[ ]:


def submit(model):
    model.fit(train_set, train_labels)
    predictions = model.predict(test_set)
    predictions = pd.DataFrame({'idhogar': test_ids,
                               'Target': predictions})

     # Make a submission dataframe
    submission = submission_base.merge(predictions, 
                                       on = 'idhogar',
                                       how = 'left').drop(columns = ['idhogar'])

    # Fill in households missing a head
    submission['Target'] = submission['Target'].fillna(4).astype(np.int8)
    
    return submission 


# # XGBOOST 

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_set, train_labels, test_size = 0.2, 
                                                    random_state =42)


# In[ ]:


print('train shape', X_train.shape, y_train.shape)
print('test shape', X_test.shape, y_test.shape)


# In[ ]:


import xgboost 
from xgboost import XGBClassifier
from sklearn.metrics import f1_score


# In[ ]:


xgb_model  = XGBClassifier(objective = 'multi:softmax', random_state = 42)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
f1_score(y_test, y_pred, average = 'macro')


# In[ ]:


test_ids = list(final.loc[final['Target'].isnull(), 'idhogar'])


# In[ ]:


xgb_submission = submit(xgb_model)


# # LightGBM

# In[ ]:


import lightgbm
from lightgbm import LGBMClassifier


# In[ ]:


lgb_model  = LGBMClassifier(objective = 'multiclass', random_state = 42)
lgb_model.fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
print(y_pred)
f1_score(y_test, y_pred, average = 'macro')


# In[ ]:


lgb_submission = submit(lgb_model)


# In[ ]:


xgb_submission.to_csv('xgb.csv', index = False)
lgb_submission.to_csv('lgb.csv', index = False)


# In[ ]:





# # Advanced LightGBM

# In[ ]:


import lightgbm as lgb 
print('Train start')

params = {"objective" : "multiclass",
          "num_class" : 4,
          "num_leaves" : 60,
          "max_depth": -1,
          "learning_rate" : 0.01,
          "bagging_fraction" : 0.9,  # subsample
          "feature_fraction" : 0.9,  # colsample_bytree
          "bagging_freq" : 5,        # subsample_freq
          "bagging_seed" : 42,
          "verbosity" : -1 }


# In[ ]:


# lgbDataset에서는 label은 0부터 시작해야함 0, 1, 2, 3 

y_tra = y_train - 1
y_tes = y_test -1


# In[ ]:


lgtrain, lgval = lgb.Dataset(data = X_train, label = y_tra), lgb.Dataset(data = X_test, label = y_tes)
print(lgtrain)
lgb_ad_model = lgb.train(params, lgtrain, num_boost_round = 2000, valid_sets=[lgtrain, lgval], 
                  early_stopping_rounds=100, verbose_eval=100)


# In[ ]:


predictions = lgb_ad_model.predict(test_set)


# In[ ]:


predict_df_1 = []
predict_df_2 = []
predict_df_3 = []
predict_df_4 = []


# In[ ]:


for i in range(len(predictions)):
    predict_df_1.append(predictions[i][0])
    predict_df_2.append(predictions[i][1])    
    predict_df_3.append(predictions[i][2])    
    predict_df_4.append(predictions[i][3])    


# In[ ]:


df_pred = pd.DataFrame(predict_df_1, columns = ['1'])


# In[ ]:


df_pred['2'] = predict_df_2
df_pred['3'] = predict_df_3
df_pred['4'] = predict_df_4
df_pred.head()


# In[ ]:


df_pred['Target'] = df_pred[['1', '2', '3', '4']].idxmax(axis = 1)


# In[ ]:


df_pred


# In[ ]:


df_pred.to_csv('result.csv', index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




