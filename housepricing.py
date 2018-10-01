# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 14:23:33 2018

@author: SKMOON
"""


"""
HeuristicTinkers Project No.002 "Housepricing"

CONTENTS
    0. Settings
        0.1 Directory Settings
        0.2 Package Imports

    1. Import Data
        1.1 Data Check
        1.2 Merge Train and Test
    
    2. Preprocessing
        2.1 Data Overview and Distribution
        2.2 Correlations
        2.3 NaN Handling
        2.4 Removing Garbage Variables
            2.4.1 Variance Threshold
        2.5 Encoding
            2.5.1 One Hot Encoding
            2.5.2 Label Encoding
        2.6 
    
     
"""

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

### 0. Settings
### 0.1 Directory Settings

# 우선 current working directory를 설정하고 생각합시다
# cwd에 본인 working directory 적어넣으세요

cwd = "C:/Python/git/housepricing"

import os
os.getcwd()
os.chdir(cwd)

### 0.2 Package Imports

# 필요하시다면 warnings 무시해 주시고
# import warnings
# warnings.filterwarnings('ignore')

# 그동안 사용했던 주요 라이브러리도 import 해옵시다
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import Imputer , Normalizer , scale, LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
import xgboost as xgb

# 우리 전용 라이브러리인 ht_utils도 import 해옵시다
import ht_utils as ht

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

### 1. Import Data
### 1.1 Data Check

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# .head()로 모든 column을 다 훑어볼 수 있게 합시다
# 변수가 오지게 많습니다
pd.options.display.max_columns = None
train.head()
test.head()

### 1.2 Merge Train and Test

# 독립변수와 종속변수로 나눠주고
X = train.drop(["SalePrice"] , axis = 1)
Y = train.SalePrice

# 나중에 할 Preprocessing을 위해 train 독립변수와 test를 붙여도 놓읍시다
# [0:1460]가 X, [1460:2919]가 test입니다
full = pd.concat([X, test], ignore_index = True)

full.iloc[0:1460]
full.iloc[1460:2919]

# (치트키) 짜증나니까 MSZoning 변수에 C (all)을 C로 바꿔줍시다
full = full.replace("C (all)" , "C")

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

### 2. Preprocessing
### 2.1 Data Overview and Distribution

# 변수 탐색을 시작합시다
# data descripion 워드파일도 같이 참조하세요
# ht_describe로 어떤 data type이 많은지 살펴봅시다

ht.ht_describe(full)

# ht_cut을 통해 data type에 따라 전체 df를 쪼갤 수 있습니다
# ht_describe에서 나왔던 순서에 따라 쪼개집니다
# ht_cut 사용법은 아래 참조

temp = ht.ht_cut(full)
df_float = temp[0][1]
df_int = temp[1][1]
df_object = temp[2][1]
del temp

# 종속변수인 SalePrice 및 기타 눈에 띄는 변수들의 분포를 볼 수 있습니다
sns.distplot(Y)
sns.distplot(full['LotArea'])
sns.distplot(full['OverallQual'])
sns.distplot(full['OverallCond'])
sns.distplot(full['YearBuilt'])

### 2.2 Correlations

# 변수 간 상관관계부터 살펴 봅시다
# 제일 바깥 줄에 있는 SalePrice와 다른 독립변수들 간의 상관관계를 볼 수 있습니다
# test는 SalePrice가 없기 때문에 여기서는 full 대신 X를 이용해야 합니다

def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Corr Matrix')
    labels=[list(train)]
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    fig.colorbar(cax)
    plt.show()

correlation_matrix(train)

# 상관관계를 숫자로 뽑아냅시다
# SalePrice와 상관관계가 높은 (0.5 이상) 변수들부터 추려봅시다
# ht_xycorr을 이용해서 쉽게 뽑아낼 수 있습니다 (Y에는 1개의 변수만, threshold는 0~1사이의 값)
ht.ht_xycorr(X, Y, 0.5)

# 10개 정도 변수가 추려집니다
#OverallQual
#YearBuilt
#YearRemodAdd
#TotalBsmtSF
#1stFlrSF
#GrLivArea
#FullBath
#TotRmsAbvGrd
#GarageCars
#GarageArea

# 이 변수들로 scatter plot을 그리다보면 우선 제거 대상 outlier들을 찾아볼 수 있습니다
plt.scatter(X['OverallQual'], Y)
plt.scatter(X['1stFlrSF'], Y)
plt.scatter(X['GrLivArea'], Y)
plt.scatter(X['TotRmsAbvGrd'], Y)
plt.scatter(X['GarageArea'], Y)

### 2.3 NaN Handling

# NaN값이 있는 변수명을 추려봅시다
NaN_vars = pd.DataFrame(columns = ['var_name', 'nan_count', 'dtype'])

for i in full:
    if full[i].isnull().values.any() == False: pass
    else:
        m = full[i].isnull().sum()
        NaN_vars = NaN_vars.append({'var_name' : i , 'nan_count' : m, 'dtype': full[i].dtype} , ignore_index = True) 

NaN_vars

# 하나씩 처리해봅시다
# 우선, NaN값이 4개 이하인 변수들은 최빈값으로 대체해줍시다

for (i,m) in NaN_vars.iterrows():
    if m['nan_count'] < 5:
        full[m['var_name']].fillna(full[m['var_name']].value_counts().index[0], inplace = True)
    else: pass

""" 여기서 논쟁 포인트
지하실이 없는 경우 면적 값은 0으로 들어가는데 딱 1개 obs.가 NaN으로 처리돼 있습니다
저 코드대로 돌리면 지하실이 없는 1개 obs.의 지하실 면적 NaN 값은 분포의 최빈값으로 대체됩니다
1개 obs.는 전체 분석에 영향을 주기 어려우니 systematic하게 처리하는 게 맞을까요?
아니면 저렇게 확실히 logical 오류가 있는 경우에는 obs.단위로 바로잡아 주는 게 맞을까요? """

# LotFrontage는 분포를 그려보고 최빈값인 60을 넣어주는 방향으로 처리합시다
LF_NotNull = full['LotFrontage'].loc[full['LotFrontage'].isnull() == False]

sns.distplot(LF_NotNull)
LF_NotNull.mode() ## 60.0

full['LotFrontage'] = full['LotFrontage'].fillna(60.0)


# 나머지 변수의 NaN은 해당 설비가 없음을 뜻합니다
# 따라서 "Unavailable"로 처리하고 이후 더미화해서 보도록 합시다
# 단, MasVnrArea와 GarageYrBlt는 float 형태이므로 각각 0과 9999를 넣어줍시다

full['MasVnrArea'].fillna(0, inplace = True)
full['GarageYrBlt'].fillna(9999, inplace = True)

for i in full:
    full[i].fillna("Unavailable", inplace = True)

# NaN값이 없음을 볼 수 있습니다
full.isnull().sum().sum()

###2.4 Removing Garbage Variables
###2.4.1 Variance Threshold

# 한 변수 내에 variance가 너무 작으면 분석에 쓸모가 없을 확률이 높습니다
# Variance Threshold 모델로 걸러봅시다
# Santander 때처럼 추정할 값이 아주 적은 비율이면 VT는 함부로 해서는 안 될 것 같습니다
# 2개 변수는 variance가 너무 적다는 결과를 확인할 수 있습니다

from sklearn.feature_selection import VarianceThreshold

def variance_threshold_selector(data, threshold = (.8*(1 - .8))):
        selector = VarianceThreshold(threshold)
        selector.fit(data)
        return data[data.columns[selector.get_support(indices=True)]]

temp = ht.ht_cut(full)
df_float = temp[0][1]
df_int = temp[1][1]
df_object = temp[2][1]
del temp

vt = pd.concat([df_float, df_int], axis = 1)
vt_result = variance_threshold_selector(vt)

list(set(vt) - set(vt_result))
#BsmtHalfBath
#KitchenAbvGr

full.BsmtHalfBath.value_counts()
full.KitchenAbvGr.value_counts()

###2.5 Encoding
###2.5.1 One Hot Encoding

# ht_makedummy를 사용하면 쉽게 one hot encoding을 해줄 수 있습니다
# str 또는 unique 값이 n개 미만인 int 변수를 dummy로 만들어줍시다

dum = ht.ht_makedummy(full, 1)
dum.head()

###2.5.2 Label Encoding

