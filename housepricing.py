# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 14:23:33 2018

@author: SKMOON
"""
# 우선 current working directory를 설정하고 생각합시다
# cwd에 본인 working directory 적어넣으세요
cwd = "C:/Python/git/housepricing"

import os
os.getcwd()
os.chdir(cwd)

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

# 변수 탐색을 시작합시다
# data descripion 파일도 같이 참조하세요
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 짜증나니까 MSZoning 변수에 C (all)을 C로 바꿔줍시다
train = train.replace("C (all)" , "C")

X = train.drop(["SalePrice"] , axis = 1)
Y = train.SalePrice

list(train)

# .head()로 모든 column을 다 훑어볼 수 있게 합시다
# 변수가 오지게 많습니다
pd.options.display.max_columns = None
train.head()

# 종속변수인 SalePrice 및 기타 눈에 띄는 변수들의 분포를 볼 수 있습니다
sns.distplot(train['SalePrice'])
sns.distplot(train['LotArea'])
sns.distplot(train['OverallQual'])
sns.distplot(train['OverallCond'])
sns.distplot(train['YearBuilt'])

# 변수 간 상관관계부터 살펴 봅시다
# 제일 바깥 줄에 있는 SalePrice와 다른 독립변수들 간의 상관관계를 볼 수 있습니다

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
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax)
    plt.show()

correlation_matrix(train)

# 상관관계를 숫자로 뽑아냅시다
# SalePrice와 상관관계가 높은 (0.5 이상) 변수들부터 추려봅시다
train.corr(method = 'pearson')

for v in list(X):
    try:
        if abs(Y.corr(X[v])) > 0.5:
            print(v)
        else: pass
    except TypeError: pass


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

# str 또는 int으로 구성된 변수를 dummy로 만들어줍시다
# 우선 Unique한 값이 10개 미만인 변수만 dummy화 해줍시다

def MakeDummy(df, limit_num):
    VarList = pd.DataFrame()
    for var in df:
        if df[var].dtypes == 'O' and len(set(df[var])) < limit_num:
            dum = pd.DataFrame()
            dum = pd.get_dummies(df[var], prefix = var)
            VarList = pd.concat([VarList, dum], axis = 1)
        elif df[var].dtypes == 'int64' and len(set(df[var])) < limit_num:
            dum = pd.DataFrame()
            dum = pd.get_dummies(df[var], prefix = var)
            VarList = pd.concat([VarList, dum], axis = 1)
        else: pass
    return VarList

dum = MakeDummy(X, 9)

# Dummy화 된 변수를 X에 붙여줍시다
X = pd.concat([X, dum], axis = 1)

train.corr(method = 'pearson')

# 앞에서 해줬던 상관관계 분석을 다시 해보면 6개 정도 변수가 더 추려집니다
#ExterQual_TA
#BsmtQual_Ex
#KitchenQual_Ex
#KitchenQual_TA
#FullBath_1
#GarageCars_3

