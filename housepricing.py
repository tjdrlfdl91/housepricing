# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 14:23:33 2018

@author: SKMOON
"""
# 우선 current working directory를 설정하고 생각합시다
# cwd에 본인 working directory 적어넣으세요
cwd = "C:/Users/SKMOON/Downloads/santander"

import os
os.getcwd()
os.chdir(cwd)

# warnings 무시하고
import warnings
warnings.filterwarnings('ignore')

# 그동안 사용했던 주요 라이브러리도 import 해옵시다
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.preprocessing import Imputer , Normalizer , scale, LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
import xgboost as xgb

