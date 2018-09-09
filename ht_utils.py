
import numpy as np
import pandas as pd

def ht_describe(df): # description for dataframe (# of columns, datatypes and classification)
    
    #general description
    print("****General Description****")
    print(df.describe())
    print("\n")
    
    #number of columns
    print("****The number of columns the dataframe contains****\n{0} columns".format(len(df.columns)))
    print("\n")
    
    #I can cut the code here to make separate functions
    
    #variable lists for each data type
    type_list = []
    for col in df:
        type_list.append(str(df[col].dtypes))
    type_list = list(set(type_list))
    type_list.sort()
    
    coltyplst = [[0]*2 for i in range(len(type_list))]
    for i in range(0, len(type_list)):
        coltyplst[i][0] = type_list[i]
        coltyplst[i][1] = list(df.select_dtypes(include = [type_list[i]]))

    for i in range(len(coltyplst)):
        print("****Data type: {0}, {1} variables****\n{2}\n".format(coltyplst[i][0], len(coltyplst[i][1]), coltyplst[i][1]))

def ht_cut(df): # cut the dataframe on datatypes and return a list of them  

    type_list = []
    
    for col in df:
        type_list.append(str(df[col].dtypes))
    type_list = list(set(type_list))
    type_list.sort()
    
    dflist = [[0]*2 for i in range(len(type_list))]
    
    for i in range(0, len(type_list)):
        dflist[i][0] = type_list[i]
        dflist[i][1] = df.select_dtypes(include = [type_list[i]])
    
    return dflist

def ht_makedummy(df, limit_num): # find variables with limited unique values and replace them into dummies 
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
        else:
            VarList = pd.concat([VarList, df[var]], axis = 1)
    return VarList

def ht_xycorr(X, Y, threshold): # find variables in X which has higher absolute corr. with Y than threshold
    for v in list(X):
        try:
            if abs(Y.corr(X[v])) > threshold:
                print(v)
            else: pass
        except TypeError: pass
