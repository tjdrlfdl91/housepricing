
import numpy as np
import pandas as pd

def ht_describe(a): #description for dataframe
    
    #general description
    print("****General Description****")
    print(a.describe())
    print("\n")
    
    #number of columns
    print("****The number of columns the dataframe contains****\n{0} columns".format(len(a.columns)))
    print("\n")
    
    #I can cut the code here to make separate functions
    
    #variable lists for each data type
    type_list = []
    for col in a:
        type_list.append(a[col].dtypes)
    type_list = list(set(type_list))
    
    coltyplst = [[0]*2 for i in range(len(type_list))]
    for i in range(0, len(type_list)):
        coltyplst[i][0] = type_list[i]
        coltyplst[i][1] = list(a.select_dtypes(include = [type_list[i]]))

    for i in range(len(coltyplst)):
        print("****Data type: {0}, {1} variables****\n{2}\n".format(coltyplst[i][0], len(coltyplst[i][1]), coltyplst[i][1]))

def ht_cut(a): #cut the dataframe on datatypes and return a list of them  

    type_list = []
    
    for col in a:
        type_list.append(a[col].dtypes)
    type_list = list(set(type_list))
    
    dflist = [[0]*2 for i in range(len(type_list))]
    
    for i in range(0, len(type_list)):
        dflist[i][0] = type_list[i]
        dflist[i][1] = a.select_dtypes(include = [type_list[i]])
    
    return dflist
