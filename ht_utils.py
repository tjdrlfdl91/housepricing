import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold


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


def ht_makedummy(df, limit_num, var_list = None): # change variables with stirngs and specific integers into dummies // if you insert var_list, this will only target the columns in the list.
    return_list = pd.DataFrame()
    if var_list == None:
        var_list = list(df)
    else: pass
    for var in list(df):
        if var in var_list:    
            if df[var].dtypes == 'O':
                dum = pd.DataFrame()
                dum = pd.get_dummies(df[var], prefix = var)
                return_list = pd.concat([return_list, dum], axis = 1)
            elif df[var].dtypes == 'int64' and len(set(df[var])) < limit_num:
                dum = pd.DataFrame()
                dum = pd.get_dummies(df[var], prefix = var)
                return_list = pd.concat([return_list, dum], axis = 1)
            else:
                return_list = pd.concat([return_list, df[var]], axis = 1)
        else:
            return_list = pd.concat([return_list, df[var]], axis = 1)
    return return_list



def ht_xycorr(X, Y, threshold): # find variables in X which has higher absolute corr. with Y than threshold
    for v in list(X):
        try:
            if abs(Y.corr(X[v])) > threshold:
                print(v)
            else: pass
        except TypeError: pass

#from the idea of StackingAveragedModels(https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)
class ht_metamodel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)