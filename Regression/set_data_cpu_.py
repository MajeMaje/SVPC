import numpy as np
import pandas as pd


def set_data():
    train = pd.read_csv('../DimensionalityReduction/result/reduction_train.csv')   
    test = pd.read_csv('../DimensionalityReduction/result/reduction_test.csv')
    
    Y = train['target']
    
    train = train.drop(['ID','target'], axis = 1)
    test = test.drop(['ID'], axis = 1)

    print(train.shape,"train")
    print(test.shape,"test")
    
    return np.asarray(train.values).astype(np.float32) ,np.asarray(test.values).astype(np.float32) , np.asarray(Y.values).astype(np.float32).reshape(Y.shape[0],1)

if __name__ == '__main__':
    set_data()
