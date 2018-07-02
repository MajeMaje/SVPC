import numpy as np
import pandas as pd


eps = 1e-8

def set_data():
    # load data
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    #remove all 0 cal
    nunique = train.nunique()
    drop_cols = nunique[nunique == 1].index.tolist()
    X = pd.concat([train, test], axis=0, ignore_index=True)
    X = X.drop(drop_cols + ['ID', 'target'], axis=1).values.astype(np.float32)

    # normalize
    positive_index = X >= eps
    X[positive_index] = np.log1p(X[positive_index])
    mean_posi, std_posi = X[positive_index].mean(), X[positive_index].std()
    X[positive_index] = (X[positive_index] - mean_posi) / std_posi
    X[~positive_index] = 0

    # train nn regressor
    X_test = X[train.shape[0]:]
    X_train = X[:train.shape[0]]
    y_train = train['target'].values.astype(np.float32)

    # normalize target
    y_train = np.log1p(y_train) - mean_posi

    return X_test,X_train,y_train



if __name__ == '__main__':
    set_data()
