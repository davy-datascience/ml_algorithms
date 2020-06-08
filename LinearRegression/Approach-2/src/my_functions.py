import pandas as pd
from pandas.core.frame import DataFrame

def normalize(X):
    ''' Apply mean normalization, so each feature is at the same scale '''
    X_normalized = pd.DataFrame()
    avg_X = X.mean()
    range_X = X.max() - X.min()
    
    for i in range(len(X.columns)):
        X_normalized[i] = (X.iloc[:, i] - avg_X[i]) / (range_X[i] if range_X[i] != 0 else 1)
    X_normalized.columns = X.columns
    return X_normalized

def feature_engineering(X):
    ''' Select interesting features and create new calculated features '''
    features = ['LotArea', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'KitchenAbvGr', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'PoolArea']
    
    X = X[features].copy()
    X["FlrSF"] = X["1stFlrSF"] + X["2ndFlrSF"]
    X["Bath"] = X["FullBath"] +  X["HalfBath"] + X["BsmtFullBath"] + X["BsmtHalfBath"]
    
    X = X.drop(columns=['1stFlrSF', '2ndFlrSF', "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"])
    
    return X

def predict(X, tetas):
    ''' Predict with the model (tetas)
        X input can be either a Series for a single example to predict or a DataFrame
    '''
    X = X.copy() # So we do not alter the input dataframe X
    
    # Check if the input is either DataFrame or a Series
    if isinstance(X, DataFrame):
        X.insert(0, "X0", 1)
        y_pred = pd.Series()
        
        for i in range(len(X.index)):
            y_pred = y_pred.append(pd.Series(tetas.dot(X.iloc[i])))
        
        return y_pred
    
    else:
        X = pd.Series([1]).append(X)
        return tetas.dot(X)
        