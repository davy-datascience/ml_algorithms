import pandas as pd

def normalize(X):
    X_normalized = pd.DataFrame()
    for i in range(len(X.columns)):
        avg_x = X.mean()[i] #  on peut le sortir du for et faire un array avc ttes les moyennes
        range_x = X.max()[i] - X.min()[i]
        X_normalized[i] = (X.iloc[:, i] - avg_x) / range_x
    X_normalized.columns = X.columns
    return X_normalized

def feature_engineering(X):
    features = ['LotArea', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'KitchenAbvGr', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'PoolArea']
    
    X = X[features].copy()
    X["FlrSF"] = X["1stFlrSF"] + X["2ndFlrSF"]
    X["Bath"] = X["FullBath"] +  X["HalfBath"] + X["BsmtFullBath"] + X["BsmtHalfBath"]
    
    X = X.drop(columns=['1stFlrSF', '2ndFlrSF', "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"])
    
    return X

def predict(X, tetas):
    y_pred = X.iloc[:, 0] * tetas[1] + tetas[0]
    for i in range(2, len(tetas)):
        y_pred += X.iloc[:, i-1] * tetas[i] 
        
    return(y_pred)