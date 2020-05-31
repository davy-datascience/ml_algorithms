import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
dataset = pd.read_csv("https://raw.githubusercontent.com/davy-datascience/portfolio/master/LinearRegression/Approach-2/dataset/house_pricing.csv")

# Separate the dataset into a training set and a test set
train, test = train_test_split(dataset, test_size = 0.2)

features = ['LotArea', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'KitchenAbvGr', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'PoolArea']

X_train = train[features].copy()
X_train["FlrSF"] = X_train["1stFlrSF"] + X_train["2ndFlrSF"]
X_train["Bath"] = X_train["FullBath"] +  X_train["HalfBath"] + X_train["BsmtFullBath"] + X_train["BsmtHalfBath"]

X_train = X_train.drop(columns=['1stFlrSF', '2ndFlrSF', "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"])

teta = []
for i in range(X_train.columns.size):
    teta.append(0.0)
    
for i in range(X_train.rows...):
    