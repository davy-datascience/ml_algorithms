import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from my_functions import normalize, feature_engineering, predict

learning_rate = 0.1
epochs = 5

# Read the data
dataset = pd.read_csv("https://raw.githubusercontent.com/davy-datascience/portfolio/master/LinearRegression/Approach-2/dataset/house_pricing.csv")

# Separate the dataset into a training set and a test set
train, test = train_test_split(dataset, test_size = 0.2)

X_train = feature_engineering(train)
X_train = normalize(X_train)
y_train = train["SalePrice"]

X_test = feature_engineering(test)
X_test = normalize(X_test)
y_test = test["SalePrice"]

tetas = np.ones(X_train.columns.size + 1)
    
def regressionFunction(tetas, X):
    X = pd.Series([1]).append(X)
    return tetas.dot(X)

def gradientDescent(X, epochs, learning_rate):
    tetas = np.ones(X.columns.size + 1)
    
    for epoch in range(epochs):
        new_tetas = []
        
        for j in range(len(tetas)):
            sum_cost = []
            for i in range(len(X.index)):
                h = regressionFunction(tetas, X_train.iloc[i])
                y = y_train.iloc[i]
                x = 1
                if j != 0:
                    x = X.iloc[i, j - 1]
                sum_cost.append((h - y) * x)
            new_tetas.append(tetas[j] - learning_rate * (1/len(X.index)) * sum(sum_cost))

        tetas = np.array(new_tetas)
        print(tetas)
        
    return tetas

tetas = gradientDescent(X_train, epochs, learning_rate)

# Predict the test set with my model and see
y_pred = predict(X_test, tetas)
print("MAE for my model: {}".format(mean_absolute_error(y_pred, y_test)))

# Predict the test set with the sklearn algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred2 = regressor.predict(X_test)
print("MAE for the algorithm of the sklearn module: {}".format(mean_absolute_error(y_pred2, y_test)))
        

