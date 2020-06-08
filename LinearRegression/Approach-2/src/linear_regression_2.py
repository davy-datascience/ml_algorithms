import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from my_functions import normalize, feature_engineering, predict
import progressbar

# Set the learning rate and the number of iterations
learning_rate = 0.1
epochs = 50

# Read the data
dataset = pd.read_csv("https://raw.githubusercontent.com/davy-datascience/portfolio/master/LinearRegression/Approach-2/dataset/house_pricing.csv")

# Separate the dataset into a training set and a test set
train, test = train_test_split(dataset, test_size = 0.2)

# Separation independent variable X - dependent variable y for the train set & the test set
# Do specific feature engineering & normalize X
X_train = feature_engineering(train)
X_train = normalize(X_train)
y_train = train["SalePrice"]

X_test = feature_engineering(test)
X_test = normalize(X_test)
y_test = test["SalePrice"]

# Arbitrary begin with all tetas = 1
tetas = np.ones(X_train.columns.size + 1)

for i in progressbar.progressbar(range(epochs)):
    # Create a variable new_tetas so we can update simulatneously variable tetas
    new_tetas = []
    
    # Iterate over all tetas to calcuate their new values that will be updated simulteously
    for j in range(len(tetas)):
        sum_cost = []
        # Iterate over all training data
        for i in range(len(X_train.index)):
            # Calculate the predicted value of the data row with our regressionFunction
            h = predict(X_train.iloc[i], tetas)
            y = y_train.iloc[i]
            x = 1
            if j != 0:
                x = X_train.iloc[i, j - 1]
            sum_cost.append((h - y) * x)
        new_tetas.append(tetas[j] - learning_rate * (1/len(X_train.index)) * sum(sum_cost))

    # Simultaneous update of thetas
    tetas = np.array(new_tetas)

# Predict the test set with my model and see
y_pred = predict(X_test, tetas)
print("MAE for my model: {}".format(mean_absolute_error(y_pred, y_test)))

# Predict the test set with the sklearn algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred2 = regressor.predict(X_test)
print("MAE for the algorithm of the sklearn module: {}".format(mean_absolute_error(y_pred2, y_test)))
        

