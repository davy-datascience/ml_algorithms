import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import progressbar
from my_functions import drawAll, transformLine, predict

# Set the learning rate and the number of iterations
learning_rate = 0.01
nb_epochs = 1000

# Read the data
dataset = pd.read_csv("https://raw.githubusercontent.com/davy-datascience/portfolio/master/LogisticRegression/Approach-1/dataset/dataset.csv", index_col='id')

# Separate the dataset into a training set and a test set
train, test = train_test_split(dataset, test_size = 0.2)

# Look for the point with the maximum and minimum value of x in the dataset to see the range of x values
# Find the point with the maximum value of x in the dataset
idx_max = train["sep_long"].idxmax()
x_max = train.loc[idx_max]["sep_long"]

# Find the point with the minimum value of x in the dataset
idx_min = train["sep_long"].idxmin()
x_min = train.loc[idx_min]["sep_long"]

# Begin with the line y = 0
lineCoefs = [0, 1, 0]

drawAll(train, x_min, x_max, lineCoefs)

# Iterate choosing a random point and moving the line with the function transformLine
for i in progressbar.progressbar(range(nb_epochs)):
    sample = train.sample()
    sepL = sample.iloc[0].sep_long
    petL = sample.iloc[0].pet_larg
    categUp = sample.iloc[0].specy
    lineCoefs = transformLine(sepL, petL, categUp, lineCoefs, learning_rate)
    #drawAll(train, x_min, x_max, lineCoefs)  # Uncomment this line to see the line at each iteration

drawAll(train, x_min, x_max, lineCoefs)

# Predict the test set with my model and print the mae
y_pred = predict(test, lineCoefs)
print("MAE : {}".format(mean_absolute_error(y_pred, test.specy)))