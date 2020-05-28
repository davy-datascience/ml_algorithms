import pandas as pd
from my_functions import drawAll, transformLine, predict
from sympy.geometry import Point, Line
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import progressbar

# Set the learning rate and the number of iterations
learning_rate = 0.01
nb_epochs = 1000

# Read the data
dataset = pd.read_csv("https://raw.githubusercontent.com/davy-datascience/portfolio/master/LinearRegression/Approach-1/dataset/Salary_Data.csv")

# Separation independent variable X - dependent variable y
X = dataset.YearsExperience
y = dataset.Salary

# Separate the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Looking for 1st line equation
# The line must have the same scope than the scatter plots from the dataset
# I decided to build the line choosing the point that has the max x-value and the point that has the min x-value

# Find the point with the maximum value of x in the dataset
idx_max = X_train.idxmax()
x_max = Point(X_train.loc[idx_max], y_train.loc[idx_max])

# Find the point with the minimum value of x in the dataset
idx_min = X_train.idxmin()
x_min = Point(X_train.loc[idx_min], y_train.loc[idx_min])

# Build the line with the 2 points 
line = Line(x_min, x_max)
drawAll(X_train, y_train, line)

# Iterate choosing a random point and moving the line with the function transformLine
for i in progressbar.progressbar(range(nb_epochs)):
    sample = dataset.sample()
    point = Point(sample.YearsExperience, sample.Salary)
    line = transformLine(point, line, X_train.median(), learning_rate)
    #drawAll(X_train, y_train, line)    # Uncomment this line to see the line at each iteration

drawAll(X_train, y_train, line)

# Predict the test set with my model and see
y_pred = predict(X_test, line)
print("MAE (Mean Absolute Error) is used to evaluate the model accuracy")
print("MAE for my model: {}".format(mean_absolute_error(y_pred, y_test)))

# Predict the test set with the sklearn algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train.to_frame(), y_train)
y_pred2 = regressor.predict(X_test.to_frame())
print("MAE for the algorithm of the sklearn module: {}".format(mean_absolute_error(y_pred2, y_test)))