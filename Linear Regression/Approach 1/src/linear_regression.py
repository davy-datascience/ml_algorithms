import pandas as pd
from my_functions import drawAll, transformLine, predict
from sympy.geometry import Point, Line
import time
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

learning_rate = 0.005
nb_epochs = 1000

# Read the data
dataset = pd.read_csv("Salary_Data.csv")

# arrondi à décimale
dataset = dataset.round(0)

#Suppression des lignes sans valeurs 
dataset.dropna(axis=0, subset=['Salary'], inplace=True)

#Add indexes (no indexes in that dataset)
indexes_dataset = [i for i in range(len(dataset))]
dataset = dataset.assign(index=pd.Series(indexes_dataset).values)
dataset = dataset[['index', 'YearsExperience', 'Salary']]
dataset.set_index('index')

#Séparation variable prédictive - variable dépendante
X = dataset.YearsExperience
y = dataset.Salary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Cherche 1ère droite équation 

# Cherche point qui a le x au max
idx_max = X_train.idxmax()
x_max = Point(X_train.loc[idx_max], y_train.loc[idx_max])

# Cherche point qui a le x au min
idx_min = X_train.idxmin()
x_min = Point(X_train.loc[idx_min], y_train.loc[idx_min])

# vecteur droite pour équation 
#x_max - x_min
line = Line(x_min, x_max)
drawAll(X_train, y_train, line)

start_time = time.time()

for i in range(nb_epochs):
    sample = dataset.sample()
    point = Point(sample.YearsExperience, sample.Salary)
    line = transformLine(point, line, X_train.median(), learning_rate)
    #drawAll(X_train, y_train, line)

drawAll(X_train, y_train, line)

y_pred = predict(X_test, line)
print("MAE my model: {}".format(mean_absolute_error(y_pred, y_test)))


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train.to_frame(), y_train)

# Faire de nouvelles prédictions
y_pred2 = regressor.predict(X_test.to_frame())
print("MAE Linear Regression: {}".format(mean_absolute_error(y_pred2, y_test)))

print("--- TOTAL --- %s seconds ---" % (time.time() - start_time))
