import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def drawAll(dataset, x_min, x_max, coefs):
    """ plot the points from the dataset and draw the actual Line """
    datasetTrue = dataset[dataset["specy"] == 1]
    datasetFalse = dataset[dataset["specy"] == 0]
    
    x = np.linspace(x_min,x_max)
    y = (-coefs[0] * x - coefs[2]) / coefs[1]
    
    plt.plot(x, y)
    
    plt.scatter(datasetFalse["sep_long"], datasetFalse["pet_larg"], color = 'red')
    plt.scatter(datasetTrue["sep_long"], datasetTrue["pet_larg"], color = 'green')
    plt.show()

def transformLine(x, y, categUp, lineCoefs, learning_rate):
    """ According to the random point, update the Line """
    
    # Check if the point is below or above the line
    # By reporting the point (x,y) to the equation ax+by+c :
    # If ax+by+c > 0 then the point is above the line, else it is below the line
    
    position = lineCoefs[0] * x + lineCoefs[1] *y + lineCoefs[2]
    
    # Look if the point is incorrectly classified, if so move the line towards point
    if position > 0 and not categUp :
        lineCoefs[0] -= x * learning_rate
        lineCoefs[1] += y * learning_rate
        lineCoefs[2] -= learning_rate
    elif position < 0 and categUp : 
        lineCoefs[0] += x * learning_rate
        lineCoefs[1] -= y * learning_rate
        lineCoefs[2] += learning_rate        
    
    return lineCoefs

def predict(X, lineCoefs):
    """ I use my model (the equation of the line) to predict which specy a new set of values belongs to  """
    prediction = []
    
    a = lineCoefs[0]
    b = lineCoefs[1]
    c = lineCoefs[2]
    
    for row in X.iterrows():
        x = row[1].loc["sep_long"]
        y = row[1].loc["pet_larg"]
        
        # The result of the equation ax + by+ c = 0 tells if the point is in category 1 (positive) or category 0 (negative)
        position = a * x + b * y + c
        prediction.append(position > 0)

    return pd.DataFrame(prediction, index= X.index)