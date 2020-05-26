import numpy as np
import matplotlib.pyplot as plt
from sympy.geometry import Point, Line

def drawAll(X, Y, line):
    coefs = line.coefficients
    x = np.linspace(X.min(),X.max())
    y = (-coefs[0] * x - coefs[2]) / coefs[1]
    plt.plot(x, y)
    plt.scatter(X, Y, color = 'red')
    plt.show()

def transformLine(point, line, x_median, learning_rate): 
    #On prend la médiane des valeurs de x pour de meilleurs résultats pour les calculs des distances horizontales
    
    #Création de la ligne verticale qui passe par le nouveau point
    ymin = line.points[0] if line.direction.y > 0 else line.points[1]
    ymax = line.points[1] if line.direction.y > 0 else line.points[0]
    vertical_line = Line(Point(point.x,ymin.y), Point(point.x,ymax.y))
    #Cherche l'intersection avec notre droite (pour calculer la distance verticale)
    I = line.intersection(vertical_line)
    vertical_distance = point.y - I[0].y
    horizontal_distance = point.x - x_median  
    
    coefs = line.coefficients

    a = coefs[0]
    b = coefs[1]
    c = coefs[2]
    
    #Calcul des points qui constituent la nouvelle ligne
    # y = - (a/b)*x + learning_rate * vertical_distance * horizontal_distance - (c/b) + learning_rate * vertical_distance
    
    x_min = line.points[0].x
    y_min = - (a/b)*x_min + (learning_rate * vertical_distance * horizontal_distance * x_min) /b - (c/b) + learning_rate * vertical_distance
    
    x_max = line.points[1].x
    y_max = - (a/b)*x_max + (learning_rate * vertical_distance * horizontal_distance * x_max) /b - (c/b) + learning_rate * vertical_distance
    
    newLine = Line(Point(x_min, y_min), Point(x_max, y_max))
    return newLine
    
def predict(X, line):
    prediction = []
    coefs = line.coefficients
    a = coefs[0]
    b = coefs[1]
    c = coefs[2]
    for x in X.values:
        y = - (a/b)*x - (c/b)
        prediction.append(y)
    return prediction