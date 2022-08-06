import numpy as np
import cv2
from sklearn.metrics import jaccard_score
from Canny import *
import matplotlib.pyplot as plt


def DetectCircles(Image, Threshold, Region, Raduis=None):
    image = Canny(Image)
    (Rows, Columns) = image.shape
    if Raduis == None:
        MaximumRaduis = np.max((Rows, Columns))
        MinimumRaduis = 3
    else:
        [MinimumRaduis,MaximumRaduis] = Raduis
    RangeOfRaduis = MaximumRaduis - MinimumRaduis

    CirclesArray1 = np.zeros((MaximumRaduis, Rows + 2 * MaximumRaduis, Columns + 2 * MaximumRaduis))
    CircelsArray = np.zeros((MaximumRaduis, Rows + 2 * MaximumRaduis, Columns + 2 * MaximumRaduis))
    Thetas = np.arange(0, 360) * np.pi / 180
    Edges = np.argwhere(image[:, :]) 
    for itr in range(RangeOfRaduis):
        raduis = MinimumRaduis + itr
        BluePrint = np.zeros((2 * (raduis+1), 2 * (raduis + 1)))
        (CenterY, CenterX) = (raduis + 1, raduis + 1) 
        for angle in Thetas:
            x = int(np.round(raduis * np.cos(angle)))
            y = int(np.round(raduis * np.sin(angle)))
            BluePrint[CenterY + x, CenterX + y] = 1
        Const = np.argwhere(BluePrint).shape[0] #To get ones Indecies in Rows
        for x, y in Edges:
            X = [x - CenterY + MaximumRaduis, x + CenterY + MaximumRaduis]  # Computing the extreme X values
            Y = [y - CenterX + MaximumRaduis, y + CenterX + MaximumRaduis]  # Computing the extreme Y values
            CirclesArray1[raduis, X[0]:X[1], Y[0]:Y[1]] += BluePrint
        CirclesArray1[raduis, CirclesArray1[raduis] < Threshold * Const / raduis] = 0

    for raduis, x, y in np.argwhere(CirclesArray1):
        Array = CirclesArray1[raduis - Region:raduis + Region, x - Region:x + Region, y - Region:y + Region]
        try:
            p, a, b = np.unravel_index(np.argmax(Array), Array.shape)
        except:
            continue

        CircelsArray[raduis + (p - Region), x + (a - Region), y + (b - Region)] = 1

    return CircelsArray[:, MaximumRaduis:-MaximumRaduis, MaximumRaduis:-MaximumRaduis]

def DisplayCircles(CirclesArray, Image):
    CircleCoordinates = np.argwhere(CirclesArray)
    for raduis, x, y in CircleCoordinates:
        cv2.circle(Image, (y, x), raduis, color=(0, 255, 0), thickness=2)
    return Image
