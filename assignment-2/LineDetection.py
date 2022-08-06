from Canny import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

def HoughLinesAccumlator(Image, RhoResolution=1, ThetaResolution=1):
    Height, Width = Image.shape
    Diagonal = np.ceil(np.sqrt(Height**2 + Width**2))
    Rhos = np.arange(-Diagonal, Diagonal + 1, RhoResolution)
    Thetas = np.deg2rad(np.arange(-90, 90, ThetaResolution))
    HoughAccumlator = np.zeros((len(Rhos), len(Thetas)))
    WhitePixelColumnNumber, WhitePixelRowNumber = np.nonzero(Image)
    Indices=np.arange(0,len(Thetas)*len(WhitePixelColumnNumber),1)       
    x=np.outer(WhitePixelRowNumber,np.cos(Thetas))
    y=np.outer(WhitePixelColumnNumber,np.sin(Thetas))
    rhos=(x.flatten()).astype(int)+(y.flatten()).astype(int)+int(Diagonal)
    for rho , index in zip(rhos , Indices):
        HoughAccumlator[rho, index%180] += 1
    return HoughAccumlator, Rhos, Thetas

def GetPeaks(HoughAccumlator, PeaksNumber, Threshold=0,NeighborhoodSize=3):
    ''' A function that returns the Indecies of the accumulator array HoughAccumlator that
        correspond to a local maxima.  If Threshold is active all values less
        than this value will be ignored, if neighborhood_size is greater than
        (1, 1) this number of Indecies around the maximum will be surpessed. '''

    Indecies = [] 
    HoughAccumlatorCopy = np.copy(HoughAccumlator)
    for i in range(PeaksNumber):
        Index = np.argmax(HoughAccumlatorCopy)
        HoughCopyIndecies = np.unravel_index(Index, HoughAccumlatorCopy.shape)
        Indecies.append(HoughCopyIndecies)
        IndexY, IndexX = HoughCopyIndecies 
        if (IndexX - (NeighborhoodSize/2)) < 0: MinimumX = 0
        else: MinimumX = int(IndexX - (NeighborhoodSize/2))
        if ((IndexX + (NeighborhoodSize/2) + 1) > HoughAccumlator.shape[1]): MaximumX = HoughAccumlator.shape[1]
        else: MaximumX =int(IndexX + (NeighborhoodSize/2) + 1)
        if (IndexY - (NeighborhoodSize/2)) < 0: MinimumY = 0
        else: MinimumY =int( IndexY - (NeighborhoodSize/2))
        if ((IndexY + (NeighborhoodSize/2) + 1) > HoughAccumlator.shape[0]): MaximumY = HoughAccumlator.shape[0]
        else: MaximumY = int(IndexY + (NeighborhoodSize/2) + 1)
        for x in range(MinimumX, MaximumX):
            for y in range(MinimumY, MaximumY):
                HoughAccumlatorCopy[y, x] = 0
                if (x == MinimumX or x == (MaximumX - 1)):
                    HoughAccumlator[y, x] = 255
                if (y == MinimumY or y == (MaximumY - 1)):
                    HoughAccumlator[y, x] = 255
    return Indecies, HoughAccumlator
    
def PlotHoughAcculmator(HoughAccumlator, plot_title='Hough Accumulator Plot'):
    ''' A function that plot a Hough Space using Matplotlib. '''
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title(plot_title) 	
    plt.imshow(HoughAccumlator, cmap='jet')
    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.tight_layout()
    plt.show()


def DrawLines(Image, Indicies, Rhos, Thetas):
    ''' A function that takes Indicies a Rhos table and Thetas table and draws
        lines on the input images that correspond to these values. '''
    DetectedRhos = Rhos[np.array(list(zip(*Indicies))[0])]
    DetectedThetas=Thetas[np.array(list(zip(*Indicies))[1])]
    Sines=np.sin(DetectedThetas)
    Cosines=np.cos(DetectedThetas)
    StartXs = (Cosines*DetectedRhos + 1000*(-Sines)).astype(int)
    EndXs = (Cosines*DetectedRhos - 1000*(-Sines)).astype(int)
    StartYs = (Sines*DetectedRhos + 1000*(Cosines)).astype(int)
    EndYs = (Sines*DetectedRhos - 1000*(Cosines)).astype(int)
    EndPoints=np.array([*zip(EndXs, EndYs)])
    StartPoints=np.array([*zip(StartXs, StartYs)])
    for i in range(len(StartPoints)):
        cv2.line(Image,StartPoints[i] ,EndPoints[i], (0, 255, 0), 2)


