from Utilities import *
import matplotlib.pyplot as plt
def Canny(image):
    Weak=50
    GaussianImage=GaussianFilter(image,1)
    Magnitude , Theta = Gradient(GaussianImage)
    Suppression=NonMaxSuppression(Magnitude,Theta)
    Thresholding=Threshold(Suppression,15,5,Weak)
    FinalImage=Hysteresis(Thresholding,Weak)  
    plt.imshow(FinalImage,cmap='gray')
    plt.show()
    return FinalImage