
from matplotlib import pyplot as plt
import numpy as np
import cv2

def GlobalDoubleThresholding(image,High,Low=0 , Weak=0):
    ResultantImage = np.zeros(image.shape)
    HighNumbersRow, HighNumbersColumn = np.where(image >= High)
    LowNumbersRow, LowNumbersColumn = np.where((image <= High) & (image >= Low))
    ResultantImage[HighNumbersRow, HighNumbersColumn] = 255
    ResultantImage[LowNumbersRow, LowNumbersColumn] =Weak 
    return ResultantImage  
def LocalThresholding(source: np.ndarray, RegionsX: int, RegionsY: int, ThresholdingFunction , Text):
    """
       Applies Local Thresholding To The Given Grayscale Image Using The Given Thresholding Callback Function
       :param source: NumPy Array of The Source Grayscale Image
       :param Regions: Number of Regions To Divide The Image To
       :param ThresholdingFunction: Function That Does The Thresholding
       :return: Thresholded Image
       """
    Image = np.copy(source)
    if len(Image.shape) > 2:
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
    MaximumRow, MaximumColumn = Image.shape
    ThresholdedImage = np.zeros((MaximumRow, MaximumColumn))
    RowsStep = MaximumRow // RegionsY
    ColumnsStep = MaximumColumn // RegionsX
    ColumnsRange = []
    RowsRange = []
    for i in range(0, RegionsX):
        ColumnsRange.append(ColumnsStep * i)
    for i in range(0, RegionsY):
        RowsRange.append(RowsStep * i)

    ColumnsRange.append(MaximumColumn)
    RowsRange.append(MaximumRow)
    for x in range(0, RegionsX):
        for y in range(0, RegionsY):
            print(ColumnsRange[x+1])
            print(RowsRange[y+1])
            ThresholdedImage[RowsRange[y]:RowsRange[y + 1], ColumnsRange[x]:ColumnsRange[x + 1]] = ThresholdingFunction(Image[RowsRange[y]:RowsRange[y + 1], ColumnsRange[x]:ColumnsRange[x + 1]])
    cv2.imwrite(Text,ThresholdedImage)
    return ThresholdedImage

def OptimalThresholding(source: np.ndarray):
    """
    Applies Thresholding To The Given Grayscale Image Using The Optimal Thresholding Method
    :param source: NumPy Array of The Source Grayscale Image
    :return: Thresholded Image
    """

    Image = np.copy(source)
    if len(Image.shape) > 2:
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
    else:
        pass
    ## Calculate Initial Thresholds Used in Iteration
    # Maximum X & Y Values For The Image
    MaxX = Image.shape[1] - 1
    MaxY = Image.shape[0] - 1
    # Mean Value of Background Intensity, Calculated From The Four Corner Pixels
    BackgroundMean = (int(Image[0, 0]) + int(Image[0, MaxX]) + int(Image[MaxY, 0]) + int(Image[MaxY, MaxX])) / 4
    Sum = 0
    Length = 0
    # Loop To Calculate Mean Value of Foreground Intensity
    for i in range(0, Image.shape[1]):
        for j in range(0, Image.shape[0]):
            # Skip The Four Corner Pixels
            if not ((i == 0 and j == 0) or (i == MaxX and j == 0) or (i == 0 and j == MaxY) or (
                    i == MaxX and j == MaxY)):
                Sum += Image[j, i]
                Length += 1
    ForegroundMean = Sum / Length
    # Get The Threshold, The Average of The Mean Background & Foreground Intensities
    InitialThreshold = (BackgroundMean + ForegroundMean) / 2
    NewThreshold = GetOptimalThreshold(Image, InitialThreshold)
    iteration = 0
    # Iterate Till The Threshold Value is Constant Across Two Iterations
    while InitialThreshold != NewThreshold:
        InitialThreshold = NewThreshold
        NewThreshold = GetOptimalThreshold(Image, InitialThreshold)
        iteration += 1
    # Return Thresholded Image Using Global Thresholding
    OptImg=GlobalDoubleThresholding(Image, NewThreshold)
    cv2.imwrite("OptImg.png", OptImg) 
    return OptImg

def GetOptimalThreshold(Image: np.ndarray, Threshold):
    """
    Calculates Optimal Threshold Based on Given Initial Threshold
    :param Image: NumPy Array of The Source Grayscale Image
    :param Threshold: Initial Threshold
    :return OptimalThreshold: Optimal Threshold Based on Given Initial Threshold
    """
    # Get Background Array, Consisting of All Pixels With Intensity Lower Than The Given Threshold
    Background = Image[np.where(Image < Threshold)]
    # Get Foreground Array, Consisting of All Pixels With Intensity Higher Than The Given Threshold
    Foreground = Image[np.where(Image > Threshold)]
    # Mean of Background & Foreground Intensities
    BackgroundMean = np.mean(Background)
    ForegroundMean = np.mean(Foreground)
    # Calculate Optimal Threshold
    OptimalThreshold = (BackgroundMean + ForegroundMean) / 2
    return OptimalThreshold



def OtsuThresholding(source: np.ndarray):
    """
     Applies Thresholding To The Given Grayscale Image Using Otsu's Thresholding Method
     :param source: NumPy Array of The Source Grayscale Image
     :return: Thresholded Image
     """
    Image = np.copy(source)

    if len(Image.shape) > 2:
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
    else:
        pass
    # Get Image Dimensions
    RowsRange, ColumnsRange = Image.shape
    # Get The Values of The Histogram Bins
    HistogramValues = np.histogram(Image.ravel(), 256)[0]
    # print(HistogramValues)
    # plt.show()
    # Calculate The Probability Density Function
    ProbabilityDenistyFunction = HistogramValues / (RowsRange * ColumnsRange)
    print(ProbabilityDenistyFunction)
    # Calculate The Cumulative Density Function
    CumulativeDenistyFunction = np.cumsum(ProbabilityDenistyFunction)
    OptimalThreshold = 1
    MaxVariance = 0
    # Loop Over All Possible Thresholds, Select One With Maximum Variance Between Background & The Object (Foreground)
    for t in range(1, 255):
        # Background Intensities Array
        Background = np.arange(0, t)
        # Object/Foreground Intensities Array
        Foreground = np.arange(t, 256)
        # Calculation Mean of Background & The Object (Foreground), Based on CumulativeDenistyFunction & ProbabilityDenistyFunction
        CumulativeDenistyFunction2 = np.sum(ProbabilityDenistyFunction[t + 1:256])
        BackgroundMean = sum(Background * ProbabilityDenistyFunction[0:t]) / CumulativeDenistyFunction[t]
        ForegroundMean = sum(Foreground * ProbabilityDenistyFunction[t:256]) / CumulativeDenistyFunction2
        # Calculate Cross-Class Variance
        Variance = CumulativeDenistyFunction[t] * CumulativeDenistyFunction2 * (ForegroundMean - BackgroundMean) ** 2
        # Filter Out Max Variance & It's Threshold
        if Variance > MaxVariance:
            MaxVariance = Variance
            OptimalThreshold = t
    OtsuImg= GlobalDoubleThresholding(Image, OptimalThreshold)       
    cv2.imwrite("OtsuImg.png", OtsuImg)        
    return OtsuImg

def SpectralThresholding(source: np.ndarray):
    """
     Applies Thresholding To The Given Grayscale Image Using Spectral Thresholding Method
     :param source: NumPy Array of The Source Grayscale Image
     :return: Thresholded Image
     """
    Image = np.copy(source)
    if len(Image.shape) > 2:
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
    # Get Image Dimensions
    RowsRange, ColumnsRange = Image.shape
    # Get The Values of The Histogram Bins
    HistogramValues = np.histogram(Image.ravel(), 256)[0]
    # plt.show()
    # Calculate The Probability Density Function
    ProbabilityDenistyFunction = HistogramValues / (RowsRange * ColumnsRange)
    # Calculate The Cumulative Density Function
    CumulativeDenistyFunction = np.cumsum(ProbabilityDenistyFunction)
    OptimalLow = 1
    OptimalHigh = 1
    MaxVariance = 0
    # Loop Over All Possible Thresholds, Select One With Maximum Variance Between Background & The Object (Foreground)
    Global = np.arange(0, 256)
    GlobalMean = sum(Global * ProbabilityDenistyFunction) / CumulativeDenistyFunction[-1]
    for LowT in range(1, 254):
        for HighT in range(LowT + 1, 255):
            # Background Intensities Array
            Background = np.arange(0, LowT)
            # Low Intensities Array
            Low = np.arange(LowT, HighT)
            # High Intensities Array
            High = np.arange(HighT, 256)
            # Get Low Intensities CumulativeDenistyFunction
            CDFL = np.sum(ProbabilityDenistyFunction[LowT:HighT])
            # Get Low Intensities CumulativeDenistyFunction
            CDFH = np.sum(ProbabilityDenistyFunction[HighT:256])
            # Calculation Mean of Background & The Object (Foreground), Based on CumulativeDenistyFunction & ProbabilityDenistyFunction
            BackgroundMean = sum(Background * ProbabilityDenistyFunction[0:LowT]) / CumulativeDenistyFunction[LowT]
            LowMean = sum(Low * ProbabilityDenistyFunction[LowT:HighT]) / CDFL
            HighMean = sum(High * ProbabilityDenistyFunction[HighT:256]) / CDFH
            # Calculate Cross-Class Variance
            Variance = (CumulativeDenistyFunction[LowT] * (BackgroundMean - GlobalMean) ** 2 + (CDFL * (LowMean - GlobalMean) ** 2) + 
            (CDFH * (HighMean - GlobalMean) ** 2))
            # Filter Out Max Variance & It's Threshold
            if Variance > MaxVariance:
                MaxVariance = Variance
                OptimalLow = LowT
                OptimalHigh = HighT
    SpectImg=GlobalDoubleThresholding(Image, OptimalHigh, OptimalLow, 128)      
    cv2.imwrite("SpectImg.png", SpectImg)        
    return SpectImg           
