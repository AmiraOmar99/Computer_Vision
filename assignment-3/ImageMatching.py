import cv2, time, numpy as np
import matplotlib.pyplot as plt
from SIFT import computeKeypointsAndDescriptors

def SumOfSquaredDifference(DescripotrOne, DescriptorTwo):
    SumSquare = 0
    # Get SSD between the 2 vectors
    for m in range(len(DescripotrOne)):
        SumSquare += (DescripotrOne[m] - DescriptorTwo[m]) ** 2
    SumSquare = - (np.sqrt(SumSquare))
    return SumSquare
def NormalizedCrossCorrelations(DescripotrOne, DescriptorTwo):
    #Vector normalization
    NormalizedOutputOne = (DescripotrOne - np.mean(DescripotrOne)) / (np.std(DescripotrOne))
    NormalizedOutputTwo = (DescriptorTwo - np.mean(DescriptorTwo)) / (np.std(DescriptorTwo))
    CorrelationVerctor = np.multiply(NormalizedOutputOne, NormalizedOutputTwo)
    #Get Correlation
    Correlation = float(np.mean(CorrelationVerctor))
    return Correlation

def GetMatchedImage(ImageOne, ImageTwo, Matcher):
    
    if(Matcher == "ssd"):
        Matcher = SumOfSquaredDifference
    else:
        Matcher = NormalizedCrossCorrelations
    
    StartTimeOne = time.perf_counter() #Starts timer for computation time of first image ISFT.
    KeyPointsOne, DescripotrOne = computeKeypointsAndDescriptors(ImageOne)
    ComputationTimeOne = (time.perf_counter() - StartTimeOne)
    SatrtTimeTwo = time.perf_counter()
    KeyPointsTwo, DescriptorTwo = computeKeypointsAndDescriptors(ImageTwo)
    CoputationTimeTwo = (time.perf_counter() - SatrtTimeTwo)
    StartTimeThree = time.perf_counter()
    Matches = FeatureMatching(DescripotrOne, DescriptorTwo, Matcher)
    ComputationTimeThree = (time.perf_counter() - StartTimeThree)
    Matches = sorted(Matches, key=lambda x: x.distance, reverse=True)
    MatchedImage = cv2.drawMatches(ImageOne, KeyPointsOne, ImageTwo, KeyPointsTwo, Matches[:20], ImageTwo, flags=2)
    print(MatchedImage.shape)
    MatchedImage = cv2.cvtColor(MatchedImage, cv2.COLOR_BGR2RGB)
    cv2.imwrite("MatchedImage.png", MatchedImage) 
    return ComputationTimeOne , CoputationTimeTwo, ComputationTimeThree

def FeatureMatching(DescripotrOne, DescriptorTwo, Matcher):
    Matches = []
    #Get number of keypoints of each image.
    KeyPointsOneNumber = DescripotrOne.shape[0]
    KeyPointsTwoNumber = DescriptorTwo.shape[0]
    for i in range(KeyPointsOneNumber):
        Distance = -np.inf
        IndexY = -1
        #Loop over each key point in image2
        for j in range(KeyPointsTwoNumber):
            value = Matcher(DescripotrOne[i], DescriptorTwo[j])
            if value > Distance:
                Distance = value
                IndexY = j

        DescripotrMatcher = cv2.DMatch()
        DescripotrMatcher.queryIdx = i
        DescripotrMatcher.trainIdx = IndexY
        DescripotrMatcher.distance = Distance
        Matches.append(DescripotrMatcher)

    return Matches
