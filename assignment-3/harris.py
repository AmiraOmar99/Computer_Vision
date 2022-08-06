import cv2
import numpy as np
from matplotlib import pyplot as plt
from ConvertToGrayscale import *
import time
from normalize import *


def readimage(img):
    #Read the image and get the height and width of it
    img = cv2.imread(img)
    h=img.shape[0]
    w=img.shape[1]
    return img,h,w


def harris(img,height,width,w_size,k,threshold):
    #k is the sensetivity factor to separate corners from edges
    #w_size is used to create a window around each pixel
    #threshold is used to get the corners of image
    src=np.copy(img)
    if len(src.shape) == 3:#if image is colored
        gray=ConvertToGaryscale(src)#get the gray scale
    else:#if image is grayscale
        gray=src
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)

    time_start = time.perf_counter()#start timer
    matrix_R = np.zeros((height,width))#matrix of zeros with the same size of image
    dy, dx = np.gradient(gray)#get the gradients 
    
    #calaculate the product and second derivative
    dx2=np.square(dx)
    dy2=np.square(dy)
    dxy=dx*dy
    offset = int( w_size / 2 )

    print ("Finding Corners")
    #loop over columns and rows of image
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):

            Sx2 = np.sum(dx2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sy2 = np.sum(dy2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(dxy[y-offset:y+1+offset, x-offset:x+1+offset])
            # Find determinant and trace, use to get corner response
            #Define the matrix H(x,y)=[[Sx2,Sxy],[Sxy,Sy2]]

            H = np.array([[Sx2,Sxy],[Sxy,Sy2]])

            det=np.linalg.det(H) #which is equal to lambda1 by lambda2
            tr=np.matrix.trace(H) #which is equal to lambda1 + lambda2

            # Harris Response R
            # Calculate the response function ( R=det(H)-k(Trace(H))^2 )
            R=det-k*(tr**2)
            matrix_R[y-offset, x-offset]=R

    matrix_R=norm(matrix_R)#normalize to be from 0 to 1
    
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
             #Apply threshold to get corners
            value=matrix_R[y, x]
            if value>threshold:
                cv2.circle(src,(x,y),3,(0,255,0))#draw the image with green circles on corners
                
                
    computation_time = (time.perf_counter() - time_start)#calculate the computation time
    src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    cv2.imwrite("Harries.png", src)
    return src,computation_time

#Test
# img,h,w=readimage("squares.png")
# img_harris,t=harris(img,h,w,5, 0.04, 0.75)
# # print(t)
# plt.figure("Harris")
# plt.imshow(cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB)), plt.title("Harris")
# plt.xticks([]), plt.yticks([])
# plt.show()




