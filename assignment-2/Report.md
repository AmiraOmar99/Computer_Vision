# SBE 404B - Computer Vision

## CV_Task 2

**Team 3**

**Submitted to: Dr. Ahmed Badwy and Eng. Mohamed Adel**

Submitted by:

|              Name              | Section | B.N. |
|:------------------------------:|:-------:|:----:|
|   Esraa Mohamed Saeed   |    1    |   10  |
|   Alaa Tarek Samir   |    1    |  12  |
| Amira Gamal Mohamed  |    1    |  15  |
|   Fatma Hussein Wageh   |    2    |  8  |
| Mariam Mohamed Osama |    2    |  26  |



**The programming langusage is Python**




### Task requirments

  #### We are asked to:
- Detect edges using Canny edge detector, detect lines, circles, ellipsed located in these images (if any). Superimpose the detected shapes on the images.
- Initialize the contour for a given object and evolve the Active
Contour Model (snake) using the greedy algorithm. Represent the output as chain code and compute the perimeter and the area inside these contours.

**Detecting Edges, Lines, and Circles**

- In this Section we used Canny for Edge detection.
- In  Hough Line Transform we used Edged image to get acculmator and used it to get all detected lines indecies.
- Then we  represented all lines in the image by finite lines darwn ny cv2.lines.
- In Hough Circle Transform we Edged image to get acculmator and used it to get all detected circle indecies.
- Then we  represented all circles in the image by finite lines darwn ny cv2.circles.


**Results**

<img src="images/lines.png" alt="lines Image" width="600" height="300"/>


<img src="images/circles.png" alt="circles Image" width="600" height="300"/>


**Active Contour (Snakes)**

**Active Contour**: Active contour is defined as an active model for the segmentation process. Contours are the boundaries that define the region of interest in an image. A contour is a collection of points that have been interpolated. The interpolation procedure might be linear, splines, or polynomial, depending on how the curve in the image is described. Here we use **spline**

1. Snake Model
The snake model is a technique that has the ability to solve a broad range of segmentation problems. The model’s primary function is to identify and outline the target object for segmentation. It requires some prior knowledge of the target object’s shape, especially for complicated things. Active snake models, often known as snakes, are generally configured by the use of spline focused on minimizing energy, followed by various forces governing the image.

**Note** you must must specify an approximate shape and starting position for the snake somewhere near the desired contour.


**Greedy Algorithm**

- The greedy approach is an energy-minimizing algorithm in-troduced for 2D contours. Global mini-mization is done by means of successive local optimization.

- The greedy method is an implementation technique used to simplify the implementation of the minimization of energy without having to perform an optimization algorithm technique such as the gradient descent. It works under the assumption that finding for each point of the contour the closest local energy minimizing neighbor will converge to the overall global minimum of the contour.

<img src="images/equation.png" alt="Equation" width="800" height="150"/>

**Results**

- Image_1

<img src="images/snake1.png" alt="snake1 Image" width="600" height="300"/>


-Image_2

<img src="images/snake2.png" alt="snake2 Image" width="600" height="300"/>

- Image_3

<img src="images/snake3.png" alt="snake3 Image" width="600" height="300"/>


-Image_4

<img src="images/snake4.png" alt="snake4 Image" width="600" height="300"/>
