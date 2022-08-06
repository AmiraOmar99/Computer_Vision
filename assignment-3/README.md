# SBE 404B - Computer Vision

## CV_Task 3

**Team 3**

**Submitted to: Dr. Ahmed Badwy and Eng. Laila Abbas**

Submitted by:

|              Name              | Section | B.N. |
|:------------------------------:|:-------:|:----:|
|   Esraa Mohamed Saeed   |    1    |   10  |
|   Alaa Tarek Samir   |    1    |  12  |
| Amira Gamal Mohamed  |    1    |  15  |
|   Fatma Hussein Wageh   |    2    |  8  |
| Mariam Mohamed Osama |    2    |  26  |

**The programming langusage is Python 3.8.8**

- **Libraries Used:**
  
  - Time
  - numpy==1.19.5
  - matplotlib==3.4.3
  - matplotlib-inline==0.1.3
  - opencv-contrib-python==4.5.3.56
  - opencv-python==4.5.3.56
  - PyQt5==5.15.6
  - PyQt5-Qt5==5.15.2
  - sys
  

**How to run:**

- Write "python gui.py" in the terminal or click on the run button
- To get the harris output :
  - In the harris section in ui, browse for an image then enter threshold, and sensetivity values
  - Finally click on harris button
- To get Sift output:
  - In the Sift section in ui, browse for an image 
  - Finally click on sift button 
- To get Matching Output:
  - In the matching section in ui, browse for image1 and image2
  - select matching type (ncc or ssd)
  - Finally click on Match button




### **1. Extract Unique Features In Images Using Harris Operator** ###
- **Code Architecture:**
  
1.  Get the image in gray scale then calculate image x and y derivatives of the gray scale image. For this we applied np.gradient <br /> <br />
    ```python
    
    if len(src.shape) == 3:#if image is colored
        gray=ConvertToGaryscale(src)#get the gray scale
    else:#if image is grayscale
        gray=src
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)

    matrix_R = np.zeros((height,width))#matrix of zeros with the same size of image
    dy, dx = np.gradient(gray)#get the gradients 

    ```
2. Derivate again the previous values to obtain the second derivative <br /> <br />   

    ```python
    #calaculate the product and second derivative
    dx2=np.square(dx)
    dy2=np.square(dy)
    dxy=dx*dy
    ```
3. For each pixel, sum the last step obtained derivatives, using the sums of the previous step we define H matrix.<br/><br/>
    ```python
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):

            Sx2 = np.sum(dx2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sy2 = np.sum(dy2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(dxy[y-offset:y+1+offset, x-offset:x+1+offset])
            #Define the matrix H(x,y)=[[Sx2,Sxy],[Sxy,Sy2]]
            H = np.array([[Sx2,Sxy],[Sxy,Sy2]])
    ```
4. Calculate the response of the detector. <br/><br/>
    ```python
            # Find determinant and trace, use to get corner response
            det=np.linalg.det(H) #which is equal to lambda1 by lambda2
            tr=np.matrix.trace(H) #which is equal to lambda1 + lambda2

            # Harris Response R
            # Calculate the response function ( R=det(H)-k(Trace(H))^2 )
            R=det-k*(tr**2)
            matrix_R[y-offset, x-offset]=R
    matrix_R=norm(matrix_R)#normalize to be from 0 to 1        
    ```
5. Use a threshold value in order to exclude some of the detections. <br/><br/>
    ```python
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
             #Apply threshold to get corners
            value=matrix_R[y, x]
            if value>threshold:
                cv2.circle(src,(x,y),3,(0,255,0))#draw the image with green circles on corners
    ```

6. To get the computation time 
   ```python
    time_start = time.perf_counter()#start timer
    #harris code
    computation_time = (time.perf_counter() - time_start)#calculate the computation time
    ```

### **2. Feature Descriptors Using SIFT Algorithm** ###
- **Code Architecture:**
1. Generate base image from input image by upsampling by 2 in both directions and blurring.
```python
def base_generator(image, sigma, initial_sigma):
    # upsampling by 2 in both directions (fx=2, fy=2)
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    
    #image is assumed to be blured by a sigma of 0.5 so we blur by differnce of both sigmas (initial and wanted sigma)
    sigma_diff = (sigma ** 2) - (initial_sigma ** 2)
    
    # min blurring sigma is 0.01
    new_sigma = sqrt(max(sigma_diff, 0.01)) 
    print('old sigma: ({}), new sigma ({})'.format(2 * initial_sigma,new_sigma))
    
    # blur the upsampled image by the new calculated sigma instead of initial sigma
    gauss_image = GaussianBlur(image, (0, 0), sigmaX=new_sigma, sigmaY=new_sigma)
```
2. Compute number of octaves in image pyramid as function of base image shape.
```python
# Octaves number is the number of halving an image up to size 3*3 
    octaves_num = (log(border_size) / log(2)) -1 # the -1 is to avoid the last halving (1*1 image)
    octaves_num = int(round(octaves_num)) 
```
3. Generate list of sigmas at which gaussian kernels will blur the input image.
```python
def kernel_sigmas(initial_sigma, intervals):
    # according to the paper, we produce s+3 images in each octave  
    
    # The initial image is incrementally convolved with Gaussians to produce images separated by a constant factor k in scale space 

    # Initialize an array with the size of images in each octave to save the blurring sigma of each image in the octave
    
    # The first image (scale) in the octave will allways hold the initial(original) sigma

    #looping over every image in the octave starting from the second image(of index 1)
    for image_index in range(1, num_images_per_octave):
        
        sigma_previous = (k ** (image_index - 1)) * initial_sigma
        sigma_total = k * sigma_previous
        new_sigma = sqrt(sigma_total ** 2 - sigma_previous ** 2)
        gaussian_sigmas[image_index] = new_sigma
    
    gaussian_sigmas2 = array(gaussian_sigmas)
```
4. Generate scale-space pyramid of Gaussian images.
```python
def generateGaussianImages(image, num_octaves, gaussian_kernels):
    # Looping over every ocatve
    for octave_index in range(num_octaves):
        # Initialize an impty list to carry the gausian images in every octave
        
        # first image in octave already has the correct blur
        
        # looping over every kernal in the same octave from the 2nd image 
        for gaussian_kernel in gaussian_kernels[1:]:
            # blur the image by the corresponding gaussian blur
            image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            # Add the image to the list of the octave
            gaussian_images_in_octave.append(image)
        # Add the octave list to the list of octaves
        gaussian_images.append(gaussian_images_in_octave)
        
        # take the third-to-last image of the previous octave as the base of the next octave
        octave_base = gaussian_images_in_octave[-3]
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
    
    gaussian_images = array(gaussian_images, dtype=object)
```
5. Generate Difference-of-Gaussians image pyramid.
```python
def DoG_images(gaussian_images):
    # initialize a list to carry the difference of gaussians 
    
    # looping over each ocatve
    for gaussian_images_in_octave in gaussian_images:
        
        # initialize a list to carry the dogs of the same octave
        dog_images_in_octave = []
        # Subtract every two successive images per octave
        rng = int(len(gaussian_images_in_octave))
        
        for i in range(rng-1):
            first = gaussian_images_in_octave[i]
            second = gaussian_images_in_octave[i+1]
            # for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            # ordinary subtraction will not work because the images are unsigned integers
            dog_images_in_octave.append(subtract(second,first))  
        # Add the dogs of the same octave to the dogs list    
        dog_images.append(dog_images_in_octave)
    dog_images = array(dog_images, dtype=object)
```
6. Find pixel positions of all scale-space extrema in the image pyramid.
```python
def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):

    threshold = floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation

    # Looping over every octave (dogs in the same octave)
    
        # Taking 3 successive images to compute the extrema
            
            # (i, j) is the center of the 3x3 array

                    # since i,j are the center we send the images from (i-1 to i+2) as the three neighnour pixels 
                    
                    # sending 3 iamges of the same octave to compare the pixel with the surrounding pixels of the same, above and below image
```
7. Sort keypoints and remove duplicate keypoints.
```python
def get_unique_keypoints(keypoints):

    #sorting keypoints through compare function
    
    #save first keypoint to unique_keypoints list
   
    #iterating over all of the keypoints and getting the unique ones
        
        #comparing between the previous unique point and the current point
        if (previous_unique_keypoint.pt[0] != current_keypoint.pt[0]) or ...
```
8. Convert keypoint point, size, and octave to input image size.
```python
def convertKeypointsToInputImageSize(keypoints):
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    print(array(converted_keypoints).shape)
```

9. Generate descriptors for each keypoint
    
10. Call all of the above functions to get the output
```python
def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    image=ConvertToGaryscale(image)
    image = image.astype('float32')
    base_image = base_generator(image, sigma, assumed_blur)
    num_octaves = octaves_numm(min(base_image.shape))
    gaussian_kernels = kernel_sigmas(sigma, num_intervals)
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    dog_images = DoG_images(gaussian_images)
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    keypoints = get_unique_keypoints(keypoints)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    descriptors = generateDescriptors(keypoints, gaussian_images)
    return keypoints, descriptors
```

11. To get the computation time
 ```python
 t1 = time.time()
        kp, dc = computeKeypointsAndDescriptors(self.images[1], sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5)
t2 = time.time()
computation time=t2-t1

 
 ```   

12. To draw the image with key points
 ```python
 #draw the image and keypoints above it
 fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(self.images[1], 'gray')
        for pnt in kp:
            ax.scatter(pnt.pt[0], pnt.pt[1], s=(pnt.size)*10, c="red")

 
 ```  

### **3.Matching the Image Set Features** ###
- **Code Architecture:**
1. Get features extraced from SIFT <br/><br/>
    ```python
    KeyPointsOne, DescripotrOne = computeKeypointsAndDescriptors(ImageOne)
    KeyPointsTwo, DescriptorTwo = computeKeypointsAndDescriptors(ImageTwo)
2. Start timer to compute matching time <br/><br/>
    ```python
    StartTimeThree = time.perf_counter()
 3. Select the technique for matching <br/><br/>    
   ```python   
    if(Matcher == "ssd"):
        Matcher = SumOfSquaredDifference
    else:
        Matcher = NormalizedCrossCorrelations    
4. Match two images with the selected techniqe <br/><br/>    
   ```python
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
  ```
5. Compute matching time and draw matches<br/><br/>    
   ```python
    ComputationTimeThree = (time.perf_counter() - StartTimeThree)
    Matches = sorted(Matches, key=lambda x: x.distance, reverse=True)
    MatchedImage = cv2.drawMatches(ImageOne, KeyPointsOne, ImageTwo, KeyPointsTwo, Matches[:20], ImageTwo, flags=2)


