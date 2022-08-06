# SBE 404B - Computer Vision

## CV_Task 4

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
  - sys
  - os
  - numpy==1.19.5
  - matplotlib==3.4.3
  - matplotlib-inline==0.1.3
  - opencv-contrib-python==4.5.3.56
  - opencv-python==4.5.3.56
  - PyQt5==5.15.6
  - PyQt5-Qt5==5.15.2

**How to run:**
- **After running the code(main.py file) mainwindow will be opened on thresholding tab and for running thresholding**
- 1.browse an image using browse button.
- 2.we need to choose thresolding method from combobox.
- 3.if we choose any type of local thresholding we have to insert values for regionx and regiony.

- **After running the code(main.py file) mainwindow will be opened, choose which tab you want, and for running LUV and segmentation clustering:**
- 1. To map the RGB image to LUV color space, you need to browse for an image by browse button, then click on convert to luv button
- 2. For Kmeans clustering, browse an image using the browse button and enter clusters number, then you have the ability to choose which space you want to apply the algorithm on (RGB or LUV) (there are two buttons).
- 3. For region growing, browse an image using the browse button and enter threshold value, then click on the segmentation button.
- 4. For meanshift, browse an image using the browse button and enter threshold value, then you have the ability to choose which space you want to apply the algorithm on (RGB or LUV) (there are two buttons).
- 5. For Aglloremative clustering, browse an image using the browse button and enter clusters number, then you have the ability to choose which space you want to apply the algorithm on (RGB or LUV) (there are two buttons).

**Note**
- For all segmentation results, threshold and number of clusters values have a big effect on the 
result.
- The image size affects the computation time.

 
**Code Architecture**<br>
## Modules: 
**1. Thresholding.py:** Contating threshold algorithms implementation [optimal thresholding, otsu and spectral] both local and global. <br>
**2. Segmenation.py:** Contains  essential and basic functions used in segmenation algorithms **such as**
- point class for all points attributes such as x, y coordinates <br>
- class for KMeans Algorithm
- class for MeanShift Algorithm
- class for Agglomerative Clustering
- function for RegionGrowing.

**3. Segmentation_LUV.py:**<br> Applies different segmenation algorithms  on LUV images.

**4. Segmentation_RGB.py:**<br>Applies different segmenation algorithms  on RGB images.

**5. LUV.py:**<br> Contains basic functions for Conversion form RGB to LUV.

**6. main.py:**<br> Main file

**7. mainWindow.py**<br> GUI Script


## 1. Thresholding
**1. Optimal Thresholding.**
- Calculate Initial Thresholds Used in Iteration
- Iterate Till The Threshold Value is Constant Across Two Iterations
- Apply thresholding using calculated value
**2. Otsu Thresholding.**
- Get the threshold with maximum variance between background and foreground
- Apply thresholding using calculated value
**3. Spectral Thresholding**
- we calculate two CDF one for high and the other for low intensties and calculate variance using both.
- Apply double thresholding using calculated values

**4. Local Thresholding for the three methods**
- we divide the image into regions and the number of regions depends on RegionX and RegionY. 
- apply any of these global thresholding for each region.

## 2. Map from RGB to LUV

**1. Convert( ) RGBToXYZ( ) using the following formulas.**

 X = 0.412453*R + 0.35758 *G + 0.180423*B<br>
 Y = 0.212671*R + 0.71516 *G + 0.072169*B<br>
 Z = 0.019334*R + 0.119193*G + 0.950227*B<br>
```python 
#convert to XYZ plane
    X = 0.412453 * copied[i, j][0] + 0.357580 * copied[i, j][1] + 0.180423 * copied[i, j][2]
    Y = 0.212671 * copied[i, j][0] + 0.715160 * copied[i, j][1] + 0.072169 * copied[i, j][2]
    Z = 0.019334 * copied[i, j][0] + 0.119193 * copied[i, j][1] + 0.950227 * copied[i, j][2]
```
 **Note:** 
 - The formulas assume that R, G, and B values are normalized to [0 to 1] for integer data types.
 - For floating point data types, the data must already be in the range [0 to 1].
 - For integer data types, the converted image data is saturated to [0 to 1] and scaled to the data type range.

**2. Convert( ) XYZ to LUV as follows.**
**CIE chromaticity coordinates:**

  - xn = 0.312713
  - yn = 0.329016

 **CIE luminance:**

Yn = 1.0

 un = 4*xn / (-2*xn + 12*yn + 3) <br>
 vn = 9*yn / (-2*xn + 12*yn + 3) <br>

 u = 4*X / (X + 15*Y + 3*Z) <br>
 v = 9*Y / (X + 15*Y + 3*Z)<br>

 L = 116 * (Y/Yn)^(1/3) - 16 <br>
 U = 13*L*(u-un) <br>
 V = 13*L*(v-vn) <br>
```python 
#to compare Y value
    numy=0.008856

    #get L
    if (Y > numy):
        L =((116.0 * (Y **(1/3)) ) - 16.0) 
    if(Y <= numy):
        L = (903.3 * Y)
    #get U and V dashed
    if(( X + (15.0*Y ) + (3.0*Z) )!=0):
        u1 = 4.0*X /( X + (15.0*Y ) + (3.0*Z) )
        v1 = 9.0*Y /( X + (15.0*Y ) + (3.0*Z) )
    #constants
    un=0.19793943
    vn=0.46831096
    #get U

    U = 13 * L * (u1 -un)
    #get V
    V = 13 * L * (v1 -vn)
```
 
 - Computed L component values are in the range [0 to 100].
 - Computed U component values are in the range [-124 to 220].
 - Computed V component values are in the range [-140 to 116].

**3. Scaling is performed as follows.**

8U data type:
 - L = L * FW_MAX_8U / 100
 - U = (U + 134) * FW_MAX_8U / 354
 - V = (V + 140) * FW_MAX_8U / 256
```python
out [i,j] [0] = ( 255.0/100) *L
out [i,j] [1] = ( 255.0/ 354) *(U+134 )
out [i,j] [2] = (255.0/ 262) *(V +140)
```

## 3. Segmentation Using Clustering
We implemented the following *four* Clustering methods:

- K-Means
- MeanShift
- Region Growing
- Agglomerative Clustering

There are *two* main parameters to determine:
- No. of Clusters: to specify how many clusters you need in the output image.
- Threshold: to threshold the output image in specific level.
  
**Code Architecture**

## 1. K-Means 
1. specify number of k clusters to assign.
``` python
def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K #number of clusters
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []
```
2. randomly initialize k centroids. 
``` python
# randomly initialize k centroids 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

```
3. **repreat** 
4. **expectation:** Assign each point to its closest centroid.
5. **Maximization:** Compute new centroid (mean) of each cluster.
6. **Until:** centroid position doesn't change. 
``` python 
for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self.CreateClusters(self.centroids)
            if self.plot_steps:
                self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self.Get_Cent(self.clusters)

            # check if clusters have changed
            if self.IsConverged(centroids_old, self.centroids)
                break
```

## 2. Mean Shift 
1. We take one single data point and perform the next steps. These steps are performed on all the data points in the space.
2. Using a bandwidth, we draw a region around the data point and take all those points that are falling within that region.
```python
 def calculate_euclidean_distance(self, current_mean_random: bool, threshold: int):

        #calculate the Euclidean distance of all the other pixels in M with the current mean.

        # we draw a region around the data point
        # and take all those points that are falling within that region.

        
        below_threshold_arr = []

        # selecting a random row from the feature space and assigning it as the current mean
        if current_mean_random:
            current_mean = np.random.randint(0, len(self.feature_space))
            self.current_mean_arr = self.feature_space[current_mean]

        for f_indx, feature in enumerate(self.feature_space):
            # Finding the euclidean distance of the randomly selected row i.e. current mean with all the other rows
            ecl_dist = euclidean_distance(self.current_mean_arr, feature)

            # Checking if the distance calculated is within the threshold. If yes taking those rows and adding
            # them to a list below_threshold_arr
            if ecl_dist < threshold:
                below_threshold_arr.append(f_indx)

        return below_threshold_arr, self.current_mean_arr

```
3. Once we have lot of other data points, we find the mean of all those data points and assign the mean of those data points to be the new mean (This is where we do the mean shift).
```python 
def get_new_mean(self, below_threshold_arr: list):
        #Once we have lot of other data points, 
        # we find the mean of all those data points and assign the 
        # mean of those data points to be the new mean
        iteration = 0.01

        # For all the rows found and placed in below_threshold_arr list, calculating the average of
        # each channel and index positions.
        mean_1 = np.mean(self.feature_space[below_threshold_arr][:, 0])
        mean_2 = np.mean(self.feature_space[below_threshold_arr][:, 1])
        mean_3 = np.mean(self.feature_space[below_threshold_arr][:, 2])
        mean_i = np.mean(self.feature_space[below_threshold_arr][:, 3])
        mean_j = np.mean(self.feature_space[below_threshold_arr][:, 4])

        # Finding the distance of these average values with the current mean and comparing it with iter
        mean_e_distance = (euclidean_distance(mean_1, self.current_mean_arr[0]) +
                           euclidean_distance(mean_2, self.current_mean_arr[1]) +
                           euclidean_distance(mean_3, self.current_mean_arr[2]) +
                           euclidean_distance(mean_i, self.current_mean_arr[3]) +
                           euclidean_distance(mean_j, self.current_mean_arr[4]))

        # If less than iter, find the row in below_threshold_arr that has i, j nearest to mean_i and mean_j
        # This is because mean_i and mean_j could be decimal values which do not correspond
        # to actual pixel in the Image array.
        if mean_e_distance < iteration:
            new_arr = np.zeros((1, 3))
            new_arr[0][0] = mean_1
            new_arr[0][1] = mean_2
            new_arr[0][2] = mean_3

            # When found, color all the rows in below_threshold_arr with
            # the color of the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
```
4. The above process is repeated until the cluster no longer includes extra data point within it.
5. Once the process is repeated for all data points, we finally reach the clustered data points as if all the points in the sample space reach its corresponding local maximas.

## 3.Region Growing 
1. Scan the image in sequence! Find the first pixel that does not belong, and set the pixel as (x0, Y0).
2. Taking (x0, Y0) as the center, consider the neighborhood pixels (x, y) of (x0, Y0), if (x0, Y0) meets the growth criteria, merge (x, y) and (x0, Y0) in the same region, and push (x, y) onto the stack.
3. Take a pixel from the stack and return it to **step 2** as (x0, Y0).
```python 
while (len(seedList) > 0):
        #Taking (x0, Y0) as the center,
        #consider the neighborhood pixels (x, y) of (x0, Y0),
        # if (x0, Y0) meets the growth criteria, merge (x, y) and 
        # (x0, Y0) in the same region, and push (x, y) onto the stack.
        #then repeat till stack is empty

        currentPoint = seedList.pop(0)
        seedMark[currentPoint.x, currentPoint.y] = label

        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y

            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width:
                continue

            grayDiff = get_diff(im, currentPoint, Point(tmpX, tmpY))

            if grayDiff < threshold and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))

    return seedMark
```
4. When the stack is empty! Return to **step 1**.
5. **Repeat** steps 1-4 until each point in the image has attribution. Growth ends.

**Note**
Initial seeds affect the the segmented image.

## 4. Agglomerative Clustering
1. Make each data point as a single-point cluster.
```python 
# Make each data point as a single-point cluster
        groups = {}
        d = int(256 / self.initial_k)
        for i in range(self.initial_k):
            j = i * d
            groups[(j, j, j)] = []
```
2. Take the two closest distance clusters by single linkage method and make them one clusters.
```python
for i, p in enumerate(points):
            #Take the two closest distance clusters by single linkage method and make them one clusters
            go = min(groups.keys(), key=lambda c: euclidean_distance(p, c))
            groups[go].append(p)
```
3. Repeat step 2 until there is only one cluster.
```python 
while len(self.clusters_list) > self.clusters_num:
            # Find the closest (most similar) pair of clusters
            cluster1, cluster2 = min(
                [(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                key=lambda c: clusters_distance_2(c[0], c[1]))

            # Remove the two clusters from the clusters list
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]

            # Merge the two clusters
            merged_cluster = cluster1 + cluster2

            # Add the merged cluster to the clusters list
            self.clusters_list.append(merged_cluster)
```
4. Create a Dendrogram to visualize the history of groupings.
5. Find optimal number of clusters from Dendrogram.
```python 
def predict_cluster(self, point):
        
        # Find cluster number of point
        
        # assuming point belongs to clusters that were computed by fit functions

        return self.cluster[tuple(point)]

    def predict_center(self, point):
        
        # Find center of the cluster that point belongs to
        
        point_cluster_num = self.predict_cluster(point)
        center = self.centers[point_cluster_num]
        return center
```
