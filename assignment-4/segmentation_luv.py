from segmentation import KMeans
from segmentation import AgglomerativeClustering
from segmentation import MeanShift
import cv2
import numpy as np
from luv import RGB2LUV
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def apply_k_means_luv(source, k=5, max_iter=100):
    
    # convert to RGB
    img=np.copy(source)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #convert to luv 
    img = RGB2LUV(img)

    # reshape image to points
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # run clusters_num-means algorithm
    model = KMeans(k, max_iters=max_iter)
    y_pred = model.predict(pixel_values)

    centers = np.uint8(model.cent())
    y_pred = y_pred.astype(int)

    # flatten labels and get segmented image
    labels = y_pred.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)
    saved=mpimg.imsave("kmeans_luv.png", segmented_image)

    return segmented_image, labels

def apply_agglomerative_luv(source: np.ndarray, clusters_numbers: int = 2, initial_clusters: int = 25):

    # convert to RGB
    src = np.copy(source)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    #convert to luv 
    src = RGB2LUV(src)
    agglomerative = AgglomerativeClustering(source=src, clusters_numbers=clusters_numbers,
                                            initial_k=initial_clusters)
    saved=mpimg.imsave("agg_luv.png", agglomerative.output_image)

    return agglomerative.output_image



def apply_mean_shift_luv(source: np.ndarray, threshold: int = 60):
    

    src = np.copy(source)

    # convert to RGB
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    #convert to luv 
    src = RGB2LUV(src)

    ms = MeanShift(source=src, threshold=threshold)
    ms.run_mean_shift()
    output = ms.get_output()
    saved=mpimg.imsave("meanshift_luv.png", output)

    return output



# if __name__ == "__main__":

#     img = cv2.imread('seg-image.png')
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     segmentedimg,labels= apply_k_means_luv(source=img)

#     plt.figure()

#     plt.subplot(1, 2, 1)
#     plt.imshow(img_rgb)
#     plt.axis('off')
#     plt.title('Original image')

#     plt.subplot(1, 2, 2)
#     plt.imshow(segmentedimg)
#     plt.axis('off')
#     plt.title(f'Segmented image')

#     plt.show()
