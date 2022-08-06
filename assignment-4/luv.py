
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def RGB2LUV(img):
    copied = img.copy()
    out=img=np.copy(img)
    #get the size of image
    width, height = copied.shape[:2]
    #fit rgb channels to float points from 0 to 1
    copied =copied /255.0
    #loop for image pixels
    for i in range(width):
        for j in range(height):
            #convert to XYZ plane
            X = 0.412453 * copied[i, j][0] + 0.357580 * copied[i, j][1] + 0.180423 * copied[i, j][2]
            Y = 0.212671 * copied[i, j][0] + 0.715160 * copied[i, j][1] + 0.072169 * copied[i, j][2]
            Z = 0.019334 * copied[i, j][0] + 0.119193 * copied[i, j][1] + 0.950227 * copied[i, j][2]

            #then we convert XYZ plane to LUV plane
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

            #convert to 8 bits scale

            out [i,j] [0] = ( 255.0/100) *L
            out [i,j] [1] = ( 255.0/ 354) *(U+134 )
            out [i,j] [2] = (255.0/ 262) *(V +140) 

    out=out.astype(np.uint8)

    
    saved=mpimg.imsave("luv.png", out)

    
    return out #return the LUV image as 8 bit image
# for i in range(5):
# img = cv2.imread("seg-image.png")
# cv_im=cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
# print(cv_im)     
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
# cop_img = np.copy(img)
# luv_img = RGB2LUV(img)

# fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# ax = axes.ravel()

# ax[0].imshow(cv_im)
# ax[0].set_title('Original luv Image')
# ax[0].set_axis_off()

# ax[1].imshow(luv_img )
# ax[1].set_title('Luv Image')
# ax[1].axis('image')

# plt.tight_layout()
# plt.show()