import matplotlib.pyplot as plt
import snake
import numpy as np
import cv2
def snakes(image,x,y,z):
    img = snake.Snake( image, closed = True )
    img.set_alpha(x)
    img.set_beta(y)
    img.set_gamma(z)
    for i in range(500):
        snakeImg = img.visualize()
        x = []
        y = []
        for i in range(len(img.points)):
            x.append(img.points[i][0])
            y.append(img.points[i][1])
        area=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
        area=np.abs(area)
        perimeter = img.get_length()
        snake_changed = img.step() 
    plt.imshow(snakeImg)
    plt.show()
    print("area=",area)
    print("............")
    print("perimeter=",perimeter)

def animation(image,x,y,z):
    img = snake.Snake( image, closed = True )
    snake_window_name = "Snakes"
    img.set_alpha(x)
    img.set_beta(y)
    img.set_gamma(z)
    while(True):
        snakeImg = img.visualize()
        cv2.imshow( snake_window_name, snakeImg )
        snake_changed = img.step() 
    #Stops looping when ESC pressed
        k = cv2.waitKey(33)
        if k == 27:
            break

cv2.destroyAllWindows()