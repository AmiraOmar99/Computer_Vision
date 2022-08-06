import cv2
import math 
import numpy as np
from Utilities import *

class Snake:
    min_distance_b_points = 5           # The minimum distance between two points other wise we consider them overlapped. 
    max_distance_b_points = 50          # The maximum distance to insert a point.
    kernel_size_search = 7              # The size of the iterative kernel

    ##### Snake Variables#####
    closed = True       # True >> closed snake(Shrinks to contain object inside) False>> open snake (expands to contain the object ) 
    alpha = 0.5         # The weight of the continuity energy.(internal)
    beta = 0.5          # The weight of the Curveture energy. (internal)
    gamma = 0.5         # The weight of the Image Energy. (gradient / External)
    starting_points = 50       # initial number of snake points
    snake_length = 0
    gray = None         # The image in grayscale.
    binarised = None       # The binary image.
    gradientX = None    # The X_gradient (sobel) of the image.
    gradientY = None    # The Y_gradient (sobel) of the image.
    points = None

    def __init__(self,image,closed = True):
        # Image Properties
        self.image = image
        self.width=image.shape[1]
        self.height = image.shape[0]
        
        self.closed = closed # set the contour to be closed loop

        self.gray = ConvertToGaryscale( self.image)
        self.binarised = local_thresh( self.gray, 11 )
        self.gradientX,self.gradientY =Sobel( self.gray )

        # Get center of the image to draw the cicle
        half_width = math.floor( self.width / 2 )
        half_height = math.floor( self.height / 2 )

        ###########If we use the closed loop, we put the large circle that will reduce untill round the edges of objects######
        if self.closed: # we use large cicle that shrinks to the boundaried of the object
            n = self.starting_points
            radius = half_width if half_width < half_height else half_height # we choose radius to be the minimum out image dimenssions 
            self.points = [ np.array([
                half_width + math.floor( math.cos( 2 * math.pi / n * x ) * radius ),
                half_height + math.floor( math.sin( 2 * math.pi / n * x ) * radius ) ])
                for x in range( 0, n )
            ]
        else:   # in case open we use a horizontal line at first
            n = self.starting_points
            factor = math.floor( half_width / (self.starting_points-1) )
            self.points = [ np.array([ math.floor( half_width / 2 ) + x * factor, half_height ])
                for x in range( 0, n )
            ]

    def visualize(self):
        img = self.image.copy()
        # To draw connected points of the contour
        point_color = ( 0, 255, 0 )     
        line_color = ( 50, 0, 75 )     
        thickness = 2                  
        
        num_points = len(self.points)

        # Draw a line between the current and the next point
        for i in range( 0, num_points - 1 ):
            cv2.line( img, tuple( self.points[ i ] ), tuple( self.points[ i + 1 ] ), line_color, thickness )

        # 0 -> N (Closes the snake)
        if self.closed:
            cv2.line(img, tuple( self.points[ 0 ] ), tuple( self.points[ num_points-1 ] ), line_color, thickness )

        # Drawing circles over points
        [ cv2.circle( img, tuple( x ), thickness, point_color, -1) for x in self.points ]

        return img
    
    def dist (first_point,second_point):
        #controls the insertion and removing points around object
        return np.sqrt(np.sum((first_point-second_point) ** 2))

    ####### Normalization funtion to normalize the kernel of search
    def normalize (kernel):
        
        abs_sum = np.sum( [ abs( x ) for x in kernel ] )
        return kernel / abs_sum if abs_sum != 0 else kernel
        
    def get_length(self): #for area and primeter calculations
        n_points = len(self.points)
        if not self.closed:
            n_points -= 1

        return np.sum( [ Snake.dist( self.points[i], self.points[ (i+1)%n_points  ] ) for i in range( 0, n_points ) ] )

# Energies(Internal and External) 
# Continuity energy 
    def cont_energy(self, p, prev):
        # The average distance between points in the snake
        avg_dist = self.snake_length / len(self.points)
        un = Snake.dist( prev, p )
        dun = abs( un - avg_dist )
        return dun**2

# Curveture energy
    def curv_energy(self, p, prev, next ):
        # previous and current point
        prev_x = p[0] - prev[0]
        prev_y = p[1] - prev[1]
        distant = math.sqrt(prev_x**2 + prev_y**2)

        # currrent and next point
        next_x = p[0] - next[0]
        next_y = p[1] - next[1]
        next_distant = math.sqrt( next_x**2 + next_y**2 )

        if distant == 0 or next_distant == 0:
            return 0

        cx = float( next_x + prev_x )  / ( distant * next_distant )
        cy = float( next_y + prev_y ) / ( distant * next_distant )
        cn = cx**2 + cy**2
        return cn


# Image (Gradient) Energy
    def Grad_energy(self,p):
        if p[0] < 0 or p[0] >= self.width or p[1] < 0 or p[1] >= self.height:
            return np.finfo(np.float64).max #return None when cached
         
        return -( self.gradientX[ p[1] ][ p[0] ]**2 + self.gradientY[ p[1] ][ p[0] ]**2  )


    def set_alpha(self,parm):
        self.alpha = parm /100

    def set_beta(self,parm):
        self.beta = parm/100

    def set_gamma(self,parm):
        self.gamma = parm/100

    def remove_overlaping_points(self):
        
        snake_size = len( self.points )

        for i in range( 0, snake_size ):
            for j in range( snake_size-1, i+1, -1 ):
                if i == j:
                    continue

                curr = self.points[ i ]
                end = self.points[ j ]

                dist = Snake.dist( curr, end )

                if dist < self.min_distance_b_points:
                    remove_indexes = range( i+1, j ) if (i!=0 and j!=snake_size-1) else [j]
                    remove_size = len( remove_indexes )
                    non_remove_size = snake_size - remove_size
                    if non_remove_size > remove_size:
                        self.points = [ p for k,p in enumerate( self.points ) if k not in remove_indexes ]
                    else:
                        self.points = [ p for k,p in enumerate( self.points ) if k in remove_indexes ]
                    snake_size = len( self.points )
                    break


    def add_missing_points(self):
        snake_size = len( self.points )
        for i in range(0,snake_size):
            curr = self.points[i]
            prev = self.points[(i+snake_size-1)%snake_size]
            next = self.points[(i+1)%snake_size]
            next2 = self.points[(i+2)%snake_size]

            if Snake.dist(curr,next) > self.max_distance_b_points:
                c0 = 0.125 / 6.0
                c1 = 2.875 / 6.0
                c2 = 2.875 / 6.0
                c3 = 0.125 / 6.0
                x = prev[0] * c3 + curr[0] * c2 + next[0] * c1 + next2[0] * c0
                y = prev[1] * c3 + curr[1] * c2 + next[1] * c1 + next2[1] * c0

                new_point = np.array( [ math.floor( 0.5 + x ), math.floor( 0.5 + y ) ] )

                self.points.insert( i+1, new_point )
                snake_size += 1


    def step(self):
        changed = False
        self.snake_length = self.get_length()
        new_snake = self.points.copy()

        search_kernel_size = (self.kernel_size_search,self.kernel_size_search)
        hks = math.floor(self.kernel_size_search/2)
        energy_cont = np.zeros(search_kernel_size)
        energy_curv = np.zeros(search_kernel_size)
        energy_grad = np.zeros(search_kernel_size)

        for i in range( 0, len( self.points ) ):
            curr = self.points[ i ]
            prev = self.points[ ( i + len( self.points )-1 ) % len( self.points ) ]
            next = self.points[ ( i + 1 ) % len( self.points ) ]

            for dx in range( -hks, hks ):
                for dy in range( -hks, hks ):
                    p = np.array( [curr[0] + dx, curr[1] + dy] )

                    # Calculates the energy functions on p
                    energy_cont[ dx + hks ][ dy + hks ] = self.cont_energy(p,prev)
                    energy_curv[dx+hks][dy+hks] = self.curv_energy(p,prev,next)
                    energy_grad[dx + hks][dy + hks] = self.Grad_energy(p)


            #Then, normalize the energies
            energy_cont = Snake.normalize(energy_cont)
            energy_curv = Snake.normalize(energy_curv)
            energy_grad = Snake.normalize(energy_grad)

            e_sum = self.alpha * energy_cont + self.beta * energy_curv + self.gamma * energy_grad
            emin = np.finfo(np.float64).max

            x,y = 0,0
            for dx in range( -hks, hks ):
                for dy in range( -hks, hks ):
                    if e_sum[ dx + hks ][ dy + hks ] < emin:
                        emin = e_sum[ dx + hks ][ dy + hks ]
                        x = curr[0] + dx
                        y = curr[1] + dy
            
            # Boundary check
            x = 1 if x < 1 else x
            x = self.width-2 if x >= self.width-1 else x
            y = 1 if y < 1 else y
            y = self.height-2 if y >= self.height-1 else y

            # Check for changes
            if curr[0] != x or curr[1] != y:
                changed = True

            new_snake[i] = np.array( [ x, y ] )

        self.points = new_snake

        # remove overlaping points and add missing points
        self.remove_overlaping_points()
        self.add_missing_points()
        return changed