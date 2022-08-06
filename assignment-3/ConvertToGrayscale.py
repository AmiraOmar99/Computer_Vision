            
def ConvertToGaryscale(ColoredImage):
    
    Red, Green, Blue = ColoredImage[:,:,0], ColoredImage[:,:,1], ColoredImage[:,:,2]
    GrayscaleImage = 0.299 * Red + 0.587 * Green + 0.114 * Blue

    return GrayscaleImage
