import cv2

"""
Accepts an image as inout and performs a series of 
preprocessing analysis on the image.

@param image: image to process

@return the processed image
"""
def preprocessEntryPoint(image):
    # Greyscale the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Canny edge detection", image)
    cv2.waitKey(0)

    # Destroying present windows on screen 
    cv2.destroyAllWindows()  
    
    return image