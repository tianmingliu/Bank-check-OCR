import cv2

"""
Accepts an image as inout and performs a series of 
preprocessing analysis on the image.

@param image: image to process

@return the processed image
"""
def preprocessEntryPoint(image):
    # TODO(Dustin): Preserve aspect ratio as much as possible
    # Rescale the image if need be
    height = image.shape[0] # keep original height
    width  = image.shape[1] # keep original width

    gbl_width = 1080
    gbl_height = 720

    if gbl_width < width:
        tmp_width = gbl_width
    else:
        tmp_width = width

    if gbl_height < height:
        tmp_height = gbl_height
    else:
        tmp_height = height
    
    dim = (tmp_width, tmp_height)

    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # cv2.imshow("Rescaled image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Greyscale the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("Canny edge detection", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return image
