import cv2
import numpy as np
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

    
    # START luminance
    """
    source = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    cv2.imshow("YUV", source)
    cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    
    y, u, v = cv2.split(source)
    cv2.equalizeHist(y, y)
    out = cv2.merge([y, u, v])
    
    cv2.imshow("YUV Equalized", out)
    cv2.waitKey(0)

    source = cv2.cvtColor(out, cv2.COLOR_YUV2BGR)
    cv2.imshow("Back to Source", source)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    dst = cv2.fastNlMeansDenoisingColored(source, None, 5, 5, 20, 15) 

    cv2.imshow(" Denoised Image", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  

    # Greyscale the image
    image = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    """
    # End LUMINANCE

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  

    return image
