import cv2
import numpy as np

"""
Attempts to downsize an image to a provided width, height. If 
the current size of the image is smaller than the desired dimensions,
the the size of the image is not changed.

@param image: image to downsize
@param new_width: desired width of the image
@param new_height: desired height of the image

@return the downsized image
@return the original width of the image
@return the original height of the image
"""
def downsize_image(image, new_width, new_height):
    height = image.shape[0] # keep original height
    width  = image.shape[1] # keep original width
    
    if new_width < width:
        tmp_width = new_width
    else:
        tmp_width = width

    if new_height < height:
        tmp_height = new_height
    else:
        tmp_height = height
    
    dim = (tmp_width, tmp_height)

    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return image, width, height

"""
Removes sharp lighting and shadow from an image.

@param image to remove lighting from

@return result: image with the values blurred to reduce lighting
                and shadowing effects.
@return norm_result: normalized image of the other return value.
"""
def remove_shadow(image):
    # Remove the shadow
    rgb_planes = cv2.split(image)

    planes = []
    norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 33)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        planes.append(diff_img)
        norm_planes.append(norm_img)

    result = cv2.merge(planes)
    norm_result = cv2.merge(norm_planes)

    # cv2.imshow("Merged Image", result)
    # cv2.imshow("Normalized Merged Image", norm_result)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return result, norm_result

"""
Accepts an image as inout and performs a series of 
preprocessing analysis on the image.

@param image: image to process

@return the processed image
"""
def preprocessEntryPoint(image):

    # TODO(Dustin): Preserve aspect ratio as much as possible
    # Rescale the image if need be
    smol_image, _, _ = downsize_image(image, 1080, 720)

    # Remove any background noise like small dots
    new_image = cv2.cvtColor(smol_image, cv2.COLOR_BGR2GRAY)

    _, blackAndWhite = cv2.threshold(new_image, 127, 255, cv2.THRESH_BINARY_INV) # this line might not be necessary

    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 5:   #filter small dotted regions
            img2[labels == i + 1] = 255
    
    res = cv2.bitwise_not(img2)

    # cv2.imshow("Dotless Image", res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  

    #new_image = remove_shadow(res)
    new_image = res

    return new_image, smol_image
