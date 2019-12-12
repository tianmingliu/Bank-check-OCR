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
width: width of the image
height: height of the image
x: x position
y: y position
direction: 0 North
           1 East
           2 South
           3 West
@return true if the (x,y) is within bounds. False otherwise

Note(Dustin): y - 1 is North
              x - 1 is West
"""


def check_neighbor(width: int, height: int, x: int, y: int, direction: int):
    # if north
    if direction == 0:
        if y - 1 < 0:
            return False
        else:
            return True
    # if east
    elif direction == 1:
        if x + 1 >= width:
            return False
        else:
            return True
    # if south
    elif direction == 2:
        if y + 1 >= height:
            return False
        else:
            return True
    # if west
    elif direction == 3:
        if x - 1 < 0:
            return False
        else:
            return True
    # wrong direction provided
    else:
        return False


"""
An experiment to determine if we can brute force finding the lines

@param img: img to find lines in
"""


def find_lines(img):
    height = img.shape[0]
    width  = img.shape[1]

    blank_image = np.zeros((height,width,3), np.uint8)

    black = [0, 0, 0]
    white = [255, 255, 255] 

    min_length = 5 # minumum length in pixels of a line

    # for each pixel
    for j in range(height):

        start_x = 0        # starting pixel of the line
        current_length = 0 # current length of the line
        for i in range(width):
            channels_xy = img[j,i]

            # check only left/right for now            
            if check_neighbor(width, height, i + 1, j, 1):
                channels_xy_right = img[j, i + 1]
                if all(channels_xy == black) and all(channels_xy_right == black):
                    current_length += 1
                else:
                    if current_length >= min_length:
                        # blank_image[j,i] = white
                        cv2.line(blank_image, (start_x, j), (start_x + current_length, j + 5), white, 1)
                    current_length = 0
                    start_x = i + 1


"""
Accepts an image as inout and performs a series of 
preprocessing analysis on the image.

@param image: image to process

@return the processed image
"""


def preprocessEntryPoint(image):
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

    # new_image = remove_shadow(res)
    new_image = res

    # find_lines(new_image)

    # A test for detecting lines
    # detect_lines(new_image) 

    return new_image, smol_image
