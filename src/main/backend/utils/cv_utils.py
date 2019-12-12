import cv2

"""
Displays an image until a key is pressend.

@param img: image to disaply
@param title: title of the image

@return nothing.
"""


def show(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
Performs mser on a given image.

@param image: the image to perform mser on
Returns the msers.
"""


def impl_mser(image):
    mser = cv2.MSER_create()
    regions = mser.detectRegions(image)
    return regions[0]
