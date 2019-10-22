import cv2

def postprocessEntryPoint(image, dim, data):
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image