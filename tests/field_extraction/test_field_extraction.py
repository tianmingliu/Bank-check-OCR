import cv2

import backend.preprocess.preprocess_main            as pr
import backend.field_extraction.field_extractor_main as fe

file_in_dir  = "../test-files/field_extract/input/"
file_out_dir = "../test-files/field_extract/output/"


filenames = [
    "a_1.jpg",
    "a_2.jpg",
    "a_3.jpg",
    "b_1.jpg",
    "b_2.jpg",
    "b_3.jpg",
]

def write_image(filename, img):
    cv2.imwrite(filename, img)

def main():

    for file in filenames:
        print("Testing: " + file_in_dir + file)
        img = cv2.imread(file_in_dir + file)

        img, old_image = pr.preprocessEntryPoint(img)
        write_image(file_out_dir + "preprocess_" + file, img)

        img, fields = fe.extractFieldsEntryPoint(old_image, img)
        write_image(file_out_dir + "field_extract_" + file, img)


if __name__ == "__main__":
    main()