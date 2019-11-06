import cv2

import backend.preprocess.preprocess_main            as pr
import backend.field_extraction.field_extractor_main as fe

file_in_dir  = "tests/test-files/field_extract/input/"
file_out_dir = "tests/test-files/field_extract/output/"


filenames = [
    "a_1.jpg",
    # "a_2.jpg",
    # "a_3.jpg",
    # "b_1.jpg",
    # "b_2.jpg",
    # "b_3.jpg",
]

def write_image(filename, img):
    cv2.imwrite(filename, img)


def test_extraction():
    file = file_in_dir + filenames[0]

    print("Testing: " + file_in_dir + file)
    img = cv2.imread(file_in_dir + file)

    cv2.imshow("fd", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img, old_image = pr.preprocessEntryPoint(img)
    write_image(file_out_dir + "preprocess_" + file, img)

    img, fields = fe.extractFieldsEntryPoint(old_image, img)
    write_image(file_out_dir + "field_extract_" + file, img)



def test_preprocess_extract():
    for file in filenames:
        print("Testing: " + file_in_dir + file)
        img = cv2.imread(file_in_dir + file)

        # cv2.imshow("fd", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        img, old_image = pr.preprocessEntryPoint(img)
        write_image(file_out_dir + "preprocess_" + file, img)

        img, fields = fe.extractFieldsEntryPoint(old_image, img)
        write_image(file_out_dir + "field_extract_" + file, img)

        for (field, image) in fields:
            cv2.imshow("Field", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print("Field type: " + str(field.field_type))

def main():
    test_preprocess_extract()
    


if __name__ == "__main__":
    main()