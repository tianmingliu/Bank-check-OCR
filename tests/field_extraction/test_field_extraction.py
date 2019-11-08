import cv2

import backend.preprocess.preprocess_main            as pr
import backend.field_extraction.field_extractor_main as fe
import backend.data_extraction.data_extraction_main  as de

file_in_dir  = "tests/test-files/field_extract/input/"
file_out_dir = "tests/test-files/field_extract/output/"


filenames = [
    # "a_1.jpg",  #
    # "a_2.jpg",    # gaps, few overlaps for date 
    "a_3.jpg",      # some overlap
    # "b_1.jpg",
    # "b_2.jpg",
    # "b_3.jpg",
]

def write_image(filename, img):
    cv2.imwrite(filename, img)

def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
TODO(Dustin): 
- Split image into sections
- Run mser on each section
    - Merge bounding boxes first by overlap, then by distance


-
"""

"""
Performs mser on a given image.

Returns the msers.
"""
def impl_mser(image):
    mser = cv2.MSER_create()
    regions = mser.detectRegions(image)
    return regions[0]

def test_mser():

    for file in filenames:
        img = cv2.imread(file_in_dir + file)
        img, _ = pr.preprocessEntryPoint(img)

        # vis = img.copy()
        # mser = cv2.MSER_create()

        # returns msers, boxes
        # bbox (left, right, top, lower)
        regions = impl_mser(img)

        # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
        # cv2.polylines(vis, hulls, 1, (0, 255, 0))

        # show the bounding box results
        img_cpy = img.copy()
        for box in regions:
            [x, y, w, h] = cv2.boundingRect(box)

            if w < 35 and h < 35:
                continue

            cv2.rectangle(img_cpy, (x, y), (x+w, y+h), (150, 0, 150), 2)
        show("Test", img_cpy)
    # show("img", vis)

def test_extraction():
    file = file_in_dir + filenames[0]

    print("Testing: " + file) 
    img = cv2.imread(file)

    img, old_image = pr.preprocessEntryPoint(img)
    # write_image(file_out_dir + "preprocess_" + file, img)

    img, fields = fe.extractFieldsEntryPoint(old_image, img)
    # write_image(file_out_dir + "field_extract_" + file, img)

    for field, field_img in fields:
        print("TYPE:")
        de.extract_data_entry_point(field_img, field)



def test_preprocess_extract():
    for file in filenames:
        print("Testing: " + file_in_dir + file)
        img = cv2.imread(file_in_dir + file)

        img, old_image = pr.preprocessEntryPoint(img)
        write_image(file_out_dir + "preprocess_" + file, img)

        img, fields = fe.extractFieldsEntryPoint(old_image, img)
        write_image(file_out_dir + "field_extract_" + file, img)

        for (field, image) in fields:
            cv2.imshow("Field", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print("Field type: " + str(field.field_type))

def test_hardcoded():
    file = "resources/images/check_example.jpg"
    img = cv2.imread(file)

    img, old_image = pr.preprocessEntryPoint(img)
    img, fields = fe.extractFieldsEntryPoint(old_image, old_image)

    for field, field_img in fields:
        print("TYPE:")
        de.extract_data_entry_point(field_img, field)


def main():
    test_preprocess_extract()
    # test_extraction()
    # test_mser()
    # test_hardcoded()
    


if __name__ == "__main__":
    main()