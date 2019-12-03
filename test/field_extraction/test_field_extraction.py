import cv2

from src.main.backend.preprocess.preprocess_main import preprocessEntryPoint
from src.main.backend.field_extraction.field_extractor_main import extractFieldsEntryPoint, check_bounding_collision, merge_overlapping_bb, merge_close_bb, crop
from src.main.backend.data_extraction.data_extraction_main import extract_data_entry_point


"""
TODO(Dustin):
- Isolate text on images (might not do this - it could get tested for extract)
- Detect lines on image (might not do this one right now)
- Extract text from an image
    - For each field:
        - Without isolated text
        - With isolated text
- Automate extraction test some. Provide a struct with preprocessing info,
  field extraction info (line detection, text isolation)


"""

file_base_dir      = "test/test-files/field_extract/"
file_in_dir        = file_base_dir + "input/"
file_out_dir       = file_base_dir + "output/"
file_processed_dir = file_base_dir + "processed/"

# TODO(Dustin): Collect all files in a directory, so they do not have to be
# manually inserted into this file.
filenames = [
    "a_1.jpg",
    "a_2.jpg",
    "a_3.jpg",
    "b_1.jpg",
    "b_2.jpg",
    "b_3.jpg",
]

# TODO(Dustin): Move these to cv_utils file for better use
"""
BEGIN CV HELPERS
"""
def write_image(filename, img):
    cv2.imwrite(filename, img)

def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
END CV HELPERS
"""

# TODO(Dustin): Move these helpers to a global test file so other
# test can take advantage of
"""
BEGIN TEST HELPERS

The following test allow for clearer output for a user.
[var_name] represents a parameter sent to the function

- test_header: prints a line in the format: TEST: [header]
    - Used for an overall test suite

- test_scenario: prints two lines in the format:

\tScenario [scenario]
\tExpected [str(expected)]

    - Used for a specific test. Lets a user know the scenario being test
      and what the expeted output should be

- test_actual: prints a line in the following format:

\t\tActual: [str(results)]

    - Used for displaying the actual results

- test_result: prints a line in the following format if [left] == [right]

\tTest [Pass/Fail]

    - Used for determining if a test passed or not
    - Pass is printed if the test passes, Fail is printed otherwise

"""
def test_header(header):
    print("TEST: " + header)

def test_scenario(scenario, expected):
    print("\tScenario: " + scenario)
    print("\t\tExpected: " + str(expected))

def test_actual(result):
    print("\t\tActual: " + str(result))

def test_result(left, right):
    if left == right:
        print("\tTest Pass")
    else:
        print("\tTest Fail")

    print()

"""
END TEST HELPERS
"""


"""
Preps the test files to be in the preprocessed state

Preprocess files are written to test/test-files/field_extract/processed
"""
def setup():
    for file in filenames:
        fullpath = file_in_dir + file
        img = cv2.imread(fullpath)
        img, _ = preprocessEntryPoint(img)

        write_image(file_processed_dir + file, img)


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
        img, _ = preprocessEntryPoint(img)

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

    img, old_image = preprocessEntryPoint(img)
    # write_image(file_out_dir + "preprocess_" + file, img)

    img, fields = extractFieldsEntryPoint(old_image, img)
    # write_image(file_out_dir + "field_extract_" + file, img)

    for field, field_img in fields:
        print("TYPE:")
        extract_data_entry_point(field_img, field)



def test_preprocess_extract():
    for file in filenames:
        print("Testing: " + file_in_dir + file)
        img = cv2.imread(file_in_dir + file)

        img, old_image = preprocessEntryPoint(img)
        write_image(file_out_dir + "preprocess_" + file, img)

        img, fields = extractFieldsEntryPoint(old_image, img)
        write_image(file_out_dir + "field_extract_" + file, img)

        # for (field, image) in fields:
        #     cv2.imshow("Field", image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        #     print("Field type: " + str(field.field_type))

def test_hardcoded():
    file = "resources/images/check_example.jpg"
    img = cv2.imread(file)

    img, old_image = preprocessEntryPoint(img)
    img, fields = extractFieldsEntryPoint(old_image, old_image)

    for field, field_img in fields:
        print("TYPE:")
        extract_data_entry_point(field_img, field)

"""
Tests the function: bounding_box_collision

TODO(Dustin): Scenarios
"""
def test_bounding_box_collision():
    test_header("Bounding Box Collision")

    # Overlap y, no overlap x: no collision
    exp = False
    test_scenario("Overlap X Axis, No Overlap Y Axis", exp)
    box_a = (0, 0, 100, 100)
    box_b = (50, 200, 150, 300)
    act = check_bounding_collision(box_a, box_b)
    test_actual(act)
    test_result(exp, act)

    # Overlap y, no overlap x: no collision
    test_scenario("Overlap Y Axis, No Overlap X Axis", False)
    box_a = (0, 0, 100, 100)
    box_b = (150, 50, 250, 150)
    act = check_bounding_collision(box_a, box_b)
    test_actual(act)
    test_result(exp, act)

    # Overlap x, overlap y: collision
    exp = True
    test_scenario("Overlap Y Axis, Overlap X Axis", exp)
    box_a = (0, 0, 100, 100)
    box_b = (50, 50, 150, 150)
    act = check_bounding_collision(box_a, box_b)
    test_actual(act)
    test_result(exp, act)

    # Box A inside Box b: collision
    exp = True
    test_scenario("Box A inside box b", exp)
    box_a = (50, 50, 100, 100)
    box_b = (0, 0, 150, 150)
    act = check_bounding_collision(box_a, box_b)
    test_actual(act)
    test_result(exp, act)

    # Box B inside Box A: collision
    exp = True
    test_scenario("Box B inside box a", exp)
    box_b = (50, 50, 100, 100)
    box_a = (0, 0, 150, 150)
    act = check_bounding_collision(box_a, box_b)
    test_actual(act)
    test_result(exp, act)

"""
Tests the function: merge_overlapping_bb

TODO(Dustin): scenarios
"""
def test_bounding_box_merge():
    test_header("Bounding Box Merge")

    exp_size  = 1
    exp_min_x  = 50
    exp_min_y  = 40
    exp_width  = 300
    exp_height = 250
    box_a = (50, 40, 150, 150)
    box_b = (100, 50, 200, 200) # 300 250
    list = [box_a, box_b]
    act = merge_overlapping_bb(None, list)

    # size
    test_scenario("Box A overlap left of Box B: Size", exp_size)
    test_actual(len(act))
    test_result(exp_size, len(act))

    (x, y, w, h) = act[0]

    # minx
    test_scenario("Box A overlap left of Box B: Min X", exp_min_x)
    test_actual(x)
    test_result(exp_min_x, x)

    # miny
    test_scenario("Box A overlap left of Box B: Min Y", exp_min_y)
    test_actual(y)
    test_result(exp_min_y, y)

    # max x
    test_scenario("Box A overlap left of Box B: Width", exp_width)
    test_actual(x+w)
    test_result(exp_width, x+w)

    # max y
    test_scenario("Box A overlap left of Box B: Height", exp_height)
    test_actual(y+h)
    test_result(exp_height, y+h)

"""
Tests whether or not bounding boxes can merge with a given x,y
threshold.

Scenario 1: Box A is left of Box B and within merge threshold for x and y.
Scenario 2: Box A is left of Box B and is outside of merge threshold for x and y.
Scenario 3: Box A is left of Box B and is within merge threshold for x and outside for y
Scenario 4: Box A is left of Box B and is within merge threshold for y and outside for x
"""
def test_bounding_box_merge_close():
    test_header("Bounding Box Merge")

    x_threshold = 10
    y_threshold = 10

    exp_size = 1

    # x, y, w, h
    box_a = (0, 0, 100, 100)
    box_b = (105, 105, 50, 50)

    act = merge_close_bb(None, [box_a, box_b], x_threshold, y_threshold)
    test_scenario("Box A to left of Box B; Should Merge; Size Test", exp_size)
    test_actual(exp_size)
    test_result(exp_size, len(act))

    box_a = (0, 0, 100, 100)
    box_b = (111, 111, 50, 50)

    exp_size = 2
    act = merge_close_bb(None, [box_a, box_b], x_threshold, y_threshold)
    test_scenario("Box A to left of Box B; Should Not Merge; Size Test", exp_size)
    test_actual(exp_size)
    test_result(exp_size, len(act))

    box_a = (0, 0, 100, 100)
    box_b = (105, 80, 50, 50)

    exp_size = 1
    act = merge_close_bb(None, [box_a, box_b], x_threshold, y_threshold)
    test_scenario("Box A to left of Box B; X within threshold; Should Merge; Size Test", exp_size)
    test_actual(exp_size)
    test_result(exp_size, len(act))

    box_a = (0, 0, 100, 100)
    box_b = (80, 105, 50, 50)

    exp_size = 1
    act = merge_close_bb(None, [box_a, box_b], x_threshold, y_threshold)
    test_scenario("Box A to left of Box B; Y within threshold; Should Merge; Size Test", exp_size)
    test_actual(exp_size)
    test_result(exp_size, len(act))

"""
Test different ways to split an image.

Upper Testing Strategy
- Crop upper third
- Crop upper fourth
- Crop upper fifth
- Crop upper third then crop right half
- Crop upper fourth then crop right half
- Crop upper fifth then crop right half
- Crop upper third then crop right half then remove upper half
- Crop upper fourth then crop right half then remove upper third
- Crop upper fifth then crop right half then remove upper fourth

Middle Testing Strategy:
- cry

Lower Testing Strategy:
- make another glass of coffee

Upper Results: Crop top third. Crop right half. Crop lower half.
   min_x = width/2
   max_x = width
   min_y = height/2
   max_y = height

Middle Results: Middle results are reported in percentages. Row denotes a new cropped image.
   Middle Region min_y: 30%
                 max_y: 70%          <- percent of entire check
   Pay Row:      min_y: 0%
                 max_y: 40%          <- percent of middle region
   Pay Field:    min_x: 12%
                 max_x: 73%          <- percent of pay region
   Amount Field: min_x: 77.5%
                 max_x: 95.0%        <- percent of pay region
   Written Row:  min_y: 40%
                 max_y: 100%         <- percent of middle region
   Written Field min_x: 5%
                 max_x: 75%          <- percent of written region

Lower Results:
    Lower Region   min_y: 70%
                   max_y: 90%        <- percent of the entire check
    Top Region     min_y: 5%
                   max_y: 55%        <- percent of the lower region
    Memo Region    min_y: 8%
                   max_x: 45%        <- percent of the top region
    Sig Region     min_x: 50%
                   max_x: 92%        <- percont of the top region
    Account Region min_x: 55%
                   max_y: 100%       <- percent of the lower region


"""
import os
def test_split_image():
    out_dir = file_out_dir + "crop_image/"

    test_header("Crop the top part of the image with varying sizes. Output name is test/test-files/field_extrac/output/crop_image/img_name_upper_crop.jpg")

    upper_out_dir = out_dir + "upper/"
    upper_split = [3, 4, 5]
    upper_split_name = [
        "top_third_",
        "top_fourth_",
        "top_fifth_",
        "top_third_half_",
        "top_fourth_half_",
        "top_fifth_half_",
        "top_third_half_half_",
        "top_fourth_half_third_",
        "top_fifth_half_fourth_"
    ]
    for file in filenames:
        fullpath = file_processed_dir + file
        image = cv2.imread(fullpath)

        name_idx = 0
        for split in upper_split:
            height = image.shape[0]
            width  = image.shape[1]

            # upper crop
            min_x = 0
            min_y = 0
            max_x = width
            max_y = int(height / split)

            new_image, old_image = crop(image, min_x, min_y, max_x, max_y)
            write_image(upper_out_dir + upper_split_name[name_idx] + file, new_image)

            # crop upper portion
            height = new_image.shape[0]
            width  = new_image.shape[1]
            min_x = int(width / 2)
            max_x = width
            max_y = height

            new_image, old_image = crop(new_image, min_x, min_y, max_x, max_y)
            write_image(upper_out_dir + upper_split_name[name_idx + len(upper_split)] + file, new_image)

            # crop upper portion again to remove check number: Half Crop
            height = new_image.shape[0]
            width  = new_image.shape[1]
            min_x = 0
            max_x = width
            min_y = int(height / (split-1))
            max_y = height

            new_image, old_image = crop(new_image, min_x, min_y, max_x, max_y)
            write_image(upper_out_dir + upper_split_name[name_idx + 2*len(upper_split)] + file, new_image)

            name_idx += 1



    test_header("Crop the middle part of the image with varying sizes. Output name is test/test-files/field_extrac/output/crop_image/img_name_middle_crop.jpg")
    middle_out_dir = out_dir + "middle/"

    # current approach splits middle 25% - 58%
    # To find the middle region, need to find the min/max y value
    # start with a min .2 and max .8
    # first increment

    # Prep output directories
    cropped_dir = os.getcwd() + "/" + middle_out_dir + "cropped/"
    pay_row_dir = os.getcwd() + "/" + middle_out_dir + "pay_row/"
    pay_col_dir = os.getcwd() + "/" + middle_out_dir + "pay_col/"
    amo_col_dir = os.getcwd() + "/" + middle_out_dir + "amo_col/"
    written_dir = os.getcwd() + "/" + middle_out_dir + "witten/"

    if not os.path.isdir(cropped_dir):
        os.mkdir(cropped_dir)
    if not os.path.isdir(pay_row_dir):
        os.mkdir(pay_row_dir)
    if not os.path.isdir(pay_col_dir):
        os.mkdir(pay_col_dir)
    if not os.path.isdir(amo_col_dir):
        os.mkdir(amo_col_dir)
    if not os.path.isdir(written_dir):
        os.mkdir(written_dir)
    # Finished prepping output directories

    for file in filenames:
        file_split = file.split(".")
        file_no_extension = file_split[0] + "/"
        fullpath = file_processed_dir + file

        image = cv2.imread(fullpath)
        height = image.shape[0]
        width  = image.shape[1]

        # Prep output directories for each test file
        # check if a directory with name: middle_out_dir + file (without extension)
        # if does not exist create it
        file_final_out_dir = cropped_dir + file_no_extension
        file_pay_row_dir = pay_row_dir + file_no_extension
        file_pay_col_dir = pay_col_dir + file_no_extension
        file_amo_col_dir = amo_col_dir + file_no_extension
        file_written_dir = written_dir + file_no_extension
        if not os.path.isdir(file_final_out_dir):
            os.mkdir(file_final_out_dir)
        if not os.path.isdir(file_pay_row_dir):
            os.mkdir(file_pay_row_dir)
        if not os.path.isdir(file_pay_col_dir):
            os.mkdir(file_pay_col_dir)
        if not os.path.isdir(file_amo_col_dir):
            os.mkdir(file_amo_col_dir)
        if not os.path.isdir(file_written_dir):
            os.mkdir(file_written_dir)
        # Finished preppring directories

        # ---------------------------------------------
        # Determine middle region cropped bounds
        # ---------------------------------------------
        start_min_percent = 0.2
        start_max_percent = 0.8
        current_min_percent = start_min_percent
        current_max_percent = start_max_percent

        min_x = 0
        max_x = width
        while current_min_percent < start_max_percent:
            current_max_percent = start_max_percent
            min_y = int(height * current_min_percent)

            while current_max_percent > current_min_percent:
                max_y = int(height * current_max_percent)

                out_filename = file_final_out_dir + str(current_min_percent) + "_" + str(current_max_percent) + ".jpg"

                # crop + write image to file
                new_image, old_image = crop(image, min_x, min_y, max_x, max_y)
                write_image(out_filename, new_image)

                current_max_percent -= 0.05
            current_min_percent += 0.05

        # verdict for full size: min_y .3; max_y .7
        verdict_min = 0.3
        verdict_max = 0.7
        min_y = int(height * verdict_min)
        max_y = int(height * verdict_max)
        cropped_image, old_image = crop(image, min_x, min_y, max_x, max_y)

        # ---------------------------------------------
        # Find bounds for height pay to the order of and amount
        # ---------------------------------------------
        cropped_height = cropped_image.shape[0]
        cropped_width  = cropped_image.shape[1]

        start_min_percent = 0.05
        start_max_percent = 0.95
        current_min_percent = start_min_percent
        current_max_percent = start_max_percent

        min_x = 0
        max_x = cropped_width
        min_y = 0

        # can assume the first row will take up at least 10% of the image
        while current_max_percent > current_min_percent + 0.1:
            max_y = int(cropped_height * current_max_percent)

            out_filename = file_pay_row_dir + str(current_max_percent) + ".jpg"

            # crop + write image to file
            new_image, old_image = crop(cropped_image, min_x, min_y, max_x, max_y)
            write_image(out_filename, new_image)

            current_max_percent -= 0.05

        # verdit for pay_row: min_y .35, max_y .40
        verdict_pay_row_min = 0.35
        verdict_pay_row_max = 0.40
        max_y = int(cropped_height * verdict_pay_row_max)
        pay_row_image, old_image = crop(cropped_image, min_x, min_y, max_x, max_y)


        # ---------------------------------------------
        # Find bounds for width pay to the order of
        # ---------------------------------------------
        # goal is to remove "pay to the order of text" and to remove the amount portion
        pay_row_height = pay_row_image.shape[0]
        pay_row_width  = pay_row_image.shape[1]

        start_min_percent = 0.05
        start_max_percent = 0.95
        current_min_percent = start_min_percent
        current_max_percent = start_max_percent

        min_y = 0
        max_y = pay_row_height
        while current_min_percent < start_max_percent:
            current_max_percent = start_max_percent
            min_x = int(pay_row_width * current_min_percent)

            while current_max_percent > current_min_percent:
                max_x = int(pay_row_width * current_max_percent)

                out_filename = file_pay_col_dir + str(current_min_percent) + "_" + str(current_max_percent) + ".jpg"

                # crop + write image to file
                new_image, old_image = crop(pay_row_image, min_x, min_y, max_x, max_y)
                write_image(out_filename, new_image)

                current_max_percent -= 0.05
            current_min_percent += 0.05

        # Verdict:
        # min_x: generally lies between 10% - 15%
        #     closer to 10% would be better, so let's use 12%
        # max_x: both 70% and 75% worked. Let's do 73%
        # This next image is a little different than the previous ones
        # Do not need to save the image, but it is saved for posterity
        verdict_min = 0.12
        verdict_max = 0.73
        min_x = int(pay_row_width * verdict_min)
        max_x = int(pay_row_width * verdict_max)
        pay_col_image, old_image = crop(cropped_image, min_x, min_y, max_x, max_y)

        # ---------------------------------------------
        # Find bounds for width amount
        # ---------------------------------------------
        # goal is to remove pay to the order of and to remove the outer border. Maybe even remove the "$"
        # Can use the max_x from pay to remove that section
        # Use the previous test results to determine the bounds rather than writing duplicate test co

        # Verdit:
        # min_x: generally list between 75% - 80%, but occasionally 80% cuts off text
        #     close to 75% will work, so let's do 77.5%
        # max_x: bounds always works with 95% so let's do that
        verdict_min = 0.775
        verdict_max = 0.950
        min_x = int(pay_row_width * verdict_min)
        max_x = int(pay_row_width * verdict_max)
        pay_col_image, old_image = crop(cropped_image, min_x, min_y, max_x, max_y)

        out_filename = file_amo_col_dir + str(verdict_min) + "_" + str(verdict_max) + ".jpg"

        # crop + write image to file
        new_image, old_image = crop(pay_row_image, min_x, min_y, max_x, max_y)
        write_image(out_filename, new_image)


        # ---------------------------------------------
        # Find bounds for written amount
        # ---------------------------------------------
        # Has to be done in two parts.
        #
        # Part 1:
        # For written amount, take the max_y of pay_height and cropped_height
        #
        # Part 2:
        # Trim the left and right sides of the image
        min_x = 0
        max_x = cropped_width
        min_y = int(cropped_height * verdict_pay_row_max)
        max_y = cropped_height

        out_filename = file_written_dir + file

        # crop the cropped_image to get just the written amount region
        written_image, old_image = crop(cropped_image, min_x, min_y, max_x, cropped_height)

        written_height = written_image.shape[0]
        written_width  = written_image.shape[1]

        start_min_percent = 0.05
        start_max_percent = 0.95
        current_min_percent = start_min_percent
        current_max_percent = start_max_percent

        min_y = 0
        max_y = pay_row_height
        while current_min_percent < start_max_percent:
            current_max_percent = start_max_percent
            min_x = int(written_width * current_min_percent)

            while current_max_percent > current_min_percent:
                max_x = int(written_width * current_max_percent)

                out_filename = file_written_dir + str(current_min_percent) + "_" + str(current_max_percent) + ".jpg"

                # crop + write image to file
                new_image, old_image = crop(written_image, min_x, min_y, max_x, max_y)
                write_image(out_filename, new_image)

                current_max_percent -= 0.05
            current_min_percent += 0.05

        # Verdict
        # min: 5% consistently was within bounds
        # max: could do between 75% - 80% but 80% often included "DOL"
        #     75% is a safe bet and does not cut off much of the written amount line
        verdict_min = 0.05
        verdict_max = 0.75


    test_header("Crop the bottom part of the image with varying sizes. Output name is test/test-files/field_extrac/output/crop_image/img_name_bottom_crop.jpg")
    lower_out_dir = out_dir + "lower/"

    # Prep output directories
    cropped_dir = os.getcwd() + "/" + lower_out_dir + "cropped/"
    top_dir     = os.getcwd() + "/" + lower_out_dir + "top/"
    memo_dir    = os.getcwd() + "/" + lower_out_dir + "memo/"
    sig_dir     = os.getcwd() + "/" + lower_out_dir + "signature/"
    account_dir = os.getcwd() + "/" + lower_out_dir + "account/"

    if not os.path.isdir(cropped_dir):
        os.mkdir(cropped_dir)
    if not os.path.isdir(top_dir):
        os.mkdir(top_dir)
    if not os.path.isdir(memo_dir):
        os.mkdir(memo_dir)
    if not os.path.isdir(sig_dir):
        os.mkdir(sig_dir)
    if not os.path.isdir(account_dir):
        os.mkdir(account_dir)
    # Finished prepping output directories

    for file in filenames:
        file_split = file.split(".")
        file_no_extension = file_split[0] + "/"
        fullpath = file_processed_dir + file
        image = cv2.imread(fullpath)

        height = image.shape[0]
        width  = image.shape[1]

        # Prep output directories for each test file
        # check if a directory with name: middle_out_dir + file (without extension)
        # if does not exist create it
        file_cropped = cropped_dir + file_no_extension
        file_top     = top_dir + file_no_extension
        file_memo    = memo_dir + file_no_extension
        file_sig     = sig_dir + file_no_extension
        file_account = account_dir + file_no_extension

        if not os.path.isdir(file_cropped):
            os.mkdir(file_cropped)
        if not os.path.isdir(file_top):
            os.mkdir(file_top)
        if not os.path.isdir(file_memo):
            os.mkdir(file_memo)
        if not os.path.isdir(file_sig):
            os.mkdir(file_sig)
        if not os.path.isdir(file_account):
            os.mkdir(file_account)
        # Finished preppring directories


        # Determine the cropped region of the image
        start_min_percent = 0.45
        start_max_percent = 1.00
        current_min_percent = start_min_percent
        current_max_percent = start_max_percent

        min_x = 0
        max_x = width
        min_y = 0
        max_y = height
        while current_min_percent < start_max_percent:
            current_max_percent = start_max_percent
            min_y = int(height * current_min_percent)

            while current_max_percent > current_min_percent:
                max_y = int(height * current_max_percent)

                out_filename = file_cropped + str(current_min_percent) + "_" + str(current_max_percent) + ".jpg"

                # crop + write image to file
                new_image, old_image = crop(image, min_x, min_y, max_x, max_y)
                write_image(out_filename, new_image)

                current_max_percent -= 0.05
            current_min_percent += 0.05

        # Verdict
        # min_y 70%
        # max_y 95%
        verdict_min = 0.7
        verdict_max = 0.95
        min_y = int(verdict_min * height)
        max_y = int(verdict_max * height)
        cropped_image, old_image = crop(image, min_x, min_y, max_x, max_y)

        cropped_height = cropped_image.shape[0]
        cropped_width  = cropped_image.shape[1]

        # Determine the region for the memo and signature
        start_min_percent = 0.05
        start_max_percent = 0.90
        current_min_percent = start_min_percent
        current_max_percent = start_max_percent

        min_x = 0
        max_x = cropped_width
        min_y = 0
        max_y = cropped_height
        while current_min_percent < start_max_percent:
            current_max_percent = start_max_percent
            min_y = int(cropped_height * current_min_percent)

            while current_max_percent > current_min_percent:
                max_y = int(cropped_height * current_max_percent)

                out_filename = file_top + str(current_min_percent) + "_" + str(current_max_percent) + ".jpg"

                # crop + write image to file
                new_image, old_image = crop(cropped_image, min_x, min_y, max_x, max_y)
                write_image(out_filename, new_image)

                current_max_percent -= 0.05
            current_min_percent += 0.05

        # Verdict for Top region:
        # min_y 5%
        # max_y 55%
        verdict_top_min = 0.05
        verdict_top_max = 0.55
        min_y = int(verdict_top_min * cropped_height)
        max_y = int(verdict_top_max * cropped_height)
        top_image, old_image = crop(cropped_image, min_x, min_y, max_x, max_y)

        top_height = top_image.shape[0]
        top_width  = top_image.shape[1]

        # Now find the bounds for the memo
        start_min_percent = 0.05
        start_max_percent = 0.90
        current_min_percent = start_min_percent
        current_max_percent = start_max_percent

        min_x = 0
        max_x = top_width
        min_y = 0
        max_y = top_height
        while current_min_percent < start_max_percent:
            current_max_percent = start_max_percent
            min_x = int(top_width * current_min_percent)

            while current_max_percent > current_min_percent:
                max_x = int(top_width * current_max_percent)

                out_filename = file_memo + str(current_min_percent) + "_" + str(current_max_percent) + ".jpg"

                # crop + write image to file
                new_image, old_image = crop(top_image, min_x, min_y, max_x, max_y)
                write_image(out_filename, new_image)

                current_max_percent -= 0.05
            current_min_percent += 0.05

        # Verdict for Memo region
        # min_x 8%
        # max_x 45%

        # Verdict for Signature region
        # min_x 50%
        # max_x 92%

        # Verdict for Account Region
        # min_y 55%
        # max_y 100%




def test_isolate_test_on_image():
    test_header("Given an preprocessed image, isolate the text on it. Output folder is")


def test_detect_lines():
    test_header("Given a preprocessed image, detect lines on the image. Output folder is")

    test_header("Given a preprocessed image with the text isolated, detect lines on the image. Output folder is")


def test_field_extraction():
    out_dir = file_out_dir + "field_extraction/"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    test_header("Test field extraction")

    for file in filenames:
        file_split = file.split(".")
        file_no_extension = file_split[0] + "/"
        fullpath = file_processed_dir + file

        file_dir = out_dir + file_no_extension
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)

        image = cv2.imread(fullpath)
        height = image.shape[0]
        width  = image.shape[1]

        old_img, fields = extractFieldsEntryPoint(None, image)


        for (field, img) in fields:
            output_filename = file_dir + str(field.field_type) + ".jpg"
            write_image(output_filename, img)

            extract_data_entry_point(img, field)

        

def main():

    # Setup to gather preprocessed images
    setup()

    print("\nUNIT TEST SUITE: Field Extraction Tests\n")

    # Bounding Box Tests
    #----------------------------------------
    test_bounding_box_collision()
    test_bounding_box_merge()
    test_bounding_box_merge_close()

    print("\nBLACK BOX TEST SUITE: Field Extraction Tests\n")

    # Image Tests
    #----------------------------------------
    test_split_image()
    test_field_extraction()

    # Bounding Box Tests
    #----------------------------------------



    # test_preprocess_extract()
    # test_extraction()
    # test_mser()
    # test_hardcoded()

if __name__ == "__main__":
    main()
