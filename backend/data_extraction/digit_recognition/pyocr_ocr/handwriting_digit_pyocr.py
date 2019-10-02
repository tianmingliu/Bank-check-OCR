from PIL import Image
import sys

import pyocr
import pyocr.builders
import cv2

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)

print("Available tools:")
for n in tools:
    print(n.get_name())
tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))


langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))
lang = langs[0]
print("Will use lang '%s'" % (lang))

f_name = "C:\\Users\\development\\2019FallTeam03\\" \
        "backend\\data_extraction\\digit_recognition\\" \
        "test_images\\cropped_field7.jpg"
image = cv2.imread(f_name)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

filename = "{}.png".format("temp")
cv2.imwrite(filename, gray)

txt = tool.image_to_string(
    Image.open(filename),
    lang=lang,
    builder=pyocr.builders.TextBuilder()
)

print(txt)
