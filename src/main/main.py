import os
import base64
from flask import Flask, flash, request, redirect, jsonify, render_template
from werkzeug.utils import secure_filename
import cv2
from backend.data_extraction.field.data.field_data import FieldData
from backend.data_extraction.data_extraction_main import validate_extracted_field
from backend.controller import controller_entry_point

UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads/')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

result = {}

"""
Defines the file types that are allowed

@param filename: the name of the file

@return True if file has one of the allowed extensions, False otherwise
"""


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


"""
Converts the image to data

@param path: the file path of the image

@return the data of the image
"""


def image_to_data(path):
    ext = path.split('.')[-1]
    prefix = f'data:image/{ext};base64,'
    encoded = base64.b64encode(open(path, "rb").read()).decode('utf-8')
    return prefix + encoded


"""
Handles file upload events

@return the resulting processed image and extracted data to the frontend in JSON
"""


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(path)
            file.save(path)

            # Send the image to the backend
            global result
            img, result = controller_entry_point(path)
            cv2.imwrite(path, img)

            # Show the processed image
            return jsonify(
                image=image_to_data(path),
                results=result
            )


"""
Handle the Re-Validate button events

@return the resulting re-validated data to the frontend in JSON
"""


@app.route('/api/revalidate', methods=['POST'])
def revalidate_data():
    new_data = request.get_json(force=True)
    for key in result:
        if result[key]["field_type"] == 1:
            result[key]["extracted_data"] = new_data['date']['data']
        elif result[key]["field_type"] == 2:
            result[key]["extracted_data"] = new_data['paytoorder']['data']
        elif result[key]["field_type"] == 3:
            result[key]["extracted_data"] = new_data['amount']['data']
        elif result[key]["field_type"] == 4:
            result[key]["extracted_data"] = new_data['writtenamount']['data']
        elif result[key]["field_type"] == 5:
            result[key]["extracted_data"] = new_data['signature']['data']
        elif result[key]["field_type"] == 6:
            result[key]["extracted_data"] = new_data['memo']['data']
        elif result[key]["field_type"] == 7:
            result[key]["extracted_data"] = new_data['routing']['data']
        elif result[key]["field_type"] == 8:
            result[key]["extracted_data"] = new_data['account']['data']

        result[key]["validation"] = validate_extracted_field(FieldData.from_dict(result[key]))

    print(result)
    return jsonify(result)


"""
Handles the event that occurs when users navigates to the home

@return the home.html template
"""


@app.route('/', methods=['GET'])
def render_home():
    return render_template('home.html')


# Runs the program
if __name__ == "__main__":
    app.run(debug=True)
