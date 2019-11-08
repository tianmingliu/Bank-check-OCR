import os
import base64
from flask import Flask, flash, request, redirect, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import json
import cv2 

from backend.controller import controller_entry_point


UPLOAD_FOLDER = os.getcwd() + "/uploads/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'pdf'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_data(path):
    ext = path.split('.')[-1]
    prefix = f'data:image/{ext};base64,'
    encoded = base64.b64encode(open(path, "rb").read()).decode('utf-8')
    return prefix + encoded

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename)


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
            print(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Send the image to the backend
            img, res = controller_entry_point(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(path, img)

            # Show the processed image
            # return redirect(url_for('uploaded_file',
            #                         filename=filename))
            return jsonify(
                image=image_to_data(path),
                results=res
            )

@app.route('/', methods=['GET'])
def render_home():
    return render_template('home.html')

# Runs the program
if __name__ == "__main__":
    app.run(debug=True)

