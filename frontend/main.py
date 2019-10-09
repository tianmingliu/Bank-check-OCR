import os
# from flask import Flask, render_template, send_file

from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename


from flask import Blueprint
from flask_restful import Api
from backend.hello import Hello

api_bp = Blueprint('api', __name__)
api = Api(api_bp)

# Route
api.add_resource(Hello, '/Hello')

UPLOAD_FOLDER = os.getcwd() + "/uploads/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # The definition for the home page
# @app.route("/", methods=['GET', 'POST'])
# def home():
#     return render_template("home.html")

# @app.route('/field-data', methods=['GET'])
# def get_fields():
#     try:
#         return send_file("..\\backend\\out.json", attachment_filename='out.json')
#     except Exception as e:
#         return str(e)

# @app.route('/image', methods=['POST'])
# def send_image():
#     pass

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/', methods=['GET', 'POST'])
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
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def create_app(config_filename):
    app = Flask(__name__)
    app.config.from_object(config_filename)
    
    app.register_blueprint(api_bp, url_prefix='/api')

    return app

# Runs the program
if __name__ == "__main__":
    # app = create_app("config")
    app.run(debug=True)

