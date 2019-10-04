import os
from flask import Flask, render_template, send_file

UPLOAD_FOLDER = "uploads/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# The definition for the home page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route('/field-data', methods=['GET'])
def get_fields():
    try:
        return send_file("..\\..\\backend\\out.json", attachment_filename='out.json')
    except Exception as e:
        return str(e)

@app.route('/image', methods=['POST'])
def send_image():
    pass

# Runs the program
if __name__ == "__main__":
    app.run(debug=True)

