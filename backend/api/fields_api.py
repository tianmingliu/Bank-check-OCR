from flask import Flask, jsonify, request, send_file

app = Flask(__name__)


@app.route('/field-data', methods=['GET'])
def get_fields():
    try:
        return send_file('../../out.json', attachment_filename='out.json')
    except Exception as e:
        return str(e)


@app.route('/image', methods=['POST'])
def send_image():
    pass




