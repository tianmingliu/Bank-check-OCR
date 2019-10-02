from flask import Flask, render_template, send_file

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'pdf'}

app = Flask(__name__)

# The definition for the home page
@app.route("/")
def home():
    return render_template("home.html")

@app.route('/field-data', methods=['GET'])
def get_fields():
    try:
        return send_file('../../out.json', attachment_filename='out.json')
    except Exception as e:
        return str(e)

# Runs the program
if __name__ == "__main__":
    app.run(debug=True)

