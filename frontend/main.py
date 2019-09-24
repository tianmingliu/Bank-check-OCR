from flask import Flask, render_template

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'pdf'}

app = Flask(__name__)

# The definition for the home page
@app.route("/")
def home():
    return render_template("home.html")

# Runs the program
if __name__ == "__main__":
    app.run(debug=True)

