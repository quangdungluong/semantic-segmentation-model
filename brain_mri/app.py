"""Flask app"""
import os
from flask import Flask, render_template, request, send_from_directory

from src.utils import create_model, predict
from PIL import Image

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', "tif"]
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = "results"
# Create model
model = create_model()
model.eval()

def allowed_file(filename):
    """Check allow file"""
    return filename.split('.')[-1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.mkdir(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['RESULT_FOLDER']):
    os.mkdir(app.config['RESULT_FOLDER'])

@app.route('/')
def home():
    """Home"""
    return render_template('index.html', label='Hiiii', imagesource='./static/img/b.png', returnJson={})


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Upload and process file"""
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            png_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename.split('.')[0] + '.png')
            result_path = os.path.join(app.config['RESULT_FOLDER'], file.filename.split('.')[0] + '.png')

            original_image = Image.open(file)
            original_image.save(png_path)

            prediction = predict(model, file)
            prediction = Image.fromarray(prediction)
            prediction.save(result_path)

            print(png_path)
            print(result_path)
    return render_template('index.html', imagesource=png_path, prediction=result_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Save uploaded file"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
