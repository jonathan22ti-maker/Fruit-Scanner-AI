from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Memastikan folder uploads sudah ada agar tidak error saat menyimpan file
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model
model = load_model('Fruits.h5')

# Class names
class_names = [
    'Avocado Green 1',
    'Pear 1',
    'apple_red_1'
]

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction)) * 100

    result = class_names[predicted_class]

    return render_template(
        'index.html',
        prediction=result,
        confidence=round(confidence, 2),
        image_file=file.filename
    )

# Run app
if __name__ == '__main__':
    app.run(debug=True)