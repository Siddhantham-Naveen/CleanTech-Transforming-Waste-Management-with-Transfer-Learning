from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('train_model/mobilenetv2_waste.h5')
labels = ['Biodegradable', 'Recyclable', 'Trash']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    filepath = os.path.join('static/uploads', file.filename)
    file.save(filepath)

    img = load_img(filepath, target_size=(96, 96))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]

    return render_template(
        'portfolio-details.html',
        prediction=predicted_class,
        uploaded_image=file.filename
    )

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
