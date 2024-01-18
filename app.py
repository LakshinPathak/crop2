from flask import Flask, render_template, request
import numpy as np
import requests
from tensorflow.keras.models import load_model
import joblib
import cv2
from io import BytesIO

app = Flask(__name__)

# Google Drive file ID
file_id = '1sOiN-RH3aTbsTM6sGrvu2AkGLXOgJOpP'

# Download the model file from Google Drive
def download_file_from_google_drive(file_id):
    URL = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(URL)
    return BytesIO(response.content)

# Load the pre-trained model and label binarizer
model_file = download_file_from_google_drive(file_id)
model = load_model(model_file)
label_binarizer = joblib.load('label_transform.pkl')  # Replace 'label_transform.pkl' with the actual file name

# Convert image to array function
def convert_image_to_array(image_path):
    try:
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (224, 224))
            return np.expand_dims(image, axis=0)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error: {e}")
        return None

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the uploaded image
            file = request.files['file']
            
            # Save the uploaded image
            image_path = 'static/uploads/' + file.filename
            file.save(image_path)

            # Convert the image to array
            image_array = convert_image_to_array(image_path)

            # Make prediction
            prediction = model.predict(image_array)
            
            # Convert one-hot encoded prediction to class label
            predicted_label = label_binarizer.inverse_transform(prediction)[0]

            return render_template('result.html', prediction=predicted_label, image_path=image_path)

        except Exception as e:
            print(f"Error: {e}")
            return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)
