from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('cnn.h5')

# Route for the home page
@app.route('/')
def home():
    return "Welcome to the Model Prediction API! Use the /predict endpoint to make predictions."

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image is part of the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']

        # If no file is selected
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Open the image file
        img = Image.open(file.stream)

        # Preprocess the image for the model (resizing and normalizing)
        img = img.resize((64, 64))  # Resize according to the model's expected input size
        img = np.array(img)  # Convert to a NumPy array
        
        # Ensure the image has 3 channels (RGB)
        if img.ndim == 2:  # If the image is grayscale
            img = np.expand_dims(img, axis=-1)  # Add channel dimension (64, 64, 1)
            img = np.repeat(img, 3, axis=-1)  # Convert grayscale to RGB (64, 64, 3)

        # Normalize the image
        img = img.astype('float32') / 255.0

        # Add batch dimension (model expects batch dimension)
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 64, 64, 3)

        # Make prediction
        prediction = model.predict(img)

        # Return prediction as JSON response
        return jsonify({"prediction percent for pneumonia": (prediction[0]*100).tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
