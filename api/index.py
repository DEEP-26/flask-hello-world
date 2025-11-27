import os
import time
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
import io

# Initialize Flask application
app = Flask(__name__)
# Mock ML Model Data
FRUITS = ["Apple", "Mango", "Banana", "Orange", "Tomato"]
CATEGORIES = ["Overripe", "Ripe", "Unripe"]

def mock_predict_ripeness(image):
    im = image.resize((224,224))
    test = np.array(im)/255.0
    test = np.expand_dims(test, axis=0)
    ans = model.predict(test)
    
    ripeness = CATEGORIES[ans.argmax()]
    confidence = ans.max()
    print(f"Mock Prediction: {ripeness}(Confidence: {confidence:.2f})")
    
    return {
        "ripeness": ripeness,
        "confidence": f"{confidence:.2f}"
    }
def b64_to_b(base64_string):
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_string)
    return image_bytes

def i_from_b(image_bytes):
   image_stream = io.BytesIO(image_bytes)
   image = Image.open(image_stream)
   return image

@app.route('/', methods=['GET'])
def index():

    """Renders the main HTML interface."""
    # This line assumes your HTML file is accessible as 'index.html'
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive base64 image data from the client and return a prediction.
    """
    try:
        # Get the JSON payload containing the 'image' key
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data (base64) provided in the request body."}), 400

        image_base64 = data['image']

        img = i_from_b(b64_to_b(image_base64))

        
        # Call the mock prediction function with the base64 data
        prediction_result = mock_predict_ripeness(img)
        
        return jsonify(prediction_result)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500

if __name__ == '__main__':
    # Running in debug mode allows for automatic reloads
    model = tf.keras.models.load_model('m2.keras')
    app.run(debug=True)
