from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

app = Flask(__name__)
model = tf.keras.models.load_model("waste_classification.h5")  # Load the trained model

# Define class labels (update these based on your model's classes)
CLASS_LABELS = ["Batteries", "Clothes", "E-waste", "Glass", "Light Blubs", "Metal", "Organic", "Paper", "Plastic"]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def predict_waste(img_path):
    img = Image.open(img_path).resize((224, 224))  # Resize to model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return CLASS_LABELS[class_index]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)
            result = predict_waste(img_path)
            return render_template("index.html", img_path=img_path, result=result)
    return render_template("index.html", img_path=None, result=None)


if __name__ == "__main__":
    app.run(debug=True)
