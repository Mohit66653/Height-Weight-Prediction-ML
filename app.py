from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ---------------------------------------------------
# Load the trained model safely (absolute path)
# ---------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------
# Home route
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------------------------------------------
# Prediction route
# ---------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get weight from form (assumed in KG)
        weight_kg = float(request.form["weight"])

        # Convert KG → Pounds (because model trained on pounds)
        

        # Prepare input for model
        input_data = np.array([[weight_kg]])

        # Predict height (output is in inches)
        predicted_height_cm = model.predict(input_data)[0]


        return jsonify({
            "height": round(predicted_height_cm, 2)
        })

    except Exception as e:
        return jsonify({
            "error": "Invalid input. Please enter a valid number."
        }), 400

# ---------------------------------------------------
# Run the Flask app
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

