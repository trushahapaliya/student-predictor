from flask import Flask, render_template, request, jsonify
import os
from model.train import train_and_save, predict_grade

app = Flask(__name__)

MODEL_PATH = "model/model.pkl"

# Auto-train model on first run
if not os.path.exists(MODEL_PATH):
    print("🎓 Training ML model on 1000 student records...")
    train_and_save(MODEL_PATH)
    print("✅ Model trained and saved!")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict student grade from form inputs."""
    data = request.get_json()
    result = predict_grade(data, MODEL_PATH)
    return jsonify(result)

@app.route("/retrain", methods=["POST"])
def retrain():
    """Retrain model with fresh synthetic data."""
    metrics = train_and_save(MODEL_PATH)
    return jsonify({"status": "Model retrained successfully", "accuracy": metrics["accuracy"]})

@app.route("/model-info", methods=["GET"])
def model_info():
    """Return model metadata."""
    import pickle
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return jsonify({
        "algorithm": "Logistic Regression",
        "features": bundle["feature_names"],
        "accuracy": bundle["accuracy"],
        "train_samples": bundle["train_samples"],
        "test_samples": bundle["test_samples"]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
