from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

app = Flask(__name__)

# Global Variables
data = None
model = None

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Dataset Upload Endpoint
@app.route('/upload', methods=['POST'])
def upload_file():
    global data
    file = request.files.get('file')

    if not file:
        return jsonify({"error": "No file provided."}), 400

    # Load the dataset into a pandas DataFrame
    data = pd.read_csv(file)
    if not {'Temperature', 'Run_Time', 'Downtime_Flag'}.issubset(data.columns):
        return jsonify({"error": "Dataset must contain 'Temperature', 'Run_Time', and 'Downtime_Flag' columns."}), 400

    return jsonify({"message": "Dataset uploaded successfully.", "columns": list(data.columns)}), 200

# Train Model Endpoint
@app.route('/train', methods=['POST'])
def train_model():
    global data, model

    if data is None:
        return jsonify({"error": "No dataset uploaded. Please upload a dataset first."}), 400

    # Prepare the data
    X = data[['Temperature', 'Run_Time']]
    y = data['Downtime_Flag']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    roc_auc = auc(*roc_curve(y_test, y_pred_proba[:, 1])[:2])

    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("static/roc_curve.png")
    plt.close()

    # Save the model
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    return jsonify({
        "message": "Model trained successfully.",
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        },
        "confusion_matrix_path": "/static/confusion_matrix.png",
        "roc_curve_path": "/static/roc_curve.png"
    }), 200

# Predict Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    global model

    if model is None:
        return jsonify({"error": "No model trained. Please train a model first."}), 400

    data = request.get_json()

    if not {'Temperature', 'Run_Time'}.issubset(data.keys()):
        return jsonify({"error": "Input must contain 'Temperature' and 'Run_Time'."}), 400

    # Prepare input for prediction
    input_data = [[data['Temperature'], data['Run_Time']]]
    prediction = model.predict(input_data)[0]
    confidence = max(model.predict_proba(input_data)[0])

    return jsonify({
        "Downtime": "Yes" if prediction == 1 else "No",
        "Confidence": confidence
    }), 200

# Serve Static Files (Visualizations)
@app.route('/static/<filename>', methods=['GET'])
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
