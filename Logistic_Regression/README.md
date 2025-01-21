# Machine Downtime Prediction API

This project is a Flask-based API to predict machine downtime based on input parameters such as temperature and runtime. The API uses a logistic regression model trained on provided data and supports endpoints for uploading datasets, training the model, and making predictions.For this project i synthetically generated dataset of 3000rows with columns names mentioned in the pdf.

## Features
- **Upload Dataset**: Accepts a CSV file for training.
- **Train Model**: Trains a logistic regression model and returns evaluation metrics.
- **Predict Downtime**: Predicts machine downtime and provides confidence scores based on input parameters.
- **Visualizations**: Generates confusion matrix heatmaps and ROC curves for evaluation.

---

## Project Structure
```
project-directory/
|-- app.py              # Flask application code
|-- data/               # Folder for storing datasets
|-- models/             # Folder for storing trained model
|-- static/             # Folder for saving visualizations
|-- requirements.txt    # Python dependencies
```

---

## Prerequisites
1. Python installed on your system (version 3.7 or higher).
2. Required Python libraries (install via `pip`).

---

## Setup Instructions
### 1. Clone the Repository
git clone <repository-url>
cd project-directory


### 2. Install Dependencies
Install the required Python packages using the following command:
pip install -r requirements.txt


### 3. Run the Flask Application
Start the Flask server by running:
python app.py
The application will be available at `http://127.0.0.1:5000/`.

---

## API Endpoints

### 1. Upload Dataset
**Endpoint:** `/upload`  
**Method:** `POST`  
**Description:** Upload a CSV dataset for training the model.

#### Example Request (Postman):
- **Type:** Form-data
- **Key:** `file`
- **Value:** Select your dataset file (e.g., `sample_data.csv`).

#### cURL Command:
curl -X POST -F 'file=@data/sample_data.csv' http://127.0.0.1:5000/upload
for my project-
curl -X POST -F "file=@C:\Users\91808\OneDrive\Desktop\TechPranee\Logistic_Regression\dataset\sample_data.csv" http://127.0.0.1:5000/upload

#### Example Response:
json
{
    "message": "Dataset uploaded successfully.",
    "columns": ["Temperature", "Run_Time", "Downtime_Flag"]
}


### 2. Train the Model
**Endpoint:** `/train`  
**Method:** `POST`  
**Description:** Train the logistic regression model on the uploaded dataset.

#### Example Request (Postman):
- **Type:** POST
- **URL:** `http://127.0.0.1:5000/train`

#### cURL Command:
curl -X POST http://127.0.0.1:5000/train

#### Example Response:
json
{
    "message": "Model trained successfully.",
    "metrics": {
        "accuracy": 0.85,
        "precision": 0.88,
        "recall": 0.82,
        "f1_score": 0.85,
        "roc_auc": 0.91
    },
    "confusion_matrix_path": "/static/confusion_matrix.png",
    "roc_curve_path": "/static/roc_curve.png"
}


---

### 3. Predict Downtime
**Endpoint:** `/predict`  
**Method:** `POST`  
**Description:** Make a prediction for machine downtime based on input parameters.

#### Example Request (Postman):
- **Type:** POST
- **URL:** `http://127.0.0.1:5000/predict`
- **Body (raw JSON):**
```json
{
    "Temperature": 80,
    "Run_Time": 120
}
```

#### cURL Command:
```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"Temperature": 80, "Run_Time": 120}'
```

#### Example Response:
```json
{
    "Downtime": "Yes",
    "Confidence": 0.85
}
```

---

### 4. View Visualizations
After training the model, visualizations are saved in the `static/` folder:
- **Confusion Matrix:** `http://127.0.0.1:5000/static/confusion_matrix.png`
- **ROC Curve:** `http://127.0.0.1:5000/static/roc_curve.png`

---



## Example Dataset
Place a CSV file (e.g., `sample_data.csv`) in the `data/` folder with the following structure:

Machine_ID,Temperature,Run_Time,Downtime_Flag
1,75,120,0
2,80,100,1
3,85,130,1
4,70,90,0
5,88,150,1
```

---

## Dependencies
- Flask
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

Install all dependencies using the following command:
pip install -r requirements.txt

