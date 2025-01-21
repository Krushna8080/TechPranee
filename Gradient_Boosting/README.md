# Machine Downtime Prediction API (Gradient Boosting)

This project is a Flask-based API to predict machine downtime using a Gradient Boosting Classifier. It supports endpoints for uploading datasets, training the model, and making predictions.

## Features
- **Upload Dataset**: Accepts a CSV file for training.
- **Train Model**: Trains a Gradient Boosting model and returns evaluation metrics.
- **Predict Downtime**: Predicts machine downtime and provides confidence scores.
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
```bash
git clone <repository-url>
cd project-directory
```

### 2. Install Dependencies
Install the required Python packages using the following command:
```bash
pip install -r requirements.txt
```

### 3. Run the Flask Application
Start the Flask server by running:
```bash
python app.py
```
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
```bash
curl -X POST -F 'file=@data/sample_data.csv' http://127.0.0.1:5000/upload
```

#### Example Response:
```json
{
    "message": "Dataset uploaded successfully.",
    "columns": ["Temperature", "Run_Time", "Downtime_Flag"]
}
```

---

### 2. Train the Model
**Endpoint:** `/train`  
**Method:** `POST`  
**Description:** Train the Gradient Boosting model on the uploaded dataset.

#### Example Request (Postman):
- **Type:** POST
- **URL:** `http://127.0.0.1:5000/train`

#### cURL Command:
```bash
curl -X POST http://127.0.0.1:5000/train
```

#### Example Response:
```json
{
    "message": "Model trained successfully.",
    "metrics": {
        "accuracy": 0.92,
        "precision": 0.93,
        "recall": 0.91,
        "f1_score": 0.92,
        "roc_auc": 0.96
    },
    "confusion_matrix_path": "/static/confusion_matrix.png",
    "roc_curve_path": "/static/roc_curve.png"
}
```

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
    "Temperature": 85,
    "Run_Time": 130
}
```

#### cURL Command:

curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"Temperature": 85, "Run_Time": 130}'


#### Example Response:
```json
{
    "Downtime": "Yes",
    "Confidence": 0.89
}
```

---

### 4. View Visualizations
After training the model, visualizations are saved in the `static/` folder:
- **Confusion Matrix:** `http://127.0.0.1:5000/static/confusion_matrix.png`
- **ROC Curve:** `http://127.0.0.1:5000/static/roc_curve.png`

---

## Notes
- Make sure the dataset contains the columns `Temperature`, `Run_Time`, and `Downtime_Flag`.
- Save your dataset in the `data/` folder for easy access.
- If you face any issues, ensure all dependencies are installed correctly using `requirements.txt`.

---

## Example Dataset
Place a CSV file (e.g., `sample_data.csv`) in the `data/` folder with the following structure:

```
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
```

