# TechPranee ML API Project

This repository showcases the implementation of Machine Learning APIs for different supervised learning algorithms. We utilized **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Gradient Boosting** to predict machine downtime based on input features like `Temperature` and `Run_Time`.

## Project Overview

- **Dataset:** Created a synthetic dataset of 3000 rows using a Python script with the required columns: `Machine_ID`, `Temperature`, `Run_Time`, and `Downtime_Flag`.
- **APIs:** Each algorithm is wrapped with Flask-based RESTful APIs to allow easy interaction.
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix were calculated and visualized for each algorithm.
- **Visualization:** Confusion Matrix and ROC Curve visualizations are generated dynamically and stored for each model.

### Algorithms Implemented

| Algorithm          | Accuracy | Folder Name      |
|--------------------|----------|------------------|
| Logistic Regression| 73%      | `Logistic_Regression` |
| Gradient Boosting  | 71%      | `Gradient_Boosting`    |
| Decision Tree      | 53%     | `Decision_Tree`       |
| Random Forest      | 57%     | `Random_Forest`       |

Logistic Regression achieved the highest accuracy of **73%**, followed by Gradient Boosting.

---

## Directory Structure
TechPranee/
│
├── Decision_Tree/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   └── dataset/
│       └── sample_data.csv
│
├── Gradient_Boosting/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   └── dataset/
│       └── sample_data.csv
│
├── Logistic_Regression/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   └── dataset/
│       └── sample_data.csv
│
├── Random_Forest/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   └── dataset/
│       └── sample_data.csv
│
└── README.md
