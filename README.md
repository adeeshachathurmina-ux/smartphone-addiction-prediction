# 📱 Smartphone Usage and Addiction Analysis & Predictor

This repository contains a complete Data Science & Machine Learning project analyzing smartphone usage habits, screen time, lifestyle patterns, and predicting smartphone addiction risk using Machine Learning.

---

## 🚀 Features

- **Exploratory Data Analysis (EDA)**: Jupyter Notebook (`38689_Notebook.ipynb`) with data cleaning, visualization, outlier checking, and distribution analyses.
- **Machine Learning Pipelines**: Models trained on 7,500 rows dataset (`Logistic Regression` & `KNN`).
- **Streamlit Web Application (`app.py`)**: An interactive, user-friendly Web UI to predict addiction risk based on custom user inputs.
- **Model Training & Artifact Export (`train_model.py`)**: Automatic training script saving model artifacts (`best_model.pkl`, `scaler.pkl`, `model_columns.pkl`).

---

## 🛠️ How to Run the Web App Locally

### 1. Prerequisites
Ensure you have Python 3.9 or higher installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

After running the command, open your browser and go to `http://localhost:8501`.

---

## 📁 Repository Structure

```
├── 38689_Notebook.ipynb                           # Main Jupyter Notebook
├── Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv # Raw Dataset
├── app.py                                         # Streamlit Web Application
├── train_model.py                                 # ML Training & Artifact Exporter
├── best_model.pkl                                 # Trained Logistic Regression Model
├── scaler.pkl                                     # Trained StandardScaler Object
├── model_columns.pkl                              # Feature Column Names
├── requirements.txt                               # Python Dependencies
└── README.md                                      # Documentation
```
