# 📱 Smartphone Usage & Addiction Analysis & Risk Predictor

An end-to-end Machine Learning project and interactive Web Application designed to analyze daily smartphone usage patterns and predict smartphone addiction risks based on user lifestyle metrics.

---

## 📌 Overview

With the rapid increase in digital device consumption, understanding healthy vs. unhealthy screen habits is essential. This project utilizes a dataset of **7,500 user records** containing daily screen time metrics, lifestyle indicators, and stress levels to build a predictive machine learning model.

The system features:
- **Data Analysis Notebook**: Comprehensive exploration and feature analysis.
- **Model Training Pipeline**: Automated script training, evaluating, and selecting the best classification model (Logistic Regression / K-Nearest Neighbors).
- **Interactive Web Application**: A user-friendly Streamlit web application providing real-time addiction risk prediction and personalized digital wellness recommendations.

---

## ✨ Features

- 📊 **Comprehensive Usage Analytics**: Evaluates daily screen time, social media usage, gaming hours, work/study hours, notification frequency, and sleep duration.
- 🤖 **Machine Learning Classification**: Accurately classifies users into **Addicted** or **Non-Addicted** categories with probability scoring.
- 🌐 **Interactive Streamlit Web Interface**: Input daily usage habits and get instant analysis, risk indicators, and customized recommendations.
- 🔄 **Reproducible ML Pipeline**: Modular training script (`train_model.py`) that preprocesses data, scales features, evaluates models, and exports model artifacts (`.pkl`).

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.x
- **Web Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning & Data Science**:
  - `pandas` - Data manipulation & analysis
  - `numpy` - Numerical computation
  - `scikit-learn` - Model training, scaling, and evaluation
  - `joblib` - Model serialization and loading

---

## 📊 Dataset Summary

The dataset `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv` consists of **7,500 samples** with the following key attributes:

| Feature Name | Description |
| :--- | :--- |
| `age` | User age (years) |
| `gender` | Gender (Male, Female, Other) |
| `daily_screen_time_hours` | Total daily smartphone usage (hours) |
| `social_media_hours` | Time spent on social media apps (hours) |
| `gaming_hours` | Time spent on mobile gaming (hours) |
| `work_study_hours` | Productive smartphone usage (hours) |
| `sleep_hours` | Daily sleep duration (hours) |
| `notifications_per_day` | Average daily notification count |
| `app_opens_per_day` | Daily app launch frequency |
| `weekend_screen_time` | Screen time during weekends (hours) |
| `stress_level` | Perceived stress level (Low, Medium, High) |
| `academic_work_impact` | Negative impact on work/studies (Yes/No) |
| **`addicted_label`** | **Target Variable**: Addiction binary status (`0`: Healthy, `1`: Addicted) |

---

## 📁 Directory Structure

```text
├── 38689_Notebook.ipynb                      # Jupyter Notebook for EDA & model experimentation
├── 38689_Report.pdf                          # Project analysis report
├── Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv  # Dataset
├── train_model.py                            # Model training & serialization script
├── app.py                                    # Streamlit web application
├── best_model.pkl                            # Serialized machine learning model
├── scaler.pkl                                # Serialized StandardScaler object
├── model_columns.pkl                         # Serialized feature column alignment list
└── README.md                                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

*Note: If `requirements.txt` is not present, install core dependencies manually:*
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

---

## 💻 Usage

### 1. Train the Machine Learning Model

To retrain the classification models and export updated artifacts (`best_model.pkl`, `scaler.pkl`, `model_columns.pkl`):

```bash
python train_model.py
```

### 2. Launch the Web Application

To run the Streamlit app locally:

```bash
streamlit run app.py
```

Once launched, open your web browser at `http://localhost:8501` to use the interactive application.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
