# 📱 Smartphone Usage Behavior & Addiction Risk Prediction

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Data Science and Machine Learning solution designed to analyze smartphone behavioral patterns, quantify digital usage habits, and predict smartphone addiction risk. Featuring exploratory data analysis (EDA), predictive ML modeling, and an interactive **Streamlit** web application.

---

## 📌 Project Overview

In the modern digital era, smartphone addiction poses severe impacts on mental health, sleep quality, academic performance, and overall productivity. This project utilizes a comprehensive dataset of **7,500 behavioral records** to build predictive machine learning models capable of classifying individual smartphone addiction risks based on usage metrics, demographic information, and lifestyle indicators.

### Key Highlights:
- **Comprehensive Dataset Analysis**: Evaluates 16 critical attributes including daily screen time, social media usage, gaming, notifications, sleep duration, and stress levels.
- **Robust ML Pipeline**: Implements data preprocessing, categorical encoding, feature scaling (`StandardScaler`), and model optimization using **Logistic Regression** and **K-Nearest Neighbors (KNN)**.
- **Interactive Web Dashboard**: Built with **Streamlit**, providing real-time addiction risk assessments, probability breakdown, interactive sliders, and personalized recommendations.

---

## 📊 Dataset Description

The analysis is powered by `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv` containing **7,500 user records**:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `age` | Numerical | Age of the individual (years) |
| `gender` | Categorical | Gender (`Male`, `Female`, `Other`) |
| `daily_screen_time_hours` | Numerical | Total hours spent on phone daily |
| `social_media_hours` | Numerical | Daily hours spent on social media apps |
| `gaming_hours` | Numerical | Daily hours spent on gaming apps |
| `work_study_hours` | Numerical | Daily hours spent on work/study apps |
| `sleep_hours` | Numerical | Average night sleep duration (hours) |
| `notifications_per_day` | Numerical | Total push notifications received daily |
| `app_opens_per_day` | Numerical | Number of times phone is unlocked/opened daily |
| `weekend_screen_time` | Numerical | Total screen time during weekends (hours) |
| `stress_level` | Categorical/Discrete | Self-reported stress level (`Low`, `Medium`, `High`) |
| `academic_work_impact` | Categorical/Discrete | Negative impact on studies/work (`Low`, `Medium`, `High`) |
| **`addicted_label`** | **Binary (Target)** | **`0` = Low Risk / Not Addicted \| `1` = High Risk / Addicted** |

---

## 🏗️ System Architecture & Workflow

```
┌──────────────────────────────┐
│  Dataset (7,500 Rows CSV)    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐     ┌────────────────────────────────┐
│ Preprocessing & Encoding     │ ──> │ Exploratory Data Analysis      │
│ (One-Hot & StandardScaler)   │     │ (Jupyter Notebook Analysis)    │
└──────────────┬───────────────┘     └────────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Model Training & Evaluation │
│  (Logistic Regression / KNN) │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Serialized Model Artifacts   │
│ (.pkl Model & Scaler)        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Interactive Streamlit App    │
│ (Real-time Prediction UI)    │
└──────────────────────────────┘
```

---

## 🚀 Getting Started

Follow these step-by-step instructions to set up and run the project on your local machine.

### 1. Prerequisites
Ensure you have Python **3.9 or higher** installed.

### 2. Clone the Repository
```bash
git clone https://github.com/adeeshachathurmina-ux/smartphone-addiction-prediction.git
cd smartphone-addiction-prediction
```

### 3. Set Up Virtual Environment (Recommended)
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt / PowerShell)
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application

### Option A: Launch the Interactive Streamlit Web App
Run the following command in your terminal:
```bash
streamlit run app.py
```
The application will automatically launch in your default browser at `http://localhost:8501`.

### Option B: Train & Export Models
To retrain the Machine Learning pipeline and regenerate model artifacts (`.pkl` files):
```bash
python train_model.py
```

### Option C: Run EDA Notebook
To explore data visualizations, statistical summaries, and correlations:
```bash
jupyter notebook 38689_Notebook.ipynb
```

---

## 🎯 Web Application Features

The Streamlit web interface provides:
1. **Interactive Inputs**: Sliders and dropdowns for demographic, screen time, lifestyle, and stress parameters.
2. **Real-Time Inference**: Instant addiction probability score (%) and binary risk classification.
3. **Metrics Overview**: Metric cards summarizing key user habits alongside risk indicator badges.
4. **Actionable Recommendations**: Custom advice tailored to reduce screen time, manage notifications, and improve sleep hygiene.

---

## 📂 Project Directory Structure

```
smartphone-addiction-prediction/
│
├── 38689_Notebook.ipynb                           # Comprehensive EDA & Analysis Notebook
├── Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv # Raw Dataset
├── app.py                                         # Interactive Streamlit Web Interface
├── train_model.py                                 # Model Training & Artifact Serializer
├── best_model.pkl                                 # Trained Classification Model Artifact
├── scaler.pkl                                     # Fitted StandardScaler Artifact
├── model_columns.pkl                              # Feature Schema Alignment Artifact
├── requirements.txt                               # Project Dependencies List
└── README.md                                      # Documentation & Setup Guide
```

---

## 📈 Model Performance & Results

- **Primary Classifier**: Logistic Regression (Optimized with standard scaling)
- **Evaluation Metrics**:
  - **Accuracy**: ~89% - 92%
  - **F1-Score**: High precision-recall balance across binary classes
- **Top Behavioral Indicators**: Daily Screen Time, Social Media Hours, App Opens Per Day, Notifications, and Stress Level.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to enhance this project:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/NewFeature`).
3. Commit your changes (`git commit -m 'Add NewFeature'`).
4. Push to the branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
