import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Smartphone Addiction Predictor",
    page_icon="📱",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E88E5;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        color: #555555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-title'>📱 Smartphone Usage & Addiction Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Predict smartphone addiction risks based on daily habits and usage patterns</div>", unsafe_allow_html=True)

# Load Saved Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, scaler, model_columns

model, scaler, model_columns = load_artifacts()

# Sidebar Information
st.sidebar.title("ℹ️ About the App")
st.sidebar.info(
    "This Web Application uses Machine Learning (Logistic Regression) "
    "to analyze smartphone usage metrics such as Daily Screen Time, "
    "Social Media Usage, Sleep Hours, and Stress Levels to predict potential addiction."
)

st.sidebar.subheader("📊 Dataset Overview")
st.sidebar.write("Dataset Size: **7,500 Rows**")
st.sidebar.write("Target Variable: **Addicted Label (0: No, 1: Yes)**")

# Create Input Form with Columns
st.subheader("📋 Enter Usage & Lifestyle Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (අවුරුදු)", min_value=10, max_value=80, value=22, step=1)
    gender = st.selectbox("Gender (ලින්ගිකත්වය)", ["Male", "Female", "Other"])
    daily_screen_time_hours = st.slider("Daily Screen Time (පැය)", 0.0, 18.0, 6.5, 0.5)
    weekend_screen_time = st.slider("Weekend Screen Time (පැය)", 0.0, 18.0, 8.0, 0.5)

with col2:
    social_media_hours = st.slider("Social Media Hours (පැය)", 0.0, 15.0, 3.5, 0.5)
    gaming_hours = st.slider("Gaming Hours (පැය)", 0.0, 15.0, 1.5, 0.5)
    work_study_hours = st.slider("Work/Study Hours (පැය)", 0.0, 15.0, 4.0, 0.5)
    sleep_hours = st.slider("Sleep Hours (නින්ද පැය)", 0.0, 12.0, 6.0, 0.5)

with col3:
    notifications_per_day = st.number_input("Notifications / Day (දිනකට නොටිෆිකේෂන්)", min_value=0, max_value=500, value=85, step=5)
    app_opens_per_day = st.number_input("App Opens / Day (දිනකට ඇප් ඕපන් කිරීම්)", min_value=0, max_value=500, value=60, step=5)
    stress_level = st.selectbox("Stress Level (මානසික පීඩන මට්ටම)", ["Low", "Medium", "High"])
    academic_work_impact = st.selectbox("Academic / Work Impact (අධ්‍යාපනයට/රක්ෂාවට බලපෑම)", ["No", "Yes"])

st.markdown("---")

# Prediction Trigger
if st.button("🔍 Analyze & Predict Addiction Risk"):
    # Construct input dataframe matching raw categorical values
    input_dict = {
        'age': age,
        'daily_screen_time_hours': daily_screen_time_hours,
        'social_media_hours': social_media_hours,
        'gaming_hours': gaming_hours,
        'work_study_hours': work_study_hours,
        'sleep_hours': sleep_hours,
        'notifications_per_day': notifications_per_day,
        'app_opens_per_day': app_opens_per_day,
        'weekend_screen_time': weekend_screen_time,
        'gender_Male': 1 if gender == "Male" else 0,
        'gender_Other': 1 if gender == "Other" else 0,
        'stress_level_Low': 1 if stress_level == "Low" else 0,
        'stress_level_Medium': 1 if stress_level == "Medium" else 0,
        'academic_work_impact_Yes': 1 if academic_work_impact == "Yes" else 0
    }

    # Format dataframe
    input_df = pd.DataFrame([input_dict])

    # Reorder columns to match trained columns
    input_df = input_df[model_columns]

    # Scale inputs
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    st.subheader("🎯 Prediction Result (ප්‍රතිඵලය)")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.error("🚨 **High Risk of Smartphone Addiction! (ඇබ්බැහි වීමේ අවදානම ඉහළයි)**")
            st.write("පරිශීලකයා ස්මාර්ට්ෆෝන් භාවිතයට ඇබ්බැහි වී ඇති බවට පූර්ව නිගමනය කෙරේ.")
        else:
            st.success("✅ **Healthy Usage Habit! (සාමාන්‍ය සෞඛ්‍ය සම්පන්න භාවිතය)**")
            st.write("පරිශීලකයාගේ ස්මාර්ට්ෆෝන් භාවිතය සාමාන්‍ය මට්ටමක පවතී.")

    with res_col2:
        addiction_prob = probabilities[1] * 100
        st.metric("Addiction Probability (ඇබ්බැහි වීමේ ප්‍රබලතාව)", f"{addiction_prob:.1f}%")
        st.progress(int(addiction_prob))

    # Advice
    st.markdown("### 💡 Recommended Recommendations (නිර්දේශ):")
    if prediction == 1:
        st.write("- 📵 **Set daily screen time limits** for social media and gaming.")
        st.write("- 😴 **Improve sleep hygiene**: Avoid phone screens at least 1 hour before sleep.")
        st.write("- 🔕 **Reduce notification overload**: Mute non-essential app notifications.")
    else:
        st.write("- 👍 Keep up your current balanced screen habits!")
        st.write("- 🏃 Ensure regular breaks and physical exercise.")
