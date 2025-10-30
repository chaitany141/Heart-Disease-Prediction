import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px
import time

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Heart Disease Predictor ❤️", page_icon="🫀", layout="wide")

# -------------------------------------------------
# CSS Styling
# -------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3, h4 {
    color: #f8f9fa;
}
.stButton>button {
    background-color: #00b4d8;
    color: white;
    border-radius: 12px;
    font-size: 1.1rem;
    height: 3rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #0096c7;
    transform: scale(1.03);
}
.card {
    background-color: rgba(255, 255, 255, 0.1);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.card:hover {
    background-color: rgba(255, 255, 255, 0.15);
    transform: scale(1.02);
    transition: 0.3s ease;
}
a {
    color: #00c2ff !important;
    text-decoration: none !important;
}
.footer {
    text-align: center;
    font-size: 0.9rem;
    color: #b0b0b0;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar Info
# -------------------------------------------------
st.sidebar.title("🩺 Heart Health Assistant")
st.sidebar.markdown("""
Welcome to **Heart Disease Predictor**!  
This app uses ML ensemble models to estimate your **heart disease risk**.

**Tabs:**
- 🏠 Home  
- 🧩 Predict  
- 📂 Bulk Predict  
- 📊 Model Info  
""")
st.sidebar.markdown("---")
st.sidebar.caption("Developed by **Chaitanya Pawar** 👨‍💻")

# -------------------------------------------------
# Helper Function for CSV Download
# -------------------------------------------------
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions CSV</a>'
    return href

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab0, tab1, tab2, tab3 = st.tabs(['🏠 Home', '🧩 Predict', '📂 Bulk Predict', '📊 Model Information'])

# -------------------------------------------------
# TAB 0: HOME PAGE
# -------------------------------------------------
with tab0:
    st.markdown("<h1 style='text-align:center;'>🫀 Heart Disease Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Leveraging Machine Learning for Early Heart Disease Detection</h4>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        ### ❤️ Understanding Heart Disease
        Heart disease is one of the **leading causes of death globally**, often resulting from factors like high blood pressure, cholesterol, diabetes, and unhealthy lifestyles.  
        Early detection is crucial — **predicting the risk** allows individuals to take preventive actions before it becomes life-threatening.
        
        ### 💡 Why Prediction Matters
        Machine learning models can analyze patterns in your health data to estimate the **probability of heart disease**.  
        These predictions are **not diagnoses**, but data-driven insights to support **early awareness and medical consultation**.
        
        ### 🌿 Take Control of Your Health:
        - Maintain a balanced diet 🥗  
        - Stay physically active 🏃‍♂️  
        - Avoid smoking 🚭  
        - Manage stress 😌  
        - Get regular check-ups 🩺  
        """, unsafe_allow_html=True)

    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=280)

    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("<div class='card'>🧩 <h3>Predict</h3><p>Enter your health data to get estimated heart disease probability.</p></div>", unsafe_allow_html=True)
    with colB:
        st.markdown("<div class='card'>📂 <h3>Bulk Predict</h3><p>Upload a CSV file to analyze multiple patient records at once.</p></div>", unsafe_allow_html=True)
    with colC:
        st.markdown("<div class='card'>📊 <h3>Model Info</h3><p>View and compare model accuracy and performance visually.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='footer'>💙 Stay Heart-Healthy | Created with ❤️ by <b>Chaitanya Pawar</b></div>", unsafe_allow_html=True)

# -------------------------------------------------
# TAB 1: Predict
# -------------------------------------------------
with tab1:
    st.markdown("### 🧠 Machine Learning-Based Heart Disease Probability Estimation 🫀")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=150)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
        cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    with col2:
        resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
        st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    sex_num = 0 if sex == "Male" else 1
    chest_pain_num = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fasting_bs_num = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg_num = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina_num = 1 if exercise_angina == "Yes" else 0
    st_slope_num = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_num],
        'ChestPainType': [chest_pain_num],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs_num],
        'RestingECG': [resting_ecg_num],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina_num],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_num]
    })

    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    modelnames = ['tree.pkl', 'logisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl']

    predictions = []
    def predict_heart_disease(data):
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions

    if st.button("🔍 Analyze Health Data"):
        with st.spinner("Analyzing your data... ⏳"):
            time.sleep(1.5)
            result = predict_heart_disease(input_data)
            model_votes = [r[0] for r in result]
            disease_votes = sum(model_votes)
            ensemble_prediction = 1 if disease_votes >= 2 else 0

        st.markdown("### 🧩 Ensemble Model Insights")
        risk_percent = (disease_votes / 4) * 100

        if ensemble_prediction == 1:
            st.error(f"🚨 Estimated chances of heart disease: **{risk_percent:.0f}%**")
        else:
            st.success(f"💚 Estimated chances of heart disease: **{risk_percent:.0f}%**")

        st.progress(int(risk_percent))
        st.caption("⚠️ Note: This prediction is based on machine learning analysis and should not replace professional medical advice.")

        with st.expander("📈 See individual model predictions"):
            for i in range(len(result)):
                icon = "❤️" if result[i][0] == 1 else "💚"
                st.write(f"{icon} **{algonames[i]}:** {'Higher chance of heart disease' if result[i][0] == 1 else 'Lower chance of heart disease'}")

# -------------------------------------------------
# TAB 2: Bulk Prediction
# -------------------------------------------------
with tab2:
    st.title("📂 Bulk Prediction from CSV")

    st.info("""
    Upload a CSV file with **11 features** in this order:
    ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
     'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope')
    """)

    uploaded_file = st.file_uploader("📎 Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)
            model = pickle.load(open('logisticRegression.pkl', 'rb'))
            input_data['Predicted Risk'] = model.predict(input_data)
            input_data['Predicted Risk'] = input_data['Predicted Risk'].apply(lambda x: "Higher chance" if x == 1 else "Lower chance")
            st.success("✅ Predictions generated successfully!")
            st.dataframe(input_data, use_container_width=True)
            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
    else:
        st.warning("Please upload a valid CSV file to proceed.")

# -------------------------------------------------
# TAB 3: Model Information
# -------------------------------------------------
with tab3:
    st.title("📊 Model Performance Overview")

    data = {
        'Decision Trees': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 84.23,
        'Support Vector Machine': 84.22
    }

    df = pd.DataFrame(list(data.items()), columns=['Model', 'Accuracy (%)'])
    fig = px.bar(df, x='Model', y='Accuracy (%)', text='Accuracy (%)', title="Model Accuracy Comparison",
                 color='Model', color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Each model was trained and tested individually. The ensemble combines predictions to improve reliability.")
