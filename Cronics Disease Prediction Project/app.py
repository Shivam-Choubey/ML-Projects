import streamlit as st
import pandas as pd
import numpy as np

# --- MOCKING DATA LOADING ---
# NOTE: Since this environment cannot access local files, the original pickle loading
# has been replaced with mock classes for demonstration. If you run this script
# locally, uncomment your original lines below and remove the mock classes.

# Original code (uncomment for local use):
# import pickle
# scaler = pickle.load(open("models/scaler.pkl", 'rb'))
# model_gbc = pickle.load(open("models/model_gbc.pkl", 'rb'))


# MOCKING CLASSES for demonstration purposes:
class MockScaler:
    def transform(self, df):
        # In a real app, this would apply standardization.
        # We only transform the categorical columns below, so this simply returns the df.
        return df

class MockModel:
    def predict(self, df):
        # MOCK PREDICTION LOGIC: Predict CKD (1) if Serum Creatinine > 1.5 AND Hemoglobin < 13.0
        
        # Find 'sc' and 'hemo' values before scaling/encoding
        # Note: In the mock logic, 'sc' and 'hemo' here are the original unscaled inputs,
        # as the MockScaler does not apply scaling.
        sc = df['sc'].iloc[0]
        hemo = df['hemo'].iloc[0]
        
        # Simple health rule for mock prediction
        if sc > 1.5 and hemo < 13.0:
            return np.array([1]) # CKD
        else:
            return np.array([0]) # No CKD
        
# Initialize mock objects:
scaler = MockScaler()
model_gbc = MockModel()
# -----------------------------


# --- PREDICTION FUNCTION ---
def predict_chronic_disease(age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc):
    """Processes input and returns a prediction (0 or 1)."""
    
    # Create a DataFrame with input variables
    df_dict = {
        'age': [age],
        'bp': [bp],
        'sg': [sg],
        'al': [al],
        'hemo': [hemo],
        'sc': [sc],
        'htn': [htn],
        'dm': [dm],
        'cad': [cad],
        'appet': [appet],
        'pc': [pc]
    }
    df = pd.DataFrame(df_dict)

    # Encode the categorical columns
    df['htn'] = df['htn'].map({'yes':1, "no":0})
    df['dm'] = df['dm'].map({'yes':1, "no":0})
    df['cad'] = df['cad'].map({'yes':1, "no":0})
    df['appet'] = df['appet'].map({'good':1, "poor":0})
    df['pc'] = df['pc'].map({'normal':1, "abnormal":0})

    # Scale the numeric columns using the previously fitted scaler
    numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
    # Use copy to avoid SettingWithCopyWarning, though safe here
    df[numeric_cols] = scaler.transform(df[numeric_cols].copy())

    # Make the prediction
    prediction = model_gbc.predict(df)

    # Return the predicted class
    return prediction[0]


# --- STREAMLIT UI ---
st.set_page_config(
    page_title="CKD Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('🩺 Chronic Kidney Disease (CKD) Risk Predictor')
st.markdown("""
Enter the patient's clinical and demographic data below to estimate the risk of Chronic Kidney Disease.
""")

# --- Sidebar for context and definitions ---
st.sidebar.header("Variable Definitions")
st.sidebar.markdown("""
- **sg (Specific Gravity):** A measure of kidney concentrating ability.
- **al (Albumin):** Presence of protein (albumin) in the urine.
- **hemo (Hemoglobin):** Level of hemoglobin in the blood.
- **sc (Serum Creatinine):** A key marker for kidney function. High levels indicate impaired function.
- **htn/dm/cad:** History of Hypertension, Diabetes Mellitus, and Coronary Artery Disease.
""")

# --- Input Form ---
with st.container(border=True):
    st.subheader("Patient Vitals and Lab Results")

    col_vitals, col_labs, col_conditions = st.columns(3)

    with col_vitals:
        st.caption("Demographics & Pressure")
        age = st.slider("Age (years)", min_value=1, max_value=100, value=50, step=1)
        bp = st.slider("Blood Pressure (mmHg)", min_value=60, max_value=200, value=120, step=5)
        
    with col_labs:
        st.caption("Key Laboratory Indicators")
        hemo = st.slider("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=14.0, step=0.1)
        sc = st.slider("Serum Creatinine (mgs/dL)", min_value=0.5, max_value=15.0, value=1.0, step=0.1)
        
    with col_conditions:
        st.caption("Urinalysis")
        sg = st.number_input("Specific Gravity", min_value=1.005, max_value=1.035, value=1.020, step=0.005, format="%.3f")
        al = st.number_input("Albumin (Scale 0-5)", min_value=0.0, max_value=5.0, value=0.0, step=0.5)


    st.markdown("---")
    st.subheader("Patient History and Lifestyle")
    
    col_history1, col_history2, col_history3 = st.columns(3)

    with col_history1:
        htn = st.selectbox("Hypertension (High BP)", ["no", 'yes'])
        dm = st.selectbox("Diabetes Mellitus", ["no", 'yes'])
    
    with col_history2:
        cad = st.selectbox("Coronary Artery Disease", ["no", 'yes'])
        appet = st.selectbox("Appetite Status", ["good", "poor"])
        
    with col_history3:
        pc = st.selectbox("Protein in Urine (PC)", ["normal", "abnormal"])
        # Placeholder for visual alignment
        st.write("") 
        
    # --- Centering the Prediction Button ---
    col_button_left, col_button_center, col_button_right = st.columns([1, 2, 1])
    
    with col_button_center:
        if st.button('Analyze Patient Data', type="primary"):
            st.session_state.run_prediction = True

    # State Initialization (Moved outside the 'with col_button_center:' block for clarity)
    if 'run_prediction' not in st.session_state:
        st.session_state.run_prediction = False

# --- RESULT DISPLAY ---
if st.session_state.run_prediction:
    st.markdown("---")
    
    # Make the prediction
    try:
        result = predict_chronic_disease(age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc)
    except Exception as e:
        # Handle cases where mock data or model transformation fails gracefully
        st.error(f"Prediction failed due to an error: {e}")
        result = -1 # Sentinel value
        
    # Display the result
    if result == 1:
        # CKD Predicted
        st.error('### 🔴 HIGH RISK: Chronic Kidney Disease (CKD) Predicted')
        with st.expander("Diagnostic Summary and Recommendations"):
            st.write("""
            Based on the input features, the model suggests a high likelihood of CKD.
            Factors like **elevated Serum Creatinine** or **proteinuria (Albumin)** often drive this result.
            
            **Recommendations:**
            - **Confirm** laboratory values and patient history.
            - **Refer** the patient to a nephrologist for comprehensive evaluation.
            - **Monitor** blood pressure and blood glucose closely if diabetic.
            """)
        
    elif result == 0:
        # No CKD Predicted
        st.success('### 🟢 LOW RISK: No Chronic Kidney Disease Predicted')
        with st.expander("Diagnostic Summary and Recommendations"):
            st.write("""
            The model indicates a low risk for Chronic Kidney Disease based on the provided data.
            Key indicators (like Serum Creatinine and Albumin) appear to be within normal limits.

            **Recommendations:**
            - Continue **regular annual screenings** for at-risk individuals (e.g., diabetics, hypertensives).
            - **Lifestyle monitoring** is crucial for prevention.
            """)
            
    else:
        # Error case
        st.warning('### ⚠️ Analysis Incomplete')

# Reset prediction state after display for cleaner interactivity
st.session_state.run_prediction = False
