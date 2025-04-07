import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

# ✅ This must come before any other Streamlit commands
st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")

# --- Load the model ---
@st.cache_resource
def load_model_once():
    model_path = Path(__file__).parent / "model3.h5"
    return load_model(model_path)

model = load_model_once()

# --- App UI ---
def main():
    st.title("🩺 Diabetes Risk Prediction (Full Model)")
    st.markdown("This model uses all available health information to predict diabetes risk.")

    # --- Input Form ---
    with st.form("input_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 1, 120, 30)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
        bmi = weight / ((height / 100) ** 2)
        st.metric("BMI", f"{bmi:.1f}")

        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        glucose = st.number_input("Blood Glucose (mg/dL)", 50.0, 500.0, 100.0)
        hba1c = st.number_input("HbA1c (%)", 3.0, 20.0, 5.5, step=0.1)

        submitted = st.form_submit_button("Predict Risk")

    # --- Prediction ---
    if submitted:
        try:
            # Prepare feature array
            features = [
                1 if gender == "Male" else 0,
                float(age),
                float(bmi),
                1 if smoking == "Never" else 0,
                1 if smoking == "Former" else 0,
                1 if smoking == "Current" else 0,
                1 if hypertension == "Yes" else 0,
                1 if heart_disease == "Yes" else 0,
                float(glucose),
                float(hba1c)
            ]

            X = np.array([features], dtype=np.float32)
            st.write("🧪 Input Features:", X)

            # Make prediction
            prediction = model.predict(X)[0][0] * 100
            st.success(f"🧾 **Predicted Diabetes Risk: {prediction:.1f}%**")

            # Interpretation
            if prediction < 5:
                st.info("🟢 Low Risk")
            elif prediction < 20:
                st.warning("🟡 Moderate Risk")
            else:
                st.error("🔴 High Risk")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
