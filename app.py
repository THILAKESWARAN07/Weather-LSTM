import streamlit as st
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model

# --------------------------------------------------
# 🎨 Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="LSTM Weather Forecast",
    page_icon="🌤️",
    layout="centered"
)

# --------------------------------------------------
# 🏷️ Header Section
# --------------------------------------------------
st.markdown("<h1 style='text-align:center; color:#4B9CD3;'>🌦️ LSTM Weather Temperature Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='text-align:center; font-size:17px; color:gray;'>
    Predict the next temperature value using a deep learning model (LSTM) trained on historical weather data.  
    Enter your latest readings to forecast future temperature trends. 🌡️
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# 🧠 Load Model and Scalers
# --------------------------------------------------
@st.cache_resource
def load_lstm_model():
    try:
        model = load_model("model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading LSTM model: {e}")
        return None


@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load("scaler.pkl")
        return scaler
    except Exception as e:
        st.error(f"❌ Error loading scaler file: {e}")
        return None

model = load_lstm_model()
scaler = load_scaler()

if model is not None and scaler is not None :
    st.success("✅ Model and Scaler Loaded Successfully!")
else:
    st.warning("⚠️ Please ensure all required files are in the same folder: `model.h5`, `scaler.pkl`.")

st.divider()

# --------------------------------------------------
# 📊 User Input Section
# --------------------------------------------------
st.subheader("🧾 Input Latest Weather Data")

feature_names = [
    'p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 'sh',
    'H2OC', 'rho', 'wv', 'max. wv', 'wd', 'rain', 'raining',
    'SWDR', 'PAR', 'max. PAR', 'Tlog'
]

st.markdown("<small>Tip: Use actual recent readings for more accurate predictions.</small>", unsafe_allow_html=True)

# Create input columns (2 per row)
user_input = {}
cols = st.columns(2)
for i, col in enumerate(feature_names):
    with cols[i % 2]:
        user_input[col] = st.number_input(f"{col}", value=0.0, format="%.4f")

# Convert input to numpy array
input_data = np.array(list(user_input.values())).reshape(1, -1)

st.divider()

# --------------------------------------------------
# 🔮 Prediction Section
# --------------------------------------------------
if st.button("✨ Generate Prediction"):
    try:
        scaled_input = scaler.transform(input_data)

        # Repeat last timestep for sequence shape
        X_input = np.repeat(scaled_input, 60, axis=0).reshape(1, 60, -1)

        # Predict directly without inverse scaling
        prediction_scaled = model.predict(X_input)
        predicted_temp = prediction_scaled[0][0]

        st.success(f"🌡️ **Predicted Next Temperature: {predicted_temp:.2f} (scaled units)**")
        st.balloons()
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

st.divider()

# --------------------------------------------------
# 🧾 Additional Info
# --------------------------------------------------
with st.expander("📘 About this App"):
    st.write("""
    - **Model Type:** LSTM (Long Short-Term Memory)
    - **Dataset:** Cleaned weather data (`cleaned_weather.csv`)
    - **Purpose:** Forecast short-term temperature trends
    - **Inputs:** 20 recent weather parameters
    """)

# --------------------------------------------------
# 🦾 Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:14px;'>
        Built with ❤️ using <b>Streamlit</b> + <b>TensorFlow</b><br>
        © 2025 Weather Intelligence Lab
    </div>
    """,
    unsafe_allow_html=True
)
