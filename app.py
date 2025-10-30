# app.py
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(page_title="🌦️ Weather Predictor", layout="centered")

# Load model & scaler
@st.cache_resource
def load_resources():
    model = load_model("model.h5", compile=False)

    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_resources()

# Load dataset (for last sequence)
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_weather.csv", parse_dates=["date"], index_col="date")
    return df

df = load_data()

# -----------------------------
# Helper: Prediction Function
# -----------------------------
def predict_next_weather(model, scaler, df, time_steps=60, target_col="T"):
    """
    Predict the next temperature value using the last time_steps from df.
    Assumes df is already scaled during training.
    """
    # Select features used during training
    features = [
        "p","T","Tpot","Tdew","rh","VPmax","VPact","VPdef",
        "sh","H2OC","rho","wv","max. wv","wd","rain",
        "raining","SWDR","PAR","max. PAR","Tlog"
    ]

    # Keep only available features
    features = [f for f in features if f in df.columns]
    data = df[features].values

    # Scale the data
    scaled = scaler.transform(data)

    # Take the last 'time_steps' sequence
    seq = scaled[-time_steps:]
    seq = np.reshape(seq, (1, time_steps, len(features)))

    # Predict
    pred_scaled = model.predict(seq, verbose=0)[0][0]

    # Inverse transform (to original scale)
    dummy = np.zeros((1, len(features)))
    dummy[0, features.index(target_col)] = pred_scaled
    pred_inv = scaler.inverse_transform(dummy)[0][features.index(target_col)]

    return round(float(pred_inv), 2)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌤️ LSTM Weather Predictor")
st.write("Predict the next temperature reading using your trained LSTM model.")

# User inputs
time_steps = st.slider("Select number of time steps used for prediction", 10, 100, 60, step=5)
target_col = st.selectbox("Select target variable to predict", options=["T", "Tdew", "Tpot", "rh"])

if st.button("🔮 Predict Next Value"):
    try:
        prediction = predict_next_weather(model, scaler, df, time_steps, target_col)
        st.success(f"✅ Predicted {target_col}: {prediction}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# -----------------------------
# Show latest data
# -----------------------------
st.subheader("📊 Latest Weather Data")
st.dataframe(df.tail(10))

st.info("Ensure 'model.h5', 'scaler.pkl', and 'weather.csv' are in the same folder as app.py before running.")
