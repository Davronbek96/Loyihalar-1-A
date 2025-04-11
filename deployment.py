import streamlit as st
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score

model = load('weather_forecast_model.joblib')
try:
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')
    y_test= y_test.squeeze()
    show_accuracy = True
except:
    show_accuracy = False
st.set_page_config(page_title="Rain Prediction", layout="centered")
st.title("🌦️ Weather Forecast - Rain Prediction")
st.markdown("Provide today's weather details to predict the chance of rain:")
temperature = st.number_input("🌡️ Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0)
humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
wind_speed = st.number_input("💨 Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=10.0)
cloud_cover = st.number_input("☁️ Cloud Cover (%)", min_value=0.0, max_value=100.0, value=50.0)
pressure = st.number_input("🌬️ Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1013.0)
if st.button("Predict Rain"):
    input_data = pd.DataFrame([{
    "Temperature": temperature,
    "Humidity": humidity,
    "Wind_Speed": wind_speed,
    "Cloud_Cover": cloud_cover,
    "Pressure": pressure
}])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🌧️ It will likely rain.")
    else:
        st.info("🌤️ No rain expected.")
        if show_accuracy:
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.metric("🔍 Model Accuracy", f"{acc * 100:.2f}%")
        else:
                st.warning("Accuracy score is not available. Please ensure `X_test.csv` and `y_test.csv` are in the directory.")



