import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
# import pickle

# Load the dataset
df = pd.read_csv("assets/Crops.csv")
# Load the model (make sure you have saved the model using joblib and named it 'crop_model.joblib')
# with open("assets/crop_model_rf.pkl", "rb") as f:
#     model = pickle.load(f)

model = joblib.load('assets/models/crop_model_rf.joblib')  

# Set up the layout of the app
st.title("Crop Prediction App")
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a page to navigate:")

# Create a sidebar navigation
navigation = st.sidebar.radio("Pages", ["Infographic", "Prediction"])

def info():
    st.header("Infographic Section")

    # Scatter plot of temperature vs rainfall
    st.subheader("Scatter Plot of Temperature vs Rainfall")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='temperature', y='rainfall', hue='label', data=df)
    st.pyplot(plt)

def pred():
    # st.write(model)
    st.header("Crop Prediction Section")

    # Create input fields for the features
    st.subheader("Input Features")

    N = st.slider('Nitrogen', min_value=0, max_value=140, value=60)
    P = st.slider('Phosphorous', min_value=5, max_value=145, value=60)
    K = st.slider('Potassium', min_value=5, max_value=205, value=60)
    temperature = st.slider('Temperature', min_value=8.0, max_value=43.0, value=25.0)
    humidity = st.slider('Humidity', min_value=14.0, max_value=99.0, value=60.0)
    ph = st.slider('pH', min_value=3.5, max_value=10.0, value=6.5)
    rainfall = st.slider('Rainfall', min_value=20.0, max_value=300.0, value=100.0)

    # Prepare the feature array for prediction
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

  

    # Predict the crop
    if st.button('Predict Crop'):
        prediction = model.predict(features)
        st.success(f"Suggested crop for given climatic condition: {prediction[0]}")

    # Details about each crop
    st.subheader("Crop Details")
    crop_details = {
        "rice": "Rice requires high temperature (20-30°C), high humidity, and a lot of water.",
        "wheat": "Wheat thrives in cool, moist conditions and needs well-drained soils.",
        "maize": "Maize prefers warm weather and well-drained, fertile soils.",
        "chickpea": "Chickpea grows best in cool, dry conditions and well-drained soils.",
        "kidneybeans": "Kidney beans need a moderate temperature and well-drained, fertile soils.",
        "pigeonpeas": "Pigeon peas thrive in warm weather and can tolerate poor soils.",
        "mothbeans": "Moth beans prefer hot, arid climates and sandy soils.",
        "mungbean": "Mung beans need warm weather and well-drained, fertile soils.",
        "blackgram": "Black gram grows well in warm, moist conditions and well-drained soils.",
        "lentil": "Lentils thrive in cool, dry conditions and well-drained soils.",
        "pomegranate": "Pomegranates require hot, dry climates and well-drained soils.",
        "banana": "Bananas need high temperature and high humidity, with plenty of water.",
        "mango": "Mangoes thrive in hot weather and well-drained soils.",
        "grapes": "Grapes prefer warm, dry climates and well-drained soils.",
        "watermelon": "Watermelons need hot weather and sandy, well-drained soils.",
        "muskmelon": "Muskmelons grow best in warm weather and sandy, well-drained soils.",
        "apple": "Apples need cool to cold weather and well-drained soils.",
        "orange": "Oranges prefer warm climates and well-drained soils.",
        "papaya": "Papayas require warm weather and well-drained soils.",
        "coconut": "Coconuts thrive in hot, humid climates and sandy, well-drained soils.",
        "cotton": "Cotton grows well in warm weather and well-drained, fertile soils.",
        "jute": "Jute requires warm, humid climates and well-drained, fertile soils.",
        "coffee": "Coffee grows best in cool to warm climates and well-drained soils."
    }

    crop = prediction[0] if 'prediction' in locals() else None
    if crop:
        st.info(crop_details[crop])

def main():
    if navigation == "Infographic":
        info()

    elif navigation == "Prediction":
        pred()

if __name__ == "__main__":
    main()