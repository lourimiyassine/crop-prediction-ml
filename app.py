import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the dataset and model
df = pd.read_csv("assets/Crops.csv")
model = joblib.load('assets/models/crop_model_rf.joblib')

# Set up the layout of the app
st.title("Crop Prediction App (created by:yassine lourimi)")
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a page to navigate:")


# Create a sidebar navigation
navigation = st.sidebar.radio("Pages", ["Overview", "Infographic", "Prediction"])

def overview():
    st.header("Overview")
    st.subheader("Project Title: Crop Prediction Using Machine Learning ")
    st.write("""
    **Description**: This project predicts the most suitable crop to plant based on specific environmental conditions such as soil nutrients, temperature, humidity, pH level, and rainfall. It helps farmers and agricultural planners make data-driven decisions to optimize crop yield and sustainability.

    **Why Introduce This Project?** 
    - Agricultural efficiency and productivity are crucial for meeting the demands of a growing population.
    - Traditional farming practices often rely on intuition and historical patterns, which can be inefficient and unreliable under changing climate conditions.
    - A data-driven approach to crop selection can improve yields, reduce waste, and increase profitability.

    **Issues Overcome**: 
    - Helps farmers choose the right crop based on current soil and weather conditions.
    - Reduces the risk of crop failure due to improper crop selection.
    - Provides a systematic method for planning agricultural practices with scientific backing.
    """)

def info():
    st.header("Infographic Section")
    st.subheader("Dataset Overview")
    st.write("""
    The dataset used in this application contains the following features:
    - **Nitrogen (N)**: Essential for the growth of leaves and overall plant development.
    - **Phosphorous (P)**: Important for the development of roots and flowers.
    - **Potassium (K)**: Helps in the overall functioning and metabolism of the plant.
    - **Temperature**: Optimal temperature range for crop growth.
    - **Humidity**: Necessary humidity levels for crop development.
    - **pH**: Soil pH level for optimal growth.
    - **Rainfall**: Required rainfall for the crop.

    There are 7 features (columns) in the dataset used for predicting crop types.
    The dataset includes the following 23 crops:
    """)

    # Display crops in two columns
    crops = [
        "Rice", "Wheat", "Maize", "Chickpea", "Kidney Beans", "Pigeon Peas",
        "Moth Beans", "Mung Bean", "Black Gram", "Lentil", "Pomegranate", "Banana",
        "Mango", "Grapes", "Watermelon", "Muskmelon", "Apple", "Orange", "Papaya",
        "Coconut", "Cotton", "Jute", "Coffee"
    ]

    col1, col2 = st.columns(2)
    for i, crop in enumerate(crops):
        if i < len(crops) // 2:
            col1.write(f"- **{crop}**")
        else:
            col2.write(f"- **{crop}**")

    st.subheader("Statistical Summary")
    # Round the statistical summary to two decimal places
    st.write(df.describe().round(2))

    st.subheader("Crops and Their Attributes")
    st.write("""
    This section provides a detailed analysis of the various crops and the climatic conditions required for their optimal growth.
    """)

    # Dropdown for crop selection
    st.subheader("Scatter Plot of Temperature vs Rainfall")
    crop_list = df['label'].unique().tolist()
    selected_crop = st.selectbox("Select Crop", crop_list)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='temperature', y='rainfall', hue='label', data=df[df['label'] == selected_crop])
    plt.title(f"Scatter Plot for {selected_crop}")
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='temperature', y='rainfall', hue='label', data=df)
    st.pyplot(plt)

    # Distribution plots for each attribute
    st.subheader("Distribution of Attributes")
    st.write("""
    The distribution plots below provide insights into the range and spread of each attribute in the dataset, helping to understand the overall data structure and variability.
    """)
    attributes = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    descriptions = {
        'N': "Nitrogen (N) distribution: Shows the range of nitrogen content in the soil samples.",
        'P': "Phosphorous (P) distribution: Indicates the phosphorous levels across different samples.",
        'K': "Potassium (K) distribution: Represents the variation in potassium levels.",
        'temperature': "Temperature distribution: Displays the temperature range suitable for crop growth.",
        'humidity': "Humidity distribution: Highlights the range of humidity levels observed.",
        'ph': "pH distribution: Shows the pH levels present in the soil samples.",
        'rainfall': "Rainfall distribution: Illustrates the range of rainfall received by the crops."
    }

    for attribute in attributes:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[attribute], kde=True)
        plt.title(f'Distribution of {attribute.capitalize()}')
        plt.xlabel(descriptions[attribute])
        st.pyplot(plt)

    # Add the new scatter plot here
    st.subheader("Crop Suitability in High Temperature Conditions")
    plt.figure(figsize=(12, 8))  # Set the plot size
    plt.rcParams['figure.dpi'] = 150  # Set the plot DPI
    sns.scatterplot(x=df['temperature'], y=df['label'], hue=df['label'], palette='viridis', s=100, alpha=0.7)
    plt.title('Crop Suitability in High Temperature Conditions', fontsize=15)
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Crop', fontsize=12)
    plt.legend(title='Crop', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    

def pred():
    st.header("Crop Prediction Section")

    # Create input fields for the features
    st.subheader("Input Features")

    N = st.slider('Nitrogen (N)', min_value=0, max_value=140, value=60)
    P = st.slider('Phosphorous (P)', min_value=5, max_value=145, value=60)
    K = st.slider('Potassium (K)', min_value=5, max_value=205, value=60)
    temperature = st.number_input('Temperature (°C)', min_value=8.0, max_value=43.0, value=25.0)
    humidity = st.number_input('Humidity (%)', min_value=14.0, max_value=99.0, value=60.0)
    ph = st.number_input('pH', min_value=3.5, max_value=10.0, value=6.5)
    rainfall = st.number_input('Rainfall (mm)', min_value=20.0, max_value=300.0, value=100.0)

    # Prepare the feature array for prediction
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Predict the crop
    if st.button('Predict Crop'):
        prediction = model.predict(features)
        st.success(f"Suggested crop for given climatic condition: {prediction[0]}")

        # Show image of the crop
        crop_images = {
            "rice": "assets/images/rice.jpg",
            "wheat": "assets/images/wheat.jpg",
            "maize": "assets/images/maize.jpg",
            "chickpea": "assets/images/chickpea.jpg",
            "kidneybeans": "assets/images/kidneybeans.jpg",
            "pigeonpeas": "assets/images/pigeonpeas.jpg",
            "mothbeans": "assets/images/mothbeans.jpg",
            "mungbean": "assets/images/mungbean.jpg",
            "blackgram": "assets/images/blackgram.jpg",
            "lentil": "assets/images/lentil.jpg",
            "pomegranate": "assets/images/pomegranate.jpg",
            "banana": "assets/images/banana.jpg",
            "mango": "assets/images/mango.jpg",
            "grapes": "assets/images/grapes.jpg",
            "watermelon": "assets/images/watermelon.jpg",
            "muskmelon": "assets/images/muskmelon.jpg",
            "apple": "assets/images/apple.jpg",
            "orange": "assets/images/orange.jpg",
            "papaya": "assets/images/papaya.jpg",
            "coconut": "assets/images/coconut.jpg",
            "cotton": "assets/images/cotton.jpg",
            "jute": "assets/images/jute.jpg",
            "coffee": "assets/images/coffee.jpg"
        }

        if prediction[0] in crop_images:
            st.image(crop_images[prediction[0]], use_column_width=True)

    # Details about each crop
    st.subheader("Crop Details")
    crop_details = {
        "rice": "Rice is a staple food crop that requires high temperatures ranging from 20°C to 30°C, high humidity, and a substantial amount of water for optimal growth. It is typically grown in flooded conditions known as paddies.",
        "wheat": "Wheat is a versatile cereal crop that thrives in temperate climates with temperatures between 10°C and 25°C. It requires well-drained loamy soil and moderate rainfall or irrigation for good yield.",
        "maize": "Maize, also known as corn, is a cereal crop that grows well in warm weather with temperatures between 21°C and 30°C. It needs moderate rainfall and well-drained, fertile soil.",
        "chickpea": "Chickpea is a legume crop that prefers cooler climates with temperatures ranging from 18°C to 30°C. It requires well-drained loamy soil and low to moderate rainfall.",
        "kidneybeans": "Kidney beans require temperatures between 15°C and 25°C, moderate rainfall, and fertile, well-drained soil for optimal growth.",
        "pigeonpeas": "Pigeon peas are a drought-tolerant legume that grows well in warm climates with temperatures between 18°C and 30°C. They require well-drained soil and moderate rainfall.",
        "mothbeans": "Moth beans are a drought-resistant legume that thrives in warm climates with temperatures between 25°C and 30°C. They require well-drained sandy soil and low to moderate rainfall.",
        "mungbean": "Mung bean is a warm-season legume that grows well in temperatures between 25°C and 35°C. It requires well-drained loamy soil and moderate rainfall.",
        "blackgram": "Black gram is a legume that prefers warm weather with temperatures between 25°C and 35°C. It requires well-drained loamy soil and moderate rainfall.",
        "lentil": "Lentil is a cool-season legume that grows well in temperatures between 18°C and 30°C. It requires well-drained loamy soil and moderate rainfall.",
        "pomegranate": "Pomegranate is a fruit-bearing shrub that thrives in warm, dry climates with temperatures between 25°C and 35°C. It requires well-drained sandy loam soil and low to moderate rainfall.",
        "banana": "Banana is a tropical fruit crop that requires warm temperatures between 25°C and 35°C, high humidity, and substantial rainfall or irrigation for optimal growth.",
        "mango": "Mango is a tropical fruit crop that thrives in warm climates with temperatures between 24°C and 30°C. It requires well-drained sandy loam soil and moderate rainfall.",
        "grapes": "Grapes are a fruit crop that grows well in warm, dry climates with temperatures between 18°C and 30°C. They require well-drained sandy loam soil and low to moderate rainfall.",
        "watermelon": "Watermelon is a warm-season fruit crop that requires temperatures between 25°C and 30°C, well-drained sandy soil, and moderate rainfall.",
        "muskmelon": "Muskmelon is a warm-season fruit crop that thrives in temperatures between 25°C and 30°C. It requires well-drained sandy soil and moderate rainfall.",
        "apple": "Apple is a temperate fruit crop that grows well in cool climates with temperatures between 18°C and 25°C. It requires well-drained loamy soil and moderate rainfall.",
        "orange": "Orange is a citrus fruit crop that thrives in warm, subtropical climates with temperatures between 20°C and 30°C. It requires well-drained sandy loam soil and moderate rainfall.",
        "papaya": "Papaya is a tropical fruit crop that grows well in warm climates with temperatures between 25°C and 30°C. It requires well-drained sandy loam soil and moderate rainfall.",
        "coconut": "Coconut is a tropical crop that thrives in warm, humid climates with temperatures between 20°C and 30°C. It requires well-drained sandy loam soil and substantial rainfall.",
        "cotton": "Cotton is a warm-season crop that grows well in temperatures between 25°C and 35°C. It requires well-drained sandy loam soil and moderate rainfall.",
        "jute": "Jute is a tropical fiber crop that requires warm, humid climates with temperatures between 25°C and 35°C. It requires well-drained loamy soil and substantial rainfall.",
        "coffee": "Coffee is a tropical crop that grows well in warm, humid climates with temperatures between 15°C and 25°C. It requires well-drained loamy soil and moderate rainfall."
    }

    selected_crop = st.selectbox("Select a crop to view details", list(crop_details.keys()))
    st.write(crop_details[selected_crop])

# Show the selected page
if navigation == "Overview":
    overview()
elif navigation == "Infographic":
    info()
elif navigation == "Prediction":
    pred()
