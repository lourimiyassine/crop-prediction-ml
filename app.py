import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the dataset
df = pd.read_csv("assets/Crops.csv")
# Load the model (make sure you have saved the model using joblib and named it 'crop_model.joblib')
model = joblib.load('assets/models/crop_model_rf.joblib')

# Set up the layout of the app
st.title("Crop Prediction App")
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a page to navigate:")

# Create a sidebar navigation
navigation = st.sidebar.radio("Pages", ["Overview", "Infographic", "Prediction"])

def overview():
    st.header("Overview")
    st.subheader("Project Title: Crop Prediction Using Machine Learning")
    st.write("""
    **Description**: This project is designed to predict the most suitable crop to plant based on specific environmental conditions such as soil nutrients, temperature, humidity, pH level, and rainfall. By utilizing machine learning, this app helps farmers and agricultural planners make data-driven decisions to optimize crop yield and sustainability.

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
    with col1:
        for crop in crops[:len(crops)//2]:
            st.write(f"- **{crop}**")

    with col2:
        for crop in crops[len(crops)//2:]:
            st.write(f"- **{crop}**")

    st.subheader("Statistical Summary")
    st.write(df.describe())

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
    st.subheader("Scatter Plot of Temperature vs Rainfall")
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

def pred():
    st.header("Crop Prediction Section")

    # Create input fields for the features
    st.subheader("Input Features")

    N = st.slider('Nitrogen (N)', min_value=0, max_value=140, value=60)
    P = st.slider('Phosphorous (P)', min_value=5, max_value=145, value=60)
    K = st.slider('Potassium (K)', min_value=5, max_value=205, value=60)
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
        "rice": "Rice is a staple food crop that requires high temperatures ranging from 20°C to 30°C, high humidity, and a substantial amount of water for optimal growth. It is typically grown in flooded conditions known as paddies, which help to control weeds and pests. Rice cultivation is labor-intensive, and it thrives in well-drained, fertile soils rich in organic matter. Proper water management is crucial for rice, as it needs consistent moisture throughout its growing season.",
        "wheat": "Wheat is a major cereal crop that grows best in cool, moist conditions and well-drained soils. It is usually planted in the fall and harvested in the summer. Optimal growth temperatures range from 12°C to 25°C. Wheat requires a period of cold weather to trigger flowering, a process known as vernalization. It is a versatile crop that can be used for various products, including bread, pasta, and cereals. Wheat cultivation benefits from moderate rainfall and deep, loamy soils.",
        "maize": "Maize, also known as corn, is a versatile crop that thrives in warm climates with temperatures ranging from 18°C to 27°C. It requires moderate rainfall and fertile, well-drained soils. Maize is a staple food in many parts of the world and is used for human consumption, animal feed, and biofuel production. Proper management of pests and diseases is essential for high maize yields, and it benefits from crop rotation practices to maintain soil fertility.",
        "chickpea": "Chickpeas are a type of legume that grow best in cool, dry climates with temperatures between 10°C and 25°C. They require well-drained, loamy soils and are relatively drought-tolerant. Chickpeas are a rich source of protein and are used in various dishes worldwide. They improve soil fertility through nitrogen fixation, making them a valuable crop in crop rotation systems. Chickpeas require minimal irrigation and are often grown as a rainfed crop.",
        "kidneybeans": "Kidney beans are a popular legume that grows best in warm climates with temperatures ranging from 18°C to 30°C. They require well-drained, sandy loam soils and moderate rainfall. Kidney beans are a rich source of protein and dietary fiber, making them a valuable addition to vegetarian diets. Proper pest and disease management is crucial for high yields, and they benefit from crop rotation practices to maintain soil health.",
        "pigeonpeas": "Pigeon peas are a legume crop that thrives in warm, semi-arid climates with temperatures between 20°C and 35°C. They are drought-tolerant and can be grown in poor soils, but they perform best in well-drained loamy soils. Pigeon peas are a rich source of protein and are used in various culinary dishes worldwide. They improve soil fertility through nitrogen fixation and are often grown as a companion crop to provide shade and support to other plants.",
        "mothbeans": "Moth beans are a drought-tolerant legume that grows well in arid and semi-arid regions with temperatures ranging from 20°C to 30°C. They require sandy, well-drained soils and minimal rainfall. Moth beans are a rich source of protein and are often used in traditional Indian cuisine. They are an important crop in dryland farming systems and help improve soil fertility through nitrogen fixation.",
        "mungbean": "Mung beans are a warm-season legume that thrives in tropical and subtropical climates with temperatures between 25°C and 35°C. They require well-drained loamy soils and moderate rainfall. Mung beans are a rich source of protein and are used in various culinary dishes, including soups and salads. They are often grown as a cover crop to improve soil fertility and prevent erosion.",
        "blackgram": "Black gram, also known as urad bean, is a legume that grows well in warm, humid climates with temperatures ranging from 25°C to 35°C. It requires well-drained loamy soils and moderate rainfall. Black gram is a rich source of protein and is widely used in Indian cuisine. It improves soil fertility through nitrogen fixation and is often grown as a companion crop with cereals.",
        "lentil": "Lentils are a cool-season legume that grows best in temperate climates with temperatures between 15°C and 25°C. They require well-drained loamy soils and moderate rainfall. Lentils are a rich source of protein and are used in various dishes worldwide. They improve soil fertility through nitrogen fixation and are an important crop in sustainable farming systems.",
        "pomegranate": "Pomegranates are a fruit crop that thrives in hot, dry climates with temperatures ranging from 25°C to 35°C. They require well-drained sandy loam soils and minimal rainfall. Pomegranates are rich in antioxidants and are used in various culinary dishes and beverages. They are drought-tolerant and can be grown in arid regions with proper irrigation.",
        "banana": "Bananas are a tropical fruit crop that grows best in warm, humid climates with temperatures between 26°C and 30°C. They require well-drained loamy soils and abundant rainfall. Bananas are rich in carbohydrates and are a staple food in many tropical regions. They require consistent moisture and proper pest and disease management for optimal yields.",
        "mango": "Mangoes are a tropical fruit crop that thrives in hot, humid climates with temperatures ranging from 24°C to 30°C. They require well-drained sandy loam soils and moderate rainfall. Mangoes are rich in vitamins and are widely consumed fresh or in processed forms. Proper pest and disease management is essential for high yields, and they require regular pruning and irrigation.",
        "grapes": "Grapes are a fruit crop that grows best in temperate climates with temperatures between 15°C and 25°C. They require well-drained loamy soils and moderate rainfall. Grapes are used for fresh consumption, wine production, and raisin making. Proper management of pests and diseases is crucial for high yields, and they benefit from trellising and pruning practices.",
        "watermelon": "Watermelons are a warm-season fruit crop that thrives in hot, dry climates with temperatures ranging from 25°C to 30°C. They require well-drained sandy loam soils and moderate rainfall. Watermelons are a rich source of vitamins and are widely consumed fresh. They require proper pest and disease management and consistent moisture for optimal yields.",
        "muskmelon": "Muskmelons are a warm-season fruit crop that grows best in hot, dry climates with temperatures between 25°C and 30°C. They require well-drained sandy loam soils and moderate rainfall. Muskmelons are rich in vitamins and are widely consumed fresh. Proper pest and disease management is essential for high yields, and they require consistent moisture throughout their growing season.",
        "apple": "Apples are a temperate fruit crop that grows best in cool climates with temperatures ranging from 15°C to 24°C. They require well-drained loamy soils and moderate rainfall. Apples are rich in vitamins and are widely consumed fresh or in processed forms. Proper pest and disease management is crucial for high yields, and they require regular pruning and irrigation.",
        "orange": "Oranges are a citrus fruit crop that thrives in warm, humid climates with temperatures between 20°C and 30°C. They require well-drained sandy loam soils and moderate rainfall. Oranges are rich in vitamin C and are widely consumed fresh or as juice. Proper pest and disease management is essential for high yields, and they require consistent moisture throughout their growing season.",
        "papaya": "Papayas are a tropical fruit crop that grows best in warm, humid climates with temperatures ranging from 25°C to 30°C. They require well-drained loamy soils and abundant rainfall. Papayas are rich in vitamins and are widely consumed fresh or in processed forms. Proper pest and disease management is crucial for high yields, and they require consistent moisture and regular pruning.",
        "coconut": "Coconuts are a tropical fruit crop that thrives in hot, humid climates with temperatures between 27°C and 32°C. They require well-drained sandy soils and abundant rainfall. Coconuts are used for various products, including coconut milk, oil, and water. They require consistent moisture and proper pest and disease management for optimal yields.",
        "cotton": "Cotton is a fiber crop that grows best in warm climates with temperatures ranging from 25°C to 35°C. It requires well-drained sandy loam soils and moderate rainfall. Cotton is used for textile production and is an important cash crop in many regions. Proper pest and disease management is crucial for high yields, and it benefits from crop rotation practices to maintain soil health.",
        "jute": "Jute is a fiber crop that thrives in hot, humid climates with temperatures between 24°C and 37°C. It requires well-drained sandy loam soils and abundant rainfall. Jute is used for making burlap, ropes, and other products. Proper water management is crucial for high yields, and it requires consistent moisture throughout its growing season.",
        "coffee": "Coffee is a tropical crop that grows best in cool, humid climates with temperatures ranging from 15°C to 24°C. It requires well-drained, fertile soils and moderate rainfall. Coffee is used for making beverages and is an important cash crop in many regions. Proper pest and disease management is crucial for high yields, and it requires regular pruning and irrigation."
    }
    
    crop = prediction[0] if 'prediction' in locals() else None
    if crop:
        st.info(crop_details[crop])

def main():
    if navigation == "Overview":
        overview()
    elif navigation == "Infographic":
        info()
    elif navigation == "Prediction":
        pred()

if __name__ == "__main__":
    main()
