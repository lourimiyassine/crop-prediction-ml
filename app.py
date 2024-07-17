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
navigation = st.sidebar.radio("Pages", ["Infographic", "Prediction"])

def info():
    st.header("Infographic Section")
    st.subheader("Dataset Overview")
    st.write("""
             The dataset used in the crop prediction app includes various features that represent the environmental conditions and soil properties essential for crop growth. It is structured to aid in predicting the most suitable crop for given climatic and soil conditions.
    The dataset used in this application contains the following features:
    - **Nitrogen (N)**: Essential for the growth of leaves and overall plant development.
    - **Phosphorous (P)**: Important for the development of roots and flowers.
    - **Potassium (K)**: Helps in the overall functioning and metabolism of the plant.
    - **Temperature**: Optimal temperature range for crop growth.
    - **Humidity**: Necessary humidity levels for crop development.
    - **pH**: Soil pH level for optimal growth.
    - **Rainfall**: Required rainfall for the crop.

    There are 7 features (columns) in the dataset used for predicting crop types.
    he dataset includes the following 23 crops:

    - **Rice**
    - **Wheat**
    - **Maize**
    - **Chickpea** 
    - **Kidney Beans**
    - **Pigeon Peas**
    - **Moth Beans**
    - **Mung Bean**
    - **Black Gram**
    - **Lentil**
    - **Pomegranate**
    - **Banana**
    - **Mango**
    - **Grapes**
    - **Watermelon**
    - **Muskmelon**
    - **Apple**
    - **Orange**
    - **Papaya**
    - **Coconut**
    - **Cotton**
    - **Jute**
    - **Coffee**        
""")

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

    # Distribution plots for each attribute
    st.subheader("Distribution of Attributes")
    attributes = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    for attribute in attributes:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[attribute], kde=True)
        plt.title(f'Distribution of {attribute.capitalize()}')
        st.pyplot(plt)

def pred():
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
        "maize": "Also known as corn, maize is a warm-season crop that prefers temperatures between 18°C and 27°C and requires well-drained, fertile soils. It is one of the most widely grown crops in the world, used for food, fodder, and industrial products. Maize has a high water requirement during its growth period, particularly during the flowering and grain-filling stages. It is sensitive to frost and grows best with moderate rainfall. Maize cultivation benefits from proper weed and pest management.",
        "chickpea": "Chickpeas are a cool-season legume that thrives in dry conditions and well-drained soils. Optimal growth temperatures range from 21°C to 26°C. Chickpeas are drought-tolerant and require minimal water compared to other crops. They are an excellent source of protein and are used in various culinary dishes worldwide. Chickpeas also have the added benefit of improving soil health by fixing nitrogen, making them an essential crop for sustainable farming.",
        "kidneybeans": "Kidney beans grow best in moderate temperatures and well-drained, fertile soils. They require a growing season of about 100-140 days and thrive in temperatures ranging from 18°C to 26°C. Kidney beans need consistent moisture, particularly during flowering and pod development stages. They are a rich source of protein, dietary fiber, and essential vitamins and minerals. Proper irrigation and pest management are crucial for a good yield.",
        "pigeonpeas": "Pigeon peas are a warm-season crop that can tolerate poor soils and thrive in temperatures between 20°C and 35°C. They are drought-resistant and can be grown in semi-arid regions. Pigeon peas are an important source of protein and are used in various traditional dishes in tropical and subtropical regions. The crop also helps in improving soil fertility through nitrogen fixation.",
        "mothbeans": "Moth beans prefer hot, arid climates and sandy soils. They are highly drought-resistant and can grow in regions with minimal rainfall. Optimal growth temperatures range from 25°C to 35°C. Moth beans are a good source of protein and are often used in traditional Indian cuisine. They require minimal inputs and are suitable for cultivation in marginal lands.",
        "mungbean": "Mung beans need warm weather and well-drained, fertile soils. They thrive in temperatures between 25°C and 35°C. Mung beans are fast-growing and typically mature within 60-75 days. They are rich in protein and essential nutrients, making them a valuable crop for both human consumption and soil improvement through nitrogen fixation.",
        "blackgram": "Black gram grows well in warm, moist conditions and well-drained soils. It prefers temperatures between 25°C and 30°C. Black gram is a short-duration crop, typically maturing in 70-90 days. It is a significant source of protein and is widely used in various culinary dishes, particularly in South Asia. The crop also contributes to soil fertility through nitrogen fixation.",
        "lentil": "Lentils thrive in cool, dry conditions and well-drained soils. Optimal growth temperatures range from 18°C to 30°C. Lentils are highly nutritious, providing protein, fiber, and essential micronutrients. They have a relatively short growing season of about 80-110 days. Lentils improve soil health by fixing nitrogen and are an essential crop in crop rotation systems.",
        "pomegranate": "Pomegranates require hot, dry climates and well-drained soils. They thrive in temperatures ranging from 20°C to 35°C and can tolerate drought conditions. Pomegranates are known for their high nutritional value, providing vitamins, minerals, and antioxidants. The crop requires proper irrigation during the flowering and fruit development stages to ensure good yields. Pomegranate cultivation also benefits from good pest and disease management practices.",
        "banana": "Bananas need high temperatures (around 26°C to 30°C) and high humidity for optimal growth. They require plenty of water but must be planted in well-drained soils to prevent waterlogging. Bananas are fast-growing, typically reaching maturity within 9-12 months. They are rich in carbohydrates, vitamins, and minerals. Proper irrigation and nutrient management are essential for healthy banana production.",
        "mango": "Mangoes thrive in hot weather with well-drained soils. Optimal growth temperatures range from 24°C to 27°C. Mangoes are highly valued for their delicious, nutrient-rich fruits, providing vitamins A and C. The crop requires adequate water during flowering and fruit development stages. Mango trees benefit from regular pruning and pest management to ensure healthy growth and good yields.",
        "grapes": "Grapes prefer warm, dry climates and well-drained soils. They thrive in temperatures between 15°C and 30°C. Grapes are used for fresh consumption, wine production, and dried products like raisins. Proper irrigation, trellising, and pruning are essential for maintaining vine health and productivity. Grapes require careful management of pests and diseases to ensure high-quality fruit production.",
        "watermelon": "Watermelons need hot weather and sandy, well-drained soils. They grow best in temperatures between 25°C and 30°C. Watermelons are rich in vitamins, minerals, and antioxidants, making them a popular summer fruit. The crop requires consistent moisture, especially during the fruit development stage. Watermelons benefit from proper spacing and pest management to ensure healthy growth.",
        "muskmelon": "Muskmelons, also known as cantaloupes, grow best in warm weather and sandy, well-drained soils. Optimal growth temperatures range from 24°C to 30°C. Muskmelons are rich in vitamins A and C and provide a sweet, nutritious fruit. The crop requires adequate water, particularly during flowering and fruit development stages. Muskmelons benefit from proper weed and pest management practices.",
        "apple": "Apples need cool to cold weather and well-drained soils. They thrive in temperatures ranging from 18°C to 24°C. Apples are rich in dietary fiber, vitamins, and antioxidants. The crop requires a period of cold weather to break dormancy and promote flowering, a process known as chilling. Apples benefit from proper pruning, pest management, and irrigation practices.",
        "orange": "Oranges prefer warm climates and well-drained soils. Optimal growth temperatures range from 15°C to 30°C. Oranges are an excellent source of vitamin C and other essential nutrients. The crop requires adequate water during flowering and fruit development stages. Oranges benefit from proper irrigation, pruning, and pest management practices to ensure healthy growth and high-quality fruit production.",
        "papaya": "Papayas require warm weather and well-drained soils. They thrive in temperatures between 21°C and 33°C. Papayas are rich in vitamins A and C, making them a nutritious fruit choice. The crop requires consistent moisture, particularly during the flowering and fruit development stages. Papayas benefit from proper spacing, pest management, and nutrient management practices.",
        "coconut": "Coconuts thrive in hot, humid climates and sandy, well-drained soils. Optimal growth temperatures range from 27°C to 32°C. Coconuts provide a versatile crop used for food, oil, and other products. The crop requires consistent moisture and benefits from regular fertilization and pest management practices. Coconuts are highly tolerant of saline soils and coastal conditions.",
        "cotton": "Cotton grows well in warm weather and well-drained, fertile soils. Optimal growth temperatures range from 20°C to 30°C. Cotton is a major fiber crop used in textile production. The crop requires adequate water during flowering and boll development stages. Cotton benefits from proper pest management, particularly to control boll weevils and other pests.",
        "jute": "Jute requires warm, humid climates and well-drained, fertile soils. Optimal growth temperatures range from 24°C to 37°C. Jute is primarily grown for its fiber, which is used in making burlap, hessian, and other products. The crop requires adequate water and benefits from regular weeding and pest management practices.",
        "coffee": "Coffee grows best in cool to warm climates and well-drained soils. Optimal growth temperatures range from 15°C to 24°C. Coffee is a major global commodity, providing a stimulant beverage enjoyed worldwide. The crop requires adequate shade, proper irrigation, and pest management to ensure healthy growth and high-quality bean production. Coffee plants benefit from regular pruning and nutrient management practices."
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


