import pandas as pd
import joblib
import streamlit as st

# load our saved pipeline once when the app starts
model = joblib.load("FERTILIZER-RECOMMENDATION-SYSTEM.joblib")

# a title of the WEB PAGE
st.title("Fertilizer Recommendation System")

# a plain line of instructional text, shown just below the title
st.write("Which Fertilizer should be used ?")

# Streamlit Slider for numeric inputs
Temperature = st.slider("Temperature", 0, 55, 5)
Humidity = st.slider("Humidity Of the AIR", 0.0, 100.0, 1.0)
Moisture = st.slider("Moisture", 0.0, 100.0, 5.0)
Nitrogen = st.slider("Nitrogen", 0, 100, 1)
Phosphorous = st.slider("Phosphorous", 0, 100, 5)
Potassium = st.slider("Potassium", 0,100,5)
pH = st.slider("pH of soil",0.0, 14.0, 0.1)
Rainfall = st.slider("Rainfall", 0, 5000, 5)

# Streamlit selectbox for categorical input
Soil_Type = st.selectbox("Soil Type", ['Sandy', 'Clay', 'Black', 'Loamy', 'Red'])
Crop_Type = st.selectbox("Crop Type", ['Wheat', 'Cotton', 'Sugarcane', 'Maize', 'Rice'])
Growth_Stage = st.selectbox("Growth Stage", ['Seedling', 'Flowering', 'Vegetative'])

if st.button("Predict"):

    # Dataframe that acts as input for predict function
    input_row = pd.DataFrame([{
        "Temperature" : Temperature,
        "Humidity" : Humidity,
        "Moisture" : Moisture,
        "Nitrogen" : Nitrogen,
        "Phosphorous" : Phosphorous,
        "Potassium" : Potassium,
        "pH" : pH,
        "Rainfall" : Rainfall,
        "Soil_Type" : Soil_Type,
        "Crop_Type" : Crop_Type,
        "Growth_Stage" : Growth_Stage

    }])

    # the pipeline cleans (impute/scale/encode) AND predicts
    prediction = model.predict(input_row)[0]

    # predict_probability gives us the model's confidence.
    probability = model.predict_proba(input_row)[0][1]

    if prediction == prediction:

        st.success(f"Prediction: {prediction} (confidence: {probability:.0%})")
    else:

        st.error(f"Prediction: {prediction} (confidence: {1 - probability:.0%})")
