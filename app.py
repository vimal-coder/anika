
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Load models
vgg_model = load_model("vgg16_feature_extractor.h5")
xgb_model = joblib.load("xgb_classifier.pkl")

# Categories
categories = ["cyclone", "earthquake", "flood", "wildfire"]

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Streamlit UI
st.title("⚡Disaster Prediction and management System")
st.write("Upload an image to predict the disaster type and get safety advice.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', width=300)
    
    if st.button("Predict"):
        # Preprocess the image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # rescale

        # Extract features using VGG16
        features = vgg_model.predict(img_array)
        features = features.reshape(features.shape[0], -1)  # flatten

        # Predict using XGBoost
        prediction = xgb_model.predict(features)
        predicted_label = categories[prediction[0]]

        st.success(f"Predicted Disaster Type: **{predicted_label.capitalize()}**")

        # Craft prompt for Gemini
        prompt_text = (
            f"The model detected a {predicted_label}. "
            f"Please provide detailed guidance on: \n"
            f"1. How to stay safe during this disaster.\n"
            f"2. Immediate actions to take if it occurs.\n"
            f"3. How to prevent or minimize the risk in the future.\n"
            f"Give practical, easy-to-follow advice."
        )

        response = llm([HumanMessage(content=prompt_text)])
        st.write(response.content)
