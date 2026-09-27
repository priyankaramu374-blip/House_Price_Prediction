import streamlit as st
import joblib
import pandas as pd
import time

# Load model
model = joblib.load("house_price_model.pkl")

# Page settings
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("🏠 House Price Prediction")
st.write("Enter the details of your house to predict its estimated price.")

st.divider()

# House Details
st.subheader("🏡 Enter House Details")

area = st.text_input(
    "📐 Area (sq ft)",
    placeholder="Example: 1200"
)

bedrooms = st.text_input(
    "🛏️ Number of Bedrooms",
    placeholder="Example: 3"
)

bathrooms = st.text_input(
    "🛁 Number of Bathrooms",
    placeholder="Example: 2"
)

floors = st.text_input(
    "🏢 Number of Floors",
    placeholder="Example: 2"
)

year_built = st.text_input(
    "📅 Year Built",
    placeholder="Example: 2015"
)

location = st.selectbox(
    "📍 Location",
    ["Downtown", "Suburban", "Urban", "Rural"]
)

condition = st.selectbox(
    "⭐ House Condition",
    ["Excellent", "Good", "Fair", "Poor"]
)

garage = st.selectbox(
    "🚗 Garage",
    ["Yes", "No"]
)

st.divider()

# Predict button
if st.button("🏠 Predict House Price", use_container_width=True):

    if area and bedrooms and bathrooms and floors and year_built:

        # Loading effect
        with st.spinner("🏠 Calculating house price..."):
            time.sleep(2)

            input_data = pd.DataFrame({
                "Area": [float(area)],
                "Bedrooms": [int(bedrooms)],
                "Bathrooms": [int(bathrooms)],
                "Floors": [int(floors)],
                "YearBuilt": [int(year_built)],
                "Location": [location],
                "Condition": [condition],
                "Garage": [garage]
            })

            prediction = model.predict(input_data)[0]

        # Result
        st.success("✅ House Price Predicted Successfully!")

        col1, col2 = st.columns([1, 2])

        # House image
        with col1:
            st.image(
                "https://cdn-icons-png.flaticon.com/512/619/619153.png",
                width=180
            )

        # Price
        with col2:
            st.subheader("🏡 Predicted House Price")
            st.markdown(
                f"# ₹{prediction:,.2f}"
            )

        st.info(
            "💡 This is an estimated price based on the details you entered."
        )

    else:
        st.warning(
            "⚠️ Please enter all house details before predicting."
        )