import pickle
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Food Safety Prediction",
        layout="centered"
)


with open("model.pkl", "rb") as file:
    model = pickle.load(file)


with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


with open("features.pkl", "rb") as file:
    required_columns = pickle.load(file)


def preprocess_input(features):

    input_data = pd.DataFrame([features])

    categorical_columns = [
        "product_type",
        "packaging_type"
    ]

    input_data = pd.get_dummies(
        input_data,
        columns=categorical_columns,
        drop_first=True
    )

    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[required_columns]

    input_data = input_data.astype(float)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_scaled)[0]
        probability = max(probabilities)
    else:
        probability = None

    return prediction[0], probability


st.title("Food Safety Prediction App")

st.write(
    "Enter the food product and inspection details "
    "to predict whether the product is approved."
)


st.header("Product Information")


product_type = st.selectbox(
    "Product Type",
    [
        "Dairy",
        "Meat",
        "Bakery",
        "Beverage",
        "Canned Food",
        "Frozen Food"
    ]
)


packaging_type = st.selectbox(
    "Packaging Type",
    [
        "Plastic",
        "Glass",
        "Metal",
        "Paper",
        "Cardboard"
    ]
)


st.header("Storage Information")


shelf_life_days = st.number_input(
    "Shelf Life (Days)",
    min_value=1,
    max_value=1000,
    value=30
)


storage_temperature_C = st.number_input(
    "Storage Temperature (°C)",
    min_value=-50.0,
    max_value=100.0,
    value=25.0,
    step=0.1
)


humidity_level_percent = st.number_input(
    "Humidity Level (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=0.1
)


moisture_content = st.number_input(
    "Moisture Content",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=0.1
)


st.header("Food Composition")


preservative_level = st.number_input(
    "Preservative Level",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=0.1
)


pH_level = st.number_input(
    "pH Level",
    min_value=0.0,
    max_value=14.0,
    value=7.0,
    step=0.1
)


st.header("Inspection Information")


inspection_score_0_100 = st.number_input(
    "Inspection Score (0-100)",
    min_value=0,
    max_value=100,
    value=75
)


customer_rating_1_5 = st.number_input(
    "Customer Rating (1-5)",
    min_value=1.0,
    max_value=5.0,
    value=4.0,
    step=0.1
)


features = {
    "product_type": product_type,
    "packaging_type": packaging_type,
    "shelf_life_days": shelf_life_days,
    "storage_temperature_C": storage_temperature_C,
    "humidity_level_percent": humidity_level_percent,
    "moisture_content": moisture_content,
    "preservative_level": preservative_level,
    "pH_level": pH_level,
    "inspection_score_0_100": inspection_score_0_100,
    "customer_rating_1_5": customer_rating_1_5
}


if st.button("Predict Food Safety", use_container_width=True):

    prediction, probability = preprocess_input(features)

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success("Food Product Approved")

        if probability is not None:
            st.info(
                f"Prediction Confidence: "
                f"{probability * 100:.2f}%"
            )

    else:

        st.error("Food Product Not Approved")

        if probability is not None:
            st.info(
                f"Prediction Confidence: "
                f"{probability * 100:.2f}%"
            )


if st.checkbox("Show Input Details"):

    st.subheader("Input Details")

    input_summary = pd.DataFrame(
        {
            "Feature": [
                "Product Type",
                "Packaging Type",
                "Shelf Life (Days)",
                "Storage Temperature (°C)",
                "Humidity (%)",
                "Moisture Content",
                "Preservative Level",
                "pH Level",
                "Inspection Score",
                "Customer Rating"
            ],
            "Value": [
                product_type,
                packaging_type,
                shelf_life_days,
                storage_temperature_C,
                humidity_level_percent,
                moisture_content,
                preservative_level,
                pH_level,
                inspection_score_0_100,
                customer_rating_1_5
            ]
        }
    )

    st.table(input_summary)


st.markdown("---")

st.caption(
    "Food Safety Prediction System | "
    "Machine Learning + Streamlit"
)