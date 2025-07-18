import streamlit as st
import requests

st.title(" Fashion Product Attribute Predictor")

uploaded_file = st.file_uploader("Upload a fashion image", type=["jpg", "png", "jpeg","webp"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', width=300)
    with st.spinner("Predicting..."):
        res = requests.post(
            "http://localhost:8000/predict",
            files={"file": uploaded_file.getvalue()}
        )
        if res.status_code == 200:
            preds = res.json()
            st.success(" Predictions:")
            st.markdown(f"- **Season:** {preds['season']}")
            st.markdown(f"- **Gender:** {preds['gender']}")
            st.markdown(f"- **Article Type:** {preds['articleType']}")
            st.markdown(f"- **Base Color:** {preds['baseColor']}")
        else:
            st.error("Prediction failed. Try again.")
