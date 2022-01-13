import streamlit as st

header = st.container()
model_validation = st.container()
model_forecast = st.container()

with header:
    st.title("NBA MVP Prediction Model")


with model_validation:
    st.subheader("Model Validation")
    st.text("Utilize this section to validate model prediction of previous years")

with model_forecast:
    st.subheader("Forecasting the 2021-2022 NBA MVP")
    st.text("This section contains the weekly forecast of the 2021-2022 season MVP")
