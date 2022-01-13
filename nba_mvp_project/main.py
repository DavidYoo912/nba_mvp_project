import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
from pathlib import Path
from model import train_test_split_by_year, run_model
from xgboost import XGBRegressor
import shap
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

data_path = Path(__file__).parents[1] / 'data/master_table.csv'
master_table = pd.read_csv(data_path)

header = st.container()
model_validation = st.container()
model_forecast = st.container()

with header:
    st.title("NBA MVP Prediction Model")
    st.text("Prediction model has been trained on historical MVP candidate data from 1980 - 2021")
    st.write("see this [article](https://medium.com/@david_yoo) to read about the approach")

with model_validation:
    st.subheader("Model Validation")
    st.text("Utilize this section to check out model prediction of previous years")

    xgb_data_path = Path(__file__).parents[1] / 'data/xgboost_summary.csv'
    xgb_summary = pd.read_csv(xgb_data_path)

    #show dataframe
    value_counts = xgb_summary['Label'].value_counts()
    xgb_accuracy = value_counts[0] / (value_counts[0] + value_counts[1])
    st.dataframe(xgb_summary)
    st.markdown(f"Overall Accuracy: ({value_counts[0]} / {value_counts[0] + value_counts[1]}) = **{round(xgb_accuracy, 4) * 100}%**")

    st.write("---------------------")

    ##VALIDATE SPECIFIC YEAR
    year_selected = st.selectbox(
        'Select year to check out (between 1980 to 2021)',
        range(2021, 1979, -1))
    st.write('showing predictions for year:', year_selected)

    X_train, y_train, X_test, y_test, cols = train_test_split_by_year(year=year_selected, df=master_table, scaling=False)
    model, mae, r2, predicted_winner, actual_winner, mvp_race = run_model(
                                                                    XGBRegressor(
                                                                        n_estimators=16,
                                                                        max_depth=5, 
                                                                        learning_rate = 0.2745,
                                                                        subsample=1,
                                                                        colsample_bytree=1),
                                              X_train, y_train, X_test, y_test, df=master_table, year=year_selected)
    st.write(f'Predicted: **{predicted_winner}**')
    st.write(f'Actual: **{actual_winner}**')

    mvp_race['predicted_share'] = mvp_race['predicted_share'].apply(lambda x: round(x, 2))

    mvp_race = mvp_race.rename(columns={'Tm':'Team', 'Share':'Actual MVP Share', 'predicted_share':'Predicted MVP Share'})
    st.dataframe(mvp_race[["Rank", "Player", "Team", "Actual MVP Share", "Predicted MVP Share"]])
    st.write("---------------------")

    top_candidates = list(mvp_race.head(3)['Player'])
    def visualize_shap_values(mvp_race, model, player):
        data_for_prediction = mvp_race[mvp_race['Player'] == player]
        data_for_prediction = data_for_prediction[list(cols)]
        data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

        # Create object that can calculate shap values
        explainer = shap.TreeExplainer(model)
        # Calculate Shap values
        shap_values = explainer.shap_values(data_for_prediction_array)
        plot = shap.force_plot(explainer.expected_value, shap_values, data_for_prediction)

        prediction = model.predict(data_for_prediction_array)
        return plot

    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
    
    for idx, player in enumerate(top_candidates):
        rank = idx + 1
        st.write(f"Rank {rank}: **{player}**")
        st.text("---SHAP values from prediction model---")
        plot = visualize_shap_values(mvp_race, model, player)
        st_shap(plot)

with model_forecast:
    st.subheader("Forecasting the 2021-2022 NBA MVP")
    st.text("This section contains the weekly forecast of the 2021-2022 season MVP")
