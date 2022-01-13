import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
from model import run_model_average, validate_year
from xgboost import XGBRegressor
import shap

st.set_page_config(layout="wide")

data_path = os.path.dirname(os.getcwd()) + '/data' + '/master_table.csv'
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

    #RUN FOR ALL YEARS
    xgb_avg_mae, xgb_avg_r2, xgb_accuracy, xgb_summary, xgb_models, xgb_cols = run_model_average(df=master_table,
                  regressor = XGBRegressor(n_estimators=16, max_depth=5, learning_rate = 0.2745, subsample=1, colsample_bytree=1),
                 scaling=False, print_metrics=True)

    #show dataframe
    st.dataframe(xgb_summary)
    st.markdown("Overall Accuracy: (35 / 42) = **83.33%**")

    #VALIDATE YEAR
    year_selected = st.selectbox(
        'Select year to check out (between 1980 to 2021)',
        (1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021))
    st.write('showing predictions for year:', year_selected)

    model, X_test_df, mvp_race = validate_year(year=year_selected, df=master_table)
    
    st.write("---------------------")


    xgb_cols = ['PTS', 'value_over_replacement_player', 'seed',	'W/L%', 'player_efficiency_rating',	'win_shares_per_48_minutes', 'offensive_box_plus_minus',
                    'usage_percentage', 'free_throw_attempt_rate']

    top_candidates = list(mvp_race.head(3)['Player'])
    def visualize_shap_values(mvp_race, model, player):
        data_for_prediction = mvp_race[mvp_race['Player'] == player]
        data_for_prediction = data_for_prediction[list(xgb_cols)]
        data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

        # Create object that can calculate shap values
        explainer = shap.TreeExplainer(model)
        # Calculate Shap values
        shap_values = explainer.shap_values(data_for_prediction_array)
        plot = shap.force_plot(explainer.expected_value, shap_values, data_for_prediction)
        return plot

    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
    
    for idx, player in enumerate(top_candidates):
        rank = idx + 1
        st.write(f"Rank: {rank} {player}")
        st.text("---SHAP values from prediction model---")
        plot = visualize_shap_values(mvp_race, model, player)
        st_shap(plot)

with model_forecast:
    st.subheader("Forecasting the 2021-2022 NBA MVP")
    st.text("This section contains the weekly forecast of the 2021-2022 season MVP")
