import streamlit as st

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
    st.markdown("Years **CORRECTLY** predicted: 1980, 1981, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1995, 1996, 1997, 1998, 1999, 2000, 2002, 2003, 2004, 2007, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021")
    st.markdown("Years **INCORRECTLY** predicted: 1982, 1993, 1994, 2001, 2005, 2006, 2008")
    st.markdown("Overall Accuracy: (35 / 42) = **83.33%**")

    option = st.selectbox(
        'Select year to check out (between 1980 to 2021)',
        (1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021))
    st.write('showing predictions for year:', option)

with model_forecast:
    st.subheader("Forecasting the 2021-2022 NBA MVP")
    st.text("This section contains the weekly forecast of the 2021-2022 season MVP")
