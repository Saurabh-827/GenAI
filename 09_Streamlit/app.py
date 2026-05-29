import streamlit as st
import pandas as pd
import numpy as np

# title of the application

st.title("Hello Streamlit")

# Display a simple text
st.write("This is a simple text written")

#creating a simple dataframe

df = pd.DataFrame({
    "Column 1": [1, 2, 3, 4, 5],
    "Column 2": ["A", "B", "C", "D", "E"]
})

# display dataframe

st.write(df)


# create a linechart

chart_data = pd.DataFrame(
    np.random.rand(20,3), columns=["A", "B", "C"]
)
st.line_chart(chart_data)