import streamlit as st
import pandas as pd

st.title("Streamlit text input")

name = st.text_input("Enter your name")

if name:
    st.write(f"Hello {name}!")

age = st.slider("Select your age", 0,100,25)

st.write(f"Your age is {age}")

options = ["Python", "Java", "C++", "JavaScript"]

selected_option =  st.selectbox("Choose your language:", options)

st.write(f"You selected {selected_option}")

data  = {
    "Name": ["Peter", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, 30, 35, 40, 45],
    "City": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
}

data_fr = pd.DataFrame(data)

st.write(data_fr)


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

