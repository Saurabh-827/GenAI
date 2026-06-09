import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle


# NOTE :  I got error of directory not found while deploying on streamlit so i am taking further numbered steps 

# 1. Ye line app.py ka current exact folder path nikal legi
working_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Ab hum saari files ka poora (absolute) rasta banayenge
model_path = os.path.join(working_dir, 'model.h5')
scaler_path = os.path.join(working_dir, 'scaler.pkl')
label_encoder_gender_path = os.path.join(working_dir,'label_encoder_gender.pkl')
onehot_encoder_geo_path = os.path.join(working_dir, 'onehot_encoder_geo.pkl')

# 3. Ab unhe safely load karenge

# load the trained model
model = tf.keras.models.load_model(model_path)

#  load 
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(label_encoder_gender_path, 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(onehot_encoder_geo_path, 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

#streamlit app

st.title('Customer Churn Prediction')


# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
'CreditScore': [credit_score],
'Gender': [label_encoder_gender . transform([gender ])[0]],
'Age': [age],
'Tenure': [tenure],
'Balance': [balance],
'NumOfProducts': [num_of_products],
'HasCrCard': [has_cr_card],
'IsActiveMember': [is_active_member],
'EstimatedSalary': [estimated_salary]

})

# onehot_encode 'Geography'

geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine one-hot encoded columns with input data

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1) 

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
predicted_probality = prediction[0][0]
# Display the result

st.write(f'Churn Probability: {predicted_probality}')

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    prediction_probability = prediction[0][0]

    st.write(f"Churn Probability: {prediction_probability:.2f}")

    if prediction_probability > 0.5:
        st.error("The customer is likely to churn")
    else:
        st.success("The customer is not likely to churn")