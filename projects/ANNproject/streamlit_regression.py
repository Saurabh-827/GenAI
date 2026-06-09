import streamlit as st
import pandas as pd
import tensorflow as tf
import os
import pickle

# 1. Setup exact paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'regression_model.h5')
scaler_path = os.path.join(working_dir, 'scaler.pkl')
label_encoder_gender_path = os.path.join(working_dir, 'label_encoder_gender.pkl')
onehot_encoder_geo_path = os.path.join(working_dir, 'onehot_encoder_geo.pkl')

# 2. Load Models & Encoders safely
model = tf.keras.models.load_model(model_path)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(label_encoder_gender_path, 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(onehot_encoder_geo_path, 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

# 3. Streamlit App UI
st.title('Customer Estimated Salary Prediction')
st.write("Enter the customer's details to predict their estimated salary.")

# User inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
credit_score = st.number_input('Credit Score')
# 'Exited' ab input ban gaya hai
exited = st.selectbox('Has Exited (Churned)?', [0, 1]) 

# 4. Prepare the input data (Order must match X_train exactly)
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray() # Added .toarray() just in case sparse_output is True
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data using the loaded scaler
input_data_scaled = scaler.transform(input_data)

# 5. Make prediction on button click
if st.button("Predict Salary"):
    prediction = model.predict(input_data_scaled)
    predicted_salary = prediction[0][0]
    
    # Regression model gives a direct continuous number, not a probability
    st.success(f"Predicted Estimated Salary: ${predicted_salary:,.2f}")