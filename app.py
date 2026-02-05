import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('logistic_model.pkl')
scaler= joblib.load('scaler.pkl')
le= joblib.load('label_encoder.pkl')
feature_names = joblib.load('feature_names.pkl')

# Set up the Streamlit app
st.title('Titanic Survival Prediction')
st.write('Enter the details of the passenger to predict survival.')

# Create input fields for user to enter passenger details
pclass = st.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=30)
sibsp = st.slider('Number of Siblings/Spouses Aboard', 0, 10, 2)
parch = st.slider('Number of Parents/Children Aboard', 0, 10, 0)
fare = st.slider('Fare', 0.0, 500.0, 35.0)
alone = st.selectbox('Alone', ['Yes', 'No'])
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])


# Create a button to make the prediction
if st.button('Predict Survival'):
    #encode the input data
    sex=le.transform([sex])[0]
    alone=le.transform([alone])[0]

    #dummy encode the embarked variable
    embarked_Q= 1 if embarked == 'Q' else 0
    embarked_S= 1 if embarked == 'S' else 0
    #emabarked_C is the reference category, so we don't need to create a dummy variable for it C=> S=0, Q=0

    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        alone,
        embarked_Q,
        embarked_S
    ]], columns=feature_names)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    # Make the prediction
    prediction = model.predict(input_data_scaled)

    # Display the prediction result
    if prediction[0] == 1:
        st.success('The passenger is predicted to survive.')
    else:
        st.error('The passenger is predicted not to survive.')