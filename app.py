import pandas as pd
import numpy as np 
import streamlit as st
import tensorflow
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from keras.models import load_model
import pickle
import os

import os
os.chdir(r'C:\Adithya\Necessary_Items\Python\GenAI - Course\annclassification')
os.getcwd()

model = load_model('model.h5')

with open('label_encode_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_encoder_geo.pkl','rb') as file:
    lable_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


#streamlit app
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography',lable_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit_Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = pd.DataFrame(
    {
        'CreditScore':[credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age' : [age],
        'Tenure' : [tenure],
        'Balance': [balance],
        'NumOfProducts' : [num_of_products],
        'HasCrCard' : [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary' : [estimated_salary]
    }
)

#one hot encode geogrpahy
geo_encoder = lable_encoder_geo.transform([[geography]])
geo_encoder_df = pd.DataFrame(geo_encoder.toarray(),columns=lable_encoder_geo.get_feature_names_out(['Geography']))

#concatentaion with onehot encoded data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoder_df],axis=1)

#scaling the input data
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)

prediction_probability = prediction[0][0]

st.write(f'churn probability: {prediction_probability:.2f}')

if prediction_probability > 0.5:
    st.write('customer will churn')
else:
    st.write('customer will not churn')