# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:24:06 2024

@author: 14915
"""

import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import normalize

# Load the model and scaler
with open('ml_sl_best_model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

with open('ml_sl_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Feature names list
feature_names = [
    "GMV STG.l",
    "GMV ITG.l",
    "z-FC aSTG.l to HG.l",
    "z-FC aSTG.l to TP.r",
    "z-FC aSTG.l to Ver45",
    "z-FC HG.l to Pallidum.l",
    "z-FC pSTG.r to iLOC.r",
    "z-FC PC to toITG.r",
    "z-FC_PC to pITG.l",
    "z-FC toMTG.r to Ver3",
    "z-FC toMTG.r to pMTG.l",
    "z-FC PT.l to IC.r",
    "z-FC PT.l to SCC.r",
    "z-FC Cuneal.l to Forb.l",
    "z-FC Ver3 to Caudate.l"
]

# Create title and instructions
st.title('ASD Severity Prediction Model')
st.write('Enter feature data for prediction')

# Create input fields
features = []
for name in feature_names:
    value = st.number_input(name, format="%.12f")
    features.append(value)

# Convert input data to numpy array
new_sample = np.array([features])

# Predict when the button is clicked
if st.button('Predict'):
    # Data preprocessing
    new_sample_scaled = scaler.transform(new_sample)
    new_sample_normalized = normalize(new_sample_scaled, norm='l2')

    # Make prediction
    prediction = best_model.predict(new_sample_normalized)
    prediction_proba = best_model.predict_proba(new_sample_normalized)

    # Display results
    if prediction[0] == 0:
        st.write('Prediction: ML')
    else:
        st.write('Prediction: SL')
    
    st.write(f'Probability of ML: {prediction_proba[0][0]:.4f}')
    st.write(f'Probability of SL: {prediction_proba[0][1]:.4f}')
