import streamlit as st
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

st.title("MBTI Type Predictor")


with open("XGboost_pipeline.pkl", mode = "rb"):
    model = joblib.load("XGboost_pipeline.pkl")

input = st.text_input('Enter your thoughts', '')
if(input):
    st.write('PROBLEM IS WITH PREDICT')
    prediction = model.predict([input])
    st.markdown(prediction)
