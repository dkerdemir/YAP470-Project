import streamlit as st
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

st.title("MBTI Type Predictor")


model = joblib.load(open("XGboost_pipeline.pkl", "rb"))

input = st.text_input('Enter your thoughts', '')
if(input):
    prediction = model.predict([input])
    st.markdown(prediction)
