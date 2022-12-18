import streamlit as st
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

st.title("MBTI Type Predictor")

# with open("TfidfVectorizer.pkl", mode = "rb"):
#     vectorizer = joblib.load("TfidfVectorizer.pkl")

# with open("KBest_selector.pkl", mode = "rb"):
#     selector = joblib.load("KBest_selector.pkl")

# scores = {}
# model_file = 'SVC_classifier.pkl'

# pickle_in = open(model_file, 'rb')
# model = joblib.load(pickle_in)
    
with open("pipeModelSVM.pkl", mode = "rb"):
    model = joblib.load("pipeModelSVM.pkl")

input = st.text_input('Enter your thoughts', '')
if(input):
    prediction = model.predict([input])
    st.markdown(prediction)
