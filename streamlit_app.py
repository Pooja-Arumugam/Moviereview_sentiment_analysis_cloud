import streamlit as st
import requests
from Scripts import s3


#Define The API endpoit

API_URL = " http://127.0.0.1:8000/api/v1/"

headers = {
    'Content-type':'application/json'
}

st.title("ML Model serving over REST API")


model = st.selectbox("Select Model", 
                     ["Sentiment Classifier","None"])

if model =="Sentiment Classifier":
    text = st.text_area("Enter your movie review")
    user_id = st.text_input("Enter your user id","email@email.com")

    data = {"text":[text],"user_id":user_id}
    model_api = "sentiment_analysis"


if st.button("Predict"):
    with st.spinner("Predicting .... please wait!!!"):
        response = requests.post(API_URL+model_api, headers = headers,
                                 json=data)
        
        output = response.json()
    st.write(output)