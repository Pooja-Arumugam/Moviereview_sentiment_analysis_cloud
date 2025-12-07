import streamlit as st
import boto3
import os
import torch
from transformers import pipeline

bucket_name = 'mlopslearn-s3bucket'
local_path = 'tinybert-sentiment-analysis'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'

s3 = boto3.client('s3', region_name="us-east-1")

def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']

                relative_path = os.path.relpath(s3_key, s3_prefix)
                local_file = os.path.join(local_path, relative_path)

                os.makedirs(os.path.dirname(local_file), exist_ok=True)

                s3.download_file(bucket_name, s3_key, local_file)


st.title("Machine Learning Model Deployment at server")

button = st.button("Download Model")

if button:
    with st.spinner("Downloading model from S3..."):
        download_dir(local_path, s3_prefix)
    st.success("Download complete!")

text = st.text_area("Enter your text here", "Type...")

predict = st.button("Predict")

# Load model only after it exists locally
if os.path.exists(local_path):
    classifier = pipeline(
        "text-classification",
        model=local_path,
        tokenizer=local_path,
        device=0 if torch.cuda.is_available() else -1
    )
else:
    classifier = None
    st.warning("Model not downloaded yet. Click 'Download Model' first.")

if predict:
    if classifier is None:
        st.error("⚠ Model not loaded! Download it first.")
    else:
        with st.spinner("Predicting..."):
            output = classifier(text)
            st.write(output)
            st.info(output)
