import streamlit as st
import boto3
import os
import torch
from transformers import pipeline

bucket_name = 'mlopslearn-s3bucket'
local_path = 'tinybert-sentiment-analysis'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'

# S3 client
s3 = boto3.client('s3', region_name="us-east-1")

# --------- DOWNLOAD DIRECTORY FROM S3 ----------
def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')

    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']

                # Relative file path inside local directory
                relative_path = os.path.relpath(s3_key, s3_prefix)
                local_file = os.path.join(local_path, relative_path)

                # Create subfolders if needed
                os.makedirs(os.path.dirname(local_file), exist_ok=True)

                # Download file
                s3.download_file(bucket_name, s3_key, local_file)


# --------- STREAMLIT UI ----------
st.title("Machine Learning Model Deployment on Cloud")

# Button: Download model
if st.button("Download Model"):
    with st.spinner("Downloading model files from S3..."):
        download_dir(local_path, s3_prefix)
    st.success("Model downloaded successfully!")

text = st.text_area("Enter your text here", "Type here...")

# --------- LOAD MODEL (CPU safe) ----------
classifier = None
if os.path.exists(local_path):
    try:
        classifier = pipeline(
            "text-classification",
            model=local_path,
            tokenizer=local_path,
            device=-1    # Force CPU (Streamlit Cloud has no GPU)
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")


# --------- INFERENCE ----------
if st.button("Predict"):
    if classifier is None:
        st.error("⚠ Model is not loaded. Please download it first.")
    else:
        with st.spinner("Predicting..."):
            try:
                # Disable gradients for safety
                with torch.no_grad():
                    output = classifier(text)

                st.success("Prediction complete!")
                st.write(output)

            except Exception as e:
                st.error(f"Prediction error: {e}")
