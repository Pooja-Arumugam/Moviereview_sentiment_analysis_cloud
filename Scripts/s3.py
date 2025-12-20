import boto3
import os


bucket_name = 'mlopslearn-s3bucket'


# S3 client
s3 = boto3.client('s3', region_name="us-east-1")

# --------- DOWNLOAD DIRECTORY FROM S3 ----------
def download_dir(local_path:str, model_name:str):
    s3_prefix = 'ml-models/'+ model_name
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