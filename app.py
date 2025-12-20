from Scripts.data_model import NLPDataInput,NLPDataOutput,ImageDataInput,ImageDataOutput 
from Scripts.s3 import download_dir
from fastapi import FastAPI
from fastapi import Request
import uvicorn
import os
import torch
from transformers import pipeline
import time



app = FastAPI()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
##### Download ML MOdels #####
force_download = True


model_name = 'tinybert-sentiment-analysis/'
local_path = 'ml-models/'+model_name
if not os.path.isdir(local_path) or force_download :
    download_dir(local_path, model_name)

sentiment_model = pipeline('text-classification', model = local_path, device=device)

##### Download ML MOdels  end #####


@app.get("/")
def read_root():
    return "Hello I am up!!"

@app.post("/api/v1/sentiment_analysis")
def sentiment_analysis(data:NLPDataInput):
    start = time.time() 
    output = sentiment_model(data.text)
    end = time.time()
    prediction_time = (end-start)*1000
    
    labels = [x['label'] for x in output]
    scores = [x['score']  for x in output]

    output = NLPDataOutput(model_name ="tinybert-sentiment-analysis",
                           text = data.text,
                           labels = labels,
                           scores = scores,
                           prediction_time = prediction_time
                            )
    return output



if __name__=="__main__":
    uvicorn.run(app="app:app", port=8000, reload=True)
