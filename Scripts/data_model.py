#acccept input params/data payload
# output the score, class and latency

from pydantic import BaseModel
from typing import List
from pydantic import EmailStr


class NLPDataInput(BaseModel):
    text:List[str]
    user_id:EmailStr

class ImageDataInput(BaseModel):
    url : List[str]
    user_id : EmailStr

class NLPDataOutput(BaseModel):
    model_name:str
    text:list[str]
    labels:list[str]
    scores:list[float]
    prediction_time:float

class ImageDataOutput(BaseModel):
    model_name:str
    url:list[str]
    target:list[str]
    score:list[float]
    prediction_time:int