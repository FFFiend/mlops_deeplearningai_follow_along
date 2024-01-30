import io
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from abc import ABC, abstractmethod
from pydantic import BaseModel
import transformers

# host huggingface LLMs on a fastapi server 

app = FastAPI(title="skeleton deployment")

@app.get("/")
def home():
    return "Welcome to the skeleton fastapi server"
    
@app.get("/predict")
def prediction():
    pipe = transformers.pipeline(model="facebook/bart-large-mnli")
    output = pipe("What is the capital of China? Tell me more about it.")
    print(output)
    return output

uvicorn.run(app, host="127.0.0.1", port=3000, root_path="/serve")


