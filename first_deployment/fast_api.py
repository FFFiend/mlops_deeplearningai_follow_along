import io
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from abc import ABC, abstractmethod
from pydantic import BaseModel
import transformers
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

# host huggingface LLMs on a fastapi server 

app = FastAPI(title="skeleton deployment")

@app.get("/")
def home():
    return "Welcome to the skeleton fastapi server"
    
@app.get("/predict/{model_id}/{prompt}")
def prediction(prompt: str, model_id: str):
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    output = pipe(prompt)
    return output

uvicorn.run(app, host="127.0.0.1", port=3000, root_path="/serve")



