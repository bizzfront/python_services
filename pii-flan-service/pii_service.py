from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import json

app = FastAPI()

# Carga modelo Flan-T5 Small
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class TextRequest(BaseModel):
    text: str

def extract_pii(text: str) -> dict:
    prompt = (
        "Extrae del siguiente texto cualquier dato personal o empresarial que encuentres y devuélvelo en formato JSON.\n"
        "Campos posibles a detectar: name, email, phone, business_name, address, document_id, business_id.\n"
        "Si no se encuentra ningún dato, devuelve un JSON vacío: {}\n\n"
        f"Texto: {text}\n\nJSON:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(inputs, max_length=256)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        result = json.loads(decoded)
    except json.JSONDecodeError:
        result = {}
    return result

@app.post("/detect")
async def detect_pii(body: TextRequest):
    result = extract_pii(body.text)
    return result


