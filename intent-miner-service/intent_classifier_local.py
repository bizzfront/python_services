from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal
from llama_cpp import Llama
import os
import json
import re

# ----------------------------
# Configuración del modelo GGUF
# ----------------------------
MODEL_PATH = "models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
VERTICALS_DIR = "verticals"

def load_local_model():
    return Llama(
        model_path=MODEL_PATH,
        #n_ctx=32768,          # 🟢 importante: explícitamente forzamos 32k
        n_ctx=9000,
        n_threads=2,   # ajusta según tu CPU
        n_batch=64,
        use_mlock=True
    )

llm = load_local_model()

# ----------------------------
# Inicializa FastAPI
# ----------------------------
app = FastAPI()

# ----------------------------
# Esquema del request
# ----------------------------
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: int

class IntentRequest(BaseModel):
    messages: List[Message]
    vertical: str

# ----------------------------
# Prompt builders
# ----------------------------
def load_vertical_definitions(vertical_name: str) -> dict:
    path = os.path.join(VERTICALS_DIR, f"{vertical_name}.json")
    if not os.path.exists(path):
        #print('respuesta del modelo', f"Vertical '{path}'")
        #print('respuesta del modelo', f"Vertical '{vertical_name}' no encontrada en {VERTICALS_DIR}")
        raise FileNotFoundError(f"Vertical '{vertical_name}' no encontrada en {VERTICALS_DIR}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_intent_block(vertical_dict: dict) -> str:
    return "\n".join([f"- {k}: {v}" for k, v in vertical_dict.items()])

def build_prompt(messages: List[Message], vertical_definitions: dict) -> str:
    intent_definitions = build_intent_block(vertical_definitions)

    sorted_msgs = sorted(messages, key=lambda m: m.timestamp)[-4:]
    conversation = "\n".join([
        f"{'Usuario' if m.role == 'user' else 'Asistente'}: {m.content.strip()}"
        for m in sorted_msgs
    ])

    prompt = f"""### Sistema:
Eres un clasificador de intención conversacional.

Tu única tarea es analizar la conversación y devolver únicamente el nombre del intent correspondiente al último mensaje del usuario.

Usa estas definiciones de referencia:
[intents]
{intent_definitions}
[/intents]

Responde SOLO con el nombre del intent. Sin comillas, sin etiquetas, sin explicaciones.

### Conversación:
{conversation}

### Intención:"""
    print('Prompt con instrucciones y contexto', prompt)
    return prompt.strip()

def extract_clean_intent(raw_text: str) -> str:
    """
    Limpia la salida del modelo y extrae solo el intent.
    Soporta respuestas entre [RESPUESTA]...[/RESPUESTA] o [INST]...[/INST] o texto plano.
    """
    # 1. Busca [RESPUESTA]...[/RESPUESTA]
    match = re.search(r"\[RESPUESTA\](.*?)\[/RESPUESTA\]", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Busca [INST]...[/INST]
    match = re.search(r"\[INST\](.*?)\[/INST\]", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. Texto plano, elimina saltos, comillas, tags sueltos
    return raw_text.strip().strip("\"'").replace("\n", "")
    
# ----------------------------
# Endpoint principal
# ----------------------------
@app.post("/detect-intent")
def detect_intent(request: IntentRequest):
    try:
        vertical_definitions = load_vertical_definitions(request.vertical)
        prompt = build_prompt(request.messages, vertical_definitions)

        response = llm(
            prompt,
            max_tokens=100,
            stop=[],
            temperature=0.0,
            top_k=1,
            top_p=0.9
        )

        output_text = response["choices"][0]["text"].strip()
        
        print('respuesta del modelo', output_text)

        if not output_text:
            raise ValueError("Respuesta no es un identificador válido")

        return {"intent": extract_clean_intent(output_text)}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Respuesta inválida del modelo: {str(e)}")
