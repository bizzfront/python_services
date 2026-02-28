from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal
from llama_cpp import Llama
import os
import json

# ----------------------------
# Configuración del modelo GGUF
# ----------------------------
MODEL_PATH = "models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
VERTICALS_DIR = "verticals"

def load_local_model():
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=4,
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
def load_vertical_config(vertical_name: str) -> dict:
    path = os.path.join(VERTICALS_DIR, f"{vertical_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vertical '{vertical_name}' no encontrada en {VERTICALS_DIR}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Compatibilidad hacia atrás con el formato legacy
    if "intents" not in data:
        return {"intents": data, "actions": []}

    data.setdefault("actions", [])
    return data

def build_intent_block(intents_dict: dict) -> str:
    return "\n".join([f"- {k}: {v}" for k, v in intents_dict.items()])

def build_prompt(messages: List[Message], vertical_config: dict) -> str:
    intent_definitions = build_intent_block(vertical_config["intents"])
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
    return prompt.strip()

# ----------------------------
# Función para extraer el intent limpio
# ----------------------------
def extract_clean_intent(raw_text: str) -> str:
    cleaned = raw_text.strip().lower()
    for tag in ["[inst]", "[/inst]", "[respuesta]", "[/respuesta]"]:
        cleaned = cleaned.replace(tag, "")
    return cleaned.split()[0].strip(" .,:\"'")

# ----------------------------
# Endpoint principal
# ----------------------------
@app.post("/detect-intent")
def detect_intent(request: IntentRequest):
    try:
        vertical_config = load_vertical_config(request.vertical)
        prompt = build_prompt(request.messages, vertical_config)

        response = llm(
            prompt,
            max_tokens=32,
            stop=["\n"],
            temperature=0.0,
            top_k=1,
            top_p=0.9
        )
        output_text = response["choices"][0]["text"].strip()
        intent = extract_clean_intent(output_text)

        if not intent:
            raise ValueError("El modelo no devolvió un intent válido")

        return {"intent": intent}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Respuesta inválida del modelo: {str(e)}")
