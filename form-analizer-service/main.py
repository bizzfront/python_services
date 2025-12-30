from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Literal
import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging
from datetime import datetime

# -----------------------------
# Configuración de logging
# -----------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = datetime.now().strftime("%Y-%m-%d_intents.log")
log_path = os.path.join(LOG_DIR, log_filename)
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -----------------------------
# Inicializa FastAPI
# -----------------------------
app = FastAPI()

# -----------------------------
# Modelo multilingüe
# -----------------------------
MODEL_NAME = "distiluse-base-multilingual-cased-v1"
model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# Rutas
# -----------------------------
INTENTS_DIR = "verticals"
INDEX_DIR = "data"
SIMILARITY_THRESHOLD = 0.70

# -----------------------------
# Esquema del request
# -----------------------------
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: int

class IntentRequest(BaseModel):
    vertical: str
    messages: List[Message]

# Cache de índices cargados
topic_indexes = {}

# -----------------------------
# Utilidades
# -----------------------------
def load_index(vertical: str):
    if vertical in topic_indexes:
        return topic_indexes[vertical]

    index_path = os.path.join(INDEX_DIR, f"intent_index_{vertical}.pkl")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No existe el índice entrenado para vertical: {vertical}")

    with open(index_path, "rb") as f:
        index = pickle.load(f)

    topic_indexes[vertical] = index
    return index

# -----------------------------
# Endpoint principal
# -----------------------------
@app.post("/detect-intent")
def detect_intent(body: IntentRequest, request: Request):
    try:
        index = load_index(body.vertical)
    except FileNotFoundError as e:
        logging.error(f"[{body.vertical}] {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.exception(f"Error cargando índice: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

    try:
        sorted_msgs = sorted(body.messages, key=lambda m: m.timestamp)

        full_prompt = ""
        last_user_msg = None
        for msg in sorted_msgs:
            prefix = "Usuario:" if msg.role == "user" else "Asistente:"
            full_prompt += f"{prefix} {msg.content.strip()}\n"
            if msg.role == "user":
                last_user_msg = msg.content.strip()

        if not last_user_msg:
            raise HTTPException(status_code=422, detail="No se encontró ningún mensaje del usuario para evaluar")

        combined_text = full_prompt.strip()
        query_vec = model.encode([combined_text])[0].reshape(1, -1)

        best_intent = "none"
        best_score = 0.0

        for intent, vectors in index.items():
            sims = cosine_similarity(query_vec, vectors)
            score = float(np.max(sims))
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_score < SIMILARITY_THRESHOLD:
            best_intent = "none"

        log_data = {
            "ip": request.client.host,
            "vertical": body.vertical,
            "context": full_prompt.strip(),
            "intent": best_intent,
            "score": round(best_score, 4)
        }
        logging.info(json.dumps(log_data, ensure_ascii=False))

        return {"intent": best_intent, "score": round(best_score, 4)}

    except Exception as e:
        logging.exception(f"Error durante clasificación: {str(e)}")
        raise HTTPException(status_code=500, detail="Error procesando el mensaje")

# -----------------------------
# Ejecutar con Uvicorn desde código (opcional)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=4002, reload=True)
