import os
import json
import pickle
from sentence_transformers import SentenceTransformer
import argparse

# Configuración
INTENTS_DIR = "verticals"
INDEX_DIR = "data"
MODEL_NAME = "distiluse-base-multilingual-cased-v1"
model = SentenceTransformer(MODEL_NAME)

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--vertical", required=True, help="Nombre del vertical")
args = parser.parse_args()

vertical = args.vertical
data_path = os.path.join(INTENTS_DIR, f"{vertical}.json")
output_path = os.path.join(INDEX_DIR, f"intent_index_{vertical}.pkl")

if not os.path.exists(data_path):
    raise Exception(f"No existe archivo de intents para: {vertical}")

with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Compatibilidad hacia atrás con formato legacy
intent_examples = raw_data.get("intents", raw_data)

vector_index = {}
for intent, examples in intent_examples.items():
    if not isinstance(examples, list):
        raise ValueError(
            f"El intent '{intent}' debe contener una lista de ejemplos para entrenar el índice"
        )
    vectors = model.encode(examples)
    vector_index[intent] = vectors

os.makedirs(INDEX_DIR, exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(vector_index, f)

print(f"Índice generado para '{vertical}' en {output_path}")
