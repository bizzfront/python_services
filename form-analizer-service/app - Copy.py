from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from llama_cpp import Llama
import os

# ----------------------------
# Configuraci¨®n del modelo GGUF
# ----------------------------
MODEL_PATH = "models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
VERTICALS_DIR = "verticals"

def load_local_model():
    return Llama(
        model_path=MODEL_PATH,
        #n_ctx=32768,
        n_ctx=4096,
        n_threads=2,
        n_batch=64,
        use_mlock=False
    )

llm = load_local_model()

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI()

# ----------------------------
# Esquema del request
# ----------------------------
class FormAnalysisRequest(BaseModel):
    questions: List[str]
    answers: List[str]
    vertical: str

# ----------------------------
# Carga de instrucciones desde archivo .txt
# ----------------------------
def load_vertical_instruction(vertical_name: str) -> str:
    path = os.path.join(VERTICALS_DIR, f"{vertical_name}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Instrucción de vertical '{vertical_name}' no encontrada en {VERTICALS_DIR}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# ----------------------------
# Constructor del prompt
# ----------------------------
def build_analysis_prompt(instruction: str, questions: List[str], answers: List[str]) -> str:
    bloques = []
    for i, (q, a) in enumerate(zip(questions, answers)):
        bloque = f"{i+1}. {q}\nRespuesta: {a}"
        bloques.append(bloque)
    cuerpo = "\n".join(bloques)

    return f"""
### INSTRUCCIÓN DEL SISTEMA:
{instruction}

Analiza las siguientes respuestas de un negocio y genera un informe profesional que incluya:

- Descripción general del negocio.
- Puntos fuertes observados.
- Áreas de mejora o debilidades.
- Recomendaciones prácticas para mejorar.
- Y una clasificación final del nivel de transformación digital: Bajo, Intermedio o Avanzado.

### RESPUESTAS:
{cuerpo}

### ANÁLISIS:
""".strip()



# ----------------------------
# Endpoint principal
# ----------------------------
@app.post("/analyze-form")
def analyze_form(request: FormAnalysisRequest):
    global llm
    if llm is None:
        llm = load_local_model()

    try:
        # 🔎 Mostrar datos recibidos
        print("=== DATOS RECIBIDOS ===")
        print("Vertical:", request.vertical)
        print("Questions:", request.questions)
        print("Answers:", request.answers)
        
        instruction = load_vertical_instruction(request.vertical)
        prompt = build_analysis_prompt(
            instruction,
            request.questions,
            request.answers
        )

        response = llm(
            prompt,
            max_tokens=512,
            stop=[],
            temperature=0.3,
            top_k=40,
            top_p=0.95
        )

        analysis = response["choices"][0]["text"].strip()
        print("===== PROMPT ENVIADO AL MODELO =====")
        print(prompt)
        print("====================================")
        print('Análisis generado:', analysis)

        return {"analysis": analysis}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando análisis: {str(e)}")