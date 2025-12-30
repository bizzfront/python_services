from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from llama_cpp import Llama
import os
import json

# ----------------------------
# Configuración del modelo GGUF
# ----------------------------
MODEL_PATH = "models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
VERTICALS_DIR = "verticals"

# Instancia global del LLM (se carga una única vez)
llm: Optional[Llama] = None

def get_llm() -> Llama:
    global llm
    if llm is None:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_gpu_layers=64,
            use_mlock=False,
            n_threads=2,
            n_batch=64
        )
    return llm

# ----------------------------
# Modelos de datos
# ----------------------------
class FormAnalysisRequest(BaseModel):
    vertical: str
    questions: List[str]
    answers: List[str]
    # Nuevo campo opcional de contexto: puede ser dict o string JSON
    context: Optional[Union[Dict[str, Any], str]] = None

# ----------------------------
# Utilidades
# ----------------------------

def load_vertical_instruction(vertical_name: str) -> str:
    """Carga la instrucción del sistema para la vertical dada."""
    path = os.path.join(VERTICALS_DIR, f"{vertical_name}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Instrucción de vertical '{vertical_name}' no encontrada en {VERTICALS_DIR}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def serialize_context(ctx: Dict[str, Any]) -> str:
    """
    Serializa el contexto como un string breve: "clave: valor; clave2: valor2; ..."
    """
    return "; ".join(f"{k}: {v}" for k, v in ctx.items())


def parse_context(ctx: Union[Dict[str, Any], str]) -> Optional[Dict[str, Any]]:
    """
    Convierte el contexto a dict si viene como string JSON, o devuelve el dict si ya lo es.
    """
    if ctx is None:
        return None
    if isinstance(ctx, dict):
        return ctx
    try:
        parsed = json.loads(ctx)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    raise ValueError("El campo 'context' debe ser un JSON válido o un objeto dict")


def build_analysis_prompt(
    instruction: str,
    questions: List[str],
    answers: List[str],
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Construye el prompt completo a enviar al modelo, incluyendo preguntas, respuestas e
    instrucciones de sistema. Añade datos adicionales si se proporciona contexto.
    """
    bloques = []
    for i, (q, a) in enumerate(zip(questions, answers)):
        bloques.append(f"{i+1}. {q}\nRespuesta: {a}")
    cuerpo = "\n".join(bloques)

    prompt = f"""
    ### INSTRUCCIÓN DEL SISTEMA:
    {instruction}

    ### RESPUESTAS AL FORMULARIO:
    {cuerpo}

    Por favor, genera un análisis de negocio que incluya:
    - Puntos fuertes.
    - Áreas de mejora.
    - Recomendaciones.
    - Clasificación del nivel de transformación digital.
    """

    if context:
        extra = serialize_context(context)
        if len(extra) > 200:
            extra = extra[:200] + "…"
        prompt += f"\n\n### DATOS ADICIONALES DEL CONTEXTO: {extra}"

    return prompt

# ----------------------------
# Inicializar FastAPI
# ----------------------------
app = FastAPI()

@app.post("/analyze-form")
def analyze_form(request: FormAnalysisRequest):
    print("===== REQUEST =====")
    print(request)
    try:
        # Carga instrucción de la vertical
        instruction = load_vertical_instruction(request.vertical)

        # Parsear context si viene como string
        parsed_context = None
        if request.context is not None:
            parsed_context = parse_context(request.context)

        # Construye el prompt (incluyendo contexto opcional)
        prompt = build_analysis_prompt(
            instruction,
            request.questions,
            request.answers,
            parsed_context
        )

        # Ejecuta la inferencia
        llm_instance = get_llm()
        response = llm_instance(
            prompt=prompt,
            max_tokens=512,
            temperature=0.2,
            top_p=0.95,
        )

        analysis = response["choices"][0]["text"].strip()

        print("===== PROMPT ENVIADO AL MODELO =====")
        print(prompt)
        print("====================================")
        print("Análisis generado:", analysis)

        return {"analysis": analysis}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando análisis: {str(e)}")
