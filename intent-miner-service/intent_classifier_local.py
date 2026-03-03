from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, List, Literal
from llama_cpp import Llama
import os
import json
import re
from urllib import parse, request as urllib_request
from urllib.error import HTTPError, URLError

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


class SlotExtractRequest(BaseModel):
    messages: List[Message]
    vertical: str
    intent: str
    execute_action: bool = True
    timeout_seconds: int = 15

# ----------------------------
# Prompt builders
# ----------------------------
def load_vertical_config(vertical_name: str) -> dict:
    path = os.path.join(VERTICALS_DIR, f"{vertical_name}.json")
    if not os.path.exists(path):
        #print('respuesta del modelo', f"Vertical '{path}'")
        #print('respuesta del modelo', f"Vertical '{vertical_name}' no encontrada en {VERTICALS_DIR}")
        raise FileNotFoundError(f"Vertical '{vertical_name}' no encontrada en {VERTICALS_DIR}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Compatibilidad hacia atrás con formato legado
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
    print('Prompt con instrucciones y contexto', prompt)
    return prompt.strip()


def build_slot_prompt(messages: List[Message], action: dict) -> str:
    sorted_msgs = sorted(messages, key=lambda m: m.timestamp)[-6:]
    conversation = "\n".join([
        f"{'Usuario' if m.role == 'user' else 'Asistente'}: {m.content.strip()}"
        for m in sorted_msgs
    ])
    slot_schema = {
        "intent": action.get("intent"),
        "slots": action.get("slots", [])
    }
    return f"""### Sistema:
Eres un extractor de slots para un flujo conversacional.

Tu tarea es devolver EXCLUSIVAMENTE un JSON válido con los valores de los slots solicitados.
Si no existe evidencia suficiente para un slot, usa null.

Schema de extracción:
{json.dumps(slot_schema, ensure_ascii=False, indent=2)}

Devuelve este formato exacto:
{{
  "slots": {{
    "slot_name": "valor o null"
  }}
}}

No agregues explicaciones, markdown, ni texto adicional.

### Conversación:
{conversation}

### JSON:""".strip()

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


def extract_first_json_object(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No se encontró un JSON válido en la respuesta del modelo")
    return json.loads(match.group(0))


def resolve_templates(obj: Any, slots: dict) -> Any:
    if isinstance(obj, dict):
        resolved = {}
        for key, value in obj.items():
            if isinstance(value, str):
                placeholders = re.findall(r"\{([^{}]+)\}", value)
                if placeholders and any(slots.get(p) in (None, "") for p in placeholders):
                    continue
            resolved[key] = resolve_templates(value, slots)
        return resolved

    if isinstance(obj, list):
        return [resolve_templates(value, slots) for value in obj]

    if isinstance(obj, str):
        placeholders = re.findall(r"\{([^{}]+)\}", obj)
        value = obj
        for placeholder_name in placeholders:
            placeholder = "{" + placeholder_name + "}"
            replacement = slots.get(placeholder_name)
            value = value.replace(placeholder, "" if replacement is None else str(replacement))
        return value

    return obj


def build_action_execution(action_config: dict, slots: dict) -> dict:
    http_cfg = action_config.get("http", {})
    base_url = http_cfg.get("base_url", "").rstrip("/")
    endpoint = http_cfg.get("endpoint", "")
    url = f"{base_url}{endpoint}"

    action_payload = {
        "method": http_cfg.get("method", "GET"),
        "url": url,
    }

    query = resolve_templates(http_cfg.get("query_params", {}), slots)
    if query:
        action_payload["query"] = query

    body = resolve_templates(http_cfg.get("body", {}), slots)
    if body:
        action_payload["body"] = body

    headers = resolve_templates(http_cfg.get("headers", {}), slots)
    if headers:
        action_payload["headers"] = headers

    return action_payload


def execute_action_http(action_payload: dict, timeout_seconds: int = 15) -> dict:
    method = str(action_payload.get("method", "GET")).upper()
    url = action_payload.get("url")
    query = action_payload.get("query", {})
    headers = action_payload.get("headers", {})
    body = action_payload.get("body")

    if not url:
        raise ValueError("La acción no contiene URL válida")

    if query:
        encoded_query = parse.urlencode(query, doseq=True)
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{encoded_query}"

    request_data = None
    request_headers = dict(headers) if isinstance(headers, dict) else {}

    if body is not None:
        request_data = json.dumps(body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    req = urllib_request.Request(
        url=url,
        data=request_data,
        headers=request_headers,
        method=method,
    )

    try:
        with urllib_request.urlopen(req, timeout=timeout_seconds) as resp:
            response_raw = resp.read().decode("utf-8", errors="replace")
            response_content_type = resp.headers.get("Content-Type", "")
            try:
                response_body = json.loads(response_raw)
            except Exception:
                response_body = response_raw

            return {
                "status_code": resp.status,
                "url": url,
                "content_type": response_content_type,
                "body": response_body,
            }
    except HTTPError as e:
        error_raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        try:
            error_body = json.loads(error_raw)
        except Exception:
            error_body = error_raw
        return {
            "status_code": e.code,
            "url": url,
            "error": True,
            "body": error_body,
        }
    except URLError as e:
        raise HTTPException(status_code=502, detail=f"No se pudo ejecutar action HTTP: {str(e)}")
    
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


@app.post("/extract-slots")
def extract_slots(request: SlotExtractRequest):
    try:
        vertical_config = load_vertical_config(request.vertical)
        action = next((a for a in vertical_config.get("actions", []) if a.get("intent") == request.intent), None)
        if not action:
            raise HTTPException(
                status_code=404,
                detail=f"No existe acción configurada para intent '{request.intent}' en vertical '{request.vertical}'"
            )

        prompt = build_slot_prompt(request.messages, action)
        response = llm(
            prompt,
            max_tokens=256,
            stop=[],
            temperature=0.0,
            top_k=1,
            top_p=0.9
        )
        output_text = response["choices"][0]["text"].strip()
        parsed_json = extract_first_json_object(output_text)

        slot_definitions = action.get("slots", [])
        extracted_slots = parsed_json.get("slots", {}) if isinstance(parsed_json, dict) else {}

        slots = {
            slot["name"]: extracted_slots.get(slot["name"])
            for slot in slot_definitions
        }

        missing = [
            slot["name"]
            for slot in slot_definitions
            if slot.get("required") and slots.get(slot["name"]) in (None, "")
        ]

        action_execution = None
        action_response = None
        if not missing:
            action_execution = build_action_execution(action, slots)
            if request.execute_action:
                action_response = execute_action_http(action_execution, timeout_seconds=request.timeout_seconds)

        return {
            "intent": request.intent,
            "slots": slots,
            "missing": missing,
            "action": action_execution,
            "action_response": action_response,
        }

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extrayendo slots: {str(e)}")
