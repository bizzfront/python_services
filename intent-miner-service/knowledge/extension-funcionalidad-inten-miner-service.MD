📄 01 — SPEC: Action Engine + Slot Extraction
Objetivo

Extender intent-miner-service para soportar:

/detect-intent (sin cambios)

/extract-slots (nuevo)

Soporte para verticals con:

intents

actions

configuración HTTP declarativa

El microservicio NO ejecuta llamadas externas.
Solo:

Clasifica

Extrae

Construye acción lista para ejecutar

📦 Nueva estructura oficial de vertical
{
  "intents": {
    "meds_check_availability": "cuando el usuario consulta disponibilidad de medicamentos",
    "none": "..."
  },
  "actions": [
    {
      "key": "meds_check_availability",
      "intent": "meds_check_availability",

      "slots": [
        { "name": "medicamento", "type": "string", "required": true },
        { "name": "presentacion", "type": "string", "required": false },
        { "name": "sede", "type": "string", "required": false }
      ],

      "http": {
        "method": "GET",
        "base_url": "http://localhost:3000",
        "endpoint": "/v1/fundase",
        "query_params": {
          "module": "medicinas",
          "col": "Medicamento",
          "like": "{medicamento}"
        }
      }
    }
  ]
}
📄 02 — Nuevo Endpoint: /extract-slots
Request
{
  "vertical": "asst_ghAPefvC6KwGzqhMvb3tvquX",
  "intent": "meds_check_availability",
  "messages": [...]
}
Response
{
  "intent": "meds_check_availability",
  "slots": {
    "medicamento": "amoxicilina",
    "presentacion": null,
    "sede": null
  },
  "missing": [],
  "action": {
    "method": "GET",
    "url": "http://localhost:3000/v1/fundase",
    "query": {
      "module": "medicinas",
      "col": "Medicamento",
      "like": "amoxicilina"
    }
  }
}
🧠 03 — Motor de Resolución de Variables
Objetivo

Reemplazar dinámicamente:

"{medicamento}" → "amoxicilina"
Reglas

Reemplazar en:

query_params

body

Omitir variables null no requeridas

Si slot requerido es null → NO construir action

Función requerida (conceptual)
def resolve_templates(obj, slots):
    if isinstance(obj, dict):
        return {
            k: resolve_templates(v, slots)
            for k, v in obj.items()
            if not (isinstance(v, str) and is_slot_null(v, slots))
        }

    if isinstance(obj, list):
        return [resolve_templates(v, slots) for v in obj]

    if isinstance(obj, str):
        for key, value in slots.items():
            placeholder = f"{{{key}}}"
            if placeholder in obj:
                return obj.replace(placeholder, value or "")
        return obj

    return obj
🔧 04 — Cambios en intent_classifier_local.py
1️⃣ Ajustar lectura de vertical

Antes:

build_intent_block(vertical_dict)

Ahora:

build_intent_block(vertical_dict["intents"])
2️⃣ Crear modelo nuevo
class SlotExtractRequest(BaseModel):
    messages: List[Message]
    vertical: str
    intent: str
3️⃣ Implementar endpoint
@app.post("/extract-slots")
def extract_slots(request: SlotExtractRequest):
4️⃣ Lógica interna

Cargar vertical

Buscar action:

action = next(a for a in vertical["actions"] if a["intent"] == request.intent)

Construir prompt con schema

Ejecutar modelo GGUF

Extraer JSON

Validar required

Construir action execution

Devolver estructura final

🧩 05 — Integración en index.js (Orquestador)
Flujo actualizado
// 1. Detectar intent
const intentRes = await axios.post(INTENT_URL + "/detect-intent", {
  vertical,
  messages
});

const intent = intentRes.data.intent;

// 2. Verificar si tiene action
if (intent !== "none") {

  const slotsRes = await axios.post(INTENT_URL + "/extract-slots", {
    vertical,
    intent,
    messages
  });

  const { slots, missing, action } = slotsRes.data;

  if (missing.length > 0) {
    return askForMissing(missing);
  }

  // 3. Ejecutar acción externa
  const apiResponse = await axios({
    method: action.method,
    url: action.url,
    params: action.query || undefined,
    data: action.body || undefined
  });

  return renderTemplate(apiResponse.data);
}
📊 06 — Flujo Arquitectónico Final
Usuario
   ↓
index.js (orquestador)
   ↓
intent-miner-service
   ├── /detect-intent
   └── /extract-slots
         ↓
    devuelve action + slots
   ↓
index.js ejecuta API externa
   ↓
Respuesta plantilla
   ↓
Usuario
🛡️ 07 — Separación de Responsabilidades
Servicio	Responsabilidad
intent-miner	Clasificación + extracción + construcción de action
Orquestador	Ejecución HTTP + manejo de errores + plantilla
API externa	Fuente de datos

Arquitectura limpia.
No acoplamiento.
Escalable.

🧪 08 — Plan de Pruebas
Caso 1 — Sin medicamento

Input:

"¿Tienen disponible?"

Resultado:

missing = ["medicamento"]

no action

Caso 2 — Con medicamento

Input:

"¿Tienen amoxicilina?"

Resultado:

slots = { medicamento: "amoxicilina" }

action construida correctamente

Orquestador ejecuta

Respuesta plantilla

Caso 3 — Slot opcional

Input:

"¿Hay amoxicilina 500mg en sede centro?"

Resultado:

slots = {
medicamento,
presentacion,
sede
}

🚀 Resultado Final

Tu sistema ahora:

Es declarativo

Es configurable por empresa

No depende de hardcoding

No usa LLM para redactar acciones deterministas

Soporta GET y POST dinámicamente

Es enterprise-ready