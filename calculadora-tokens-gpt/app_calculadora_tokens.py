"""
Servicio REST - Calculadora de Tokens por Conversación
Autor: Saúl Camacho
Versión: 4.0

Nuevas características:
-----------------------
✅ Endpoint /use-cases → casos de uso predefinidos con tokens promedio.
✅ Tipificación por nivel de complejidad (1-6).
✅ Compatibilidad total con cálculo de costos y modelos preconfigurados.
✅ Puerto personalizable por variable de entorno (APP_PORT).
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn


# --- Datos base de modelos GPT (precios por 1K tokens + descripción) ---
MODELOS_PRECONFIGURADOS: Dict[str, Dict[str, float | str]] = {
    # GPT-3.5 Series
    "gpt-3.5-turbo": {
        "precio_input": 0.001,   # $1.00 por 1M tokens → $0.001 por 1K
        "precio_output": 0.002,  # $2.00 por 1M tokens → $0.002 por 1K
        "descripcion": "Modelo económico y rápido ideal para chatbots básicos, atención al cliente y tareas de texto simples. Excelente balance entre costo y rendimiento."
    },
    "gpt-3.5-turbo-1106": {
        "precio_input": 0.001,
        "precio_output": 0.002,
        "descripcion": "Versión mejorada de GPT-3.5 con mayor coherencia en conversaciones largas y mejor compresión contextual. Ideal para soporte automatizado y flujos con alta rotación de mensajes."
    },

    # GPT-4 Turbo / Preview
    "gpt-4-turbo-2025-04-09": {
        "precio_input": 0.010,   # $10.00 por 1M tokens → $0.010 por 1K
        "precio_output": 0.030,  # $30.00 por 1M tokens → $0.030 por 1K
        "descripcion": "Versión turbo de GPT-4 optimizada para rendimiento y costo. Ideal para asistentes empresariales complejos, análisis de texto avanzado y generación de contenido con contexto extenso."
    },
    "gpt-4-0125-preview": {
        "precio_input": 0.010,
        "precio_output": 0.030,
        "descripcion": "Versión preview de GPT-4 con mejoras en razonamiento lógico y consistencia de salida. Recomendado para prototipos y proyectos que necesiten capacidad avanzada sin perder velocidad."
    },
    "gpt-4-0613": {
        "precio_input": 0.030,   # $30.00 por 1M tokens → $0.030 por 1K
        "precio_output": 0.060,  # $60.00 por 1M tokens → $0.060 por 1K
        "descripcion": "Modelo GPT-4 estable de mediados de 2023, reconocido por su precisión y calidad de texto. Ideal para análisis técnicos, redacción de alto nivel y soporte especializado."
    },

    # GPT-4 Omni Series
    "gpt-4o": {
        "precio_input": 0.005,   # $5.00 por 1M tokens → $0.005 por 1K
        "precio_output": 0.015,  # $15.00 por 1M tokens → $0.015 por 1K
        "descripcion": "GPT-4 Omni (Omni-directional): modelo multimodal rápido y eficiente, con comprensión avanzada de texto, imágenes y datos. Ideal para proyectos corporativos y analíticos."
    },
    "gpt-4o-mini": {
        "precio_input": 0.00015, # $0.15 por 1M tokens → $0.00015 por 1K
        "precio_output": 0.00060,# $0.60 por 1M tokens → $0.00060 por 1K
        "descripcion": "Versión ligera de GPT-4 Omni, optimizada para grandes volúmenes de interacción a bajo costo. Perfecta para chatbots, soporte masivo y automatización de tareas repetitivas."
    }
}



# --- Casos de uso predefinidos (tokens estimados) ---
USE_CASES: List[Dict] = [
    {
        "id": "atencion_cliente",
        "nombre": "Atención al cliente",
        "descripcion": "Responde consultas frecuentes, solicitudes simples o información de productos/servicios.",
        "tokens_inbound": 120,
        "tokens_outbound": 180,
        "sesiones_promedio": 4,
        "nivel": 1
    },
    {
        "id": "soporte_basico",
        "nombre": "Atender tickets de soporte al cliente",
        "descripcion": "Maneja reportes de incidencias simples, actualizaciones de estado y respuestas estándar.",
        "tokens_inbound": 180,
        "tokens_outbound": 250,
        "sesiones_promedio": 5,
        "nivel": 2
    },
    {
        "id": "soporte_tecnico_medio",
        "nombre": "Soporte técnico medio",
        "descripcion": "Resuelve problemas intermedios con procedimientos técnicos definidos.",
        "tokens_inbound": 250,
        "tokens_outbound": 350,
        "sesiones_promedio": 6,
        "nivel": 3
    },
    {
        "id": "soporte_tecnico_especializado",
        "nombre": "Soporte técnico especializado",
        "descripcion": "Analiza y soluciona incidencias complejas con razonamiento y contexto técnico detallado.",
        "tokens_inbound": 350,
        "tokens_outbound": 500,
        "sesiones_promedio": 8,
        "nivel": 4
    },
    {
        "id": "multiproposito",
        "nombre": "Multipropósito (soporte + atención al cliente)",
        "descripcion": "Combina atención al cliente con soporte técnico, capaz de alternar entre distintos tipos de tareas.",
        "tokens_inbound": 400,
        "tokens_outbound": 600,
        "sesiones_promedio": 7,
        "nivel": 5
    },
    {
        "id": "asistente_empresarial",
        "nombre": "Asistente empresarial integrado",
        "descripcion": "Integrado con sistemas internos (CRM, ERP, etc.), maneja múltiples fuentes de datos y consultas complejas.",
        "tokens_inbound": 600,
        "tokens_outbound": 900,
        "sesiones_promedio": 10,
        "nivel": 6
    }
]


# --- Modelos de entrada ---
class ParametrosConversacionInput(BaseModel):
    tokens_inbound: int
    tokens_outbound: int
    sesiones_promedio: int
    conversaciones_mensuales: int
    usa_base_conocimiento: Optional[bool] = False


from fastapi.middleware.cors import CORSMiddleware

# --- Inicialización del servicio ---
app = FastAPI(
    title="Calculadora de Tokens GPT",
    description="API REST para estimar costos mensuales de agentes GPT según modelo y caso de uso.",
    version="4.0"
)

# --- Habilitar CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes reemplazar "*" por ["http://127.0.0.1:5500"] o tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Lógica de cálculo ---
def calcular_costos(modelo_nombre: str, params: ParametrosConversacionInput):
    if modelo_nombre not in MODELOS_PRECONFIGURADOS:
        raise HTTPException(status_code=404, detail=f"Modelo '{modelo_nombre}' no encontrado.")

    precios = MODELOS_PRECONFIGURADOS[modelo_nombre]

    tokens_outbound_ajustado = params.tokens_outbound * (1.2 if params.usa_base_conocimiento else 1.0)
    tokens_totales_sesion = params.tokens_inbound + tokens_outbound_ajustado

    costo_sesion = (
        (params.tokens_inbound / 1000) * precios["precio_input"] +
        (tokens_outbound_ajustado / 1000) * precios["precio_output"]
    )

    costo_conversacion = costo_sesion * params.sesiones_promedio
    costo_mensual = costo_conversacion * params.conversaciones_mensuales

    return {
        "modelo": modelo_nombre,
        "usa_base_conocimiento": params.usa_base_conocimiento,
        "tokens_inbound": params.tokens_inbound,
        "tokens_outbound_ajustado": round(tokens_outbound_ajustado, 2),
        "tokens_totales_sesion": round(tokens_totales_sesion, 2),
        "costo_sesion_usd": round(costo_sesion, 8),
        "costo_conversacion_usd": round(costo_conversacion, 8),
        "costo_mensual_usd": round(costo_mensual, 2)
    }


# --- Endpoints REST ---

@app.get("/models")
def listar_modelos():
    """Devuelve los modelos GPT preconfigurados con sus precios."""
    return {"status": "success", "modelos_disponibles": MODELOS_PRECONFIGURADOS}


@app.get("/use-cases")
def listar_casos_de_uso():
    """Devuelve la lista de escenarios de uso predefinidos con sus valores de tokens promedio."""
    return {"status": "success", "casos_de_uso": USE_CASES}


@app.post("/calculate-cost")
def calcular_costo(modelo: str, parametros: ParametrosConversacionInput):
    """Calcula el costo mensual estimado para el modelo y parámetros proporcionados."""
    resultado = calcular_costos(modelo, parametros)
    return {"status": "success", "data": resultado}


# --- Ejecución directa ---
if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", 5005))
    print(f"🚀 Iniciando Calculadora de Tokens GPT en puerto {port} ...")
    uvicorn.run("app_calculadora_tokens:app", host="0.0.0.0", port=port, reload=True)


