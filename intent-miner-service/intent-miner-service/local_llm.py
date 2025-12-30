# local_llm.py
from llama_cpp import Llama

MODEL_PATH = "models/mistral-7b-instruct-4096.tq2_0.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=6,  # ajusta según CPU
    n_batch=32,
    use_mlock=True
)

def infer_intent_from_context(messages: list[str]) -> str:
    prompt = (
        "Eres un clasificador de intención.\n"
        "Analiza el siguiente historial de mensajes y responde solo con el intent detectado en JSON:\n"
        '{"intent": "<valor>"}\n'
        "Historial de conversación:\n\n"
    )
    for msg in messages:
        prompt += f"- {msg}\n"

    prompt += "\nIntent detectado:"

    output = llm(prompt, stop=["}"], echo=False)
    return output["choices"][0]["text"] + "}"
