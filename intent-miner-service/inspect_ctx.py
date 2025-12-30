from llama_cpp import Llama

MODEL_PATH = "models/phi-2.Q4_K_M.gguf"

llm = Llama(model_path=MODEL_PATH)

print("✅ Modelo cargado correctamente")
print(f"🧠 Tamaño máximo de contexto soportado: {llm.context_params.n_ctx} tokens")