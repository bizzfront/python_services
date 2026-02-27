# intent-miner-service

Servicio FastAPI para detección de intención conversacional.

## Entorno con `uv`

### 1) Sincronizar dependencias

```bash
uv sync
```

### 2) Ejecutar servicio principal

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 4002
```

### 3) Generar índice por vertical

```bash
uv run python train_index.py --vertical bizzfront
```

## Notas

- Ya no es necesario crear un `venv` manualmente con `python -m venv`.
- `uv` administra el entorno virtual y el lockfile automáticamente.
