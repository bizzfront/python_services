@echo off
REM — Cambia a la carpeta de tu proyecto
cd /d C:\intent-miner-service

REM — Sincroniza dependencias con uv (crea/actualiza .venv automáticamente)
uv sync
if errorlevel 1 exit /b %errorlevel%

REM — Ejecuta uvicorn con uv
uv run uvicorn main:app --host 0.0.0.0 --port 4002
