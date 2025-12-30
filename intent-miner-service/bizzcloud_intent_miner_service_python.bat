@echo off
REM — Cambia a la carpeta de tu proyecto
cd /d C:\intent-miner-service

REM — Activa el entorno virtual
call venv\Scripts\activate

REM — Ejecuta uvicorn
uvicorn intent_classifier_local:app --host 0.0.0.0 --port 4002
