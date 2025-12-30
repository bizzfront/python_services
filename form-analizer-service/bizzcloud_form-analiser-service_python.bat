@echo off
REM — Cambia a la carpeta de tu proyecto
cd /d C:\form-analizer-service

REM — Activa el entorno virtual
call venv\Scripts\activate

REM — Ejecuta uvicorn
uvicorn app:app --host 0.0.0.0 --port 4003
