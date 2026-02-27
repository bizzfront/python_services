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

## Despliegue en Linux AlmaLinux

### 1) Instalar dependencias del sistema

```bash
sudo dnf update -y
sudo dnf install -y curl git
```

### 2) Instalar `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

> Si `uv` no queda disponible en nuevas sesiones, agrega `export PATH="$HOME/.local/bin:$PATH"` a tu `~/.bashrc`.

### 3) Descargar el proyecto y sincronizar dependencias

```bash
git clone <URL_DEL_REPOSITORIO>
cd python_services/intent-miner-service
uv sync --frozen
```

### 4) Probar ejecución manual

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 4002
```

### 5) Crear servicio `systemd`

Crear el archivo `/etc/systemd/system/intent-miner.service`:

```ini
[Unit]
Description=Intent Miner Service (FastAPI + uv)
After=network.target

[Service]
Type=simple
User=<USUARIO_LINUX>
WorkingDirectory=/ruta/a/python_services/intent-miner-service
ExecStart=/home/<USUARIO_LINUX>/.local/bin/uv run uvicorn main:app --host 0.0.0.0 --port 4002
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Aplicar y arrancar el servicio:

```bash
sudo systemctl daemon-reload
sudo systemctl enable intent-miner.service
sudo systemctl start intent-miner.service
sudo systemctl status intent-miner.service
```

### 6) (Opcional) Abrir puerto en firewall

```bash
sudo firewall-cmd --permanent --add-port=4002/tcp
sudo firewall-cmd --reload
```
