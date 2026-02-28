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

## Despliegue en Linux AlmaLinux (ruta fija: `/var/www/services/intent-miner-service`)

> Este instructivo asume que el código del servicio ya está en: `/var/www/services/intent-miner-service`.

### 1) Instalar dependencias del sistema

```bash
sudo dnf update -y
sudo dnf install -y curl git
```

### 2) Verificar Python y luego instalar `uv`

```bash
python3 --version
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Si `uv` no aparece en una sesión nueva, agrega esta línea en `~/.bashrc` y vuelve a cargar el shell:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 3) Entrar al directorio real del servicio y sincronizar dependencias

```bash
cd /var/www/services/intent-miner-service
uv sync --frozen
```

> `uv sync --frozen` usa el lockfile existente y falla si hay desajustes, lo que ayuda a mantener un despliegue reproducible.

### 4) Validar arranque manual del servicio

```bash
cd /var/www/services/intent-miner-service
uv run uvicorn main:app --host 0.0.0.0 --port 4002
```

En otra terminal, validar que responde:

```bash
curl -i http://127.0.0.1:4002/
```

### 5) Usuario de ejecución: `intentminer` vs `root`

Puedes ejecutar el servicio como `root`, pero **no es lo recomendado** para producción.

- **Recomendado (`intentminer`)**: principio de mínimo privilegio. Si el proceso se compromete, el impacto queda limitado al usuario del servicio.
- **`root`**: solo si tienes una razón operativa fuerte (entorno muy controlado o migración temporal).

Si vas por la opción recomendada, crea el usuario:

```bash
sudo useradd --system --create-home --shell /sbin/nologin intentminer
sudo chown -R intentminer:intentminer /var/www/services/intent-miner-service
```

### 6) Crear servicio `systemd`

Crear el archivo `/etc/systemd/system/intent-miner.service`:

```ini
[Unit]
Description=Intent Miner Service (FastAPI + uv)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# Opción recomendada (mínimo privilegio):
User=intentminer
Group=intentminer
Environment=HOME=/home/intentminer

# Si decides correr como root, comenta las 3 líneas anteriores y usa:
# User=root
# Group=root
# Environment=HOME=/root

WorkingDirectory=/var/www/services/intent-miner-service
Environment=PYTHONUNBUFFERED=1
ExecStart=/var/www/services/intent-miner-service/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 4002
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

> **Por qué así:** en `systemd` es más estable ejecutar el binario ya instalado en `.venv` (en lugar de invocar `uv run` en cada arranque). Esto evita fallos típicos por `PATH`/`HOME` y permisos de caché.

Antes de iniciar el servicio, crea el entorno con el mismo usuario que ejecutará `systemd`:

```bash
cd /var/www/services/intent-miner-service
# Si usarás intentminer:
sudo -u intentminer uv sync --frozen

# Si usarás root:
sudo uv sync --frozen

ls -l .venv/bin/uvicorn
```

Recargar, habilitar e iniciar:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now intent-miner.service
sudo systemctl status intent-miner.service
```

Ver logs en tiempo real:

```bash
sudo journalctl -u intent-miner.service -f
```

Si no arranca, revisar diagnóstico rápido:

```bash
sudo systemctl cat intent-miner.service
# Si usarás intentminer:
sudo -u intentminer /var/www/services/intent-miner-service/.venv/bin/python -V
sudo -u intentminer /var/www/services/intent-miner-service/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 4002

# Si usarás root:
sudo /var/www/services/intent-miner-service/.venv/bin/python -V
sudo /var/www/services/intent-miner-service/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 4002
```

### 7) (Opcional) Abrir puerto en `firewalld`

```bash
sudo firewall-cmd --permanent --add-port=4002/tcp
sudo firewall-cmd --reload
sudo firewall-cmd --list-ports
```

### 8) Actualizar a una nueva versión del servicio

```bash
cd /var/www/services/intent-miner-service
git pull
uv sync --frozen
sudo systemctl restart intent-miner.service
sudo systemctl status intent-miner.service
```
