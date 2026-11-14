#!/usr/bin/env bash
# One-time setup of BacBuddy on a fresh Vultr Ubuntu 24.04 server.
# Usage (as root on the server):
#   bash setup.sh
set -euo pipefail

REPO="https://github.com/timotei-coll/bacbuddy.git"
APP_DIR="/opt/bacbuddy"

echo "==> Installing system packages"
apt-get update -y
apt-get install -y python3-venv python3-pip git caddy

echo "==> Getting the code"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull
else
  git clone "$REPO" "$APP_DIR"
fi
cd "$APP_DIR"

echo "==> Python environment"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if [ ! -f .env ]; then
  echo "==> Creating .env (your Vultr Inference settings)"
  read -rp "INFERENCE_API_KEY: " KEY
  read -rp "INFERENCE_MODEL: " MODEL
  printf 'INFERENCE_API_KEY=%s\nINFERENCE_MODEL=%s\nINFERENCE_BASE_URL=https://api.vultrinference.com/v1\n' "$KEY" "$MODEL" > .env
  chmod 600 .env
fi

echo "==> Building the search index"
.venv/bin/python -m app.ingest

echo "==> Background service"
cat > /etc/systemd/system/bacbuddy.service <<UNIT
[Unit]
Description=BacBuddy web app
After=network.target

[Service]
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/uvicorn app.server:app --host 127.0.0.1 --port 8000 --proxy-headers
Restart=always

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now bacbuddy
systemctl restart bacbuddy

echo "==> HTTPS with Caddy (free certificate via sslip.io)"
IP=$(curl -4 -s https://ifconfig.me || hostname -I | awk '{print $1}')
DOMAIN="${IP//./-}.sslip.io"
cat > /etc/caddy/Caddyfile <<CADDY
$DOMAIN {
    encode gzip
    reverse_proxy 127.0.0.1:8000
}
CADDY
systemctl restart caddy

if command -v ufw >/dev/null; then
  ufw allow 80/tcp || true
  ufw allow 443/tcp || true
fi

echo ""
echo "✅ BacBuddy is live at: https://$DOMAIN"
echo "   (the HTTPS certificate can take ~1 minute the first time)"
