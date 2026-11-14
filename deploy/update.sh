#!/usr/bin/env bash
# Pull the latest code from GitHub and restart BacBuddy. Usage: bash /opt/bacbuddy/deploy/update.sh
set -euo pipefail
cd /opt/bacbuddy
git pull
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m app.ingest
systemctl restart bacbuddy
echo "✅ Updated and restarted."
