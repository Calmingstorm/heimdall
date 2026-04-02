#!/bin/bash
set -e

INSTALL_DIR="/opt/heimdall"
CONFIG_DIR="/etc/heimdall"
DATA_DIR="/var/lib/heimdall"
LOG_DIR="/var/log/heimdall"
SERVICE_USER="heimdall"
SERVICE_GROUP="heimdall"

# --- Detect upgrade vs fresh install ---
IS_UPGRADE="false"
if [ -f "$CONFIG_DIR/config.yml" ] && [ -f "$CONFIG_DIR/.env" ]; then
    IS_UPGRADE="true"
fi

# --- Create system user and group ---
if ! getent group "$SERVICE_GROUP" >/dev/null 2>&1; then
    groupadd --system "$SERVICE_GROUP"
fi

if ! getent passwd "$SERVICE_USER" >/dev/null 2>&1; then
    useradd --system --gid "$SERVICE_GROUP" \
        --home-dir "$INSTALL_DIR" --no-create-home \
        --shell /usr/sbin/nologin \
        "$SERVICE_USER"
fi

# --- Create directories ---
mkdir -p "$CONFIG_DIR"
mkdir -p "$DATA_DIR"/{sessions,context,skills,search,knowledge}
mkdir -p "$LOG_DIR"

# --- Install default config if not present (preserve on upgrade) ---
if [ ! -f "$CONFIG_DIR/config.yml" ]; then
    cp "$INSTALL_DIR/config.yml" "$CONFIG_DIR/config.yml"
fi

if [ ! -f "$CONFIG_DIR/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$CONFIG_DIR/.env"
fi

# --- Create symlinks (app expects config and data in working directory) ---
ln -sf "$CONFIG_DIR/config.yml" "$INSTALL_DIR/config.yml"
ln -sf "$CONFIG_DIR/.env" "$INSTALL_DIR/.env"
ln -sfn "$DATA_DIR" "$INSTALL_DIR/data"
ln -sfn "$LOG_DIR" "$INSTALL_DIR/logs"

# --- Set up Python virtual environment ---
if [ ! -d "$INSTALL_DIR/.venv" ]; then
    python3.12 -m venv "$INSTALL_DIR/.venv"
fi

"$INSTALL_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/.venv/bin/pip" install --quiet "$INSTALL_DIR"

# --- Set ownership ---
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$DATA_DIR"
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$LOG_DIR"
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
chown "$SERVICE_USER:$SERVICE_GROUP" "$CONFIG_DIR/config.yml"
chown "$SERVICE_USER:$SERVICE_GROUP" "$CONFIG_DIR/.env"
chmod 600 "$CONFIG_DIR/.env"

# --- Enable and reload systemd ---
systemctl daemon-reload
systemctl enable heimdall.service

if [ "$IS_UPGRADE" = "true" ]; then
    echo ""
    echo "=================================================="
    echo "  Heimdall upgraded successfully!"
    echo "=================================================="
    echo ""
    echo "  Existing configuration preserved:"
    echo "    Config:  $CONFIG_DIR/config.yml"
    echo "    Env:     $CONFIG_DIR/.env"
    echo ""
    echo "  Restart to apply the update:"
    echo "    sudo systemctl restart heimdall"
    echo ""
    echo "  To reconfigure: sudo -u heimdall $INSTALL_DIR/.venv/bin/python -m src.setup wizard --reconfigure"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "  Heimdall installed successfully!"
    echo "=================================================="
    echo ""
    echo "  Config:  $CONFIG_DIR/config.yml"
    echo "  Env:     $CONFIG_DIR/.env"
    echo "  Data:    $DATA_DIR"
    echo "  Logs:    $LOG_DIR"
    echo ""
    echo "  Next steps:"
    echo "    1. Edit $CONFIG_DIR/.env and set DISCORD_TOKEN"
    echo "    2. Run the setup wizard: sudo -u heimdall $INSTALL_DIR/.venv/bin/python -m src.setup wizard"
    echo "    3. Start the service: sudo systemctl start heimdall"
    echo ""
    echo "  Or use the web wizard at http://localhost:3000/ui/ after starting."
    echo "=================================================="
fi
