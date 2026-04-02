#!/bin/bash
set -e

# --- Stop and disable the service ---
if systemctl is-active --quiet heimdall.service 2>/dev/null; then
    systemctl stop heimdall.service
fi

if systemctl is-enabled --quiet heimdall.service 2>/dev/null; then
    systemctl disable heimdall.service
fi

systemctl daemon-reload
