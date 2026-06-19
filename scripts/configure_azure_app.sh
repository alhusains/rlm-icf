#!/usr/bin/env bash
# =============================================================================
# Post-deploy Azure Container Apps tuning for the Streamlit ICF app.
#
# Run after every new environment setup, and again if upload/download issues
# return. Safe to re-run — all commands are idempotent.
#
# Usage:
#   chmod +x scripts/configure_azure_app.sh
#   ./scripts/configure_azure_app.sh
#
# For 6–8 concurrent users during a trial window, run with:
#   CONCURRENT_MODE=1 ./scripts/configure_azure_app.sh
# =============================================================================

set -euo pipefail

RG="${RG:-rgUHN-aihub}"
APP="${APP:-ca-uhn-icf}"
ENV="${ENV:-cae-uhn-icf}"

# Default: light scaling. CONCURRENT_MODE=1 pins 8 replicas for parallel ICF runs.
if [[ "${CONCURRENT_MODE:-0}" == "1" ]]; then
    MIN_REPLICAS=8
    MAX_REPLICAS=8
    CPU="2.0"
    MEMORY="4Gi"
else
    MIN_REPLICAS=2
    MAX_REPLICAS=8
    CPU="1.0"
    MEMORY="2Gi"
fi

log() { printf "\n==> %s\n" "$*"; }

log "1/5  Enable sticky sessions (required for Streamlit upload/download)"
az containerapp ingress sticky-sessions set \
    -g "$RG" -n "$APP" --affinity sticky --output none

log "2/5  Set ingress transport to HTTP (WebSocket-safe; avoid http2 upstream)"
az containerapp ingress update \
    -g "$RG" -n "$APP" --transport http --output none

log "3/5  Scale + resources (min=$MIN_REPLICAS max=$MAX_REPLICAS cpu=$CPU memory=$MEMORY)"
az containerapp update \
    -g "$RG" -n "$APP" \
    --min-replicas "$MIN_REPLICAS" \
    --max-replicas "$MAX_REPLICAS" \
    --cpu "$CPU" \
    --memory "$MEMORY" \
    --scale-rule-name ws-concurrency \
    --scale-rule-http-concurrency 100 \
    --output none

log "4/5  Current ingress + scale settings"
az containerapp show -g "$RG" -n "$APP" \
    --query "{sticky:properties.configuration.ingress.stickySessions,transport:properties.configuration.ingress.transport,scale:properties.template.scale,cpu:properties.template.containers[0].resources.cpu,memory:properties.template.containers[0].resources.memory}" \
    -o json

log "5/5  Replica count"
az containerapp replica list -g "$RG" -n "$APP" -o table 2>/dev/null || true

cat <<EOF

Done.

Concurrent mode: $([[ "${CONCURRENT_MODE:-0}" == "1" ]] && echo ON || echo OFF)
  - Each replica runs at most 1 ICF pipeline at a time (queued if busy).
  - For 6–8 parallel runs, use CONCURRENT_MODE=1 so 8 replicas are always warm.
  - Users are NOT assigned a dedicated replica; load is spread by ingress + sticky cookies.

Code deploy must include app.py queue + job_store changes.

Optional (extra cost): Premium ingress with 30-minute idle timeout. See:
  https://learn.microsoft.com/en-us/azure/container-apps/ingress-environment-configuration

EOF
