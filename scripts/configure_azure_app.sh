#!/usr/bin/env bash
# =============================================================================
# Post-deploy Azure Container Apps tuning for the Streamlit ICF UI.
#
# The UI only enqueues jobs; parallel pipeline runs are handled by the worker
# job (ca-uhn-aiicf-worker). This script tunes ingress for Streamlit uploads.
#
# Usage:
#   chmod +x scripts/configure_azure_app.sh
#   ./scripts/configure_azure_app.sh
# =============================================================================

set -euo pipefail

RG="${RG:-rgUHN-aihub}"
APP="${APP:-ca-uhn-icf}"

# UI tier: light scaling (upload + polling only). Pipeline parallelism is on
# the worker job — see scripts/setup_azure_storage_worker.sh.
MIN_REPLICAS=2
MAX_REPLICAS=4
CPU="1.0"
MEMORY="2Gi"

log() { printf "\n==> %s\n" "$*"; }

log "1/4  Enable sticky sessions (required for Streamlit upload/download)"
az containerapp ingress sticky-sessions set \
    -g "$RG" -n "$APP" --affinity sticky --output none

log "2/4  Set ingress transport to HTTP (WebSocket-safe)"
az containerapp ingress update \
    -g "$RG" -n "$APP" --transport http --output none

log "3/4  Scale UI replicas (min=$MIN_REPLICAS max=$MAX_REPLICAS cpu=$CPU memory=$MEMORY)"
az containerapp update \
    -g "$RG" -n "$APP" \
    --min-replicas "$MIN_REPLICAS" \
    --max-replicas "$MAX_REPLICAS" \
    --cpu "$CPU" \
    --memory "$MEMORY" \
    --scale-rule-name ws-concurrency \
    --scale-rule-http-concurrency 100 \
    --output none

log "4/4  Current settings"
az containerapp show -g "$RG" -n "$APP" \
    --query "{sticky:properties.configuration.ingress.stickySessions,transport:properties.configuration.ingress.transport,scale:properties.template.scale,cpu:properties.template.containers[0].resources.cpu,memory:properties.template.containers[0].resources.memory,storage:properties.template.containers[0].env[?name=='AZURE_STORAGE_ACCOUNT'].value | [0]}" \
    -o json

cat <<EOF

Done. UI ingress configured for Streamlit.

Parallel ICF generation is controlled by the worker job
(scripts/setup_azure_storage_worker.sh), not UI replica count.

EOF
