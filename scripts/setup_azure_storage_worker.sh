#!/usr/bin/env bash
# =============================================================================
# One-time / idempotent setup for the queue + worker architecture.
#
# Creates storage primitives (if missing), grants RBAC to the UI app and worker
# job managed identities, and creates or updates the event-driven worker job.
#
# Prerequisites:
#   - az login, correct subscription
#   - Storage account already exists (default: aiicfstorage)
#   - UI Container App (ca-uhn-icf) already deployed with system-assigned MI
#   - Container image already built in ACR (same image for UI + worker)
#
# Usage:
#   chmod +x scripts/setup_azure_storage_worker.sh
#   IMAGE_TAG=v1234567890 ./scripts/setup_azure_storage_worker.sh
#
# Optional env overrides:
#   RG, ENV, APP, WORKER_JOB, STORAGE_ACCOUNT, ACR_NAME, IMAGE_TAG,
#   WORKER_MAX_EXECUTIONS, QUEUE_CONNECTION_STRING (for KEDA scaler auth)
# =============================================================================

set -euo pipefail

RG="${RG:-rgUHN-aihub}"
ENV="${ENV:-cae-uhn-icf}"
APP="${APP:-ca-uhn-icf}"
WORKER_JOB="${WORKER_JOB:-ca-uhn-aiicf-worker}"
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-aiicfstorage}"
ACR_NAME="${ACR_NAME:-uhnicfacr26769}"
IMAGE_TAG="${IMAGE_TAG:-}"
QUEUE_NAME="${ICF_QUEUE_NAME:-icf-jobs}"
WORKER_MAX_EXECUTIONS="${WORKER_MAX_EXECUTIONS:-8}"
REPLICA_TIMEOUT="${WORKER_REPLICA_TIMEOUT:-7200}"

OPENAI_RESOURCE="${OPENAI_RESOURCE:-rebicf}"
OPENAI_DEPLOYMENT="${OPENAI_DEPLOYMENT:-gpt-5.4}"
OPENAI_ENDPOINT="${OPENAI_ENDPOINT:-https://rebicf.openai.azure.com/}"
OPENAI_API_VERSION="${OPENAI_API_VERSION:-2024-12-01-preview}"

log() { printf "\n==> %s\n" "$*"; }

if [[ -z "$IMAGE_TAG" ]]; then
    echo "ERROR: set IMAGE_TAG to the ACR tag you just built (e.g. IMAGE_TAG=v\$(date +%s))."
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-}"
fi
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "ERROR: export AZURE_OPENAI_API_KEY before running."
    exit 1
fi

IMAGE="${ACR_NAME}.azurecr.io/rlm-icf:${IMAGE_TAG}"

log "1/7  Verify storage account: $STORAGE_ACCOUNT"
STORAGE_ID=$(az storage account show -g "$RG" -n "$STORAGE_ACCOUNT" --query id -o tsv)

log "2/7  Create queue + blob containers + tables (idempotent)"
az storage queue create --account-name "$STORAGE_ACCOUNT" --name "$QUEUE_NAME" --auth-mode login -o none 2>/dev/null || true
az storage container create --account-name "$STORAGE_ACCOUNT" --name icf-input --auth-mode login -o none 2>/dev/null || true
az storage container create --account-name "$STORAGE_ACCOUNT" --name icf-output --auth-mode login -o none 2>/dev/null || true
az storage table create --account-name "$STORAGE_ACCOUNT" --name jobs --auth-mode login -o none 2>/dev/null || true

log "3/7  Grant storage RBAC to UI app identity: $APP"
UI_PRINCIPAL=$(az containerapp show -g "$RG" -n "$APP" --query identity.principalId -o tsv)
for ROLE in "Storage Blob Data Contributor" "Storage Queue Data Contributor" "Storage Table Data Contributor"; do
    az role assignment create \
        --assignee "$UI_PRINCIPAL" \
        --role "$ROLE" \
        --scope "$STORAGE_ID" \
        --output none 2>/dev/null || true
done

log "4/7  Create or update worker Container Apps Job: $WORKER_JOB"
if ! az containerapp job show -g "$RG" -n "$WORKER_JOB" &>/dev/null; then
    # KEDA azure-queue scaler needs queue depth auth. Connection string is the
    # most reliable option; the worker container itself uses managed identity.
  QUEUE_CONN="${QUEUE_CONNECTION_STRING:-}"
  if [[ -z "$QUEUE_CONN" ]]; then
      QUEUE_CONN=$(az storage account show-connection-string -g "$RG" -n "$STORAGE_ACCOUNT" --query connectionString -o tsv)
  fi
  CREATE_SECRETS=( "openai-api-key=$OPENAI_API_KEY" "queue-connection-string=$QUEUE_CONN" )
  CREATE_ENVS=(
      "AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT"
      "AZURE_OPENAI_ENDPOINT=$OPENAI_ENDPOINT"
      "AZURE_OPENAI_DEPLOYMENT=$OPENAI_DEPLOYMENT"
      "AZURE_OPENAI_API_VERSION=$OPENAI_API_VERSION"
      "AZURE_OPENAI_API_KEY=secretref:openai-api-key"
  )
  if [[ -n "${ACS_CONNECTION_STRING:-}" ]]; then
      CREATE_SECRETS+=( "acs-connection-string=$ACS_CONNECTION_STRING" )
      CREATE_ENVS+=( "ACS_CONNECTION_STRING=secretref:acs-connection-string" )
  fi
  if [[ -n "${ACS_SENDER_ADDRESS:-}" ]]; then
      CREATE_ENVS+=( "ACS_SENDER_ADDRESS=$ACS_SENDER_ADDRESS" )
  fi
  CREATE_ENVS+=(
      "ACS_SENDER_NAME=${ACS_SENDER_NAME:-UHN AI-Hub}"
      "ACS_REPLY_TO=${ACS_REPLY_TO:-AIHub@uhn.ca}"
  )

  az containerapp job create \
      -g "$RG" \
      -n "$WORKER_JOB" \
      --environment "$ENV" \
      --trigger-type Event \
      --replica-timeout "$REPLICA_TIMEOUT" \
      --replica-retry-limit 0 \
      --polling-interval 30 \
      --min-executions 0 \
      --max-executions "$WORKER_MAX_EXECUTIONS" \
      --image "$IMAGE" \
      --cpu 2.0 \
      --memory 4Gi \
      --registry-server "${ACR_NAME}.azurecr.io" \
      --registry-identity system \
      --system-assigned \
      --secrets "${CREATE_SECRETS[@]}" \
      --env-vars "${CREATE_ENVS[@]}" \
      --command "python" "worker.py" \
      --scale-rule-name icf-queue \
      --scale-rule-type azure-queue \
      --scale-rule-metadata \
          "accountName=$STORAGE_ACCOUNT" \
          "queueName=$QUEUE_NAME" \
          "queueLength=1" \
          "activationQueueLength=0" \
      --scale-rule-auth "connection=queue-connection-string" \
      --output none
else
  az containerapp job update \
      -g "$RG" \
      -n "$WORKER_JOB" \
      --image "$IMAGE" \
      --cpu 2.0 \
      --memory 4Gi \
      --replica-timeout "$REPLICA_TIMEOUT" \
      --max-executions "$WORKER_MAX_EXECUTIONS" \
      --set-env-vars \
          "AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT" \
      --output none
fi

log "5/7  Grant storage RBAC to worker job identity"
WORKER_PRINCIPAL=$(az containerapp job show -g "$RG" -n "$WORKER_JOB" --query identity.principalId -o tsv)
for ROLE in "Storage Blob Data Contributor" "Storage Queue Data Contributor" "Storage Table Data Contributor"; do
    az role assignment create \
        --assignee "$WORKER_PRINCIPAL" \
        --role "$ROLE" \
        --scope "$STORAGE_ID" \
        --output none 2>/dev/null || true
done

ACR_ID=$(az acr show -g "$RG" -n "$ACR_NAME" --query id -o tsv)
az role assignment create \
    --assignee "$WORKER_PRINCIPAL" \
    --role "AcrPull" \
    --scope "$ACR_ID" \
    --output none 2>/dev/null || true

OPENAI_ID=$(az cognitiveservices account show -g "$RG" -n "$OPENAI_RESOURCE" --query id -o tsv)
az role assignment create \
    --assignee "$WORKER_PRINCIPAL" \
    --role "Cognitive Services User" \
    --scope "$OPENAI_ID" \
    --output none 2>/dev/null || true

log "6/7  Point UI app at shared storage (remove OpenAI from UI tier if still set)"
az containerapp update \
    -g "$RG" \
    -n "$APP" \
    --set-env-vars "AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT" \
    --remove-env-vars "AZURE_OPENAI_API_KEY" "AZURE_OPENAI_ENDPOINT" "AZURE_OPENAI_DEPLOYMENT" "AZURE_OPENAI_API_VERSION" 2>/dev/null || \
az containerapp update \
    -g "$RG" \
    -n "$APP" \
    --set-env-vars "AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT" \
    --output none

log "7/7  Summary"
az containerapp job show -g "$RG" -n "$WORKER_JOB" \
    --query "{name:name,image:properties.template.containers[0].image,maxExecutions:properties.configuration.eventTriggerConfig.replicaCompletionCount,scale:properties.configuration.eventTriggerConfig.scale}" \
    -o json 2>/dev/null || az containerapp job show -g "$RG" -n "$WORKER_JOB" -o json

cat <<EOF

Done.

Redeploy checklist:
  1. az acr build + az containerapp update (UI)
  2. IMAGE_TAG=$IMAGE_TAG ./scripts/setup_azure_storage_worker.sh  (worker image + RBAC)
  3. ./scripts/configure_azure_app.sh  (Streamlit ingress tuning)

Worker processes one queue message per execution. Scale parallel runs via
WORKER_MAX_EXECUTIONS (default $WORKER_MAX_EXECUTIONS), not UI replica count.

EOF
