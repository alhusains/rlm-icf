#!/usr/bin/env bash
# =============================================================================
# UHN ICF Automation — Azure deployment script
# =============================================================================
# Run this from the rlm-icf repo root after Phase 0/1 are done:
#   - az login completed, subscription set
#   - app.py, Dockerfile, .dockerignore, requirements.txt all in place
#   - Local Streamlit run validated end-to-end
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# The script is idempotent for create-if-not-exists where Azure CLI supports
# it; if a resource already exists, the create call will fail loudly. Re-run
# only the steps you need to redo.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIGURATION — edit only this block
# ---------------------------------------------------------------------------
RG="rgUHN-aihub"
LOCATION="canadacentral"

# Existing Azure OpenAI resource (do NOT change unless you know why)
OPENAI_RESOURCE="rebicf"
OPENAI_DEPLOYMENT="gpt-5.4"
OPENAI_ENDPOINT="https://rebicf.openai.azure.com/"
OPENAI_API_VERSION="2024-12-01-preview"

# New resources we'll create. ACR name must be globally unique, lowercase,
# alphanumeric only.
ACR_NAME="uhnicfacr$RANDOM"
LAW_NAME="law-uhn-icf"
ENV_NAME="cae-uhn-icf"
APP_NAME="ca-uhn-icf"

# Sizing
CPU="1.0"
MEMORY="2.0Gi"
MIN_REPLICAS=0
MAX_REPLICAS=2

# Image tag — bump for new versions
IMAGE_TAG="v1"

# ---------------------------------------------------------------------------
# IMPORTANT — paste your existing Azure OpenAI API key here once, then never
# commit this file with the value populated. Treat it like a password.
# ---------------------------------------------------------------------------
OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-}"
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "ERROR: AZURE_OPENAI_API_KEY env var is not set."
    echo "Run: export AZURE_OPENAI_API_KEY='your-key-here'  before running this script."
    echo "(Don't paste the key directly into this file — it could leak via git.)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
ok()  { printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }

# ---------------------------------------------------------------------------
# STEP 1 — Verify CLI context
# ---------------------------------------------------------------------------
log "Step 1/9  Verify Azure CLI context"
SUB_ID=$(az account show --query id -o tsv)
SUB_NAME=$(az account show --query name -o tsv)
echo "Subscription: $SUB_NAME ($SUB_ID)"
echo "Resource group: $RG"
echo "Location: $LOCATION"
ok "Context confirmed"

# ---------------------------------------------------------------------------
# STEP 2 — Container Registry
# ---------------------------------------------------------------------------
log "Step 2/9  Create Azure Container Registry: $ACR_NAME"
az acr create \
    --resource-group "$RG" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled false \
    --location "$LOCATION" \
    --output none
ok "ACR created"

# ---------------------------------------------------------------------------
# STEP 3 — Log Analytics workspace
# ---------------------------------------------------------------------------
log "Step 3/9  Create Log Analytics workspace: $LAW_NAME"
az monitor log-analytics workspace create \
    --resource-group "$RG" \
    --workspace-name "$LAW_NAME" \
    --location "$LOCATION" \
    --output none

LAW_ID=$(az monitor log-analytics workspace show \
    -g "$RG" -n "$LAW_NAME" --query customerId -o tsv)
LAW_KEY=$(az monitor log-analytics workspace get-shared-keys \
    -g "$RG" -n "$LAW_NAME" --query primarySharedKey -o tsv)
ok "Log Analytics workspace ready (id: ${LAW_ID:0:8}…)"

# ---------------------------------------------------------------------------
# STEP 4 — Container Apps environment
# ---------------------------------------------------------------------------
log "Step 4/9  Create Container Apps environment: $ENV_NAME"
az containerapp env create \
    --resource-group "$RG" \
    --name "$ENV_NAME" \
    --location "$LOCATION" \
    --logs-workspace-id "$LAW_ID" \
    --logs-workspace-key "$LAW_KEY" \
    --output none
ok "Environment ready"

# ---------------------------------------------------------------------------
# STEP 5 — Build the image inside ACR (no local Docker required)
# ---------------------------------------------------------------------------
log "Step 5/9  Build image $ACR_NAME.azurecr.io/rlm-icf:$IMAGE_TAG via az acr build"
echo "   (this uploads your repo as the build context and builds in Azure)"
az acr build \
    --registry "$ACR_NAME" \
    --image "rlm-icf:$IMAGE_TAG" \
    --file Dockerfile \
    .
ok "Image built and pushed to ACR"

# ---------------------------------------------------------------------------
# STEP 6 — Create the Container App with system-assigned managed identity
# ---------------------------------------------------------------------------
log "Step 6/9  Create Container App: $APP_NAME"
az containerapp create \
    --resource-group "$RG" \
    --name "$APP_NAME" \
    --environment "$ENV_NAME" \
    --image "$ACR_NAME.azurecr.io/rlm-icf:$IMAGE_TAG" \
    --target-port 8000 \
    --ingress external \
    --transport auto \
    --min-replicas "$MIN_REPLICAS" \
    --max-replicas "$MAX_REPLICAS" \
    --cpu "$CPU" \
    --memory "$MEMORY" \
    --system-assigned \
    --registry-server "$ACR_NAME.azurecr.io" \
    --registry-identity system \
    --secrets "openai-api-key=$OPENAI_API_KEY" \
    --env-vars \
        "AZURE_OPENAI_ENDPOINT=$OPENAI_ENDPOINT" \
        "AZURE_OPENAI_DEPLOYMENT=$OPENAI_DEPLOYMENT" \
        "AZURE_OPENAI_API_VERSION=$OPENAI_API_VERSION" \
        "AZURE_OPENAI_API_KEY=secretref:openai-api-key" \
    --output none
ok "Container App created"

# ---------------------------------------------------------------------------
# STEP 7 — Grant the app's identity access to ACR and Azure OpenAI
# ---------------------------------------------------------------------------
log "Step 7/9  Assign roles to the Container App's managed identity"

PRINCIPAL_ID=$(az containerapp show \
    -g "$RG" -n "$APP_NAME" \
    --query identity.principalId -o tsv)
echo "   Principal ID: $PRINCIPAL_ID"

ACR_ID=$(az acr show -g "$RG" -n "$ACR_NAME" --query id -o tsv)
az role assignment create \
    --assignee "$PRINCIPAL_ID" \
    --scope "$ACR_ID" \
    --role "AcrPull" \
    --output none
ok "AcrPull granted on $ACR_NAME"

OPENAI_ID=$(az cognitiveservices account show \
    -g "$RG" -n "$OPENAI_RESOURCE" --query id -o tsv)
az role assignment create \
    --assignee "$PRINCIPAL_ID" \
    --scope "$OPENAI_ID" \
    --role "Cognitive Services User" \
    --output none
ok "Cognitive Services User granted on $OPENAI_RESOURCE"
echo "   (Not used today — we ship v1 with the API key secret. This sets us up"
echo "    to migrate to Managed Identity later without re-doing IAM.)"

# ---------------------------------------------------------------------------
# STEP 8 — Print the public URL (BEFORE auth is enabled)
# ---------------------------------------------------------------------------
log "Step 8/9  Retrieve app URL"
FQDN=$(az containerapp show \
    -g "$RG" -n "$APP_NAME" \
    --query properties.configuration.ingress.fqdn -o tsv)
APP_URL="https://$FQDN"
ok "App URL: $APP_URL"

echo ""
echo "------------------------------------------------------------------------"
echo "  At this point the app is LIVE and PUBLICLY REACHABLE without auth."
echo "  Test it once with a public protocol to confirm the deploy worked:"
echo ""
echo "    open $APP_URL"
echo ""
echo "  Then run Step 9 to enable Microsoft Entra ID authentication."
echo "------------------------------------------------------------------------"

# ---------------------------------------------------------------------------
# STEP 9 — Enable Entra Easy Auth
# ---------------------------------------------------------------------------
# Easy Auth setup is fiddly via CLI (requires creating an app registration
# and configuring it). The portal flow is much simpler and visible.
#
# After confirming the app works publicly:
#
#   1. Azure Portal → Container App "$APP_NAME" → Authentication
#   2. Click "Add identity provider"
#   3. Select "Microsoft"
#   4. Tenant type:               Workforce
#   5. App registration:          Create new
#   6. Name:                      $APP_NAME-auth (or any name)
#   7. Supported account types:   "Current tenant - Single tenant"
#   8. Restrict access:           Require authentication
#   9. Unauthenticated requests:  HTTP 302 Found (redirect to login)
#  10. Token store:               Enabled (default)
#  11. Click "Add"
#
# After this, every visit to $APP_URL redirects to UHN sign-in. Only users
# in the UHN tenant can authenticate.

log "Step 9/9  Manual step — enable Entra Easy Auth in the portal"
echo "   See the comment block above this step in deploy.sh for click-by-click."
echo ""
echo "========================================================================"
echo "  DEPLOYMENT COMPLETE (auth pending)"
echo "  ACR:        $ACR_NAME"
echo "  Env:        $ENV_NAME"
echo "  App:        $APP_NAME"
echo "  Image:      $ACR_NAME.azurecr.io/rlm-icf:$IMAGE_TAG"
echo "  URL:        $APP_URL"
echo "========================================================================"
echo ""
echo "Save the resource names above — you'll need them for redeploys and"
echo "for the Entra Easy Auth configuration in the portal."
