# syntax=docker/dockerfile:1.6
#
# UHN ICF Automation — production container image.
# Built in Azure via `az acr build`; never built locally.
#
# Image goal: small, reproducible, fast to start. Python 3.11 to match the
# repo's pyproject.toml (requires-python = ">=3.11").

FROM python:3.11-slim

# ---- OS-level setup --------------------------------------------------------
# build-essential: needed for any Python deps that compile native extensions
#                  during pip install (e.g. some PDF parsing libs).
# We remove apt lists after install to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. Container Apps doesn't *require* this, but it's
# good hygiene and some org policies enforce it.
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# ---- Python dependencies ---------------------------------------------------
# Copy only the dependency manifests first so Docker can cache the pip-install
# layer when application code changes but deps don't.
COPY --chown=app:app pyproject.toml requirements.txt ./

# We need the package metadata (rlm/, icf/) on disk before `pip install -e .`
# can succeed. Copying them here is fine; subsequent code changes still
# invalidate this layer, but for a 4-min build that's acceptable.
COPY --chown=app:app rlm/ ./rlm/
COPY --chown=app:app icf/ ./icf/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Application code ------------------------------------------------------
COPY --chown=app:app data/ ./data/
COPY --chown=app:app app.py ./

# ---- Runtime config --------------------------------------------------------
# Streamlit settings tuned for running behind a reverse proxy (Container Apps
# ingress sits in front of the app and handles TLS termination).
ENV STREAMLIT_SERVER_PORT=8000 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    PYTHONUNBUFFERED=1

USER app

EXPOSE 8000

# Use exec form so signals (SIGTERM from Container Apps on shutdown) reach
# Streamlit directly instead of being swallowed by a shell.
CMD ["streamlit", "run", "app.py"]
