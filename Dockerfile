# =============================================================================
# PRODUCTION Dockerfile — Cloud Run Job for LiveKit Worker
# =============================================================================
#
# WHY CLOUD RUN JOBS (NOT SERVICES):
# ==================================
# - Services are request-driven: containers killed when no HTTP traffic
# - Jobs are task-driven: containers run until task completes
# - LiveKit workers maintain WebSocket connections, not HTTP requests
# - Jobs can run for hours without artificial keep-alive hacks
#
# USAGE:
# ======
# Build:  docker build -f Dockerfile.job -t dental-agent-worker .
# Deploy: gcloud run jobs create dental-agent-worker --image <IMAGE>
#

# -----------------------------------------------------------------------------
# Stage 1: Builder (identical to Service Dockerfile)
# -----------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Force binary-only install for livekit to ensure the .so is included
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --only-binary=:all: livekit==1.0.23 && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production Runtime
# -----------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS runtime

# libgomp1 is REQUIRED for the LiveKit binary (.so) to load
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN useradd -m agent
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY worker_main.py .
COPY agent.py .
COPY agent_v2.py .
COPY config.py .
COPY models/ ./models/
COPY services/ ./services/
COPY tools/ ./tools/
COPY prompts/ ./prompts/
COPY utils/ ./utils/
COPY supabase_calendar_store.py .

# Python environment settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production

USER agent

# NO EXPOSE — Cloud Run Jobs don't need ports
# NO health probes — Jobs are not HTTP-based

# Run the worker entry point directly
# WHY NOT cli.run_app:
# - CLI helpers manage their own event loops and signals
# - They're designed for development, not containerized production
# - worker_main.py uses asyncio.run() for clean lifecycle control
CMD ["python", "worker_main.py"]
