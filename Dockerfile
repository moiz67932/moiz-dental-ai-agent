# =============================================================================
# PRODUCTION Dockerfile — Optimized for Railway & LiveKit
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
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

# FIX: Force binary-only install for livekit to ensure the .so is included
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --only-binary=:all: livekit==1.0.23 && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production Runtime
# -----------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS runtime

# FIX: libgomp1 is REQUIRED for the LiveKit binary (.so) to load
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

# Copy modular application structure
COPY main.py .
COPY agent.py .
COPY config.py .
COPY models/ ./models/
COPY services/ ./services/
COPY tools/ ./tools/
COPY prompts/ ./prompts/
COPY utils/ ./utils/

# Copy legacy files only if still needed (for compatibility)
# These can be removed once fully migrated
COPY supabase_calendar_store.py .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production

USER agent
EXPOSE 8080

# Run the new modular entry point
CMD ["python", "main.py", "dev"]