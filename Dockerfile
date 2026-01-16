# =============================================================================
# PRODUCTION Dockerfile — Optimized for Railway & LiveKit
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder (Stable Bookworm)
# -----------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS builder

# Install only the absolute necessities for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production Runtime (Stable Bookworm)
# -----------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS runtime

# libgomp1 is REQUIRED for LiveKit RTC FFI to load
# ffmpeg is needed for audio processing, ca-certificates for API security
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a non-root user for security
RUN useradd -m agent
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy all required local files
COPY agent_v2.py .
COPY contact_utils.py .
COPY calendar_client.py .
COPY supabase_calendar_store.py .

# Environment config
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production

USER agent
EXPOSE 8080

# Start the agent
CMD ["python", "agent_v2.py", "dev"]