# =============================================================================
# PRODUCTION Dockerfile — Agent V2 Voice AI
# =============================================================================
# Optimized for Railway deployment with LiveKit audio processing.
# Uses multi-stage build pattern for smaller final image.

# -----------------------------------------------------------------------------
# Stage 1: Builder (install dependencies with build tools)
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production Runtime
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# =============================================================================
# SYSTEM DEPENDENCIES
# =============================================================================
# ffmpeg: Required for LiveKit audio processing (encoding/decoding)
# ca-certificates: SSL/TLS for secure API connections
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =============================================================================
# SECURITY: Non-root user
# =============================================================================
RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid agent --shell /bin/bash --create-home agent

# =============================================================================
# APPLICATION SETUP
# =============================================================================
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only production files
COPY agent_v2.py .
COPY contact_utils.py .
COPY calendar_client.py .
COPY supabase_calendar_store.py .

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
# Ensure Python output is sent straight to terminal (no buffering)
ENV PYTHONUNBUFFERED=1
# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Set production environment
ENV ENVIRONMENT=production

# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================
# Switch to non-root user for security
USER agent

# Expose port (Railway assigns PORT automatically, but document for clarity)
EXPOSE 8080

# =============================================================================
# ENTRYPOINT
# =============================================================================
# 'dev' mode connects to LiveKit Cloud; remove if using 'start' for self-hosted
# Railway will inject environment variables at runtime
CMD ["python", "agent_v2.py", "dev"]
