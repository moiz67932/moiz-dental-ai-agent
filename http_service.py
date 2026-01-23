"""
Minimal HTTP Service for Cloud Run (Optional).

This is a stripped-down FastAPI service that provides ONLY health checks.
The LiveKit worker runs as a separate Cloud Run Job (worker_main.py).

WHY SEPARATE:
=============
- Cloud Run Services are request-driven (need HTTP traffic to stay alive)
- Cloud Run Jobs are task-driven (run until completion)
- LiveKit workers need Jobs, not Services
- Health checks may still be useful for monitoring dashboards

IF YOU DON'T NEED HTTP ENDPOINTS:
=================================
You can safely delete this file. The worker_main.py is self-contained.
This HTTP service is purely optional for external monitoring.
"""

from __future__ import annotations

import os
from fastapi import FastAPI

# Minimal app with no docs endpoint (reduces attack surface)
app = FastAPI(docs_url=None, redoc_url=None)


@app.get("/healthz")
def health_check():
    """
    Health check endpoint for external monitoring.
    
    Note: This does NOT check the LiveKit worker status because
    the worker runs in a separate Cloud Run Job.
    """
    return {"status": "ok", "service": "dental-agent-http"}


@app.get("/")
def root():
    """Root endpoint for basic connectivity check."""
    return {
        "service": "dental-agent-http",
        "worker": "runs as separate Cloud Run Job",
        "healthz": "/healthz",
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
