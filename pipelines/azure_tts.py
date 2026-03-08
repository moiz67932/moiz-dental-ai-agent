"""
Azure TTS wrapper — Thin convenience wrapper around livekit-plugins-azure.

This exists solely for centralized config. In production, we use the official
`livekit-plugins-azure` package which provides a LiveKit-compatible TTS plugin.

Install: pip install livekit-plugins-azure==1.3.11

Env vars required:
  AZURE_SPEECH_KEY     — Azure Speech Services subscription key
  AZURE_SPEECH_REGION  — Azure region (e.g., "eastus", "westeurope")
"""

from __future__ import annotations

import os
import logging

try:
    from livekit.plugins import azure as azure_plugin
except ImportError:
    azure_plugin = None

logger = logging.getLogger("snappy_agent")


def create_azure_tts(voice: str = "ur-PK-UzmaNeural"):
    """
    Create an Azure TTS instance using the official LiveKit Azure plugin.

    Args:
        voice: Azure Neural Voice name. Urdu options:
               - "ur-PK-UzmaNeural"  (Female, recommended)
               - "ur-PK-AsadNeural"  (Male)
    
    Returns:
        livekit.plugins.azure.TTS instance (LiveKit-compatible)
    
    Raises:
        RuntimeError: If AZURE_SPEECH_KEY or AZURE_SPEECH_REGION are not set.
        ImportError: If livekit-plugins-azure is not installed.
    """
    speech_key = os.getenv("AZURE_SPEECH_KEY", "")
    speech_region = os.getenv("AZURE_SPEECH_REGION", "")

    if not speech_key or not speech_region:
        raise RuntimeError(
            "Urdu TTS requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION env vars. "
            "Get them from https://portal.azure.com → Speech Services."
        )

    if azure_plugin is None:
        raise ImportError(
            "livekit-plugins-azure is not installed. "
            "Run: pip install livekit-plugins-azure==1.3.11"
        )

    tts = azure_plugin.TTS(voice=voice)
    logger.info(f"[URDU-TTS] ✓ Azure TTS initialized with voice={voice}, region={speech_region}")
    return tts
