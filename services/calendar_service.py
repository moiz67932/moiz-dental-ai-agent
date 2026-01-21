"""
Calendar service for Google Calendar integration.

Handles:
- OAuth token management
- Token refresh callbacks
- Calendar authentication resolution
"""

from __future__ import annotations

import os
import json
import asyncio
from typing import Optional, Dict, Callable, Tuple

from .calendar_client import CalendarAuth, set_token_refresh_callback
from config import supabase, logger, GOOGLE_OAUTH_TOKEN_PATH, GOOGLE_CALENDAR_AUTH_MODE, GOOGLE_CALENDAR_ID_DEFAULT

# Global to track agent settings ID for token refresh
_GLOBAL_AGENT_SETTINGS_ID: Optional[str] = None


def _load_env_oauth_token() -> Optional[dict]:
    """Load OAuth token from ENV-configured file path (fallback for local dev)."""
    token_path = GOOGLE_OAUTH_TOKEN_PATH
    if not token_path or not os.path.exists(token_path):
        return None
    try:
        with open(token_path, "r", encoding="utf-8") as f:
            token_data = json.load(f)
        logger.info("[CALENDAR_AUTH] Loaded OAuth token from local file.")
        return token_data
    except Exception as e:
        logger.error(f"[CALENDAR_AUTH] Failed to load local token: {e}")
        return None


async def fetch_oauth_token_from_db(agent_settings_id: str) -> Optional[dict]:
    """
    Fetch Google OAuth token from agent_settings.google_oauth_token column.
    
    Uses asyncio.to_thread for non-blocking database access.
    Returns the token dict if found and valid, None otherwise.
    """
    if not agent_settings_id:
        logger.warning("[CALENDAR_AUTH] No agent_settings_id provided for DB token fetch.")
        return None
    
    try:
        result = await asyncio.to_thread(
            lambda: supabase.table("agent_settings")
            .select("google_oauth_token")
            .eq("id", agent_settings_id)
            .limit(1)
            .execute()
        )
        
        if not result.data:
            logger.warning(f"[CALENDAR_AUTH] No agent_settings found for id={agent_settings_id}")
            return None
        
        token_json = result.data[0].get("google_oauth_token")
        
        if not token_json:
            logger.debug("[CALENDAR_AUTH] google_oauth_token column is empty in DB.")
            return None
        
        # Parse JSON if stored as string
        if isinstance(token_json, str):
            token_data = json.loads(token_json)
        else:
            token_data = token_json
        
        # Validate required fields
        if not token_data.get("refresh_token"):
            logger.warning("[CALENDAR_AUTH] DB token missing refresh_token - may not be able to refresh.")
        
        logger.info("[CALENDAR_AUTH] Loaded OAuth token from database.")
        return token_data
        
    except Exception as e:
        logger.error(f"[CALENDAR_AUTH] DB token fetch error: {e}")
        return None


async def save_refreshed_token_to_db(agent_settings_id: str, token_data: dict):
    """
    Save refreshed OAuth token back to agent_settings.google_oauth_token.
    
    Called asynchronously when Google refreshes an access token.
    Uses asyncio.create_task in the callback to avoid blocking the voice conversation.
    """
    if not agent_settings_id:
        logger.warning("[CALENDAR_AUTH] Cannot save token - no agent_settings_id.")
        return
    
    try:
        await asyncio.to_thread(
            lambda: supabase.table("agent_settings")
            .update({"google_oauth_token": json.dumps(token_data)})
            .eq("id", agent_settings_id)
            .execute()
        )
        logger.info("[CALENDAR_AUTH] Refreshed OAuth token saved to database.")
    except Exception as e:
        logger.error(f"[CALENDAR_AUTH] Failed to save refreshed token to DB: {e}")


def _create_token_refresh_callback(agent_settings_id: str) -> Callable[[dict], None]:
    """
    Create a callback that saves refreshed tokens to the database.
    
    Uses asyncio.create_task for non-blocking persistence so the
    voice conversation is never interrupted by token refresh saves.
    """
    def on_token_refresh(new_token_dict: dict):
        try:
            # Get the current event loop, create task for non-blocking save
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(
                    save_refreshed_token_to_db(agent_settings_id, new_token_dict)
                )
            else:
                # Fallback for edge case where loop isn't running
                asyncio.run(save_refreshed_token_to_db(agent_settings_id, new_token_dict))
        except Exception as e:
            logger.error(f"[CALENDAR_AUTH] Token refresh callback error: {e}")
    
    return on_token_refresh


async def resolve_calendar_auth_async(
    clinic_info: Optional[dict],
    settings: Optional[dict] = None,
) -> Tuple[Optional[CalendarAuth], str, Optional[Callable[[dict], None]]]:
    """
    Resolve calendar auth with DATABASE-FIRST priority.
    
    Priority order:
    1. Pre-fetched token from settings.google_oauth_token (already loaded in initial query)
    2. Database fetch: agent_settings.google_oauth_token column (if not pre-loaded)
    3. Local file: ENV-configured GOOGLE_OAUTH_TOKEN path (fallback for dev)
    
    Returns: (CalendarAuth, calendar_id, refresh_callback)
    
    The refresh_callback should be passed to _get_calendar_service so that
    token refreshes are persisted back to the database non-blocking.
    """
    global _GLOBAL_AGENT_SETTINGS_ID
    
    calendar_id = GOOGLE_CALENDAR_ID_DEFAULT
    refresh_callback = None
    
    # Extract agent_settings_id for DB operations
    agent_settings_id = (settings or {}).get("id") if settings else None
    if agent_settings_id:
        _GLOBAL_AGENT_SETTINGS_ID = agent_settings_id
    
    # PRIORITY 1A: Pre-fetched OAuth Token (already in settings from initial query)
    token_data = None
    
    if settings:
        pre_fetched_token = settings.get("google_oauth_token")
        if pre_fetched_token:
            # Parse JSON if stored as string
            if isinstance(pre_fetched_token, str):
                try:
                    token_data = json.loads(pre_fetched_token)
                    logger.debug("[CALENDAR_AUTH] Using pre-fetched OAuth token from settings.")
                except json.JSONDecodeError as e:
                    logger.warning(f"[CALENDAR_AUTH] Failed to parse pre-fetched token: {e}")
            elif isinstance(pre_fetched_token, dict):
                token_data = pre_fetched_token
                logger.debug("[CALENDAR_AUTH] Using pre-fetched OAuth token dict from settings.")
    
    # PRIORITY 1B: Database OAuth Token fetch (if not pre-loaded)
    if not token_data and agent_settings_id:
        logger.debug(f"[CALENDAR_AUTH] Fetching OAuth token from DB (settings_id={agent_settings_id})")
        token_data = await fetch_oauth_token_from_db(agent_settings_id)
    
    # Build CalendarAuth if we have token data from DB
    if token_data:
        try:
            token_data.setdefault("token_uri", "https://oauth2.googleapis.com/token")
            auth = CalendarAuth(
                auth_type="oauth_user",
                secret_json=token_data,
                delegated_user=None,
            )
            
            # Create refresh callback for non-blocking token persistence
            if agent_settings_id:
                refresh_callback = _create_token_refresh_callback(agent_settings_id)
            
            logger.info("[CALENDAR_AUTH] Using DATABASE OAuth token (production mode).")
            return auth, calendar_id, refresh_callback
            
        except Exception as e:
            logger.error(f"[CALENDAR_AUTH] DB token parse error: {e}")
    
    # PRIORITY 2: Local File OAuth Token (Development fallback)
    if GOOGLE_CALENDAR_AUTH_MODE == "oauth":
        token_data = _load_env_oauth_token()
        if token_data:
            try:
                token_data.setdefault("token_uri", "https://oauth2.googleapis.com/token")
                auth = CalendarAuth(
                    auth_type="oauth_user",
                    secret_json=token_data,
                    delegated_user=None,
                )
                logger.info("[CALENDAR_AUTH] Using LOCAL FILE OAuth token (dev mode).")
                return auth, calendar_id, None
            except Exception as e:
                logger.error(f"[CALENDAR_AUTH] Local file OAuth failed: {e}")
    
    # NO TOKEN FOUND — CRITICAL ERROR
    logger.critical(
        "[CALENDAR_AUTH] CRITICAL: No Google OAuth token found. "
        "Please run oauth_bootstrap.py and upload the token to Supabase "
        "(agent_settings.google_oauth_token column)."
    )
    return None, calendar_id, None


def resolve_calendar_auth(clinic_info: Optional[dict]) -> Tuple[Optional[CalendarAuth], str]:
    """
    LEGACY SYNC WRAPPER — For backwards compatibility with existing code.
    
    Prefer using resolve_calendar_auth_async() in async contexts for
    non-blocking database access and refresh callback support.
    """
    # Fallback to ENV OAuth (sync path for legacy callers)
    if GOOGLE_CALENDAR_AUTH_MODE == "oauth":
        token_data = _load_env_oauth_token()
        if token_data:
            try:
                token_data.setdefault("token_uri", "https://oauth2.googleapis.com/token")
                auth = CalendarAuth(
                    auth_type="oauth_user",
                    secret_json=token_data,
                    delegated_user=None,
                )
                return auth, GOOGLE_CALENDAR_ID_DEFAULT
            except Exception as e:
                logger.error(f"[CALENDAR_AUTH] ENV OAuth failed: {e}")
    
    logger.warning("[CALENDAR_AUTH] No OAuth token available (sync path).")
    return None, GOOGLE_CALENDAR_ID_DEFAULT
