"""
TTL-based caching utilities for clinic context.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple, Any, Optional

from config import CLINIC_CONTEXT_CACHE_TTL


class TTLCache:
    """
    Simple TTL-based cache for clinic context.
    NEVER cache: availability, schedule conflicts, appointment slots.
    ONLY cache: static clinic info, agent settings, greetings.
    """
    
    def __init__(self, ttl_seconds: int = 60):
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key not in self._cache:
            return None
        timestamp, value = self._cache[key]
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None
        return value
    
    def set(self, key: str, value: Any):
        """Cache a value with TTL."""
        self._cache[key] = (time.time(), value)
    
    def invalidate(self, key: Optional[str] = None):
        """Invalidate specific key or entire cache."""
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()
    
    def size(self) -> int:
        return len(self._cache)


# Global clinic context cache
_clinic_cache = TTLCache(ttl_seconds=CLINIC_CONTEXT_CACHE_TTL)
