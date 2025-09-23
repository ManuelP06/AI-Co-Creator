import asyncio
import hashlib
import json
from functools import wraps
from typing import Any, Optional

import redis.asyncio as redis

from app.config import settings
from app.core.logging_config import get_logger

logger = get_logger("cache")

# Redis client (optional)
redis_client: Optional[redis.Redis] = None


async def init_redis():
    """Initialize Redis connection if configured."""
    global redis_client
    if settings.redis_url:
        try:
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            await redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {e}")
            redis_client = None
    else:
        logger.info("Redis not configured, using in-memory cache")


# In-memory cache fallback
_memory_cache = {}
_cache_ttl = {}


def _generate_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from function arguments."""
    key_data = {"args": args, "kwargs": sorted(kwargs.items())}
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_string.encode()).hexdigest()


async def set_cache(key: str, value: Any, ttl: int = 3600) -> None:
    """Set a cache value."""
    try:
        if redis_client:
            await redis_client.setex(key, ttl, json.dumps(value, default=str))
        else:
            # Fallback to memory cache
            import time

            _memory_cache[key] = value
            _cache_ttl[key] = time.time() + ttl
    except Exception as e:
        logger.error(f"Failed to set cache: {e}")


async def get_cache(key: str) -> Optional[Any]:
    """Get a cache value."""
    try:
        if redis_client:
            cached = await redis_client.get(key)
            return json.loads(cached) if cached else None
        else:
            # Fallback to memory cache
            import time

            if key in _memory_cache:
                if time.time() < _cache_ttl.get(key, 0):
                    return _memory_cache[key]
                else:
                    # Expired
                    _memory_cache.pop(key, None)
                    _cache_ttl.pop(key, None)
            return None
    except Exception as e:
        logger.error(f"Failed to get cache: {e}")
        return None


async def delete_cache(key: str) -> None:
    """Delete a cache value."""
    try:
        if redis_client:
            await redis_client.delete(key)
        else:
            _memory_cache.pop(key, None)
            _cache_ttl.pop(key, None)
    except Exception as e:
        logger.error(f"Failed to delete cache: {e}")


def cache_result(ttl: int = 3600, key_prefix: str = ""):
    """Decorator to cache function results."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = (
                f"{key_prefix}:{func.__name__}:{_generate_cache_key(*args, **kwargs)}"
            )

            # Try to get from cache
            cached_result = await get_cache(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache the result
            await set_cache(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, use asyncio
            async def async_exec():
                cache_key = f"{key_prefix}:{func.__name__}:{_generate_cache_key(*args, **kwargs)}"

                cached_result = await get_cache(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result

                result = func(*args, **kwargs)
                await set_cache(cache_key, result, ttl)
                logger.debug(f"Cached result for {func.__name__}")

                return result

            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_exec())
            except RuntimeError:
                # No event loop running
                return asyncio.run(async_exec())

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


async def clear_cache_pattern(pattern: str) -> None:
    """Clear cache entries matching a pattern."""
    try:
        if redis_client:
            keys = await redis_client.keys(pattern)
            if keys:
                await redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries matching {pattern}")
        else:
            # Memory cache pattern matching
            import fnmatch

            keys_to_delete = [
                key for key in _memory_cache.keys() if fnmatch.fnmatch(key, pattern)
            ]
            for key in keys_to_delete:
                _memory_cache.pop(key, None)
                _cache_ttl.pop(key, None)
            logger.info(f"Cleared {len(keys_to_delete)} memory cache entries")
    except Exception as e:
        logger.error(f"Failed to clear cache pattern {pattern}: {e}")


# Cache invalidation helpers
async def invalidate_video_cache(video_id: int) -> None:
    """Invalidate all cache entries for a specific video."""
    await clear_cache_pattern(f"*video:{video_id}*")
