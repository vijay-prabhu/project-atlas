"""Multi-tenant isolation using ContextVar and FastAPI middleware.

Ensures every request is scoped to a tenant. Vector DB queries, LLM calls,
and data access are all filtered by tenant_id automatically.
"""

from contextvars import ContextVar
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

_current_tenant: ContextVar[Optional[str]] = ContextVar(
    "current_tenant", default=None
)

# API key to tenant mapping — in production this would be a database lookup
API_KEY_TENANT_MAP: dict[str, dict] = {
    "sk_test_tenant_a": {
        "tenant_id": "tenant_a",
        "name": "Thornton Architects",
        "namespace": "tenant_a",
    },
    "sk_test_tenant_b": {
        "tenant_id": "tenant_b",
        "name": "Morrison Engineering Group",
        "namespace": "tenant_b",
    },
    "sk_demo": {
        "tenant_id": "demo",
        "name": "Demo Tenant",
        "namespace": "demo",
    },
}

# Per-tenant configuration overrides
TENANT_CONFIGS: dict[str, dict] = {
    "tenant_a": {
        "confidence_auto_threshold": 0.85,
        "confidence_review_threshold": 0.5,
        "preferred_model": "gpt-4o-mini",
        "max_agent_steps": 5,
    },
    "tenant_b": {
        "confidence_auto_threshold": 0.90,
        "confidence_review_threshold": 0.6,
        "preferred_model": "gpt-4o",
        "max_agent_steps": 7,
    },
}

DEFAULT_TENANT_CONFIG = {
    "confidence_auto_threshold": 0.85,
    "confidence_review_threshold": 0.5,
    "preferred_model": "gpt-4o-mini",
    "max_agent_steps": 5,
}


def get_current_tenant() -> Optional[str]:
    return _current_tenant.get()


def set_current_tenant(tenant_id: str) -> None:
    _current_tenant.set(tenant_id)


def get_tenant_config(tenant_id: Optional[str] = None) -> dict:
    tid = tenant_id or get_current_tenant()
    if tid and tid in TENANT_CONFIGS:
        return {**DEFAULT_TENANT_CONFIG, **TENANT_CONFIGS[tid]}
    return DEFAULT_TENANT_CONFIG.copy()


def get_tenant_namespace(tenant_id: Optional[str] = None) -> str:
    """Get the vector DB namespace for the current tenant."""
    tid = tenant_id or get_current_tenant()
    if not tid:
        raise ValueError("No tenant context set")
    # Look up namespace from API key map
    for key_info in API_KEY_TENANT_MAP.values():
        if key_info["tenant_id"] == tid:
            return key_info["namespace"]
    return tid


class TenantContext:
    """Context manager for scoped tenant operations."""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self._token = None

    def __enter__(self):
        self._token = _current_tenant.set(self.tenant_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token is not None:
            _current_tenant.reset(self._token)
        return False


class TenantMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that extracts tenant from request headers."""

    SKIP_PATHS = {"/health", "/", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # Extract tenant from headers
        tenant_id = None

        # Option 1: API key header
        api_key = request.headers.get("X-API-Key")
        if api_key and api_key in API_KEY_TENANT_MAP:
            tenant_id = API_KEY_TENANT_MAP[api_key]["tenant_id"]

        # Option 2: Direct tenant header
        if not tenant_id:
            tenant_id = request.headers.get("X-Tenant-ID")

        # Option 3: Query parameter (for testing)
        if not tenant_id:
            tenant_id = request.query_params.get("tenant_id")

        if not tenant_id:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing tenant context. Provide X-API-Key or X-Tenant-ID header."},
            )

        set_current_tenant(tenant_id)
        response = await call_next(request)
        return response


def tenant_required(func):
    """Decorator that ensures a tenant context is active."""
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        tenant = get_current_tenant()
        if not tenant:
            raise ValueError("Operation requires tenant context")
        return await func(*args, **kwargs)

    return wrapper
