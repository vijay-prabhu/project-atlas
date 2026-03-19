"""FastAPI dependency injection."""

from src.agents.checkpoints import CheckpointStore, get_checkpoint_store
from src.core.config import Settings, get_settings
from src.core.multi_tenant import get_current_tenant


def get_tenant_id() -> str:
    """Get the current tenant ID from context."""
    tenant = get_current_tenant()
    if not tenant:
        raise ValueError("No tenant context")
    return tenant


def get_settings_dep() -> Settings:
    return get_settings()


def get_checkpoint_store_dep() -> CheckpointStore:
    return get_checkpoint_store()
