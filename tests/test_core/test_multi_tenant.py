"""Tests for multi-tenant isolation.

Covers Q9: Multi-tenant data isolation at the index level.
"""

import pytest

from src.core.multi_tenant import (
    TenantContext,
    get_current_tenant,
    get_tenant_config,
    get_tenant_namespace,
    set_current_tenant,
)


class TestTenantContext:
    """Tests for tenant context management."""

    def test_context_sets_and_restores_tenant(self):
        assert get_current_tenant() is None

        with TenantContext("tenant_a"):
            assert get_current_tenant() == "tenant_a"

        assert get_current_tenant() is None

    def test_nested_contexts(self):
        with TenantContext("tenant_a"):
            assert get_current_tenant() == "tenant_a"
            with TenantContext("tenant_b"):
                assert get_current_tenant() == "tenant_b"
            assert get_current_tenant() == "tenant_a"

    def test_get_namespace_returns_tenant_id(self):
        with TenantContext("tenant_a"):
            ns = get_tenant_namespace()
            assert ns == "tenant_a"

    def test_get_namespace_without_context_raises(self):
        with pytest.raises(ValueError):
            get_tenant_namespace()


class TestTenantConfig:
    """Tests for per-tenant configuration."""

    def test_default_config(self):
        config = get_tenant_config("unknown_tenant")
        assert config["confidence_auto_threshold"] == 0.85
        assert config["preferred_model"] == "gpt-4o-mini"

    def test_tenant_specific_config(self):
        config = get_tenant_config("tenant_b")
        # tenant_b has stricter thresholds
        assert config["confidence_auto_threshold"] == 0.90
        assert config["confidence_review_threshold"] == 0.6

    def test_config_from_context(self):
        with TenantContext("tenant_a"):
            config = get_tenant_config()
            assert config["confidence_auto_threshold"] == 0.85
