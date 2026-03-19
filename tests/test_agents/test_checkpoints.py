"""Tests for HITL checkpoints and state persistence.

Covers Q4: State, persistence, and human-in-the-loop checkpoints.
"""

import pytest

from src.agents.checkpoints import CheckpointStore, should_request_human_review


class TestCheckpointStore:
    """Tests for the checkpoint persistence layer."""

    def setup_method(self):
        self.store = CheckpointStore()

    def test_save_and_load_checkpoint(self):
        state = {
            "email_id": "test_001",
            "classification": "rfi",
            "filing_confidence": 0.65,
            "filing_action": "needs_review",
        }

        record = self.store.save("test_001", state, "tenant_a")
        assert record.status == "paused"

        loaded = self.store.load("test_001", "tenant_a")
        assert loaded is not None
        assert loaded.state["email_id"] == "test_001"
        assert loaded.status == "paused"

    def test_resume_with_approval(self):
        state = {
            "email_id": "test_002",
            "filing_action": "needs_review",
            "filing_confidence": 0.65,
            "filing_project_id": "proj_001",
            "needs_human_review": True,
        }

        self.store.save("test_002", state, "tenant_a")

        updated = self.store.resume(
            "test_002", "approve", tenant_id="tenant_a"
        )
        assert updated is not None
        assert updated["filing_action"] == "auto_file"
        assert updated["filing_confidence"] == 1.0
        assert updated["needs_human_review"] is False

    def test_resume_with_correction(self):
        state = {
            "email_id": "test_003",
            "filing_action": "needs_review",
            "filing_project_id": "proj_001",
            "needs_human_review": True,
        }

        self.store.save("test_003", state, "tenant_a")

        updated = self.store.resume(
            "test_003", "correct",
            corrected_project_id="proj_002",
            tenant_id="tenant_a",
        )
        assert updated is not None
        assert updated["filing_project_id"] == "proj_002"
        assert updated["filing_action"] == "auto_file"

    def test_resume_with_rejection(self):
        state = {
            "email_id": "test_004",
            "filing_action": "needs_review",
            "needs_human_review": True,
        }

        self.store.save("test_004", state, "tenant_a")

        updated = self.store.resume("test_004", "reject", tenant_id="tenant_a")
        assert updated["filing_action"] == "flagged"
        assert updated["filing_confidence"] == 0.0

    def test_resume_nonexistent_returns_none(self):
        result = self.store.resume("nonexistent", "approve", tenant_id="tenant_a")
        assert result is None

    def test_cannot_resume_already_resumed(self):
        state = {"email_id": "test_005", "needs_human_review": True}
        self.store.save("test_005", state, "tenant_a")
        self.store.resume("test_005", "approve", tenant_id="tenant_a")

        # Second resume should fail
        result = self.store.resume("test_005", "approve", tenant_id="tenant_a")
        assert result is None

    def test_list_pending_reviews(self):
        self.store.save("pending_1", {"email_id": "p1"}, "tenant_a")
        self.store.save("pending_2", {"email_id": "p2"}, "tenant_a")
        self.store.save("pending_3", {"email_id": "p3"}, "tenant_b")

        # Resume one
        self.store.resume("pending_1", "approve", tenant_id="tenant_a")

        all_pending = self.store.list_pending()
        assert len(all_pending) == 2

        tenant_a_pending = self.store.list_pending("tenant_a")
        assert len(tenant_a_pending) == 1

    def test_tenant_isolation(self):
        """Tenant A's checkpoints should not be visible to Tenant B."""
        self.store.save("shared_id", {"data": "tenant_a"}, "tenant_a")

        loaded_a = self.store.load("shared_id", "tenant_a")
        loaded_b = self.store.load("shared_id", "tenant_b")

        assert loaded_a is not None
        assert loaded_b is None


class TestHITLRouting:
    """Tests for the human-in-the-loop routing decision."""

    def test_needs_review_triggers_hitl(self):
        state = {"filing_action": "needs_review", "filing_confidence": 0.65}
        assert should_request_human_review(state) is True

    def test_auto_file_skips_hitl(self):
        state = {"filing_action": "auto_file", "filing_confidence": 0.92}
        assert should_request_human_review(state) is False

    def test_flagged_skips_hitl(self):
        state = {"filing_action": "flagged", "filing_confidence": 0.3}
        assert should_request_human_review(state) is False

    def test_medium_confidence_triggers_hitl(self):
        state = {"filing_action": "auto_file", "filing_confidence": 0.6}
        assert should_request_human_review(state) is True
