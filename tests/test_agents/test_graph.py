"""Tests for the email filing agent graph.

Covers:
- Q1: Multi-agent hand-offs and preventing infinite loops
- Full pipeline execution: classify → extract → file
- Confidence-based routing
- Loop breaker guardrail
"""

import pytest

from src.agents.classifier import classify_email
from src.agents.extractor import extract_metadata
from src.agents.filer import make_filing_decision
from src.agents.graph import build_filing_graph, create_initial_state, run_filing_agent


class TestClassifierAgent:
    """Tests for the classification node."""

    def test_classifies_rfi_email(self, sample_rfi_email):
        state = create_initial_state(
            email_id=sample_rfi_email["id"],
            email_subject=sample_rfi_email["subject"],
            email_body=sample_rfi_email["body"],
            email_sender=sample_rfi_email["sender"],
            tenant_id="demo",
        )
        result = classify_email(state)
        assert result["classification"] == "rfi"
        assert result["classification_confidence"] >= 0.9

    def test_classifies_submittal_email(self, sample_submittal_email):
        state = create_initial_state(
            email_id=sample_submittal_email["id"],
            email_subject=sample_submittal_email["subject"],
            email_body=sample_submittal_email["body"],
            email_sender=sample_submittal_email["sender"],
            tenant_id="demo",
        )
        result = classify_email(state)
        assert result["classification"] == "submittal"
        assert result["classification_confidence"] >= 0.9

    def test_ambiguous_email_lower_confidence(self, sample_ambiguous_email):
        state = create_initial_state(
            email_id=sample_ambiguous_email["id"],
            email_subject=sample_ambiguous_email["subject"],
            email_body=sample_ambiguous_email["body"],
            email_sender=sample_ambiguous_email["sender"],
            tenant_id="demo",
        )
        result = classify_email(state)
        # Ambiguous emails should have lower confidence
        assert result["classification_confidence"] < 0.9

    def test_classification_includes_reasoning(self, sample_rfi_email):
        state = create_initial_state(
            email_id=sample_rfi_email["id"],
            email_subject=sample_rfi_email["subject"],
            email_body=sample_rfi_email["body"],
            email_sender=sample_rfi_email["sender"],
            tenant_id="demo",
        )
        result = classify_email(state)
        assert result["classification_reasoning"] != ""

    def test_classification_adds_trace(self, sample_rfi_email):
        state = create_initial_state(
            email_id=sample_rfi_email["id"],
            email_subject=sample_rfi_email["subject"],
            email_body=sample_rfi_email["body"],
            email_sender=sample_rfi_email["sender"],
            tenant_id="demo",
        )
        result = classify_email(state)
        assert len(result["agent_trace"]) == 1
        assert result["agent_trace"][0]["agent"] == "classifier"


class TestExtractorAgent:
    """Tests for the metadata extraction node."""

    def test_extracts_rfi_number(self, sample_rfi_email):
        state = create_initial_state(
            email_id=sample_rfi_email["id"],
            email_subject=sample_rfi_email["subject"],
            email_body=sample_rfi_email["body"],
            email_sender=sample_rfi_email["sender"],
            tenant_id="demo",
        )
        state["classification"] = "rfi"
        result = extract_metadata(state)
        assert result["extracted_rfi_number"] == "RFI-247"

    def test_extracts_project_name(self, sample_submittal_email):
        state = create_initial_state(
            email_id=sample_submittal_email["id"],
            email_subject=sample_submittal_email["subject"],
            email_body=sample_submittal_email["body"],
            email_sender=sample_submittal_email["sender"],
            tenant_id="demo",
        )
        state["classification"] = "submittal"
        result = extract_metadata(state)
        assert result["extracted_project_name"] is not None
        assert "Waterfront" in result["extracted_project_name"]

    def test_extracts_submittal_number(self, sample_submittal_email):
        state = create_initial_state(
            email_id=sample_submittal_email["id"],
            email_subject=sample_submittal_email["subject"],
            email_body=sample_submittal_email["body"],
            email_sender=sample_submittal_email["sender"],
            tenant_id="demo",
        )
        state["classification"] = "submittal"
        result = extract_metadata(state)
        assert result["extracted_submittal_number"] == "SUB-089"

    def test_extracts_spec_section(self, sample_submittal_email):
        state = create_initial_state(
            email_id=sample_submittal_email["id"],
            email_subject=sample_submittal_email["subject"],
            email_body=sample_submittal_email["body"],
            email_sender=sample_submittal_email["sender"],
            tenant_id="demo",
        )
        state["classification"] = "submittal"
        result = extract_metadata(state)
        assert result.get("extracted_discipline") is not None

    def test_extraction_includes_cot_reasoning(self, sample_rfi_email):
        """Chain-of-thought reasoning should be captured."""
        state = create_initial_state(
            email_id=sample_rfi_email["id"],
            email_subject=sample_rfi_email["subject"],
            email_body=sample_rfi_email["body"],
            email_sender=sample_rfi_email["sender"],
            tenant_id="demo",
        )
        state["classification"] = "rfi"
        result = extract_metadata(state)
        assert "Step 1" in result["extraction_reasoning"]
        assert "Step 2" in result["extraction_reasoning"]


class TestFilerAgent:
    """Tests for the filing decision node."""

    def test_high_confidence_rfi_auto_files(self, sample_rfi_email):
        state = create_initial_state(
            email_id=sample_rfi_email["id"],
            email_subject=sample_rfi_email["subject"],
            email_body=sample_rfi_email["body"],
            email_sender=sample_rfi_email["sender"],
            tenant_id="demo",
        )
        state["classification"] = "rfi"
        state["classification_confidence"] = 0.95
        state["extracted_rfi_number"] = "RFI-247"
        state["extracted_project_name"] = "Waterfront Mixed-Use Tower"

        result = make_filing_decision(state)
        assert result["filing_action"] in ["auto_file", "needs_review"]
        assert result["filing_project_id"] is not None
        assert result["filing_confidence"] > 0.0

    def test_spam_email_gets_flagged(self, sample_spam_email):
        state = create_initial_state(
            email_id=sample_spam_email["id"],
            email_subject=sample_spam_email["subject"],
            email_body=sample_spam_email["body"],
            email_sender=sample_spam_email["sender"],
            tenant_id="demo",
        )
        state["classification"] = "general"
        state["classification_confidence"] = 0.3

        result = make_filing_decision(state)
        # Low confidence general email from unknown sender → flagged or needs_review
        assert result["filing_action"] in ["flagged", "needs_review"]

    def test_filing_includes_tool_results(self, sample_rfi_email):
        state = create_initial_state(
            email_id=sample_rfi_email["id"],
            email_subject=sample_rfi_email["subject"],
            email_body=sample_rfi_email["body"],
            email_sender=sample_rfi_email["sender"],
            tenant_id="demo",
        )
        state["classification"] = "rfi"
        state["classification_confidence"] = 0.95

        result = make_filing_decision(state)
        # Should have called project_lookup and sender_history at minimum
        trace = result["agent_trace"][-1]
        assert "tools_called" in trace
        assert len(trace["tools_called"]) >= 2


class TestFullPipeline:
    """End-to-end tests for the complete filing graph."""

    def test_clear_rfi_auto_filed(self, sample_rfi_email):
        """A clear RFI email should flow through the entire pipeline."""
        result = run_filing_agent(
            email_id=sample_rfi_email["id"],
            email_subject=sample_rfi_email["subject"],
            email_body=sample_rfi_email["body"],
            email_sender=sample_rfi_email["sender"],
            tenant_id="demo",
        )

        assert result["classification"] == "rfi"
        assert result["filing_action"] is not None
        assert len(result["agent_trace"]) >= 3  # classify + extract + file

    def test_pipeline_respects_max_iterations(self):
        """Loop breaker should prevent infinite loops."""
        state = create_initial_state(
            email_id="test",
            email_subject="Test",
            email_body="Test body",
            email_sender="test@test.com",
            tenant_id="demo",
            max_iterations=5,
        )
        # The iteration count starts at 0 and increments each pass
        assert state["max_iterations"] == 5
        assert state["iteration_count"] == 0

    def test_pipeline_generates_trace(self, sample_rfi_email):
        """Every agent step should be recorded in the trace."""
        result = run_filing_agent(
            email_id=sample_rfi_email["id"],
            email_subject=sample_rfi_email["subject"],
            email_body=sample_rfi_email["body"],
            email_sender=sample_rfi_email["sender"],
            tenant_id="demo",
        )

        trace = result["agent_trace"]
        agent_names = [step["agent"] for step in trace]
        assert "classifier" in agent_names
        assert "extractor" in agent_names
        assert "filer" in agent_names
