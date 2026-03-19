"""Shared test fixtures for Project Atlas."""

import json
import os

import pytest


@pytest.fixture
def sample_rfi_email():
    """A clear RFI email — should be auto-classified with high confidence."""
    return {
        "id": "test_email_001",
        "sender": "s.chen@pacificsteel.com",
        "sender_name": "Sarah Chen",
        "subject": "RE: RFI-247 - Structural Steel Connection Detail at Grid Line J-7",
        "body": "Hi Alex,\n\nFollowing up on RFI-247 regarding the steel connection detail at Grid J-7. We've reviewed the structural drawings (S-401 and S-402) and have concerns about the moment connection.\n\nCan you confirm the design intent? We need to order connection material by end of next week.\n\nThanks,\nSarah Chen\nPacific Steel Erectors",
        "attachments": ["J7_markup.pdf"],
        "recipients": ["a.thornton@thorntonarch.com"],
    }


@pytest.fixture
def sample_submittal_email():
    """A submittal email."""
    return {
        "id": "test_email_002",
        "sender": "k.walsh@summitmech.com",
        "sender_name": "Karen Walsh",
        "subject": "Submittal SUB-089 - HVAC Diffusers per Spec 23 37 00",
        "body": "Please find attached the submittal package for the HVAC linear slot diffusers per Spec Section 23 37 00 for the Waterfront Mixed-Use Tower.\n\nRequested review period: 10 business days",
        "attachments": ["SUB-089_product_data.pdf"],
        "recipients": ["p.patel@thorntonarch.com"],
    }


@pytest.fixture
def sample_ambiguous_email():
    """An email that's hard to classify — should trigger HITL."""
    return {
        "id": "test_email_003",
        "sender": "unknown@newcontractor.com",
        "sender_name": "New Person",
        "subject": "Question about the project",
        "body": "Hi,\n\nI have a question about the specifications. Can someone point me to the right document?\n\nThanks",
        "attachments": [],
        "recipients": ["a.thornton@thorntonarch.com"],
    }


@pytest.fixture
def sample_spam_email():
    """A spam/marketing email — should be flagged."""
    return {
        "id": "test_email_004",
        "sender": "noreply@acmesupply.com",
        "sender_name": "ACME Construction Supply",
        "subject": "Your Weekly Deals - Save 15% on Safety Equipment!",
        "body": "SPRING SALE! Save 15% on all safety equipment this week only!\n\nShop now at acmesupply.com\n\nUnsubscribe: reply STOP",
        "attachments": [],
        "recipients": ["a.thornton@thorntonarch.com"],
    }


@pytest.fixture
def sample_projects():
    """Load sample project data."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_projects.json"
    )
    with open(data_path) as f:
        return json.load(f)


@pytest.fixture
def tenant_id():
    """Default test tenant."""
    return "demo"
