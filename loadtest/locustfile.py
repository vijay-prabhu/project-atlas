"""Locust load test for Project Atlas.

Simulates multi-tenant traffic against the email filing and search APIs.

Usage:
    # Web UI mode
    locust -f loadtest/locustfile.py --host http://localhost:8000

    # Headless mode — 50 users, 5/sec spawn rate, 2 minutes
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 5 -t 2m

    # Multi-tenant simulation
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 100 -r 10 -t 5m
"""

import random
import uuid

from locust import HttpUser, between, task

from src.data.generator import generate_email


class TenantAUser(HttpUser):
    """Simulates a large tenant (architecture firm) with heavy email filing."""

    weight = 3  # 3x more traffic than small tenants
    wait_time = between(0.5, 2)
    tenant_id = "tenant_a"
    api_key = "sk_test_tenant_a"

    def on_start(self):
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

    @task(5)
    def file_email(self):
        """File a randomly generated AECO email."""
        email = generate_email(self.tenant_id)
        payload = {
            "email": {
                "id": email["id"],
                "sender": email["sender"],
                "sender_name": email["sender_name"],
                "subject": email["subject"],
                "body": email["body"],
                "attachments": email["attachments"],
                "recipients": email["recipients"],
            }
        }
        with self.client.post(
            "/api/v1/emails/file",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/emails/file",
        ) as response:
            if response.status_code == 200:
                result = response.json()
                filing = result.get("filing_result", {})
                if filing.get("status") in ("completed", "needs_review"):
                    response.success()
                else:
                    response.failure(f"Unexpected status: {filing.get('status')}")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def search(self):
        """Run a search query."""
        queries = [
            "What was the resolution on RFI-247?",
            "structural steel connection details",
            "HVAC diffuser conflict with beams",
            "waterproofing at parking level",
            "change order for soil conditions",
            "curtain wall mock-up schedule",
            "RFI-113 anchor bolt spacing",
            "elevator pit drainage",
        ]
        payload = {
            "query": random.choice(queries),
            "top_k": 5,
        }
        self.client.post(
            "/api/v1/search",
            json=payload,
            headers=self.headers,
            name="/api/v1/search",
        )

    @task(1)
    def health_check(self):
        """Check API health."""
        self.client.get("/health", name="/health")

    @task(1)
    def submit_feedback(self):
        """Submit feedback on a search result."""
        payload = {
            "query": "test search query",
            "result_id": f"result_{uuid.uuid4().hex[:8]}",
            "rating": random.randint(1, 5),
            "comment": random.choice([None, "Good result", "Not relevant", "Wrong project"]),
        }
        self.client.post(
            "/api/v1/feedback",
            json=payload,
            headers=self.headers,
            name="/api/v1/feedback",
        )


class TenantBUser(HttpUser):
    """Simulates a smaller tenant (engineering firm) with moderate traffic."""

    weight = 1
    wait_time = between(1, 4)
    tenant_id = "tenant_b"
    api_key = "sk_test_tenant_b"

    def on_start(self):
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

    @task(3)
    def file_email(self):
        """File a randomly generated AECO email."""
        email = generate_email(self.tenant_id)
        payload = {
            "email": {
                "id": email["id"],
                "sender": email["sender"],
                "subject": email["subject"],
                "body": email["body"],
                "attachments": email["attachments"],
                "recipients": email["recipients"],
            }
        }
        self.client.post(
            "/api/v1/emails/file",
            json=payload,
            headers=self.headers,
            name="/api/v1/emails/file",
        )

    @task(2)
    def search(self):
        """Run a search query."""
        queries = [
            "seismic retrofit progress",
            "heritage window replacement",
            "bridge deck reinforcing",
            "medical gas piping route",
        ]
        payload = {
            "query": random.choice(queries),
            "top_k": 5,
        }
        self.client.post(
            "/api/v1/search",
            json=payload,
            headers=self.headers,
            name="/api/v1/search",
        )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")
