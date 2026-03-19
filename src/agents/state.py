"""Agent state definition for the email filing multi-agent system.

The state is the single source of truth across all agents in the graph.
Every node reads from and writes to this TypedDict. Hand-offs between
agents happen through state — not direct calls.
"""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class EmailFilingState(TypedDict):
    """Shared state for the email filing agent graph.

    Fields are populated progressively as the email moves through
    the pipeline: classify → extract → file.
    """

    # Message history for LLM context
    messages: Annotated[list, add_messages]

    # Input
    tenant_id: str
    email_id: str
    email_subject: str
    email_body: str
    email_sender: str
    email_attachments: list[str]

    # Classification agent output
    classification: Optional[str]          # EmailCategory value
    classification_confidence: float
    classification_reasoning: str

    # Extraction agent output
    extracted_project_number: Optional[str]
    extracted_project_name: Optional[str]
    extracted_rfi_number: Optional[str]
    extracted_submittal_number: Optional[str]
    extracted_discipline: Optional[str]
    extracted_urgency: str
    extraction_reasoning: str              # Chain-of-thought trace

    # Tool results
    project_lookup_results: list[dict]     # From project_lookup tool
    rfi_match_results: list[dict]          # From rfi_matcher tool
    sender_history_results: list[dict]     # From sender_history tool

    # Filing decision
    filing_action: Optional[str]           # auto_file / needs_review / flagged
    filing_project_id: Optional[str]
    filing_folder_path: Optional[str]
    filing_confidence: float
    filing_reasoning: str

    # Control flow
    current_agent: str                     # Which agent is running
    iteration_count: int                   # For loop breaking
    max_iterations: int                    # Configurable per tenant
    needs_human_review: bool               # HITL flag
    human_feedback: Optional[str]          # Human's decision after review

    # Observability
    agent_trace: list[dict]                # Execution trace for debugging
    total_cost_usd: float                  # Accumulated LLM cost
