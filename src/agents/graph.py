"""LangGraph state graph assembly for the email filing multi-agent system.

This is the orchestration layer that wires together:
- Classifier (Node 1): determine email type
- Extractor (Node 2): pull structured metadata
- Filer (Node 3): make filing decision with tool-calling

Routing logic:
- After classification: always proceed to extraction
- After extraction: always proceed to filing
- After filing: check confidence → auto_file / needs_review / flagged
- Loop breaker: if iteration_count > max_iterations, force a decision

Hand-off between agents happens through the shared state — no direct
function calls between agents. Each agent reads from state and writes
back to state.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.checkpoints import should_request_human_review
from src.agents.classifier import classify_email
from src.agents.extractor import extract_metadata
from src.agents.filer import make_filing_decision
from src.agents.state import EmailFilingState
from src.core.observability import get_logger

logger = get_logger(__name__)


def _route_after_filing(state: EmailFilingState) -> str:
    """Conditional edge after the filing decision.

    Routes based on:
    1. Confidence thresholds → auto_file / needs_review / flagged
    2. Loop breaker → force decision if max iterations exceeded
    3. Priority hierarchy → flagged > needs_review > auto_file
    """
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)

    # Loop breaker: force completion if we've iterated too many times
    if iteration >= max_iterations:
        logger.info(
            "loop_breaker_triggered",
            extra={
                "iteration_count": iteration,
                "max_iterations": max_iterations,
                "forced_action": state.get("filing_action", "flagged"),
            },
        )
        return "complete"

    action = state.get("filing_action", "flagged")

    if action == "auto_file":
        return "complete"
    elif action == "needs_review":
        return "human_review"
    else:
        return "complete"


def _increment_iteration(state: EmailFilingState) -> dict:
    """Simple node that increments the iteration counter."""
    return {
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def _mark_complete(state: EmailFilingState) -> dict:
    """Terminal node that marks the filing as complete."""
    action = state.get("filing_action", "flagged")
    confidence = state.get("filing_confidence", 0.0)
    project_id = state.get("filing_project_id")

    logger.info(
        "filing_complete",
        extra={
            "email_id": state.get("email_id"),
            "action": action,
            "confidence": confidence,
            "project_id": project_id,
        },
    )

    return {
        "current_agent": "complete",
        "agent_trace": state.get("agent_trace", []) + [{
            "agent": "graph",
            "action": "filing_complete",
            "filing_action": action,
            "confidence": confidence,
            "project_id": project_id,
        }],
    }


def _human_review_node(state: EmailFilingState) -> dict:
    """HITL node — pauses the graph for human review.

    When the filing confidence is between 0.5 and 0.85, the graph
    pauses here. The state is checkpointed, and a notification is
    sent for human review. The graph resumes when the human provides
    feedback via the API.
    """
    return {
        "needs_human_review": True,
        "current_agent": "human_review",
        "agent_trace": state.get("agent_trace", []) + [{
            "agent": "graph",
            "action": "requesting_human_review",
            "confidence": state.get("filing_confidence", 0.0),
            "suggested_project": state.get("filing_project_id"),
        }],
    }


def build_filing_graph(checkpointer=None):
    """Build and compile the email filing LangGraph state graph.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
                      Uses MemorySaver by default (in-memory).
                      For production, use DynamoDB-backed checkpointer.

    Returns:
        Compiled LangGraph that can be invoked with .invoke() or .stream()
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    workflow = StateGraph(EmailFilingState)

    # ── Add nodes ──────────────────────────────────────────
    workflow.add_node("increment_iteration", _increment_iteration)
    workflow.add_node("classify", classify_email)
    workflow.add_node("extract", extract_metadata)
    workflow.add_node("file", make_filing_decision)
    workflow.add_node("human_review", _human_review_node)
    workflow.add_node("complete", _mark_complete)

    # ── Set entry point ────────────────────────────────────
    workflow.set_entry_point("increment_iteration")

    # ── Add edges ──────────────────────────────────────────
    # Linear flow: increment → classify → extract → file
    workflow.add_edge("increment_iteration", "classify")
    workflow.add_edge("classify", "extract")
    workflow.add_edge("extract", "file")

    # Conditional edge after filing: route based on confidence
    workflow.add_conditional_edges(
        "file",
        _route_after_filing,
        {
            "complete": "complete",
            "human_review": "human_review",
        },
    )

    # Human review leads to completion (after human provides input)
    workflow.add_edge("human_review", "complete")

    # Terminal node
    workflow.add_edge("complete", END)

    # ── Compile with checkpointer ──────────────────────────
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],  # Pause before human review for HITL
    )

    return graph


def create_initial_state(
    email_id: str,
    email_subject: str,
    email_body: str,
    email_sender: str,
    tenant_id: str,
    email_attachments: list[str] = None,
    max_iterations: int = 5,
) -> EmailFilingState:
    """Create the initial state for a new email filing run."""
    return {
        "messages": [],
        "tenant_id": tenant_id,
        "email_id": email_id,
        "email_subject": email_subject,
        "email_body": email_body,
        "email_sender": email_sender,
        "email_attachments": email_attachments or [],
        # Classification (populated by classifier)
        "classification": None,
        "classification_confidence": 0.0,
        "classification_reasoning": "",
        # Extraction (populated by extractor)
        "extracted_project_number": None,
        "extracted_project_name": None,
        "extracted_rfi_number": None,
        "extracted_submittal_number": None,
        "extracted_discipline": None,
        "extracted_urgency": "normal",
        "extraction_reasoning": "",
        # Tool results (populated by filer)
        "project_lookup_results": [],
        "rfi_match_results": [],
        "sender_history_results": [],
        # Filing decision (populated by filer)
        "filing_action": None,
        "filing_project_id": None,
        "filing_folder_path": None,
        "filing_confidence": 0.0,
        "filing_reasoning": "",
        # Control flow
        "current_agent": "start",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "needs_human_review": False,
        "human_feedback": None,
        # Observability
        "agent_trace": [],
        "total_cost_usd": 0.0,
    }


def run_filing_agent(
    email_id: str,
    email_subject: str,
    email_body: str,
    email_sender: str,
    tenant_id: str,
    email_attachments: list[str] = None,
    thread_id: str = None,
) -> dict:
    """Run the full email filing pipeline.

    This is the main entry point. Creates the graph, builds initial state,
    and runs the agent to completion (or until HITL pause).

    Args:
        thread_id: Unique ID for this run. Used for checkpointing.
                   If resuming after HITL, pass the same thread_id.
    """
    graph = build_filing_graph()

    initial_state = create_initial_state(
        email_id=email_id,
        email_subject=email_subject,
        email_body=email_body,
        email_sender=email_sender,
        tenant_id=tenant_id,
        email_attachments=email_attachments,
    )

    config = {"configurable": {"thread_id": thread_id or email_id}}

    # Run the graph — may pause at human_review interrupt
    result = graph.invoke(initial_state, config=config)

    # If the graph paused for human review, save to our checkpoint store
    # so the Pending Reviews page can show it
    if result.get("filing_action") == "needs_review":
        from src.agents.checkpoints import get_checkpoint_store
        store = get_checkpoint_store()
        store.save(
            thread_id=thread_id or email_id,
            state=dict(result),
            tenant_id=tenant_id,
        )
        result["needs_human_review"] = True

    return result
