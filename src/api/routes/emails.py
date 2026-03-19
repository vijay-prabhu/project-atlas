"""Email filing API routes."""

from fastapi import APIRouter, Depends

from src.agents.checkpoints import get_checkpoint_store
from src.agents.graph import run_filing_agent
from src.api.deps import get_tenant_id
from src.api.schemas import (
    ApproveFilingRequest,
    EmailClassification,
    ExtractedMetadata,
    FileEmailRequest,
    FileEmailResponse,
    FilingAction,
    FilingDecision,
    FilingResult,
    ProcessingStatus,
)

router = APIRouter(prefix="/emails", tags=["emails"])


@router.post("/file", response_model=FileEmailResponse)
async def file_email(
    request: FileEmailRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """File an email using the multi-agent system.

    Runs the full pipeline: classify → extract → file.
    Returns the filing result with agent trace.
    """
    email = request.email

    result = run_filing_agent(
        email_id=email.id or "auto",
        email_subject=email.subject,
        email_body=email.body,
        email_sender=email.sender,
        tenant_id=tenant_id,
        email_attachments=email.attachments,
    )

    # Build response from agent state
    classification = None
    if result.get("classification"):
        classification = EmailClassification(
            category=result["classification"],
            confidence=result.get("classification_confidence", 0.0),
            reasoning=result.get("classification_reasoning", ""),
        )

    metadata = None
    if result.get("extracted_project_number") or result.get("extracted_project_name"):
        metadata = ExtractedMetadata(
            project_number=result.get("extracted_project_number"),
            project_name=result.get("extracted_project_name"),
            discipline=result.get("extracted_discipline"),
            rfi_number=result.get("extracted_rfi_number"),
            submittal_number=result.get("extracted_submittal_number"),
            urgency=result.get("extracted_urgency", "normal"),
            reasoning_trace=result.get("extraction_reasoning", ""),
        )

    decision = None
    if result.get("filing_action"):
        decision = FilingDecision(
            action=FilingAction(result["filing_action"]),
            project_id=result.get("filing_project_id"),
            folder_path=result.get("filing_folder_path"),
            confidence=result.get("filing_confidence", 0.0),
            reasoning=result.get("filing_reasoning", ""),
        )

    status = ProcessingStatus.COMPLETED
    if result.get("needs_human_review"):
        status = ProcessingStatus.NEEDS_REVIEW

    filing_result = FilingResult(
        email_id=email.id or "auto",
        status=status,
        classification=classification,
        metadata=metadata,
        decision=decision,
        trace=result.get("agent_trace", []),
        cost_usd=result.get("total_cost_usd", 0.0),
    )

    return FileEmailResponse(filing_result=filing_result)


@router.post("/{email_id}/approve")
async def approve_filing(
    email_id: str,
    request: ApproveFilingRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """Approve or correct a filing that needs human review.

    This is the HITL callback — it resumes the checkpointed agent graph
    with the human's decision.
    """
    store = get_checkpoint_store()

    decision = "approve" if request.approved else "reject"
    if request.corrected_project_id:
        decision = "correct"

    updated_state = store.resume(
        thread_id=email_id,
        human_decision=decision,
        corrected_project_id=request.corrected_project_id,
        tenant_id=tenant_id,
    )

    if not updated_state:
        return {"error": "No pending review found for this email"}

    return {
        "status": "resumed",
        "email_id": email_id,
        "decision": decision,
        "filing_action": updated_state.get("filing_action"),
        "filing_project_id": updated_state.get("filing_project_id"),
    }


@router.get("/{email_id}")
async def get_filing_status(
    email_id: str,
    tenant_id: str = Depends(get_tenant_id),
):
    """Get the filing status for an email."""
    store = get_checkpoint_store()
    record = store.load(email_id, tenant_id)

    if not record:
        return {"error": "Filing not found"}

    return {
        "email_id": email_id,
        "status": record.status,
        "checkpoint_id": record.checkpoint_id,
        "state": record.state,
    }
