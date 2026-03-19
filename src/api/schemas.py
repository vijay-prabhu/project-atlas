"""Pydantic models for all domain objects and API request/response schemas."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────


class EmailCategory(str, Enum):
    RFI = "rfi"
    SUBMITTAL = "submittal"
    CHANGE_ORDER = "change_order"
    TRANSMITTAL = "transmittal"
    MEETING_MINUTES = "meeting_minutes"
    DAILY_REPORT = "daily_report"
    SHOP_DRAWING = "shop_drawing"
    GENERAL = "general"


class FilingAction(str, Enum):
    AUTO_FILE = "auto_file"
    NEEDS_REVIEW = "needs_review"
    FLAGGED = "flagged"


class ProcessingStatus(str, Enum):
    RECEIVED = "received"
    CLASSIFYING = "classifying"
    EXTRACTING = "extracting"
    FILING = "filing"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class Discipline(str, Enum):
    ARCHITECTURAL = "architectural"
    STRUCTURAL = "structural"
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    CIVIL = "civil"
    LANDSCAPE = "landscape"
    GENERAL = "general"


# ─── Domain Models ────────────────────────────────────────


class AECOEmail(BaseModel):
    """An email from the AECO industry to be classified and filed."""

    id: str = Field(default="", description="Unique email identifier")
    sender: str = Field(..., description="Sender email address")
    sender_name: str = Field(default="", description="Sender display name")
    recipients: list[str] = Field(default_factory=list)
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body text")
    attachments: list[str] = Field(default_factory=list, description="Attachment filenames")
    received_at: datetime = Field(default_factory=datetime.now)
    thread_id: Optional[str] = Field(default=None, description="Email thread ID for context")
    project_hint: Optional[str] = Field(default=None, description="Project reference if obvious from subject")


class Project(BaseModel):
    """An AECO project that emails get filed to."""

    id: str
    number: str = Field(..., description="Project number like P-2024-0847")
    name: str = Field(..., description="Project name like 'Waterfront Mixed-Use Tower'")
    client: str = Field(default="")
    status: str = Field(default="active")
    description: str = Field(default="")
    disciplines: list[Discipline] = Field(default_factory=list)


class RFI(BaseModel):
    """Request for Information tied to a project."""

    id: str
    number: str = Field(..., description="RFI number like RFI-247")
    subject: str
    from_company: str = Field(default="")
    to_company: str = Field(default="")
    status: str = Field(default="open")
    project_id: str = Field(...)
    due_date: Optional[datetime] = None


class Submittal(BaseModel):
    """A submittal log entry tied to a project."""

    id: str
    number: str = Field(..., description="Submittal number like SUB-089")
    spec_section: str = Field(default="", description="Spec section like '23 37 00'")
    description: str = Field(default="")
    status: str = Field(default="pending")
    project_id: str = Field(...)


class Contact(BaseModel):
    """A person involved in AECO projects."""

    name: str
    email: str
    company: str = ""
    role: str = ""
    project_ids: list[str] = Field(default_factory=list)


# ─── Agent Output Models ─────────────────────────────────


class EmailClassification(BaseModel):
    """Output of the classifier agent."""

    category: EmailCategory
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Why this classification was chosen")


class ExtractedMetadata(BaseModel):
    """Output of the extractor agent."""

    project_number: Optional[str] = None
    project_name: Optional[str] = None
    discipline: Optional[Discipline] = None
    rfi_number: Optional[str] = None
    submittal_number: Optional[str] = None
    spec_section: Optional[str] = None
    urgency: str = Field(default="normal")
    action_items: list[str] = Field(default_factory=list)
    reasoning_trace: str = Field(default="", description="Chain-of-thought reasoning")


class FilingDecision(BaseModel):
    """Output of the filer agent."""

    action: FilingAction
    project_id: Optional[str] = None
    folder_path: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(default="")


class FilingResult(BaseModel):
    """Complete result of the email filing pipeline."""

    email_id: str
    status: ProcessingStatus
    classification: Optional[EmailClassification] = None
    metadata: Optional[ExtractedMetadata] = None
    decision: Optional[FilingDecision] = None
    trace: list[dict] = Field(default_factory=list, description="Agent execution trace")
    cost_usd: float = Field(default=0.0, description="Total LLM cost for this filing")


# ─── Search Models ────────────────────────────────────────


class SearchQuery(BaseModel):
    """Incoming search request."""

    query: str = Field(..., min_length=3)
    filters: dict = Field(default_factory=dict, description="Optional metadata filters")
    top_k: int = Field(default=10, ge=1, le=50)
    search_type: str = Field(default="hybrid", description="hybrid, semantic, or keyword")


class Citation(BaseModel):
    """A citation linking an answer claim to a source document."""

    claim: str
    source_document: str
    source_chunk: str
    page_or_section: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    verified: bool = Field(default=False, description="Whether claim was verified against source")


class SearchResult(BaseModel):
    """A single search result with source information."""

    chunk_text: str
    score: float
    source_type: str = Field(default="email", description="email, document, rfi, submittal")
    source_id: str = ""
    source_title: str = ""
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Complete search response with answer, results, and citations."""

    answer: str = Field(default="", description="Generated answer from RAG")
    results: list[SearchResult] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    search_type_used: str = "hybrid"
    total_found: int = 0
    confidence: float = Field(default=0.0)


# ─── API Request/Response ─────────────────────────────────


class FileEmailRequest(BaseModel):
    email: AECOEmail


class FileEmailResponse(BaseModel):
    filing_result: FilingResult


class ApproveFilingRequest(BaseModel):
    approved: bool
    corrected_project_id: Optional[str] = None
    feedback: Optional[str] = None


class FeedbackRequest(BaseModel):
    query: str
    result_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
