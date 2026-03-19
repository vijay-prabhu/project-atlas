"""AECO-specific document chunking with strategies for emails, specs, and drawings.

Each document type gets its own chunking logic because AECO documents have very
different structures. Emails have reply chains, specs have numbered sections,
and drawing notes are short callouts.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.observability import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """A single chunk of text with position info and metadata."""

    content: str
    start_index: int
    end_index: int
    chunk_type: str
    metadata: dict = field(default_factory=dict)


# ─── Email Chunking ─────────────────────────────────────


# Patterns that mark reply chain boundaries in AECO email threads
_REPLY_PATTERNS = [
    re.compile(r"^-{3,}\s*Original Message\s*-{3,}", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^On\s+.+wrote:\s*$", re.MULTILINE),
    re.compile(r"^From:\s+", re.MULTILINE),
    re.compile(r"^>{3,}", re.MULTILINE),
]

# Common email signature markers
_SIGNATURE_PATTERNS = [
    re.compile(r"^--\s*$", re.MULTILINE),
    re.compile(r"^Regards,?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Best regards,?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Thanks,?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Sent from my", re.IGNORECASE | re.MULTILINE),
]


def _find_earliest_match(text: str, patterns: list[re.Pattern]) -> Optional[int]:
    """Find the position of the earliest pattern match in text."""
    earliest = None
    for pattern in patterns:
        match = pattern.search(text)
        if match and (earliest is None or match.start() < earliest):
            earliest = match.start()
    return earliest


def _split_email_sections(body: str) -> list[tuple[str, str]]:
    """Split email body into labeled sections: header, body, signature, replies.

    Returns list of (section_label, section_text) tuples.
    """
    sections: list[tuple[str, str]] = []

    # Find where the reply chain starts
    reply_pos = _find_earliest_match(body, _REPLY_PATTERNS)

    if reply_pos is not None:
        main_part = body[:reply_pos].strip()
        reply_part = body[reply_pos:].strip()
    else:
        main_part = body.strip()
        reply_part = ""

    # Split the main part into body and signature
    sig_pos = _find_earliest_match(main_part, _SIGNATURE_PATTERNS)
    if sig_pos is not None:
        body_text = main_part[:sig_pos].strip()
        sig_text = main_part[sig_pos:].strip()
        if body_text:
            sections.append(("body", body_text))
        if sig_text:
            sections.append(("signature", sig_text))
    else:
        if main_part:
            sections.append(("body", main_part))

    # Split reply chain into individual replies
    if reply_part:
        # Split on any reply boundary pattern
        reply_splits = re.split(
            r"(?=(?:^-{3,}\s*Original Message|^On\s+.+wrote:|^From:\s+))",
            reply_part,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        for i, reply in enumerate(reply_splits):
            reply = reply.strip()
            if reply:
                sections.append((f"reply_{i}", reply))

    return sections


def chunk_email(email_body: str, subject: str = "") -> list[TextChunk]:
    """Chunk an email body preserving reply chain boundaries.

    Splits the email into sections (body, signature, replies) first, then
    runs RecursiveCharacterTextSplitter on each section. Subject line is
    prepended to the first chunk's content for context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    sections = _split_email_sections(email_body)
    chunks: list[TextChunk] = []
    running_offset = 0

    for section_label, section_text in sections:
        # Find where this section starts in the original body
        section_start = email_body.find(section_text, running_offset)
        if section_start == -1:
            section_start = running_offset

        split_texts = splitter.split_text(section_text)

        for i, text in enumerate(split_texts):
            # Prepend subject to the very first chunk of the body section
            content = text
            if section_label == "body" and i == 0 and subject:
                content = f"Subject: {subject}\n\n{text}"

            # Calculate position within the original body
            text_pos = section_text.find(text)
            start_idx = section_start + (text_pos if text_pos >= 0 else 0)
            end_idx = start_idx + len(text)

            chunks.append(TextChunk(
                content=content,
                start_index=start_idx,
                end_index=end_idx,
                chunk_type="email",
                metadata={
                    "section": section_label,
                    "subject": subject,
                },
            ))

        running_offset = section_start + len(section_text)

    logger.info(
        "Email chunked",
        extra={"chunk_count": len(chunks), "subject": subject[:80]},
    )
    return chunks


# ─── Specification Chunking ──────────────────────────────


# AECO specification section header patterns (CSI MasterFormat style)
_SPEC_SECTION_PATTERN = re.compile(
    r"^(?:"
    r"PART\s+\d+\s*[-–—]\s*\w+"           # PART 1 - GENERAL
    r"|\d+\.\d+\s+[A-Z]"                   # 2.01 MATERIALS
    r"|SECTION\s+\d+"                       # SECTION 23 37 00
    r"|ARTICLE\s+\d+"                       # ARTICLE 5
    r"|END\s+OF\s+SECTION"                  # END OF SECTION
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def chunk_specification(text: str) -> list[TextChunk]:
    """Chunk specification text respecting section boundaries.

    AECO specs are structured with numbered sections (PART 1 - GENERAL,
    2.01 MATERIALS, etc). We split on those headers first so we never
    break mid-section, then apply RecursiveCharacterTextSplitter within
    sections that are still too long.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Find all section boundaries
    matches = list(_SPEC_SECTION_PATTERN.finditer(text))

    if not matches:
        # No section headers found — fall back to basic splitting
        split_texts = splitter.split_text(text)
        return [
            TextChunk(
                content=t,
                start_index=text.find(t),
                end_index=text.find(t) + len(t),
                chunk_type="specification",
                metadata={},
            )
            for t in split_texts
        ]

    # Build sections from header positions
    sections: list[tuple[str, int, int]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((section_text, start, end))

    # Include any text before the first section header
    if matches and matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.insert(0, (preamble, 0, matches[0].start()))

    chunks: list[TextChunk] = []
    for section_text, section_start, section_end in sections:
        # Extract the section header for metadata
        header_match = _SPEC_SECTION_PATTERN.match(section_text)
        header = header_match.group(0).strip() if header_match else ""

        # If the section fits in one chunk, keep it whole
        if len(section_text) <= 1024:
            chunks.append(TextChunk(
                content=section_text,
                start_index=section_start,
                end_index=section_start + len(section_text),
                chunk_type="specification",
                metadata={"section_header": header},
            ))
        else:
            # Section is too long — split it but prepend the header to each chunk
            split_texts = splitter.split_text(section_text)
            for text_part in split_texts:
                pos = section_text.find(text_part)
                abs_start = section_start + (pos if pos >= 0 else 0)

                # Prepend header if this sub-chunk doesn't already start with it
                content = text_part
                if header and not text_part.startswith(header):
                    content = f"{header}\n{text_part}"

                chunks.append(TextChunk(
                    content=content,
                    start_index=abs_start,
                    end_index=abs_start + len(text_part),
                    chunk_type="specification",
                    metadata={"section_header": header},
                ))

    logger.info(
        "Specification chunked",
        extra={"chunk_count": len(chunks), "section_count": len(sections)},
    )
    return chunks


# ─── Drawing Notes Chunking ─────────────────────────────


# Patterns for drawing sheet references and callout numbers
_DRAWING_SEPARATORS = [
    r"\n(?=Sheet\s+[A-Z]?\d+)",      # Sheet A1, Sheet 1
    r"\n(?=DWG\s+\d+)",               # DWG 001
    r"\n(?=Detail\s+\d+)",            # Detail 1
    r"\n(?=\d+[.)]\s+)",              # 1. or 1) callout
    r"\n(?=[A-Z]\d+[.:]\s)",          # A1: callout style
    r"\n\n",
]


def chunk_drawing_notes(text: str) -> list[TextChunk]:
    """Chunk drawing note text by sheet references and callout numbers.

    Drawing notes come from multimodal extraction of architectural/engineering
    drawings. They tend to be short so we use a small chunk size (256).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=25,
        separators=_DRAWING_SEPARATORS + ["\n", ". ", " ", ""],
    )

    split_texts = splitter.split_text(text)
    chunks: list[TextChunk] = []

    for part in split_texts:
        pos = text.find(part)
        start_idx = pos if pos >= 0 else 0

        # Try to extract a sheet or detail reference for metadata
        sheet_match = re.search(
            r"(Sheet\s+[A-Z]?\d+|DWG\s+\d+|Detail\s+\d+)",
            part,
            re.IGNORECASE,
        )
        sheet_ref = sheet_match.group(0) if sheet_match else ""

        chunks.append(TextChunk(
            content=part,
            start_index=start_idx,
            end_index=start_idx + len(part),
            chunk_type="drawing",
            metadata={"sheet_reference": sheet_ref},
        ))

    logger.info(
        "Drawing notes chunked",
        extra={"chunk_count": len(chunks)},
    )
    return chunks


# ─── Auto Router ─────────────────────────────────────────


_CHUNKERS = {
    "email": lambda text: chunk_email(text),
    "specification": chunk_specification,
    "spec": chunk_specification,
    "drawing": chunk_drawing_notes,
    "drawing_notes": chunk_drawing_notes,
}


def chunk_auto(text: str, doc_type: str) -> list[TextChunk]:
    """Route to the right chunking strategy based on document type.

    Supported doc_type values: email, specification, spec, drawing, drawing_notes.
    Falls back to specification chunking for unknown types since it's the most
    general-purpose.
    """
    chunker = _CHUNKERS.get(doc_type.lower())
    if chunker is None:
        logger.warning(
            "Unknown doc_type, falling back to specification chunker",
            extra={"doc_type": doc_type},
        )
        chunker = chunk_specification

    return chunker(text)
