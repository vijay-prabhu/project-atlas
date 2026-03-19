"""Project Atlas — Demo Dashboard

Interactive UI for demonstrating the email filing and search system.

Run: streamlit run ui/app.py
"""

import json
import os
import sys
import time

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.graph import run_filing_agent
from src.agents.search_agent import run_search_agent
from src.agents.checkpoints import get_checkpoint_store
from src.data.generator import generate_email

# ─── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Project Atlas",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("🏗️ Project Atlas")
st.sidebar.markdown("Smart Email Filing & Semantic Search for the AECO Industry")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["📧 File Email", "🔍 Search", "⚡ Batch Demo", "📊 Agent Trace Viewer", "📋 Pending Reviews"],
    index=0,
)

tenant_id = st.sidebar.selectbox(
    "Tenant",
    ["demo", "tenant_a", "tenant_b"],
    index=0,
    help="Simulates multi-tenant isolation. Each tenant's data is namespaced separately.",
)

st.sidebar.divider()
st.sidebar.markdown("""
**Tech Stack**
- LangGraph multi-agent system
- Hybrid search (BM25 + vector)
- RAG with citation tracking
- Multi-tenant isolation
- AWS (DynamoDB, Lambda, SQS)
""")

# ─── Helper Functions ─────────────────────────────────────────────────


def render_confidence_bar(confidence: float, label: str = "Confidence"):
    """Render a colored confidence bar."""
    if confidence >= 0.85:
        color = "🟢"
        action = "Auto-file"
    elif confidence >= 0.5:
        color = "🟡"
        action = "Needs Review"
    else:
        color = "🔴"
        action = "Flagged"
    st.metric(label, f"{confidence:.0%}", delta=f"{color} {action}")


def render_trace(trace: list[dict]):
    """Render the agent execution trace as a timeline."""
    if not trace:
        return

    for i, step in enumerate(trace):
        agent = step.get("agent", "unknown")
        action = step.get("action", "")
        duration = step.get("duration_ms", 0)

        icon = {
            "classifier": "🏷️",
            "extractor": "📋",
            "filer": "📁",
            "graph": "✅",
        }.get(agent, "⚙️")

        with st.expander(f"{icon} Step {i+1}: {agent} → {action} ({duration:.1f}ms)", expanded=(i == 0)):
            # Remove agent/action from display since they're in the header
            display = {k: v for k, v in step.items() if k not in ("agent", "action")}
            st.json(display)


# ─── Page: File Email ─────────────────────────────────────────────────

if page == "📧 File Email":
    st.title("📧 File an Email")
    st.markdown("Submit an AECO email and watch the multi-agent system classify, extract metadata, and make a filing decision.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Email Input")

        # Quick-fill buttons — set the widget keys directly, then rerun
        st.markdown("**Quick fill with sample:**")
        qf1, qf2, qf3, qf4 = st.columns(4)

        def _fill(sender, subject, body):
            st.session_state["input_sender"] = sender
            st.session_state["input_subject"] = subject
            st.session_state["input_body"] = body

        qf1.button("🔧 RFI", use_container_width=True, on_click=_fill, args=(
            "s.chen@pacificsteel.com",
            "RE: RFI-247 - Structural Steel Connection Detail at Grid Line J-7",
            "Hi Alex,\n\nFollowing up on RFI-247 regarding the steel connection detail at Grid J-7. We've reviewed the structural drawings (S-401 and S-402) and have concerns about the moment connection.\n\nCan you confirm the design intent? We need to order connection material by end of next week for the Waterfront Mixed-Use Tower.\n\nThanks,\nSarah Chen\nPacific Steel Erectors",
        ))

        qf2.button("📄 Submittal", use_container_width=True, on_click=_fill, args=(
            "k.walsh@summitmech.com",
            "Submittal SUB-089 - HVAC Diffusers per Spec 23 37 00 - Waterfront Tower",
            "Please find attached the submittal package for the HVAC linear slot diffusers per Spec Section 23 37 00 for the Waterfront Mixed-Use Tower.\n\nRequested review period: 10 business days\n\nKaren Walsh\nSummit Mechanical",
        ))

        qf3.button("💰 Change Order", use_container_width=True, on_click=_fill, args=(
            "s.chen@pacificsteel.com",
            "CO #12 - Additional Structural Steel at Roof Level - Waterfront Tower",
            "Per our discussion at the OAC meeting, we're submitting Change Order #12 for the additional structural steel at the roof level.\n\nAdditional steel weight: 18.5 tons\nProposed price: $148,000\nSchedule impact: 5 working days\n\nSarah Chen\nPacific Steel Erectors",
        ))

        qf4.button("❓ Ambiguous", use_container_width=True, on_click=_fill, args=(
            "unknown@newcontractor.com",
            "Question about the project",
            "Hi,\n\nI have a question about the specifications. Can someone point me to the right document?\n\nThanks",
        ))

        sender = st.text_input("From", key="input_sender")
        subject = st.text_input("Subject", key="input_subject")
        body = st.text_area("Body", height=200, key="input_body")

        file_button = st.button("🚀 File Email", type="primary", use_container_width=True)

    with col2:
        st.subheader("Filing Result")

        if file_button and subject and body:
            with st.spinner("Running agent pipeline: classify → extract → file..."):
                start = time.perf_counter()
                result = run_filing_agent(
                    email_id=f"ui_{int(time.time())}",
                    email_subject=subject,
                    email_body=body,
                    email_sender=sender,
                    tenant_id=tenant_id,
                )
                duration = (time.perf_counter() - start) * 1000

            st.success(f"Pipeline complete in {duration:.0f}ms")

            # Classification
            st.markdown("#### 🏷️ Classification")
            c1, c2 = st.columns(2)
            c1.metric("Category", result.get("classification", "unknown").upper())
            c2.metric("Confidence", f"{result.get('classification_confidence', 0):.0%}")
            st.caption(result.get("classification_reasoning", ""))

            st.divider()

            # Extracted Metadata
            st.markdown("#### 📋 Extracted Metadata")
            metadata_cols = st.columns(3)
            metadata_cols[0].metric("Project", result.get("extracted_project_name") or "—")
            metadata_cols[1].metric("RFI #", result.get("extracted_rfi_number") or "—")
            metadata_cols[2].metric("Discipline", (result.get("extracted_discipline") or "—").title())

            meta2_cols = st.columns(3)
            meta2_cols[0].metric("Submittal #", result.get("extracted_submittal_number") or "—")
            meta2_cols[1].metric("Urgency", (result.get("extracted_urgency") or "normal").title())
            meta2_cols[2].metric("Project #", result.get("extracted_project_number") or "—")

            if result.get("extraction_reasoning"):
                with st.expander("Chain-of-thought reasoning"):
                    steps = result["extraction_reasoning"].split(" → ")
                    for step in steps:
                        st.markdown(f"- {step}")

            st.divider()

            # Filing Decision
            st.markdown("#### 📁 Filing Decision")
            render_confidence_bar(result.get("filing_confidence", 0))

            d1, d2 = st.columns(2)
            action = result.get("filing_action", "unknown")
            d1.metric("Action", action.replace("_", " ").title())
            d2.metric("Target Project", result.get("filing_project_id") or "—")

            if result.get("filing_reasoning"):
                st.info(result["filing_reasoning"])

            st.divider()

            # Agent Trace
            st.markdown("#### 🔍 Agent Execution Trace")
            render_trace(result.get("agent_trace", []))

        elif file_button:
            st.warning("Please fill in the subject and body fields.")


# ─── Page: Search ─────────────────────────────────────────────────────

elif page == "🔍 Search":
    st.title("🔍 Smart Search")
    st.markdown("Search across project communications using hybrid retrieval (BM25 + semantic) with RAG and citation tracking.")

    query = st.text_input(
        "Search query",
        placeholder="e.g., What is the status of RFI-247? / structural steel connection / SUB-089",
    )

    sample_queries = st.columns(4)
    if sample_queries[0].button("RFI-247 status", use_container_width=True):
        query = "What is the status of RFI-247 steel connection?"
    if sample_queries[1].button("HVAC conflict", use_container_width=True):
        query = "HVAC diffuser conflict with structural beams"
    if sample_queries[2].button("Change orders", use_container_width=True):
        query = "What change orders have been submitted for the Waterfront Tower?"
    if sample_queries[3].button("Soil conditions", use_container_width=True):
        query = "unforeseen soil conditions City Hall"

    if query:
        with st.spinner("Searching..."):
            result = run_search_agent(query=query, tenant_id=tenant_id)

        # Answer
        st.markdown("### Answer")
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)
        st.markdown(result.answer)

        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{result.confidence:.0%}")
        col2.metric("Search Type", result.search_type.title())
        col3.metric("Results Found", len(result.source_chunks))

        # Source Documents
        if result.source_chunks:
            st.markdown("### Source Documents")
            for i, chunk in enumerate(result.source_chunks):
                with st.expander(f"📄 {chunk.get('title', 'Untitled')} (score: {chunk.get('score', 0):.2f})"):
                    st.markdown(chunk.get("text", "")[:500])
                    st.caption(f"Source type: {chunk.get('source_type', 'unknown')} | ID: {chunk.get('id', '')}")

        # Citations
        if result.citations:
            st.markdown("### Citations")
            for citation in result.citations:
                verified = "✅ Verified" if citation.get("verified") else "⚠️ Unverified"
                st.markdown(f"- {verified} — *{citation.get('source_document', '')}* (relevance: {citation.get('relevance_score', 0):.2f})")

        # Trace
        with st.expander("🔍 Search Trace"):
            st.json(result.trace)


# ─── Page: Batch Demo ────────────────────────────────────────────────

elif page == "⚡ Batch Demo":
    st.title("⚡ Batch Filing Demo")
    st.markdown("Generate and file random AECO emails to see the system handle volume with different email types.")

    count = st.slider("Number of emails to process", 5, 50, 10)

    if st.button("🚀 Run Batch", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        results_container = st.container()

        results = []
        start = time.perf_counter()

        for i in range(count):
            email = generate_email(tenant_id)
            result = run_filing_agent(
                email_id=email["id"],
                email_subject=email["subject"],
                email_body=email["body"],
                email_sender=email["sender"],
                tenant_id=tenant_id,
            )
            results.append({
                "email_id": email["id"],
                "subject": email["subject"][:60],
                "expected": email["expected_category"],
                "classified_as": result.get("classification", "—"),
                "confidence": result.get("classification_confidence", 0),
                "action": result.get("filing_action", "—"),
                "project": result.get("filing_project_id") or "—",
                "filing_confidence": result.get("filing_confidence", 0),
            })

            progress.progress((i + 1) / count)
            status.text(f"Processing {i+1}/{count}...")

        total_time = (time.perf_counter() - start) * 1000
        status.empty()
        progress.empty()

        st.success(f"Processed {count} emails in {total_time:.0f}ms ({count / (total_time/1000):.0f} emails/sec)")

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        actions = [r["action"] for r in results]
        m1.metric("Auto-filed", actions.count("auto_file"))
        m2.metric("Needs Review", actions.count("needs_review"))
        m3.metric("Flagged", actions.count("flagged"))

        correct = sum(1 for r in results if r["expected"] == r["classified_as"])
        m4.metric("Classification Accuracy", f"{correct/len(results):.0%}")

        # Category distribution
        st.markdown("### Classification Distribution")
        import collections
        cat_dist = collections.Counter(r["classified_as"] for r in results)
        chart_data = {cat: count for cat, count in sorted(cat_dist.items())}
        st.bar_chart(chart_data)

        # Results table
        st.markdown("### Detailed Results")
        st.dataframe(
            results,
            column_config={
                "confidence": st.column_config.ProgressColumn("Cls Conf", min_value=0, max_value=1, format="%.0f%%"),
                "filing_confidence": st.column_config.ProgressColumn("File Conf", min_value=0, max_value=1, format="%.0f%%"),
            },
            use_container_width=True,
        )


# ─── Page: Agent Trace Viewer ────────────────────────────────────────

elif page == "📊 Agent Trace Viewer":
    st.title("📊 Agent Trace Viewer")
    st.markdown("Visualize the multi-agent pipeline execution. Generate a filing and inspect every step.")

    if st.button("Generate & Trace a Random Email", type="primary"):
        email = generate_email(tenant_id)

        st.markdown("### Input Email")
        st.markdown(f"**From:** {email['sender']}")
        st.markdown(f"**Subject:** {email['subject']}")
        st.text(email["body"][:500])
        st.caption(f"Expected category: `{email['expected_category']}`")

        st.divider()

        with st.spinner("Running agent pipeline..."):
            start = time.perf_counter()
            result = run_filing_agent(
                email_id=email["id"],
                email_subject=email["subject"],
                email_body=email["body"],
                email_sender=email["sender"],
                tenant_id=tenant_id,
            )
            duration = (time.perf_counter() - start) * 1000

        st.markdown("### Pipeline Results")
        st.success(f"Completed in {duration:.0f}ms")

        # Visual pipeline flow
        st.markdown("### Agent Flow")
        cols = st.columns(4)

        with cols[0]:
            st.markdown("**🏷️ Classify**")
            st.metric("Category", (result.get("classification") or "—").upper())
            st.metric("Confidence", f"{result.get('classification_confidence', 0):.0%}")

        with cols[1]:
            st.markdown("**📋 Extract**")
            st.metric("Project", result.get("extracted_project_name") or "—")
            st.metric("Discipline", (result.get("extracted_discipline") or "—").title())

        with cols[2]:
            st.markdown("**📁 File**")
            render_confidence_bar(result.get("filing_confidence", 0), "Filing")
            st.metric("Project ID", result.get("filing_project_id") or "—")

        with cols[3]:
            st.markdown("**✅ Result**")
            action = result.get("filing_action", "—")
            if action == "auto_file":
                st.success("AUTO-FILED")
            elif action == "needs_review":
                st.warning("NEEDS REVIEW")
            else:
                st.error("FLAGGED")

        st.divider()

        # Full trace
        st.markdown("### Execution Trace")
        render_trace(result.get("agent_trace", []))

        # Raw state
        with st.expander("Raw Agent State"):
            display_state = {k: v for k, v in result.items() if k != "messages"}
            st.json(display_state)


# ─── Page: Pending Reviews ───────────────────────────────────────────

elif page == "📋 Pending Reviews":
    st.title("📋 Pending Human Reviews")
    st.markdown("Emails that need human approval before filing. This is the human-in-the-loop (HITL) interface.")

    store = get_checkpoint_store()
    pending = store.list_pending(tenant_id)

    if not pending:
        st.info("No pending reviews. File some emails with medium confidence to see them here.")
        st.markdown("Try filing an ambiguous email from the **📧 File Email** page.")
    else:
        for record in pending:
            state = record.state
            with st.expander(
                f"📧 {state.get('email_subject', 'Unknown')} — Confidence: {state.get('filing_confidence', 0):.0%}",
                expanded=True,
            ):
                st.markdown(f"**From:** {state.get('email_sender', '—')}")
                st.markdown(f"**Classification:** {state.get('classification', '—')} ({state.get('classification_confidence', 0):.0%})")
                st.markdown(f"**Suggested Project:** {state.get('filing_project_id', '—')}")
                st.markdown(f"**Reasoning:** {state.get('filing_reasoning', '—')}")

                col1, col2, col3 = st.columns(3)
                if col1.button("✅ Approve", key=f"approve_{record.thread_id}", use_container_width=True):
                    store.resume(record.thread_id, "approve", tenant_id=tenant_id)
                    st.success("Approved and filed!")
                    st.rerun()

                if col2.button("❌ Reject", key=f"reject_{record.thread_id}", use_container_width=True):
                    store.resume(record.thread_id, "reject", tenant_id=tenant_id)
                    st.warning("Rejected and flagged.")
                    st.rerun()

                correction = col3.text_input("Correct to project:", key=f"correct_{record.thread_id}", placeholder="proj_001")
                if col3.button("🔄 Correct", key=f"correct_btn_{record.thread_id}", use_container_width=True):
                    if correction:
                        store.resume(record.thread_id, "correct", corrected_project_id=correction, tenant_id=tenant_id)
                        st.success(f"Corrected to {correction} and filed!")
                        st.rerun()
