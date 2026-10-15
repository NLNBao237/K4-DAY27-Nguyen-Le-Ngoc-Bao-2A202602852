"""Shared LangGraph wiring for the HITL PR review agent.

Both the terminal exercises (`exercises/exercise_4_audit.py`) and the Streamlit
UI (`app.py`) import `build_graph` from here so they share one canonical
implementation of the routing/audit/checkpointer logic.
"""

from __future__ import annotations

import os
import time

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from common.db import write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)

AGENT_ID = "pr-review-agent@v0.1"


# ─── audit helper ─────────────────────────────────────────────────────────
async def audit(state, entry: AuditEntry) -> None:
    await write_audit_event(
        thread_id=state.get("thread_id", "fanout"),
        pr_url=state["pr_url"],
        entry=entry,
    )


# ─── nodes ────────────────────────────────────────────────────────────────
async def node_fetch_pr(state):
    t0 = time.monotonic()
    pr = fetch_pr(state["pr_url"])
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="fetch_pr",
        confidence=0.0, risk_level="med", decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {
        "pr_title": pr.title, "pr_diff": pr.diff,
        "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha,
    }


async def node_analyze(state):
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    a: PRAnalysis = await llm.ainvoke([
        {"role": "system", "content": (
            "You are a senior, security-conscious code reviewer producing a "
            "structured PRAnalysis.\n\n"
            "CALIBRATION GUIDE for the `confidence` field (0.0–1.0):\n"
            "  0.90+ : the change is mechanical, no security/semantic concerns, "
            "all review comments are obvious and self-contained.\n"
            "  0.70–0.89: small features with at most one open design question; "
            "comments are concrete and well-supported.\n"
            "  0.55–0.69: introduces a new feature with multiple risk vectors "
            "(auth, migrations, schema, dependency bumps) and you have ≥1 open "
            "question about intent, threat model, or compatibility.\n"
            "  <0.55 : risky areas (password hashing, cryptography, SQL, "
            "tokens, migrations, or new public API surface) where you cannot "
            "infer intent from the diff alone.\n\n"
            "Be CONSERVATIVE. If the diff touches auth, password storage, "
            "cryptography, raw SQL, or new dependencies, drop confidence to "
            "0.45–0.60 unless the implementation clearly follows best practice.\n\n"
            "If your confidence is below 0.60, you MUST populate "
            "`escalation_questions` with 2–4 specific, context-rich questions "
            "referencing concrete files/sections of the diff — name the file, "
            "the function, and what you need the reviewer to clarify."
        )},
        {"role": "user", "content": (
            f"Title: {state['pr_title']}\n"
            f"Files: {', '.join(state.get('pr_files', []))}\n\n"
            f"Diff:\n{state['pr_diff']}"
        )},
    ])
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="analyze",
        confidence=a.confidence, risk_level=risk_level_for(a.confidence),
        decision="pending", reason=a.confidence_reasoning,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": a}


async def node_route(state):
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="route",
        confidence=c, risk_level=risk_level_for(c),
        decision=decision, reason=f"confidence={c:.2f} → {decision}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"decision": decision}


async def node_human_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="human_approval",
        confidence=a.confidence, risk_level=risk_level_for(a.confidence),
        decision="pending", reason="interrupted — awaiting reviewer choice",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    t0 = time.monotonic()
    resp = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    choice = (resp.get("choice") or "").strip().lower()
    feedback = (resp.get("feedback") or "").strip()
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="human_approval",
        confidence=a.confidence, risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision=choice or "pending",
        reason=feedback or a.confidence_reasoning,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"human_choice": choice or None, "human_feedback": feedback or None}


async def node_escalate(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="escalate",
        confidence=a.confidence, risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=f"interrupted — {len(questions)} question(s) awaiting reviewer",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    t0 = time.monotonic()
    answers = interrupt({
        "kind": "escalation", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary, "risk_factors": a.risk_factors, "questions": questions,
    })
    answers = answers or {}
    summary = "; ".join(f"Q:{q[:40]}… A:{ans[:60]}" for q, ans in answers.items()) or "(no answers)"
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="escalate",
        confidence=a.confidence, risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision="escalate", reason=summary,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state):
    t0 = time.monotonic()
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    llm = get_llm().with_structured_output(PRAnalysis)
    refined: PRAnalysis = await llm.ainvoke([
        {"role": "system", "content": "Refine review with reviewer answers."},
        {"role": "user", "content": f"Diff:\n{state['pr_diff']}\n\nQ&A:\n{qa}"},
    ])
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="synthesize",
        confidence=refined.confidence, risk_level=risk_level_for(refined.confidence),
        decision="pending",
        reason=f"refined review with reviewer answers: {refined.confidence_reasoning}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": refined}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        return "committed"
    except Exception:
        return "commit_failed"


async def node_auto_approve(state):
    t0 = time.monotonic()
    a = state["analysis"]
    action = _post(state)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="auto_approve",
        confidence=a.confidence, risk_level=risk_level_for(a.confidence),
        decision="auto", reason=f"auto-posted review (no human in the loop): {action}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"final_action": f"auto_{action}"}


async def node_commit(state):
    t0 = time.monotonic()
    if state.get("escalation_answers") or state.get("human_choice") == "approve":
        action = _post(state)
    else:
        action = "rejected"
    a = state["analysis"]
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="commit",
        confidence=a.confidence, risk_level=risk_level_for(a.confidence),
        decision=action,
        reviewer_id=os.environ.get("GITHUB_USER") if state.get("human_choice") else None,
        reason="posted review to PR" if action == "committed"
                else ("refined review posted after escalation" if state.get("escalation_answers")
                      else f"skipped (choice={state.get('human_choice')})"),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"final_action": action}


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    return g.compile(checkpointer=checkpointer)
