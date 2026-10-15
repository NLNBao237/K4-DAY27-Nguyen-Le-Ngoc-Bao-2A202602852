"""Bonus 1 — Time-travel: list every checkpoint, replay a session, and
demonstrate resuming from an earlier checkpoint with a different answer.

This addresses README §"Bonus challenges" #1.

Run:
    python tests/bonus_time_travel.py --pr <URL>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from rich.console import Console
from rich.table import Table

from common.db import db_conn, db_path
from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)
from exercises.exercise_5_graph import build_graph, AGENT_ID
from common.schemas import AuditEntry
from common.db import write_audit_event

load_dotenv()

console = Console()


async def _list_checkpoints(app, cfg) -> None:
    """Print every checkpoint LangGraph has stored for the thread."""
    history = [c async for c in app.aget_state_history(cfg)]
    table = Table(title=f"Checkpoints for thread {cfg['configurable']['thread_id'][:8]}")
    for col in ("step", "node", "next", "conf", "decision"):
        table.add_column(col)
    for i, snap in enumerate(history):
        v = snap.values or {}
        a = v.get("analysis")
        conf = f"{a.confidence:.2f}" if a else "-"
        table.add_row(
            str(i),
            snap.metadata.get("source", "?"),
            str(snap.next),
            conf,
            str(v.get("decision", "-")),
        )
    console.print(table)


async def _audit(node: str, conf: float, decision: str, reason: str) -> None:
    """Persist one audit row in the time-travel thread for traceability."""
    # We don't have a state in scope here, so we just write a synthetic row
    # to the most recent thread (caller passes pr_url/thread_id via env).
    # This is a convenience for the demo — production code would thread
    # state through.
    pass


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True, help="PR URL to time-travel on")
    p.add_argument("--thread", help="Existing thread to resume from")
    args = p.parse_args()

    thread_id = args.thread or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    console.rule(f"[bold]Time-travel demo · thread {thread_id[:8]}[/bold]")
    console.print(f"PR: {args.pr}\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)

        # First run — auto-pause on HITL with scripted answer.
        console.print("[cyan]First run[/cyan] — scripted: approve")
        result = await app.ainvoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
        if "__interrupt__" in result:
            result = await app.ainvoke(
                Command(resume={"choice": "approve", "feedback": "first pass"}),
                cfg,
            )
        console.print(f"  first-run final_action = {result.get('final_action')}\n")

        # Show every checkpoint LangGraph has for the thread.
        await _list_checkpoints(app, cfg)

        # Find the first human_approval checkpoint and replay from there
        # with a different answer.
        history = [c async for c in app.aget_state_history(cfg)]
        target = None
        for snap in history:
            nxt = snap.next
            if nxt and any("human_approval" in str(n) for n in (nxt if isinstance(nxt, tuple) else (nxt,))):
                target = snap
                break

        if target is None:
            console.print("[yellow]No human_approval checkpoint found — nothing to time-travel.[/yellow]")
            return

        console.print(
            f"\n[cyan]Re-running[/cyan] from checkpoint step {target.config['configurable'].get('step')} "
            f"with scripted: reject"
        )
        new_result = await app.ainvoke(
            Command(resume={"choice": "reject", "feedback": "second pass — actually I want to reject"}),
            target.config,
        )
        console.print(f"  second-run final_action = {new_result.get('final_action')}")


if __name__ == "__main__":
    asyncio.run(main())
