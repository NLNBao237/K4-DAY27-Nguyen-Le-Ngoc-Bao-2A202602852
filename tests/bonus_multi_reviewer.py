"""Bonus 3 — Multi-reviewer fan-out (async gather, no sub-graphs).

Spawn N independent "reviewer" coroutines in parallel, each simulating
one human reviewer's choice, then merge verdicts with the most-cautious
rule (any reject → reject, else any edit → edit, else approve).

This addresses README §"Bonus challenges" #3.

Run:  python tests/bonus_multi_reviewer.py --pr <URL> --reviewers 6
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from typing import Literal

from dotenv import load_dotenv
load_dotenv()
import common.github as _gh
_gh.post_review_comment = lambda *a, **kw: None  # type: ignore[assignment]

from rich.console import Console

from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import PRAnalysis
from common.db import db_conn

console = Console()
AGENT = "pr-review-agent@v0.1-fanout"

Verdict = Literal["approve", "reject", "edit", "no_response"]


# Cycle: 0=approve, 1=reject, 2=edit, 3=approve, 4=reject, 5=approve
SIM_CYCLE = ["approve", "reject", "edit", "approve", "reject", "approve"]


async def one_reviewer(pr_url: str, pr_title: str, pr_diff: str,
                       pr_files: list[str], pr_head_sha: str,
                       analysis: PRAnalysis, i: int) -> tuple[str, Verdict]:
    """Simulate a single reviewer.

    Real implementation would: open the diff, draft a response, and submit
    {choice, feedback} to the graph via Command(resume=...). Here we just
    pick deterministically so the merge logic is exercised.
    """
    # Optional: re-prompt the LLM to generate a reviewer-style justification
    # (this would be a per-reviewer comment, but we keep it cheap).
    choice: Verdict = SIM_CYCLE[i % len(SIM_CYCLE)]  # type: ignore[assignment]
    feedback = (
        f"Reviewer {i}: simulated {choice}. (LGTM / concerns / changes requested.)"
    )
    console.print(f"  reviewer-{i} → {choice}")
    return f"reviewer-{i}", choice, feedback


def merge_most_cautious(verdicts: dict[str, Verdict]) -> Verdict:
    order = {"reject": 3, "edit": 2, "approve": 1, "no_response": 0}
    return max(verdicts.values(), key=lambda v: order.get(v, 0))


async def fanout_and_merge(pr_url: str, n: int) -> dict[str, Verdict]:
    pr = fetch_pr(pr_url)
    llm = get_llm().with_structured_output(PRAnalysis)
    a: PRAnalysis = await llm.ainvoke([
        {"role": "system", "content": "Senior security reviewer."},
        {"role": "user", "content": f"Title: {pr.title}\n\nDiff:\n{pr.diff}"},
    ])
    console.print(f"  analysis: confidence={a.confidence:.0%}, {len(a.comments)} comments")
    console.print(f"  fanning out to {n} reviewers…\n")

    results = await asyncio.gather(*(
        one_reviewer(pr_url, pr.title, pr.diff, pr.files_changed, pr.head_sha, a, i)
        for i in range(n)
    ))
    verdicts = {name: choice for name, choice, _ in results}
    return verdicts


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--reviewers", type=int, default=6)
    args = p.parse_args()

    console.rule(f"[bold]Multi-reviewer fan-out · {args.reviewers} reviewers[/bold]")
    verdicts = await fanout_and_merge(args.pr, args.reviewers)
    final = merge_most_cautious(verdicts)

    console.print("\n  per-reviewer verdicts:")
    for k, v in verdicts.items():
        console.print(f"    {k}: [bold]{v}[/bold]")
    console.print(f"\n  [cyan]most-cautious verdict:[/cyan] [bold]{final}[/bold]")

    # Persist one summary audit row.
    async with db_conn() as conn:
        await conn.execute(
            """INSERT INTO audit_events (
                 timestamp, thread_id, pr_url, agent_id, action, confidence,
                 risk_level, reviewer_id, decision, reason, execution_time_ms)
               VALUES (strftime('%Y-%m-%dT%H:%M:%fZ','now'), ?, ?, ?, 'fanout', 0.0,
                       'med', NULL, ?, ?, 0)""",
            (str(uuid.uuid4()), args.pr, AGENT, final, f"verdicts={verdicts}"),
        )
        await conn.commit()
    console.print("  [green]✓[/green] fan-out row persisted to audit_events")


if __name__ == "__main__":
    asyncio.run(main())
