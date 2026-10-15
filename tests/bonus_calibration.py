"""Bonus 2 — Confidence calibration: SQL analytics on `audit_events`.

Computes:
  • average confidence per decision
  • risk-level distribution
  • how often the LLM marked itself "auto" with high risk
  • escalation rate per PR
  • drift: was the post-synthesis confidence higher than the original?

Run:  python tests/bonus_calibration.py
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.table import Table

from common.db import db_conn


console = Console()


async def _q(sql: str, *params) -> list[dict]:
    async with db_conn() as conn:
        async with conn.execute(sql, params) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def main() -> None:
    console.rule("[bold]Confidence calibration report[/bold]")

    rows = await _q("SELECT COUNT(DISTINCT thread_id) AS threads, COUNT(*) AS events FROM audit_events")
    if not rows or rows[0]["events"] == 0:
        console.print("[yellow]No audit data yet — run the exercises first.[/yellow]")
        return
    console.print(
        f"Total threads: [bold]{rows[0]['threads']}[/bold]  ·  events: {rows[0]['events']}\n"
    )

    # ── avg confidence by decision ────────────────────────────────────
    console.print("[cyan]Average confidence by decision[/cyan]")
    by_decision = await _q("""
        SELECT decision, ROUND(AVG(confidence), 3) AS avg_conf,
               COUNT(*) AS n
          FROM audit_events
         GROUP BY decision
         ORDER BY avg_conf DESC
    """)
    t = Table()
    for c in ("decision", "avg_conf", "n"): t.add_column(c)
    for r in by_decision: t.add_row(r["decision"] or "-", str(r["avg_conf"]), str(r["n"]))
    console.print(t)

    # ── risk-level distribution ───────────────────────────────────────
    console.print("\n[cyan]Risk-level distribution[/cyan]")
    by_risk = await _q("""
        SELECT risk_level, COUNT(*) AS n
          FROM audit_events
         GROUP BY risk_level
    """)
    t = Table()
    for c in ("risk_level", "n"): t.add_column(c)
    for r in by_risk: t.add_row(r["risk_level"], str(r["n"]))
    console.print(t)

    # ── SHOULD-NEVER-HAPPEN check: high risk with auto decision ───────
    console.print("\n[cyan]Sanity check: high-risk + auto decision (should be 0)[/cyan]")
    bad = await _q("""
        SELECT thread_id, action, confidence, reason
          FROM audit_events
         WHERE risk_level = 'high' AND decision = 'auto'
    """)
    if bad:
        for r in bad:
            console.print(f"  [red]✗[/red] {r}")
    else:
        console.print("  [green]✓[/green] no violations")

    # ── per-PR escalation rate ────────────────────────────────────────
    console.print("\n[cyan]Per-PR routing[/cyan]")
    prs = await _q("""
        SELECT pr_url,
               MAX(CASE WHEN decision='auto_approve' OR action='auto_approve' THEN 1 ELSE 0 END) AS auto,
               MAX(CASE WHEN decision='escalate'    OR action='escalate'    THEN 1 ELSE 0 END) AS esc,
               MAX(CASE WHEN decision='approve'    OR action='human_approval' THEN 1 ELSE 0 END) AS hum
          FROM audit_events
         GROUP BY pr_url
    """)
    t = Table()
    for c in ("pr_url", "auto", "human_approval", "escalate"): t.add_column(c)
    for r in prs:
        t.add_row(r["pr_url"], str(r["auto"]), str(r["hum"]), str(r["esc"]))
    console.print(t)

    # ── drift: post-synthesis confidence vs. original ─────────────────
    console.print("\n[cyan]Confidence lift after escalation Q&A[/cyan]")
    drift = await _q("""
        WITH first_analyze AS (
            SELECT thread_id, confidence AS original_conf
              FROM audit_events
             WHERE action = 'analyze'
        ),
        synth AS (
            SELECT thread_id, confidence AS refined_conf
              FROM audit_events
             WHERE action = 'synthesize'
        )
        SELECT s.thread_id,
               f.original_conf,
               s.refined_conf,
               ROUND(s.refined_conf - f.original_conf, 3) AS lift
          FROM synth s
          JOIN first_analyze f USING (thread_id)
    """)
    if drift:
        t = Table()
        for c in ("thread_id", "original", "refined", "lift"): t.add_column(c)
        for r in drift:
            t.add_row(r["thread_id"][:8], str(r["original_conf"]), str(r["refined_conf"]),
                       f"{r['lift']:+.2f}")
        console.print(t)
        avg_lift = sum(r["lift"] for r in drift) / len(drift)
        console.print(f"  average lift: [bold]{avg_lift:+.2f}[/bold]")
    else:
        console.print("  [dim]no escalation threads yet[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
