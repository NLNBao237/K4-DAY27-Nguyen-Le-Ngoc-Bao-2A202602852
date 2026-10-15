"""Drive exercises 2/3/4 end-to-end with scripted answers.

Why this script exists:
  The LLM's natural confidence is too volatile to reliably land in the
  58-72% band on a real PR.  We monkey-patch the AUTO_APPROVE_THRESHOLD /
  ESCALATE_THRESHOLD so the right branch is forced for each test, then
  pipe scripted human answers into the prompt_human / handle_interrupt
  hooks.  This is the deterministic way to verify the HITL flow works
  end-to-end without depending on LLM mood.

Run:  python tests/run_lab_tests.py

NOTE: We monkey-patch `common.github.post_review_comment` so the lab
tests don't actually write comments to GitHub.  Real PRs ARE still
fetched (read-only) so the agent has a real diff to analyze.
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import io
import sys
import uuid
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Disable real writes to GitHub before any lab code imports.
import common.github as _gh
_real_post = _gh.post_review_comment

def _fake_post(pr_url: str, body: str) -> None:
    """No-op stand-in for post_review_comment — see module docstring."""
    print(f"  [stubbed post] would comment on {pr_url} (body={len(body)} chars)")

_gh.post_review_comment = _fake_post  # type: ignore[assignment]
# Also patch the symbols already imported into exercise modules.
import sys as _sys
for _name in ("exercises.exercise_2_hitl", "exercises.exercise_3_escalation",
              "exercises.exercise_4_audit", "exercises.exercise_5_graph"):
    _mod = _sys.modules.get(_name)
    if _mod is not None and hasattr(_mod, "post_review_comment"):
        _mod.post_review_comment = _fake_post  # type: ignore[attr-defined]
del _name, _mod

import common.schemas as schemas
from common.db import db_conn

# ─── forced thresholds (override after importing so all nodes see them) ───
schemas.AUTO_APPROVE_THRESHOLD = 0.95   # only truly obvious PRs auto-approve
schemas.ESCALATE_THRESHOLD = 0.62       # anything ≤ 0.61 is escalated


PR_AUTO = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"   # mechanical
PR_RISK = "https://github.com/VinUni-AI20k/PR-Demo/pull/2"   # risky


# ─── helpers ──────────────────────────────────────────────────────────────
@contextlib.contextmanager
def fake_stdin(scripted: list[str]):
    """Replace sys.stdin with a StringIO buffer that yields the lines."""
    buf = io.StringIO("\n".join(scripted) + "\n")
    old = sys.stdin
    sys.stdin = buf
    try:
        yield
    finally:
        sys.stdin = old


def make_console_input_printer(scripted: list[str]):
    """Patch rich.Console.input to pull from `scripted` queue.

    rich's Console.input is the only stdin reader in prompt_human /
    handle_interrupt, so swapping that is enough.
    """
    from rich.console import Console
    original = Console.input

    def fake_input(self, prompt="", *a, **kw):  # type: ignore[no-untyped-def]
        if not scripted:
            raise EOFError("scripted answers exhausted")
        line = scripted.pop(0)
        # echo what the user "typed" so the transcript is readable
        sys.stderr.write(f"  (scripted) {prompt}{line}\n")
        sys.stderr.flush()
        return line

    Console.input = fake_input  # type: ignore[assignment]
    return lambda: setattr(Console, "input", original)


# ─── Exercise 2 test ──────────────────────────────────────────────────────
def test_exercise_2() -> None:
    print("\n=== Exercise 2 — HITL with scripted answers ===")
    import exercises.exercise_2_hitl as ex2
    import common.schemas as s
    s.AUTO_APPROVE_THRESHOLD = 0.95
    s.ESCALATE_THRESHOLD = 0.62
    # re-import constants used inside node_route
    ex2.AUTO_APPROVE_THRESHOLD = 0.95
    ex2.ESCALATE_THRESHOLD = 0.62

    restore = make_console_input_printer(["approve", "LGTM, ship it."])
    try:
        ex2.main.__wrapped__ if hasattr(ex2.main, "__wrapped__") else None  # noqa
        # Call main() with sys.argv mocked so it picks up --pr and the right thread.
        old_argv = sys.argv
        sys.argv = ["exercise_2_hitl.py", "--pr", PR_AUTO]
        try:
            ex2.main()
        finally:
            sys.argv = old_argv
    finally:
        restore()
    print("✓ Exercise 2 finished cleanly")


# ─── Exercise 3 test ──────────────────────────────────────────────────────
def test_exercise_3() -> None:
    print("\n=== Exercise 3 — escalation with scripted answers ===")
    import exercises.exercise_3_escalation as ex3
    ex3.AUTO_APPROVE_THRESHOLD = 0.95
    ex3.ESCALATE_THRESHOLD = 0.62
    import common.schemas as s
    s.AUTO_APPROVE_THRESHOLD = 0.95
    s.ESCALATE_THRESHOLD = 0.62

    # PR #2 typically generates 3–5 escalation questions; pad generously.
    restore = make_console_input_printer([
        "MD5 is a temporary placeholder; migration to bcrypt is tracked in JIRA-123.",
        "Tokens are short-lived and stored hashed at rest in a separate secrets table.",
        "Yes, the SQL query should be parameterized — fixing in the follow-up PR.",
        "Sync URL is HTTPS in production; the .env value here is a dev fallback.",
        "Storage of user_id is intentional for offline merge; the v2 schema fixes this.",
    ])
    try:
        old_argv = sys.argv
        sys.argv = ["exercise_3_escalation.py", "--pr", PR_RISK]
        try:
            ex3.main()
        finally:
            sys.argv = old_argv
    finally:
        restore()
    print("✓ Exercise 3 finished cleanly")


# ─── Exercise 4 test ──────────────────────────────────────────────────────
async def test_exercise_4() -> None:
    print("\n=== Exercise 4 — audit trail with scripted answers ===")
    import exercises.exercise_4_audit as ex4
    import common.schemas as s
    s.AUTO_APPROVE_THRESHOLD = 0.95
    s.ESCALATE_THRESHOLD = 0.62
    ex4.AUTO_APPROVE_THRESHOLD = 0.95
    ex4.ESCALATE_THRESHOLD = 0.62

    restore = make_console_input_printer([
        "MD5 is a temporary placeholder; migration to bcrypt is tracked separately.",
        "Tokens are short-lived; stored hashed at rest in a separate secrets table.",
        "Yes, the SQL query should be parameterized — fixing in the follow-up PR.",
        "Sync URL is HTTPS in production; the .env value here is a dev fallback.",
        "Storage of user_id is intentional for offline merge; the v2 schema fixes this.",
    ])
    try:
        thread_id = str(uuid.uuid4())
        await ex4.run(PR_RISK, thread_id)
        print(f"  thread_id = {thread_id}")

        # Read the audit rows back
        async with db_conn() as conn:
            async with conn.execute(
                "SELECT action, confidence, decision, reviewer_id, risk_level "
                "FROM audit_events WHERE thread_id=? ORDER BY id",
                (thread_id,),
            ) as cur:
                rows = await cur.fetchall()
        print(f"\n  audit_events rows ({len(rows)}):")
        for r in rows:
            print(f"    {dict(r)}")
        assert len(rows) >= 7, f"expected ≥7 audit rows, got {len(rows)}"
        actions = [r["action"] for r in rows]
        for need in ("fetch_pr", "analyze", "route", "escalate", "synthesize", "commit"):
            assert need in actions, f"missing action {need!r} in audit trail"
        print("✓ Exercise 4 audit trail complete (all required actions present)")
    finally:
        restore()


# ─── Exercise 5 / Streamlit import-only smoke test ───────────────────────
def test_exercise_5_imports() -> None:
    print("\n=== Exercise 5 — Streamlit app import smoke test ===")
    import importlib
    importlib.invalidate_caches()
    # We don't actually run streamlit here (it needs a browser / port);
    # we just confirm app.py imports cleanly and exposes the key symbols.
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", "app.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Don't execute — just import to catch syntax / name errors.
    # app.py calls st.set_page_config at import time which is fine under
    # a headless streamlit, so we let it run.
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    for name in ("render_approval_card", "render_escalation_card", "run_graph"):
        assert hasattr(mod, name), f"app.py missing {name!r}"
    print("✓ app.py imports cleanly, all renderers present")


# ─── Exercise 1 (already validated manually) ──────────────────────────────
def test_exercise_1_smoke() -> None:
    print("\n=== Exercise 1 — smoke (uses real LLM, real PRs) ===")
    import exercises.exercise_1_confidence as ex1
    import common.schemas as s
    s.AUTO_APPROVE_THRESHOLD = 0.73   # restore README defaults for the smoke
    s.ESCALATE_THRESHOLD = 0.58
    ex1.AUTO_APPROVE_THRESHOLD = 0.73
    ex1.ESCALATE_THRESHOLD = 0.58

    restore = make_console_input_printer([])
    try:
        old_argv = sys.argv
        sys.argv = ["exercise_1_confidence.py", "--pr", PR_AUTO]
        try:
            ex1.main()
        finally:
            sys.argv = old_argv
    finally:
        restore()
    print("✓ Exercise 1 ran on PR #1")


def main() -> None:
    test_exercise_1_smoke()
    test_exercise_2()
    test_exercise_3()
    asyncio.run(test_exercise_4())
    test_exercise_5_imports()
    print("\n=== ALL EXERCISES PASSED ===")


if __name__ == "__main__":
    main()
