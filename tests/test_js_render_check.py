"""Runs the dashboard's JavaScript render check under pytest.

``tests/js/dashboard_render_check.mjs`` is the only check that exercises the
page's actual rendering logic. It needs node, which is not a dependency of this
project, so it is skipped rather than required when node is absent.

The skip is reported explicitly. A render check that silently does not run is
indistinguishable from a render check that silently passes, and the failure mode
of a dashboard is exactly the kind of thing nobody notices until it misleads.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

CHECK = Path(__file__).parent / "js" / "dashboard_render_check.mjs"
PAGE = Path(__file__).parent.parent / "quantforge" / "api" / "static" / "dashboard.html"


def test_the_check_script_exists():
    assert CHECK.is_file(), f"{CHECK} is missing; the render check would never run"
    assert PAGE.is_file(), f"{PAGE} is missing"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not installed; the dashboard's rendering "
                           "logic is unverified in this environment")
def test_dashboard_renders_under_node():
    result = subprocess.run(
        ["node", str(CHECK)],
        capture_output=True, text=True, timeout=120, cwd=str(CHECK.parent),
    )
    if result.returncode != 0:
        pytest.fail(
            "the dashboard render check failed\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    # The script prints one line per check. Assert it actually ran them, so a
    # truncated or stubbed script cannot pass by producing no output.
    checked = result.stdout.count("  ok ")
    assert checked >= 15, (
        f"the render check reported only {checked} passing assertions; it may "
        f"have exited early without running its checks.\n{result.stdout}"
    )
    assert "all render checks passed" in result.stdout
