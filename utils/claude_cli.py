"""Claude CLI helper — shells out to the authenticated `claude` CLI.

Requires a Claude Max subscription with the `claude` CLI installed and
authenticated. Gracefully returns None if the CLI is not available.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any


def is_claude_available() -> bool:
    """Check if the claude CLI is installed and on PATH."""
    return shutil.which("claude") is not None


def call_claude(
    prompt: str,
    *,
    max_turns: int = 1,
    timeout_seconds: int = 180,
) -> str | None:
    """Call Claude via the CLI and return the response text.

    Uses the existing Max subscription authentication.

    Args:
        prompt: The prompt to send to Claude.
        max_turns: Maximum conversation turns (1 for single-shot).
        timeout_seconds: Timeout for the subprocess call.

    Returns:
        The response text on success, None on error.
    """
    if not is_claude_available():
        return None

    try:
        env = {**os.environ}
        # Prevent nested session detection
        env["CLAUDE_CODE_ENTRYPOINT"] = "cli"
        env["DISABLE_AUTOUPDATER"] = "1"

        result = subprocess.run(
            [
                "claude",
                "-p", prompt,
                "--output-format", "json",
                "--max-turns", str(max_turns),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )

        if result.returncode != 0:
            return None

        response = json.loads(result.stdout)

        if response.get("is_error"):
            return None

        return response.get("result", "")

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None
