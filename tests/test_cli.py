from __future__ import annotations

import subprocess
import sys


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "News impact alert pipeline CLI." in result.stdout
