"""Run real SDK contracts outside the suite's global OpenAI module stubs."""

import os
import subprocess
import sys
from pathlib import Path


def test_real_sdk_contracts() -> None:
    cases = Path(__file__).parent / "real_sdk" / "contract_cases.py"
    # A fresh interpreter and confcutdir exclude tests/conftest.py's sys.modules
    # replacement. Explicit transport mocks plus a socket guard prohibit traffic.
    env = {key: value for key, value in os.environ.items() if not key.startswith(("OPENAI_", "AZURE_")) and "proxy" not in key.lower()}
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-ra", f"--confcutdir={cases.parent}", str(cases)],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
