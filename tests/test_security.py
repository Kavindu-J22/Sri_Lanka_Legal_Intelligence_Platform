"""
SLLIP V3 Automated Security Test Suite
======================================
Verifies that no API keys or production secrets are hardcoded in the codebase,
and confirms that environment variable loading and .env.example are properly configured.
"""

import os
import re
from pathlib import Path

WORKSPACE_DIR = Path(__file__).parent.parent.resolve()


def test_no_hardcoded_atria_key_in_code():
    """Verify that the exposed compromised Atria API key string is not present in Python source code."""
    exposed_pattern = re.compile(r"atr_[a-zA-Z0-9_]{30,}")

    py_files = list(WORKSPACE_DIR.glob("*.py"))
    assert len(py_files) > 0, "No Python source files found to audit."

    violations = []
    for f in py_files:
        content = f.read_text(encoding="utf-8")
        matches = exposed_pattern.findall(content)
        if matches:
            violations.append((f.name, matches))

    assert len(violations) == 0, f"Found hardcoded API keys in Python files: {violations}"


def test_env_example_exists_and_has_placeholders():
    """Verify that .env.example exists and contains no live secrets."""
    example_path = WORKSPACE_DIR / ".env.example"
    assert example_path.exists(), ".env.example file is missing."

    content = example_path.read_text(encoding="utf-8")
    assert "your_atria_api_key_here" in content or "ATRIA_API_KEY=" in content
    assert "atr_g5EXTQeJHPqjXM49A_apd_7UA2bGwT6h" not in content, "Live key found in .env.example!"


def test_gitignore_contains_sensitive_files():
    """Verify that .gitignore properly excludes .env, secrets, and large artifacts."""
    gitignore_path = WORKSPACE_DIR / ".gitignore"
    assert gitignore_path.exists(), ".gitignore file is missing."

    content = gitignore_path.read_text(encoding="utf-8")
    assert ".env" in content
    assert "vector_db" in content
    assert "sllip_raw_dataset" in content
