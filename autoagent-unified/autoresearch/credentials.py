# Ported from autoresearch-unified/tui/credentials.py (MIT)
"""Credential management — unified across platforms.

Resolves API keys from multiple sources with priority:
1. ANTHROPIC_API_KEY environment variable
2. macOS Keychain entry "autoresearch-agent" (macOS only)
3. File-based credential store (~/.config/autoresearch/api_key)
4. Claude Code's stored credentials (macOS only)
"""

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


KEYCHAIN_SERVICE = "autoresearch-agent"
CLAUDE_CODE_SERVICE = "Claude Code-credentials"

CONFIG_DIR = Path.home() / ".config" / "autoresearch"
API_KEY_FILE = CONFIG_DIR / "api_key"


@dataclass
class CredentialSource:
    """Describes where a credential came from."""
    api_key: str
    source: str  # "env", "keychain", "file", "claude-code"


def _get_keychain_password(service: str, account: str | None = None) -> Optional[str]:
    """Read a password from the macOS Keychain."""
    if platform.system() != "Darwin":
        return None
    cmd = ["security", "find-generic-password", "-s", service, "-w"]
    if account:
        cmd.insert(-1, "-a")
        cmd.insert(-1, account)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _try_claude_code_keychain() -> Optional[str]:
    """Attempt to read Claude Code's OAuth token from the macOS Keychain."""
    raw = _get_keychain_password(CLAUDE_CODE_SERVICE)
    if not raw:
        return None
    try:
        creds = json.loads(raw)
        if "apiKey" in creds:
            return creds["apiKey"]
        oauth = creds.get("claudeAiOauth", {})
        token = oauth.get("accessToken", "")
        if token.startswith("sk-ant-"):
            return token
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def _read_file_credential() -> Optional[str]:
    """Read API key from file-based credential store."""
    if API_KEY_FILE.exists():
        key = API_KEY_FILE.read_text().strip()
        if key:
            return key
    return None


def resolve_api_key() -> CredentialSource:
    """Resolve an Anthropic API key from available sources."""
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return CredentialSource(api_key=env_key, source="env")

    keychain_key = _get_keychain_password(KEYCHAIN_SERVICE, account="api-key")
    if keychain_key:
        return CredentialSource(api_key=keychain_key, source="keychain")

    file_key = _read_file_credential()
    if file_key:
        return CredentialSource(api_key=file_key, source="file")

    cc_key = _try_claude_code_keychain()
    if cc_key:
        return CredentialSource(api_key=cc_key, source="claude-code")

    raise RuntimeError(
        "No Anthropic API key found. Set up credentials:\n"
        "  export ANTHROPIC_API_KEY=sk-ant-...\n"
    )
