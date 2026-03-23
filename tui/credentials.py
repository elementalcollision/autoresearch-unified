"""Credential management for the autoresearch agent — unified across platforms.

Resolves API keys from multiple sources with the following priority:
1. ANTHROPIC_API_KEY environment variable (explicit, always wins)
2. macOS Keychain entry "autoresearch-agent" (secure, persistent — macOS only)
3. File-based credential store (~/.config/autoresearch/api_key — any OS)
4. Claude Code's stored credentials (if available — macOS only)

Provides --setup-key to store an API key persistently (Keychain on macOS,
file on Linux), so subsequent runs just work without env vars.
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
    """Read a password from the macOS Keychain. Returns None if not found."""
    if platform.system() != "Darwin":
        return None

    cmd = ["security", "find-generic-password", "-s", service, "-w"]
    if account:
        cmd.insert(-1, "-a")
        cmd.insert(-1, account)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _set_keychain_password(service: str, account: str, password: str) -> bool:
    """Store a password in the macOS Keychain. Returns True on success."""
    if platform.system() != "Darwin":
        return False

    # Delete existing entry first (if any)
    subprocess.run(
        ["security", "delete-generic-password", "-s", service, "-a", account],
        capture_output=True,
        timeout=5,
    )

    try:
        result = subprocess.run(
            [
                "security", "add-generic-password",
                "-s", service,
                "-a", account,
                "-w", password,
                "-U",  # update if exists
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _delete_keychain_password(service: str, account: str) -> bool:
    """Delete a password from the macOS Keychain. Returns True on success."""
    if platform.system() != "Darwin":
        return False

    try:
        result = subprocess.run(
            ["security", "delete-generic-password", "-s", service, "-a", account],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _try_claude_code_keychain() -> Optional[str]:
    """Attempt to read Claude Code's OAuth token from the macOS Keychain.

    Claude Code stores credentials as JSON in the keychain under
    service "Claude Code-credentials". The structure contains an OAuth
    access token that can be used with the Anthropic API.

    Note: OAuth tokens may expire and need refresh. This is a best-effort
    fallback — a proper API key is preferred.
    """
    raw = _get_keychain_password(CLAUDE_CODE_SERVICE)
    if not raw:
        return None

    try:
        creds = json.loads(raw)

        # Check for API key first (if user configured one in Claude Code)
        if "apiKey" in creds:
            return creds["apiKey"]

        # Check for OAuth access token
        oauth = creds.get("claudeAiOauth", {})
        token = oauth.get("accessToken", "")

        # OAuth tokens start with sk-ant-oat and work with the API
        # only if the subscription supports API access
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
    """Resolve an Anthropic API key from available sources.

    Priority:
    1. ANTHROPIC_API_KEY environment variable
    2. macOS Keychain (autoresearch-agent service) — macOS only
    3. File-based credential store (~/.config/autoresearch/api_key) — any OS
    4. Claude Code keychain credentials — macOS only

    Raises RuntimeError if no credentials found.
    """
    # 1. Environment variable (highest priority)
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return CredentialSource(api_key=env_key, source="env")

    # 2. Our own keychain entry (macOS)
    keychain_key = _get_keychain_password(KEYCHAIN_SERVICE, account="api-key")
    if keychain_key:
        return CredentialSource(api_key=keychain_key, source="keychain")

    # 3. File-based credential store (any OS)
    file_key = _read_file_credential()
    if file_key:
        return CredentialSource(api_key=file_key, source="file")

    # 4. Claude Code credentials (macOS fallback)
    cc_key = _try_claude_code_keychain()
    if cc_key:
        return CredentialSource(api_key=cc_key, source="claude-code")

    raise RuntimeError(
        "No Anthropic API key found. Set up credentials using one of:\n"
        "\n"
        "  Option 1 — Store persistently (recommended, one-time setup):\n"
        "    uv run dashboard.py --setup-key\n"
        "\n"
        "  Option 2 — Environment variable:\n"
        "    export ANTHROPIC_API_KEY=sk-ant-...\n"
        "\n"
        "  Option 3 — Per-invocation:\n"
        "    ANTHROPIC_API_KEY=sk-ant-... uv run dashboard.py --agent\n"
    )


def setup_api_key() -> None:
    """Interactive setup: prompt for API key and store persistently.

    Uses macOS Keychain on Darwin, file-based store on Linux/other.
    """
    print("Autoresearch Agent — API Key Setup")
    print("=" * 40)
    print()

    is_macos = platform.system() == "Darwin"

    if is_macos:
        print("This will store your Anthropic API key in the encrypted macOS Keychain.")
    else:
        print("This will store your Anthropic API key in ~/.config/autoresearch/api_key")

    print("Get a key at: https://console.anthropic.com/settings/keys")
    print()

    # Check for existing key
    if is_macos:
        existing = _get_keychain_password(KEYCHAIN_SERVICE, account="api-key")
    else:
        existing = _read_file_credential()

    if existing:
        masked = existing[:12] + "..." + existing[-4:]
        print(f"Existing key found: {masked}")
        response = input("Replace it? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Keeping existing key.")
            return

    api_key = input("Paste your API key (sk-ant-...): ").strip()

    if not api_key:
        print("No key entered. Aborting.")
        sys.exit(1)

    if not api_key.startswith("sk-ant-"):
        print("Warning: key doesn't start with 'sk-ant-' — are you sure this is correct?")
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborting.")
            sys.exit(1)

    if is_macos:
        if _set_keychain_password(KEYCHAIN_SERVICE, "api-key", api_key):
            print()
            print("API key stored in macOS Keychain.")
            print(f"  Service: {KEYCHAIN_SERVICE}")
            print(f"  Account: api-key")
            print()
            print("You can now run: uv run dashboard.py --agent")
            print("(No ANTHROPIC_API_KEY env var needed)")
        else:
            print("Failed to store in Keychain. Set ANTHROPIC_API_KEY instead.")
            sys.exit(1)
    else:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        API_KEY_FILE.write_text(api_key + "\n")
        os.chmod(API_KEY_FILE, 0o600)
        print()
        print("API key stored.")
        print(f"  File: {API_KEY_FILE}")
        print()
        print("You can now run the agent without ANTHROPIC_API_KEY env var.")


def clear_api_key() -> None:
    """Remove the stored API key from the platform-appropriate store."""
    cleared = False

    if platform.system() == "Darwin":
        if _delete_keychain_password(KEYCHAIN_SERVICE, "api-key"):
            print("API key removed from macOS Keychain.")
            cleared = True

    if API_KEY_FILE.exists():
        API_KEY_FILE.unlink()
        print("API key file removed.")
        cleared = True

    if not cleared:
        print("No stored API key found (or already removed).")
