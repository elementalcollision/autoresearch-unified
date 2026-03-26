"""Tests for tui.credentials — API key resolution priority chain."""

import json
import subprocess
from unittest.mock import patch

import pytest

from tui.credentials import (
    CredentialSource,
    _get_keychain_password,
    _read_file_credential,
    _try_claude_code_keychain,
    resolve_api_key,
)


# ---------------------------------------------------------------------------
# _read_file_credential
# ---------------------------------------------------------------------------

class TestReadFileCredential:
    def test_file_exists_with_key(self, tmp_path):
        key_file = tmp_path / "api_key"
        key_file.write_text("sk-ant-test123\n")
        with patch("tui.credentials.API_KEY_FILE", key_file):
            assert _read_file_credential() == "sk-ant-test123"

    def test_file_not_exists(self, tmp_path):
        key_file = tmp_path / "nonexistent"
        with patch("tui.credentials.API_KEY_FILE", key_file):
            assert _read_file_credential() is None

    def test_empty_file(self, tmp_path):
        key_file = tmp_path / "api_key"
        key_file.write_text("")
        with patch("tui.credentials.API_KEY_FILE", key_file):
            assert _read_file_credential() is None

    def test_whitespace_stripped(self, tmp_path):
        key_file = tmp_path / "api_key"
        key_file.write_text("  sk-ant-abc  \n")
        with patch("tui.credentials.API_KEY_FILE", key_file):
            assert _read_file_credential() == "sk-ant-abc"


# ---------------------------------------------------------------------------
# _get_keychain_password
# ---------------------------------------------------------------------------

class TestGetKeychainPassword:
    @patch("tui.credentials.platform.system", return_value="Linux")
    def test_non_darwin_returns_none(self, _mock_platform):
        assert _get_keychain_password("autoresearch-agent") is None

    @patch("tui.credentials.subprocess.run")
    @patch("tui.credentials.platform.system", return_value="Darwin")
    def test_darwin_success(self, _mock_platform, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="sk-ant-test\n"
        )
        assert _get_keychain_password("autoresearch-agent") == "sk-ant-test"

    @patch("tui.credentials.subprocess.run")
    @patch("tui.credentials.platform.system", return_value="Darwin")
    def test_darwin_not_found(self, _mock_platform, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=44, stdout=""
        )
        assert _get_keychain_password("autoresearch-agent") is None

    @patch("tui.credentials.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=[], timeout=5))
    @patch("tui.credentials.platform.system", return_value="Darwin")
    def test_timeout_returns_none(self, _mock_platform, _mock_run):
        assert _get_keychain_password("autoresearch-agent") is None

    @patch("tui.credentials.subprocess.run")
    @patch("tui.credentials.platform.system", return_value="Darwin")
    def test_account_param_included(self, _mock_platform, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="sk-ant-test\n"
        )
        _get_keychain_password("autoresearch-agent", account="api-key")
        cmd = mock_run.call_args[0][0]
        assert "-a" in cmd
        assert "api-key" in cmd


# ---------------------------------------------------------------------------
# _try_claude_code_keychain
# ---------------------------------------------------------------------------

class TestTryClaudeCodeKeychain:
    @patch("tui.credentials._get_keychain_password", return_value=None)
    def test_no_keychain_entry(self, _mock):
        assert _try_claude_code_keychain() is None

    @patch("tui.credentials._get_keychain_password")
    def test_apikey_in_json(self, mock_kc):
        mock_kc.return_value = json.dumps({"apiKey": "sk-ant-abc123"})
        assert _try_claude_code_keychain() == "sk-ant-abc123"

    @patch("tui.credentials._get_keychain_password")
    def test_oauth_token(self, mock_kc):
        mock_kc.return_value = json.dumps({
            "claudeAiOauth": {"accessToken": "sk-ant-oat-xyz"}
        })
        assert _try_claude_code_keychain() == "sk-ant-oat-xyz"

    @patch("tui.credentials._get_keychain_password")
    def test_non_ant_token_ignored(self, mock_kc):
        mock_kc.return_value = json.dumps({
            "claudeAiOauth": {"accessToken": "not-a-valid-token"}
        })
        assert _try_claude_code_keychain() is None

    @patch("tui.credentials._get_keychain_password", return_value="not json")
    def test_invalid_json(self, _mock):
        assert _try_claude_code_keychain() is None


# ---------------------------------------------------------------------------
# resolve_api_key — priority chain
# ---------------------------------------------------------------------------

class TestResolveApiKey:
    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        result = resolve_api_key()
        assert result == CredentialSource(api_key="sk-ant-from-env", source="env")

    @patch("tui.credentials._get_keychain_password", return_value="sk-ant-from-kc")
    def test_keychain_second(self, _mock_kc, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = resolve_api_key()
        assert result.source == "keychain"
        assert result.api_key == "sk-ant-from-kc"

    @patch("tui.credentials._try_claude_code_keychain", return_value=None)
    @patch("tui.credentials._read_file_credential", return_value="sk-ant-from-file")
    @patch("tui.credentials._get_keychain_password", return_value=None)
    def test_file_third(self, _kc, _file, _cc, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = resolve_api_key()
        assert result.source == "file"

    @patch("tui.credentials._try_claude_code_keychain", return_value="sk-ant-from-cc")
    @patch("tui.credentials._read_file_credential", return_value=None)
    @patch("tui.credentials._get_keychain_password", return_value=None)
    def test_claude_code_fourth(self, _kc, _file, _cc, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = resolve_api_key()
        assert result.source == "claude-code"

    @patch("tui.credentials._try_claude_code_keychain", return_value=None)
    @patch("tui.credentials._read_file_credential", return_value=None)
    @patch("tui.credentials._get_keychain_password", return_value=None)
    def test_no_credentials_raises(self, _kc, _file, _cc, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="No Anthropic API key found"):
            resolve_api_key()

    @patch("tui.credentials._get_keychain_password", return_value="sk-ant-from-kc")
    def test_env_takes_precedence_over_keychain(self, _mock_kc, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        result = resolve_api_key()
        assert result.source == "env"
