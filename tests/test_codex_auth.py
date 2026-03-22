"""Tests for llm/codex_auth.py — OAuth PKCE flow for OpenAI Codex."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.codex_auth import (
    CLIENT_ID,
    REDIRECT_URI,
    REFRESH_MARGIN,
    SCOPES,
    CodexAuth,
    _decode_jwt_payload,
    _generate_pkce,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_jwt(payload: dict) -> str:
    """Build a fake JWT (header.payload.signature) with the given payload."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header}.{body}.fakesig"


def _mock_aiohttp_session(mock_resp):
    """Build a mock aiohttp.ClientSession that returns mock_resp from .post().

    Handles the double async-context-manager pattern:
        async with aiohttp.ClientSession(...) as session:
            async with session.post(...) as resp:
    """
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post_ctx)

    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    return mock_session_ctx


def _mock_token_response(access_token: str, **extra) -> MagicMock:
    """Build a mock aiohttp response for a token endpoint call."""
    data = {"access_token": access_token, **extra}
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read = AsyncMock(return_value=json.dumps(data).encode())
    return mock_resp


def _mock_error_response(status: int, body: str = "error") -> MagicMock:
    """Build a mock aiohttp response for a failed token endpoint call."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read = AsyncMock(return_value=body.encode())
    return mock_resp


# ---------------------------------------------------------------------------
# _generate_pkce
# ---------------------------------------------------------------------------
class TestGeneratePKCE:
    def test_returns_verifier_and_challenge(self):
        """PKCE generates a verifier/challenge pair of the correct format."""
        verifier, challenge = _generate_pkce()
        assert isinstance(verifier, str) and len(verifier) > 0
        assert isinstance(challenge, str) and len(challenge) > 0

    def test_challenge_is_s256_of_verifier(self):
        """The challenge must be the base64url-encoded SHA-256 of the verifier."""
        verifier, challenge = _generate_pkce()
        expected_hash = hashlib.sha256(verifier.encode()).digest()
        expected_challenge = base64.urlsafe_b64encode(expected_hash).rstrip(b"=").decode()
        assert challenge == expected_challenge

    def test_each_call_generates_unique_pair(self):
        """Consecutive calls produce different verifiers (uses os.urandom)."""
        v1, _ = _generate_pkce()
        v2, _ = _generate_pkce()
        assert v1 != v2


# ---------------------------------------------------------------------------
# _decode_jwt_payload
# ---------------------------------------------------------------------------
class TestDecodeJwtPayload:
    def test_decodes_valid_jwt(self):
        """Extracts the payload dict from a well-formed JWT."""
        payload = {"sub": "user123", "email": "test@example.com"}
        assert _decode_jwt_payload(_make_jwt(payload)) == payload

    def test_returns_empty_for_single_part(self):
        """A token with no dots has fewer than 2 parts → empty dict."""
        assert _decode_jwt_payload("nodots") == {}

    def test_returns_empty_for_invalid_base64(self):
        """Corrupted payload section → empty dict, no crash."""
        assert _decode_jwt_payload("header.!!!invalid!!!.sig") == {}

    def test_handles_padding_correctly(self):
        """Payload requiring base64 padding is decoded correctly."""
        payload = {"x": "a"}
        assert _decode_jwt_payload(_make_jwt(payload)) == payload

    def test_returns_empty_for_non_json_payload(self):
        """Valid base64 but not valid JSON → empty dict."""
        not_json = base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode()
        assert _decode_jwt_payload(f"header.{not_json}.sig") == {}


# ---------------------------------------------------------------------------
# CodexAuth: init / is_configured / _load / _save
# ---------------------------------------------------------------------------
class TestCodexAuthBasics:
    def test_init_sets_path(self, tmp_path):
        """Constructor stores the credentials path."""
        auth = CodexAuth(str(tmp_path / "creds.json"))
        assert auth._path == tmp_path / "creds.json"

    def test_is_configured_false_when_no_file(self, tmp_path):
        """Returns False when credentials file doesn't exist."""
        assert CodexAuth(str(tmp_path / "creds.json")).is_configured() is False

    def test_is_configured_true_when_cached(self, tmp_path):
        """Returns True when credentials are cached in memory."""
        auth = CodexAuth(str(tmp_path / "creds.json"))
        auth._credentials = {"access_token": "tok"}
        assert auth.is_configured() is True

    def test_is_configured_true_when_file_has_token(self, tmp_path):
        """Returns True when file exists and contains access_token."""
        creds_file = tmp_path / "creds.json"
        creds_file.write_text(json.dumps({"access_token": "tok123"}))
        assert CodexAuth(str(creds_file)).is_configured() is True

    def test_is_configured_false_when_file_missing_token(self, tmp_path):
        """Returns False when file exists but access_token is absent/empty."""
        creds_file = tmp_path / "creds.json"
        creds_file.write_text(json.dumps({"refresh_token": "rt"}))
        assert CodexAuth(str(creds_file)).is_configured() is False

    def test_is_configured_false_when_file_is_corrupt(self, tmp_path):
        """Returns False when file exists but contains invalid JSON."""
        creds_file = tmp_path / "creds.json"
        creds_file.write_text("NOT JSON")
        assert CodexAuth(str(creds_file)).is_configured() is False

    def test_load_returns_cached_credentials(self, tmp_path):
        """_load returns cached credentials without reading file."""
        auth = CodexAuth(str(tmp_path / "creds.json"))
        auth._credentials = {"access_token": "cached"}
        assert auth._load() == {"access_token": "cached"}

    def test_load_reads_from_file(self, tmp_path):
        """_load reads and caches credentials from file."""
        creds_file = tmp_path / "creds.json"
        creds = {"access_token": "fromfile", "refresh_token": "rt"}
        creds_file.write_text(json.dumps(creds))
        auth = CodexAuth(str(creds_file))
        assert auth._load() == creds
        assert auth._credentials == creds

    def test_load_raises_when_no_file(self, tmp_path):
        """_load raises RuntimeError when file doesn't exist."""
        with pytest.raises(RuntimeError, match="Codex credentials not found"):
            CodexAuth(str(tmp_path / "nonexistent.json"))._load()

    def test_save_writes_and_caches(self, tmp_path):
        """_save writes JSON to disk and updates the in-memory cache."""
        creds_file = tmp_path / "subdir" / "creds.json"
        auth = CodexAuth(str(creds_file))
        creds = {"access_token": "new_tok", "refresh_token": "new_rt"}
        auth._save(creds)
        assert json.loads(creds_file.read_text()) == creds
        assert auth._credentials == creds

    def test_save_creates_parent_directories(self, tmp_path):
        """_save creates parent directories if they don't exist."""
        deep_path = tmp_path / "a" / "b" / "c" / "creds.json"
        auth = CodexAuth(str(deep_path))
        auth._save({"access_token": "tok"})
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# CodexAuth.get_access_token
# ---------------------------------------------------------------------------
class TestGetAccessToken:
    async def test_returns_token_when_not_expired(self, tmp_path):
        """Returns the cached access token when it hasn't expired."""
        creds_file = tmp_path / "creds.json"
        creds = {"access_token": "valid_token", "expires_at": int(time.time()) + 3600}
        creds_file.write_text(json.dumps(creds))
        assert await CodexAuth(str(creds_file)).get_access_token() == "valid_token"

    async def test_refreshes_when_expired(self, tmp_path):
        """Triggers refresh when token has expired."""
        creds_file = tmp_path / "creds.json"
        creds = {
            "access_token": "old_token",
            "refresh_token": "rt",
            "expires_at": int(time.time()) - 100,
        }
        creds_file.write_text(json.dumps(creds))
        auth = CodexAuth(str(creds_file))

        new_tok = _make_jwt({"sub": "user"})
        mock_resp = _mock_token_response(new_tok, refresh_token="new_rt", expires_in=3600)
        mock_ctx = _mock_aiohttp_session(mock_resp)

        with patch("src.llm.codex_auth.aiohttp.ClientSession", return_value=mock_ctx):
            token = await auth.get_access_token()
        assert token == new_tok

    async def test_refreshes_within_margin(self, tmp_path):
        """Triggers refresh when token is within REFRESH_MARGIN of expiry."""
        creds_file = tmp_path / "creds.json"
        creds = {
            "access_token": "soon_expired",
            "refresh_token": "rt",
            "expires_at": int(time.time()) + REFRESH_MARGIN - 10,
        }
        creds_file.write_text(json.dumps(creds))
        auth = CodexAuth(str(creds_file))

        new_tok = _make_jwt({"sub": "user"})
        mock_resp = _mock_token_response(new_tok, expires_in=3600)
        mock_ctx = _mock_aiohttp_session(mock_resp)

        with patch("src.llm.codex_auth.aiohttp.ClientSession", return_value=mock_ctx):
            token = await auth.get_access_token()
        assert token == new_tok


# ---------------------------------------------------------------------------
# CodexAuth.get_account_id
# ---------------------------------------------------------------------------
class TestGetAccountId:
    def test_returns_account_id(self, tmp_path):
        """Returns account_id from stored credentials."""
        creds_file = tmp_path / "creds.json"
        creds_file.write_text(json.dumps({"access_token": "tok", "account_id": "acct_123"}))
        assert CodexAuth(str(creds_file)).get_account_id() == "acct_123"

    def test_returns_none_when_no_account_id(self, tmp_path):
        """Returns None when account_id is not in credentials."""
        creds_file = tmp_path / "creds.json"
        creds_file.write_text(json.dumps({"access_token": "tok"}))
        assert CodexAuth(str(creds_file)).get_account_id() is None


# ---------------------------------------------------------------------------
# CodexAuth._refresh
# ---------------------------------------------------------------------------
class TestRefresh:
    async def test_refresh_extracts_jwt_fields(self, tmp_path):
        """Refresh stores new tokens and extracts account_id/email from JWT."""
        auth = CodexAuth(str(tmp_path / "creds.json"))
        new_tok = _make_jwt({"chatgpt_account_id": "acct_new", "email": "user@example.com"})
        mock_resp = _mock_token_response(new_tok, refresh_token="rt_new", expires_in=7200)
        mock_ctx = _mock_aiohttp_session(mock_resp)

        with patch("src.llm.codex_auth.aiohttp.ClientSession", return_value=mock_ctx):
            await auth._refresh({"access_token": "old", "refresh_token": "rt_old"})

        saved = json.loads((tmp_path / "creds.json").read_text())
        assert saved["access_token"] == new_tok
        assert saved["refresh_token"] == "rt_new"
        assert saved["account_id"] == "acct_new"
        assert saved["email"] == "user@example.com"

    async def test_refresh_preserves_old_account_id(self, tmp_path):
        """If JWT lacks account_id, preserves the old one from creds."""
        auth = CodexAuth(str(tmp_path / "creds.json"))
        new_tok = _make_jwt({"sub": "user"})  # no account_id/email
        mock_resp = _mock_token_response(new_tok, expires_in=3600)
        mock_ctx = _mock_aiohttp_session(mock_resp)

        old_creds = {
            "access_token": "old", "refresh_token": "rt",
            "account_id": "acct_preserved", "email": "old@example.com",
        }
        with patch("src.llm.codex_auth.aiohttp.ClientSession", return_value=mock_ctx):
            await auth._refresh(old_creds)

        saved = json.loads((tmp_path / "creds.json").read_text())
        assert saved["account_id"] == "acct_preserved"
        assert saved["email"] == "old@example.com"

    async def test_refresh_keeps_old_refresh_token(self, tmp_path):
        """If response omits refresh_token, keeps the old one."""
        auth = CodexAuth(str(tmp_path / "creds.json"))
        new_tok = _make_jwt({"sub": "user"})
        mock_resp = _mock_token_response(new_tok, expires_in=3600)
        # No refresh_token in response
        mock_ctx = _mock_aiohttp_session(mock_resp)

        with patch("src.llm.codex_auth.aiohttp.ClientSession", return_value=mock_ctx):
            await auth._refresh({"access_token": "old", "refresh_token": "rt_keep"})

        saved = json.loads((tmp_path / "creds.json").read_text())
        assert saved["refresh_token"] == "rt_keep"

    async def test_refresh_raises_when_no_refresh_token(self, tmp_path):
        """Raises RuntimeError if no refresh_token is available."""
        auth = CodexAuth(str(tmp_path / "creds.json"))
        with pytest.raises(RuntimeError, match="No refresh token"):
            await auth._refresh({"access_token": "old"})

    async def test_refresh_raises_on_http_error(self, tmp_path):
        """Raises RuntimeError on non-200 response from token endpoint."""
        auth = CodexAuth(str(tmp_path / "creds.json"))
        mock_resp = _mock_error_response(400, "bad request")
        mock_ctx = _mock_aiohttp_session(mock_resp)

        with patch("src.llm.codex_auth.aiohttp.ClientSession", return_value=mock_ctx):
            with pytest.raises(RuntimeError, match="token refresh failed"):
                await auth._refresh({"access_token": "old", "refresh_token": "rt"})

    async def test_refresh_sets_expires_at(self, tmp_path):
        """Refresh correctly computes expires_at from expires_in."""
        auth = CodexAuth(str(tmp_path / "creds.json"))
        new_tok = _make_jwt({"sub": "u"})
        before = int(time.time())
        mock_resp = _mock_token_response(new_tok, expires_in=7200)
        mock_ctx = _mock_aiohttp_session(mock_resp)

        with patch("src.llm.codex_auth.aiohttp.ClientSession", return_value=mock_ctx):
            await auth._refresh({"access_token": "old", "refresh_token": "rt"})

        saved = json.loads((tmp_path / "creds.json").read_text())
        assert saved["expires_at"] >= before + 7200
        assert saved["expires_at"] <= before + 7210


# ---------------------------------------------------------------------------
# CodexAuth.build_auth_url
# ---------------------------------------------------------------------------
class TestBuildAuthUrl:
    def test_returns_url_and_verifier(self):
        """Returns a (url, code_verifier) tuple."""
        url, verifier = CodexAuth.build_auth_url()
        assert isinstance(url, str) and len(url) > 0
        assert isinstance(verifier, str) and len(verifier) > 0

    def test_url_contains_required_oauth_params(self):
        """Auth URL includes all required OAuth parameters."""
        url, _ = CodexAuth.build_auth_url()
        assert "response_type=code" in url
        assert f"client_id={CLIENT_ID}" in url
        assert f"redirect_uri={REDIRECT_URI}" in url
        assert "code_challenge=" in url
        assert "code_challenge_method=S256" in url
        assert "state=" in url

    def test_url_contains_scopes(self):
        """Auth URL includes all required scopes."""
        url, _ = CodexAuth.build_auth_url()
        for scope in SCOPES.split():
            assert scope in url

    def test_each_call_produces_unique_state(self):
        """Each call generates a unique state parameter (CSRF prevention)."""
        url1, _ = CodexAuth.build_auth_url()
        url2, _ = CodexAuth.build_auth_url()
        state1 = re.search(r"state=([^&]+)", url1).group(1)
        state2 = re.search(r"state=([^&]+)", url2).group(1)
        assert state1 != state2


# ---------------------------------------------------------------------------
# CodexAuth.exchange_code
# ---------------------------------------------------------------------------
class TestExchangeCode:
    async def test_exchange_success_with_jwt_fields(self):
        """Exchanges auth code for tokens and extracts JWT fields."""
        new_tok = _make_jwt({"chatgpt_account_id": "acct_456", "email": "user@test.com"})
        mock_resp = _mock_token_response(new_tok, refresh_token="rt_exchanged", expires_in=3600)
        mock_ctx = _mock_aiohttp_session(mock_resp)

        with patch("src.llm.codex_auth.aiohttp.ClientSession", return_value=mock_ctx):
            creds = await CodexAuth.exchange_code("auth_code_123", "verifier_abc")

        assert creds["access_token"] == new_tok
        assert creds["refresh_token"] == "rt_exchanged"
        assert creds["account_id"] == "acct_456"
        assert creds["email"] == "user@test.com"
        assert "expires_at" in creds

    async def test_exchange_without_jwt_fields(self):
        """Exchange works when JWT has no account_id or email."""
        new_tok = _make_jwt({"sub": "user"})
        mock_resp = _mock_token_response(new_tok, refresh_token="rt", expires_in=7200)
        mock_ctx = _mock_aiohttp_session(mock_resp)

        with patch("src.llm.codex_auth.aiohttp.ClientSession", return_value=mock_ctx):
            creds = await CodexAuth.exchange_code("code", "verifier")

        assert creds["access_token"] == new_tok
        assert "account_id" not in creds
        assert "email" not in creds

    async def test_exchange_defaults_refresh_token_to_empty(self):
        """If response omits refresh_token, defaults to empty string."""
        new_tok = _make_jwt({"sub": "user"})
        mock_resp = _mock_token_response(new_tok)  # no refresh_token
        mock_ctx = _mock_aiohttp_session(mock_resp)

        with patch("src.llm.codex_auth.aiohttp.ClientSession", return_value=mock_ctx):
            creds = await CodexAuth.exchange_code("code", "verifier")
        assert creds["refresh_token"] == ""

    async def test_exchange_raises_on_http_error(self):
        """Raises RuntimeError when token exchange fails."""
        mock_resp = _mock_error_response(403, "forbidden")
        mock_ctx = _mock_aiohttp_session(mock_resp)

        with patch("src.llm.codex_auth.aiohttp.ClientSession", return_value=mock_ctx):
            with pytest.raises(RuntimeError, match="Token exchange failed"):
                await CodexAuth.exchange_code("code", "verifier")

    async def test_exchange_default_expires_in(self):
        """Uses default 3600 seconds if expires_in not in response."""
        new_tok = _make_jwt({"sub": "user"})
        before = int(time.time())
        mock_resp = _mock_token_response(new_tok)  # no expires_in → default 3600
        mock_ctx = _mock_aiohttp_session(mock_resp)

        with patch("src.llm.codex_auth.aiohttp.ClientSession", return_value=mock_ctx):
            creds = await CodexAuth.exchange_code("code", "verifier")

        assert creds["expires_at"] >= before + 3600
        assert creds["expires_at"] <= before + 3610
