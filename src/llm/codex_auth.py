from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path

import asyncio

import aiohttp

from ..logging import get_logger

log = get_logger("codex_auth")

# OAuth constants for OpenAI Codex CLI
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPES = "openid profile email offline_access"

# Refresh 5 minutes before expiry
REFRESH_MARGIN = 300


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier_bytes = os.urandom(32)
    code_verifier = base64.urlsafe_b64encode(verifier_bytes).rstrip(b"=").decode()
    challenge_hash = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(challenge_hash).rstrip(b"=").decode()
    return code_verifier, code_challenge


def _decode_jwt_payload(token: str) -> dict:
    """Decode the payload section of a JWT without verification."""
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    # Add padding
    padding = 4 - len(payload) % 4
    if padding != 4:
        payload += "=" * padding
    try:
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


class CodexAuth:
    def __init__(self, credentials_path: str) -> None:
        self._path = Path(credentials_path)
        self._credentials: dict | None = None
        self._refresh_lock = asyncio.Lock()

    def is_configured(self) -> bool:
        """Check if credentials file exists and has tokens."""
        if self._credentials:
            return True
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                return bool(data.get("access_token"))
            except Exception:
                return False
        return False

    def _load(self) -> dict:
        if self._credentials:
            return self._credentials
        if not self._path.exists():
            raise RuntimeError("Codex credentials not found. Run scripts/codex_login.py first.")
        self._credentials = json.loads(self._path.read_text())
        return self._credentials

    def _save(self, creds: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(creds, indent=2))
        self._credentials = creds

    async def get_access_token(self) -> str:
        """Return a valid access token, refreshing if needed.

        Uses a lock to prevent concurrent refresh attempts — OpenAI
        refresh tokens are single-use, so two simultaneous refreshes
        cause the second to fail with 'refresh_token_reused'.
        """
        creds = self._load()
        expires_at = creds.get("expires_at", 0)

        if time.time() >= expires_at - REFRESH_MARGIN:
            async with self._refresh_lock:
                # Re-check after acquiring lock — another coroutine may have refreshed
                creds = self._load()
                if time.time() >= creds.get("expires_at", 0) - REFRESH_MARGIN:
                    log.info("Access token expired or expiring soon, refreshing...")
                    await self._refresh(creds)
                creds = self._credentials

        return creds["access_token"]

    def get_account_id(self) -> str | None:
        """Return the ChatGPT account ID from stored credentials."""
        creds = self._load()
        return creds.get("account_id")

    async def _refresh(self, creds: dict) -> None:
        """Refresh the access token using the refresh token."""
        refresh_token = creds.get("refresh_token")
        if not refresh_token:
            raise RuntimeError("No refresh token available. Run scripts/codex_login.py again.")

        async with aiohttp.ClientSession(auto_decompress=False) as session:
            async with session.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": CLIENT_ID,
                    "refresh_token": refresh_token,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept-Encoding": "identity",
                },
            ) as resp:
                if resp.status != 200:
                    body = (await resp.read()).decode("utf-8", errors="replace")
                    log.error("Token refresh failed (%d): %s", resp.status, body)
                    raise RuntimeError(
                        f"Codex token refresh failed (HTTP {resp.status}). "
                        "Run scripts/codex_login.py to re-authenticate."
                    )
                raw = await resp.read()
                data = json.loads(raw)

        new_creds = {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", refresh_token),
            "expires_at": int(time.time()) + data.get("expires_in", 3600),
        }

        # Extract account ID from JWT
        payload = _decode_jwt_payload(data["access_token"])
        if "chatgpt_account_id" in payload:
            new_creds["account_id"] = payload["chatgpt_account_id"]
        elif creds.get("account_id"):
            new_creds["account_id"] = creds["account_id"]

        if "email" in payload:
            new_creds["email"] = payload["email"]
        elif creds.get("email"):
            new_creds["email"] = creds["email"]

        self._save(new_creds)
        log.info("Codex tokens refreshed successfully")

    @staticmethod
    def build_auth_url() -> tuple[str, str]:
        """Build the authorization URL and return (url, code_verifier)."""
        code_verifier, code_challenge = _generate_pkce()
        state = base64.urlsafe_b64encode(os.urandom(16)).rstrip(b"=").decode()

        params = {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": "pi",
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{AUTH_URL}?{query}", code_verifier

    @staticmethod
    async def exchange_code(code: str, code_verifier: str) -> dict:
        """Exchange authorization code for tokens."""
        async with aiohttp.ClientSession(auto_decompress=False) as session:
            async with session.post(
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": REDIRECT_URI,
                    "client_id": CLIENT_ID,
                    "code_verifier": code_verifier,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept-Encoding": "identity",
                },
            ) as resp:
                if resp.status != 200:
                    body = (await resp.read()).decode("utf-8", errors="replace")
                    raise RuntimeError(f"Token exchange failed ({resp.status}): {body}")
                raw = await resp.read()
                data = json.loads(raw)

        creds = {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "expires_at": int(time.time()) + data.get("expires_in", 3600),
        }

        payload = _decode_jwt_payload(data["access_token"])
        if "chatgpt_account_id" in payload:
            creds["account_id"] = payload["chatgpt_account_id"]
        if "email" in payload:
            creds["email"] = payload["email"]

        return creds
