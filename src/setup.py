"""
Interactive setup for Heimdall — authenticate with OpenAI Codex.

Usage:
    python -m src.setup                  # Add a Codex account (browser mode)
    python -m src.setup --headless       # Add a Codex account (paste-the-URL mode)
    python -m src.setup --list           # List configured accounts
    python -m src.setup --remove N       # Remove account at index N

Credentials are saved to data/codex_auth.json as an array of account objects.
The bot's CodexAuthPool reads this file at startup and rotates between accounts
on rate limits.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .llm.codex_auth import CodexAuth, _decode_jwt_payload

DEFAULT_CREDS_PATH = Path("data/codex_auth.json")


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth callback."""

    auth_code: str | None = None

    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        if "code" in params:
            _CallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Authentication successful!</h2>"
                b"<p>You can close this tab.</p></body></html>"
            )
        elif "error" in params:
            error = params.get("error", ["unknown"])[0]
            desc = params.get("error_description", [""])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<html><body><h2>Auth failed: {error}</h2>"
                f"<p>{desc}</p></body></html>".encode()
            )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def _load_accounts(path: Path) -> list[dict]:
    """Load existing accounts from credentials file."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and raw.get("access_token"):
        return [raw]
    return []


def _save_accounts(path: Path, accounts: list[dict]) -> None:
    """Save accounts array to credentials file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(accounts, indent=2))
    path.chmod(0o600)


def _get_auth_code_browser() -> str:
    """Open browser, listen for OAuth callback, return auth code."""
    auth_url, code_verifier = CodexAuth.build_auth_url()

    print("\nOpening browser for authentication...")
    print(f"\nIf the browser doesn't open, visit:\n{auth_url}\n")
    webbrowser.open(auth_url)

    print("Waiting for callback on http://localhost:1455 ...")
    _CallbackHandler.auth_code = None
    server = HTTPServer(("127.0.0.1", 1455), _CallbackHandler)
    server.timeout = 120

    while _CallbackHandler.auth_code is None:
        server.handle_request()
        if _CallbackHandler.auth_code is None:
            print("Timed out. Try again.")
            sys.exit(1)

    server.server_close()
    return _CallbackHandler.auth_code, code_verifier


def _get_auth_code_headless() -> str:
    """Print auth URL, user pastes callback URL, return auth code."""
    auth_url, code_verifier = CodexAuth.build_auth_url()

    print("\n=== Headless Authentication ===\n")
    print("1. Open this URL in any browser:\n")
    print(f"   {auth_url}\n")
    print("2. After logging in, you'll be redirected to localhost (which will fail).")
    print("   Copy the FULL URL from your browser's address bar and paste it below.\n")

    callback_url = input("Paste the callback URL: ").strip()
    if not callback_url:
        print("No URL provided.")
        sys.exit(1)

    params = parse_qs(urlparse(callback_url).query)
    if "code" not in params:
        print("No authorization code found in URL.")
        sys.exit(1)

    return params["code"][0], code_verifier


def cmd_add(args: argparse.Namespace) -> None:
    """Add a new Codex account."""
    creds_path = Path(args.credentials_path)
    accounts = _load_accounts(creds_path)

    print(f"\n=== Heimdall Setup — Add Codex Account ===")
    print(f"Currently configured: {len(accounts)} account(s)\n")

    if args.headless:
        auth_code, code_verifier = _get_auth_code_headless()
    else:
        auth_code, code_verifier = _get_auth_code_browser()

    print("Exchanging code for tokens...")
    creds = asyncio.run(CodexAuth.exchange_code(auth_code, code_verifier))

    email = creds.get("email", "unknown")
    account_id = creds.get("account_id", "unknown")

    # Check for duplicate
    for i, existing in enumerate(accounts):
        if existing.get("account_id") == account_id:
            print(f"\nAccount {email} already exists at index {i}. Updating tokens.")
            accounts[i] = creds
            _save_accounts(creds_path, accounts)
            print(f"Credentials updated in {creds_path}")
            return

    accounts.append(creds)
    _save_accounts(creds_path, accounts)

    print(f"\nAccount added successfully!")
    print(f"  Email:      {email}")
    print(f"  Account ID: {account_id}")
    print(f"  Index:      {len(accounts) - 1}")
    print(f"  Saved to:   {creds_path}")
    print(f"\nTotal accounts: {len(accounts)}")
    print("Tokens auto-refresh at runtime. Re-run setup if bot is offline >7 days.")


def cmd_list(args: argparse.Namespace) -> None:
    """List configured accounts."""
    accounts = _load_accounts(Path(args.credentials_path))
    if not accounts:
        print("No accounts configured. Run: python -m src.setup")
        return

    print(f"\nConfigured Codex accounts ({len(accounts)}):\n")
    for i, acct in enumerate(accounts):
        email = acct.get("email", "unknown")
        account_id = acct.get("account_id", "unknown")[:12] + "..."
        has_refresh = "yes" if acct.get("refresh_token") else "no"
        print(f"  [{i}] {email}  (id: {account_id}, refresh: {has_refresh})")
    print()


def cmd_remove(args: argparse.Namespace) -> None:
    """Remove an account by index."""
    creds_path = Path(args.credentials_path)
    accounts = _load_accounts(creds_path)

    if args.index < 0 or args.index >= len(accounts):
        print(f"Invalid index {args.index}. Valid range: 0-{len(accounts) - 1}")
        sys.exit(1)

    removed = accounts.pop(args.index)
    email = removed.get("email", "unknown")
    _save_accounts(creds_path, accounts)
    print(f"Removed account [{args.index}] ({email}). {len(accounts)} remaining.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.setup",
        description="Heimdall setup — manage Codex authentication",
    )
    parser.add_argument(
        "--credentials-path",
        default=str(DEFAULT_CREDS_PATH),
        help=f"Path to credentials file (default: {DEFAULT_CREDS_PATH})",
    )

    sub = parser.add_subparsers(dest="command")

    # Default (no subcommand) = add
    add_parser = sub.add_parser("add", help="Add a Codex account")
    add_parser.add_argument("--headless", action="store_true",
                            help="Paste callback URL manually (no local browser needed)")

    list_parser = sub.add_parser("list", help="List configured accounts")

    remove_parser = sub.add_parser("remove", help="Remove an account by index")
    remove_parser.add_argument("index", type=int, help="Account index to remove")

    # Also support --headless and --list as top-level flags for convenience
    parser.add_argument("--headless", action="store_true",
                        help="Paste callback URL manually (no local browser needed)")
    parser.add_argument("--list", action="store_true",
                        help="List configured accounts")
    parser.add_argument("--remove", type=int, default=None, metavar="N",
                        help="Remove account at index N")

    args = parser.parse_args()

    # Route top-level flags to commands
    if args.list:
        cmd_list(args)
    elif args.remove is not None:
        args.index = args.remove
        cmd_remove(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "remove":
        cmd_remove(args)
    else:
        cmd_add(args)


if __name__ == "__main__":
    main()
