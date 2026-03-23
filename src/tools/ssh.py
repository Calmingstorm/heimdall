from __future__ import annotations

import asyncio

from ..logging import get_logger

log = get_logger("ssh")

MAX_OUTPUT_CHARS = 16000

# Addresses considered "local" — commands run via subprocess, not SSH.
_LOCAL_ADDRESSES = frozenset({"127.0.0.1", "localhost", "::1"})


def is_local_address(address: str) -> bool:
    """Return True if *address* points to the local machine."""
    return address in _LOCAL_ADDRESSES


def _truncate_output(output: str) -> str:
    """Truncate output exceeding MAX_OUTPUT_CHARS, keeping head and tail."""
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    half = MAX_OUTPUT_CHARS // 2
    return output[:half] + "\n\n... (output truncated) ...\n\n" + output[-half:]


async def run_local_command(
    command: str,
    timeout: int = 30,
) -> tuple[int, str]:
    """Run a command locally via subprocess. Returns (exit_code, output).

    Used for localhost hosts — no SSH overhead, no key needed.
    """
    log.info("Local exec: %s", command)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode("utf-8", errors="replace")
        return proc.returncode or 0, _truncate_output(output)

    except asyncio.TimeoutError:
        proc.kill()
        return 1, f"Command timed out after {timeout} seconds"
    except Exception as e:
        log.error("Local command failed: %s", e)
        return 1, f"Local exec error: {e}"


async def run_ssh_command(
    host: str,
    command: str,
    ssh_key_path: str,
    known_hosts_path: str,
    timeout: int = 30,
    ssh_user: str = "root",
) -> tuple[int, str]:
    """Run a command on a remote host via SSH. Returns (exit_code, output)."""
    ssh_args = [
        "ssh",
        "-i", ssh_key_path,
        "-o", f"UserKnownHostsFile={known_hosts_path}",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        f"{ssh_user}@{host}",
        command,
    ]

    log.info("SSH to %s@%s: %s", ssh_user, host, command)

    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode("utf-8", errors="replace")
        return proc.returncode or 0, _truncate_output(output)

    except asyncio.TimeoutError:
        proc.kill()
        return 1, f"Command timed out after {timeout} seconds"
    except Exception as e:
        log.error("SSH command failed: %s", e)
        return 1, f"SSH error: {e}"
