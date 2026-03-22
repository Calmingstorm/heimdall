from __future__ import annotations

import asyncio

from ..logging import get_logger

log = get_logger("ssh")

MAX_OUTPUT_CHARS = 16000


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

        if len(output) > MAX_OUTPUT_CHARS:
            half = MAX_OUTPUT_CHARS // 2
            output = output[:half] + "\n\n... (output truncated) ...\n\n" + output[-half:]

        return proc.returncode or 0, output

    except asyncio.TimeoutError:
        proc.kill()
        return 1, f"Command timed out after {timeout} seconds"
    except Exception as e:
        log.error("SSH command failed: %s", e)
        return 1, f"SSH error: {e}"
