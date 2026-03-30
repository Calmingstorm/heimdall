from __future__ import annotations

import asyncio
import base64
import json
import shlex
from pathlib import Path


from ..config.schema import ToolsConfig
from ..logging import get_logger
from .ssh import is_local_address, run_local_command, run_ssh_command

log = get_logger("tools")

# Maximum lines of output from run_command / run_command_multi before
# truncation.  The LLM can always re-run with head/tail/grep to see
# specific portions.
_RUN_COMMAND_MAX_LINES = 200


def _truncate_lines(text: str, max_lines: int = _RUN_COMMAND_MAX_LINES) -> str:
    """Truncate command output to *max_lines*, keeping first and last halves.

    Unlike the central character-based ``truncate_tool_output`` in
    ``client.py``, this cuts at line boundaries so the LLM always sees
    complete lines.  A notice is inserted in the middle telling the LLM
    how to get more specific output.
    """
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    keep = max_lines // 2
    omitted = len(lines) - max_lines
    return "\n".join(
        lines[:keep]
        + [f"[... {omitted} lines omitted — pipe through head/tail/grep for specific output ...]"]
        + lines[-keep:]
    )


class ToolExecutor:
    def __init__(
        self, config: ToolsConfig, memory_path: str | None = None,
        browser_manager: object | None = None,
    ) -> None:
        self.config = config
        self._memory_path = Path(memory_path) if memory_path else None
        self._browser_manager = browser_manager

    def _resolve_host(self, alias: str) -> tuple[str, str, str] | None:
        """Resolve host alias to (address, ssh_user, os). Returns None if not allowed."""
        host = self.config.hosts.get(alias)
        if not host:
            return None
        return host.address, host.ssh_user, host.os

    async def execute(self, tool_name: str, tool_input: dict, *, user_id: str | None = None) -> str:
        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            return f"Unknown tool: {tool_name}"

        try:
            if tool_name in ("memory_manage", "manage_list"):
                return await handler(tool_input, user_id=user_id)
            return await handler(tool_input)
        except Exception as e:
            log.error("Tool %s failed: %s", tool_name, e)
            return f"Error executing {tool_name}: {e}"

    def _host_os(self, alias: str) -> str:
        host = self.config.hosts.get(alias)
        return host.os if host else "linux"

    async def _exec_command(
        self,
        address: str,
        command: str,
        ssh_user: str = "root",
        timeout: int | None = None,
    ) -> tuple[int, str]:
        """Execute a command locally or via SSH depending on host address.

        Local hosts (127.0.0.1, localhost, ::1) use direct subprocess —
        no SSH key needed, no network overhead.
        """
        if timeout is None:
            timeout = self.config.command_timeout_seconds
        if is_local_address(address):
            return await run_local_command(command, timeout=timeout)
        return await run_ssh_command(
            host=address,
            command=command,
            ssh_key_path=self.config.ssh_key_path,
            known_hosts_path=self.config.ssh_known_hosts_path,
            timeout=timeout,
            ssh_user=ssh_user,
        )

    async def _run_on_host(self, alias: str, command: str) -> str:
        resolved = self._resolve_host(alias)
        if not resolved:
            return f"Unknown or disallowed host: {alias}"
        address, ssh_user, _os = resolved
        code, output = await self._exec_command(address, command, ssh_user)
        if code != 0:
            return f"Command failed (exit {code}):\n{output}"
        return output

    async def _handle_run_command(self, inp: dict) -> str:
        output = await self._run_on_host(inp["host"], inp["command"])
        return _truncate_lines(output)

    async def _handle_run_script(self, inp: dict) -> str:
        """Write a script to a temp file, execute it, and clean up."""
        host = inp["host"]
        script = inp["script"]
        interpreter = inp.get("interpreter", "bash")

        # Map interpreter to file extension
        ext_map = {
            "bash": ".sh", "sh": ".sh", "python3": ".py", "python": ".py",
            "node": ".js", "ruby": ".rb", "perl": ".pl",
        }
        ext = ext_map.get(interpreter, ".sh")
        filename = inp.get("filename") or f"heimdall_script{ext}"

        # Sanitize interpreter to prevent injection
        allowed_interpreters = {"bash", "sh", "python3", "python", "node", "ruby", "perl"}
        if interpreter not in allowed_interpreters:
            return f"Unsupported interpreter: {interpreter}. Use one of: {', '.join(sorted(allowed_interpreters))}"

        resolved = self._resolve_host(host)
        if not resolved:
            return f"Unknown or disallowed host: {host}"
        address, ssh_user, _os = resolved

        # Base64-encode script to avoid all quoting/heredoc issues
        encoded = base64.b64encode(script.encode()).decode()

        safe_filename = shlex.quote(filename)
        # Write to temp file, execute, capture output, clean up
        cmd = (
            f"TMPF=$(mktemp /tmp/{safe_filename}.XXXXXXXX) && "
            f"echo '{encoded}' | base64 -d > \"$TMPF\" && "
            f"chmod +x \"$TMPF\" && "
            f"{interpreter} \"$TMPF\" 2>&1; EXIT=$?; "
            f"rm -f \"$TMPF\"; exit $EXIT"
        )

        code, output = await self._exec_command(address, cmd, ssh_user)
        if code != 0:
            return f"Script failed (exit {code}):\n{_truncate_lines(output)}"
        return _truncate_lines(output)

    async def _handle_read_file(self, inp: dict) -> str:
        path = inp["path"]
        try:
            lines = min(int(inp.get("lines", 200)), 1000)
        except (TypeError, ValueError):
            lines = 200
        safe_path = shlex.quote(path)
        return await self._run_on_host(
            inp["host"],
            f"head -n {lines} {safe_path}",
        )

    async def _handle_write_file(self, inp: dict) -> str:
        path = inp["path"]
        content = inp["content"]
        safe_path = shlex.quote(path)
        # Base64-encode content to avoid shell injection via heredoc delimiter
        encoded = base64.b64encode(content.encode()).decode()
        cmd = f"mkdir -p $(dirname {safe_path}) && echo '{encoded}' | base64 -d > {safe_path}"
        return await self._run_on_host(inp["host"], cmd)

    # --- Multi-host tools ---

    async def _handle_run_command_multi(self, inp: dict) -> str:
        import asyncio
        hosts = inp["hosts"]
        command = inp["command"]

        # Expand "all"
        if hosts == ["all"] or hosts == "all":
            hosts = list(self.config.hosts.keys())

        async def _run_one(alias: str) -> str:
            result = await self._run_on_host(alias, command)
            result = _truncate_lines(result)
            return f"### {alias}\n```\n{result.strip()}\n```"

        tasks = [_run_one(h) for h in hosts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        parts = []
        for h, r in zip(hosts, results):
            if isinstance(r, Exception):
                parts.append(f"### {h}\n```\nError: {r}\n```")
            else:
                parts.append(r)
        return "\n\n".join(parts)

    # --- Browser tools (text-returning, screenshot handled in client.py) ---

    async def _handle_browser_read_page(self, inp: dict) -> str:
        if not self._browser_manager:
            return "Browser automation is not enabled. Set browser.enabled=true in config."
        from .browser import handle_browser_read_page
        return await handle_browser_read_page(self._browser_manager, inp)

    async def _handle_browser_read_table(self, inp: dict) -> str:
        if not self._browser_manager:
            return "Browser automation is not enabled. Set browser.enabled=true in config."
        from .browser import handle_browser_read_table
        return await handle_browser_read_table(self._browser_manager, inp)

    async def _handle_browser_click(self, inp: dict) -> str:
        if not self._browser_manager:
            return "Browser automation is not enabled. Set browser.enabled=true in config."
        from .browser import handle_browser_click
        return await handle_browser_click(self._browser_manager, inp)

    async def _handle_browser_fill(self, inp: dict) -> str:
        if not self._browser_manager:
            return "Browser automation is not enabled. Set browser.enabled=true in config."
        from .browser import handle_browser_fill
        return await handle_browser_fill(self._browser_manager, inp)

    async def _handle_browser_evaluate(self, inp: dict) -> str:
        if not self._browser_manager:
            return "Browser automation is not enabled. Set browser.enabled=true in config."
        from .browser import handle_browser_evaluate
        return await handle_browser_evaluate(self._browser_manager, inp)

    # --- Web tools ---

    async def _handle_web_search(self, inp: dict) -> str:
        from .web import web_search
        max_results = min(inp.get("max_results", 5), 10)
        return await web_search(inp["query"], max_results=max_results)

    async def _handle_fetch_url(self, inp: dict) -> str:
        from .web import fetch_url
        return await fetch_url(inp["url"])

    # --- PDF analysis ---

    @staticmethod
    def _parse_page_range(pages: str, total: int) -> list[int]:
        """Parse a page range string like '1-5' or '3' into 0-indexed page indices."""
        pages = pages.strip()
        if "-" in pages:
            parts = pages.split("-", 1)
            try:
                start = max(int(parts[0]) - 1, 0)
                end = min(int(parts[1]), total)
                return list(range(start, end))
            except ValueError:
                return list(range(total))
        else:
            try:
                idx = int(pages) - 1
                if 0 <= idx < total:
                    return [idx]
                return list(range(total))
            except ValueError:
                return list(range(total))

    async def _handle_analyze_pdf(self, inp: dict) -> str:
        url = inp.get("url")
        host = inp.get("host")
        path = inp.get("path")
        pages_str = inp.get("pages")

        # Validate URL scheme early (before heavy imports) to prevent SSRF
        if url and not url.startswith(("http://", "https://")):
            return "Only http:// and https:// URLs are supported."

        import fitz

        pdf_bytes: bytes | None = None

        if url:
            # Download PDF from URL
            import aiohttp
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status != 200:
                            return f"Failed to fetch PDF from URL (HTTP {resp.status})"
                        pdf_bytes = await resp.read()
            except Exception as e:
                return f"Failed to fetch PDF from URL: {e}"
        elif host and path:
            # Fetch from host via base64
            resolved = self._resolve_host(host)
            if not resolved:
                return f"Unknown or disallowed host: {host}"
            address, ssh_user, _os = resolved
            safe_path = shlex.quote(path)
            code, output = await self._exec_command(
                address, f"base64 -w0 {safe_path}", ssh_user,
            )
            if code != 0:
                return f"Failed to read PDF from host: {output}"
            try:
                pdf_bytes = base64.b64decode(output.strip())
            except Exception as e:
                return f"Failed to decode PDF data: {e}"
        else:
            return "Provide either 'url' or both 'host' and 'path'."

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            return f"Failed to open PDF: {e}"

        try:
            total = doc.page_count
            if pages_str:
                indices = self._parse_page_range(pages_str, total)
            else:
                indices = list(range(total))

            parts = []
            for i in indices:
                page = doc[i]
                text = page.get_text()
                parts.append(f"## Page {i + 1}\n{text}")

            result = "\n\n".join(parts)
            # Truncate to TOOL_OUTPUT_MAX_CHARS (handled by caller, but be safe)
            if len(result) > 12000:
                result = result[:12000] + "\n\n[... truncated — use pages parameter for specific pages ...]"
            return result if result.strip() else "PDF contains no extractable text. Try browser_screenshot for image-heavy PDFs."
        finally:
            doc.close()

    # --- Process management ---

    async def _handle_manage_process(self, inp: dict) -> str:
        # Lazy-init the process registry
        if not hasattr(self, "_process_registry"):
            from .process_manager import ProcessRegistry
            self._process_registry = ProcessRegistry()

        action = inp.get("action", "list")
        registry = self._process_registry

        if action == "start":
            command = inp.get("command")
            host = inp.get("host")
            if not command:
                return "command is required for start action."
            if not host:
                return "host is required for start action."
            # Validate host against configured hosts
            resolved = self._resolve_host(host)
            if not resolved:
                return f"Unknown or disallowed host: {host}"
            # Periodic cleanup
            registry.cleanup()
            return await registry.start(host, command)

        elif action == "poll":
            pid = inp.get("pid")
            if pid is None:
                return "pid is required for poll action."
            return registry.poll(int(pid))

        elif action == "write":
            pid = inp.get("pid")
            text = inp.get("input_text", "")
            if pid is None:
                return "pid is required for write action."
            if not text:
                return "input_text is required for write action."
            return await registry.write(int(pid), text)

        elif action == "kill":
            pid = inp.get("pid")
            if pid is None:
                return "pid is required for kill action."
            return await registry.kill(int(pid))

        elif action == "list":
            return registry.list_all()

        return f"Unknown action: {action}"

    # --- Claude Code ---

    async def _handle_claude_code(self, inp: dict) -> str:
        host = inp.get("host") or self.config.claude_code_host
        if not host:
            return "claude_code_host not configured in tools config"
        working_dir = inp["working_directory"]
        prompt = inp["prompt"]
        allowed_tools = inp.get("allowed_tools")
        allow_edits = inp.get("allow_edits", False)

        resolved = self._resolve_host(host)
        if not resolved:
            return f"Unknown or disallowed host: {host}"
        address, ssh_user, _os = resolved

        claude_user = self.config.claude_code_user
        tmpdir = ""
        import os
        _already_claude_user = (os.getenv("USER", "") == claude_user) if claude_user else False
        if allow_edits:
            if not claude_user:
                return "claude_code_user not configured — required for allow_edits=true"
            safe_user = shlex.quote(claude_user)
            if _already_claude_user:
                mktemp_cmd = "mktemp -d /tmp/claude_code_XXXXXXXX"
            else:
                mktemp_cmd = f"su - {safe_user} -c 'mktemp -d /tmp/claude_code_XXXXXXXX'"
            _, tmpdir = await self._exec_command(
                address, mktemp_cmd, ssh_user, timeout=10,
            )
            tmpdir = tmpdir.strip()
            if not tmpdir or not tmpdir.startswith("/tmp/claude_code_"):
                return f"Failed to create temp directory: {tmpdir}"
            prompt = prompt.replace(working_dir + "/", "./")
            prompt = prompt.replace(working_dir, ".")
            prompt = (
                f"IMPORTANT: Write ALL files relative to the current directory (.)."
                f" Do NOT use absolute paths. The current directory represents"
                f" {working_dir}.\n\n{prompt}"
            )

        import base64 as b64mod
        encoded_prompt = b64mod.b64encode(prompt.encode()).decode()

        claude_args = [
            "claude",
            "--print",
            "--output-format text",
            "--no-session-persistence",
        ]
        if allow_edits:
            claude_args.append("--dangerously-skip-permissions")
        if allowed_tools:
            claude_args.append(f"--allowedTools {shlex.quote(allowed_tools)}")

        claude_cmd = " ".join(claude_args)

        if allow_edits:
            safe_tmpdir = shlex.quote(tmpdir)
            inner = f"cd {safe_tmpdir} && echo '{encoded_prompt}' | base64 -d | timeout 3600 {claude_cmd}"
            if _already_claude_user:
                cmd = inner
            else:
                cmd = f"su - {safe_user} -c {shlex.quote(inner)}"
        else:
            safe_wd = shlex.quote(working_dir)
            cmd = f"cd {safe_wd} && echo '{encoded_prompt}' | base64 -d | timeout 3600 {claude_cmd}"

        code, output = await self._exec_command(
            address, cmd, ssh_user, timeout=3660,
        )

        file_manifest = ""

        if allow_edits:
            if code == 0:
                _, file_list = await self._exec_command(
                    address,
                    f"find {safe_tmpdir} -type f -not -path '*/.git/*' -not -name '*.pyc' | sort",
                    ssh_user, timeout=10,
                )
                files_found = file_list.strip()
                if files_found:
                    safe_target = shlex.quote(working_dir)
                    cp_code, cp_output = await self._exec_command(
                        address,
                        f"mkdir -p {safe_target} && cp -a {safe_tmpdir}/. {safe_target}/",
                        ssh_user, timeout=30,
                    )
                    if cp_code != 0:
                        file_manifest = (
                            f"\n\nWARNING: File copy to {working_dir} failed "
                            f"(exit {cp_code}): {cp_output[:300]}"
                        )
                    else:
                        target_files = files_found.replace(tmpdir, working_dir)
                        file_manifest = (
                            "\n\n--- FILES ON DISK ---\n"
                            "claude_code wrote these files directly to disk at "
                            f"{working_dir}. Do NOT rewrite them with write_file.\n"
                            f"{target_files}"
                        )
            # Clean up temp dir
            await self._exec_command(
                address, f"rm -rf {safe_tmpdir}", ssh_user, timeout=10,
            )

        if code != 0:
            return f"Claude Code failed (exit {code}):\n{output[-2000:]}"

        max_output = inp.get("max_output_chars", 3000)
        if len(output) > max_output:
            half = max_output // 2
            output = output[:half] + "[... truncated ...]" + output[-half:]
        return output + file_manifest

    def set_user_context(self, user_id: str | None) -> None:
        """Deprecated: user_id is now passed directly to execute().

        Kept for backward compatibility with tests. Prefer passing user_id
        as a keyword argument to execute() instead.
        """
        self._current_user_id = user_id

    def _load_all_memory(self) -> dict[str, dict[str, str]]:
        """Load the full scoped memory structure.

        Returns {"global": {...}, "user_<id>": {...}, ...}.
        Auto-migrates old flat format to scoped format.
        """
        if not self._memory_path or not self._memory_path.exists():
            return {"global": {}}
        try:
            data = json.loads(self._memory_path.read_text())
        except Exception:
            return {"global": {}}
        if not isinstance(data, dict):
            return {"global": {}}
        # Migrate old flat format: if no "global" key, treat entire dict as global
        if "global" not in data:
            migrated = {"global": data}
            self._save_all_memory(migrated)
            return migrated
        return data

    def _save_all_memory(self, data: dict[str, dict[str, str]]) -> None:
        if not self._memory_path:
            return
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_path.write_text(json.dumps(data, indent=2))

    def _load_memory(self) -> dict[str, str]:
        """Load merged global memory (backward-compatible for system prompt)."""
        return self._load_all_memory().get("global", {})

    def _load_memory_for_user(self, user_id: str | None) -> dict[str, str]:
        """Load merged global + user-specific memory for system prompt injection."""
        all_mem = self._load_all_memory()
        merged = dict(all_mem.get("global", {}))
        if user_id:
            user_key = f"user_{user_id}"
            merged.update(all_mem.get(user_key, {}))
        return merged

    async def _handle_memory_manage(self, inp: dict, *, user_id: str | None = None) -> str:
        action = inp["action"]
        scope = inp.get("scope", "personal")

        if action == "list":
            all_mem = await asyncio.to_thread(self._load_all_memory)
            global_mem = all_mem.get("global", {})
            user_mem = all_mem.get(f"user_{user_id}", {}) if user_id else {}
            lines = []
            if global_mem:
                lines.append("**Global notes:**")
                lines.extend(f"- **{k}**: {v}" for k, v in global_mem.items())
            if user_mem:
                lines.append("**Your personal notes:**")
                lines.extend(f"- **{k}**: {v}" for k, v in user_mem.items())
            return "\n".join(lines) if lines else "No notes saved yet."

        elif action == "save":
            key = inp.get("key")
            value = inp.get("value")
            if not key or not value:
                return "Both 'key' and 'value' are required for save."
            all_mem = await asyncio.to_thread(self._load_all_memory)
            if scope == "global":
                section = "global"
            elif user_id:
                section = f"user_{user_id}"
            else:
                section = "global"
            if section not in all_mem:
                all_mem[section] = {}
            all_mem[section][key] = value
            await asyncio.to_thread(self._save_all_memory, all_mem)
            scope_label = "global" if section == "global" else "personal"
            return f"Saved {scope_label} note '{key}'."

        elif action == "delete":
            key = inp.get("key")
            if not key:
                return "'key' is required for delete."
            all_mem = await asyncio.to_thread(self._load_all_memory)
            # Try user section first, then global
            user_key = f"user_{user_id}" if user_id else None
            if user_key and key in all_mem.get(user_key, {}):
                del all_mem[user_key][key]
                await asyncio.to_thread(self._save_all_memory, all_mem)
                return f"Deleted personal note '{key}'."
            elif key in all_mem.get("global", {}):
                del all_mem["global"][key]
                await asyncio.to_thread(self._save_all_memory, all_mem)
                return f"Deleted global note '{key}'."
            return f"No note found with key '{key}'."

        return f"Unknown memory action: {action}"

    # ------------------------------------------------------------------
    # Universal list management
    # ------------------------------------------------------------------

    def _lists_path(self) -> Path | None:
        """Return path to data/lists.json (sibling of memory.json)."""
        if not self._memory_path:
            return None
        return self._memory_path.parent / "lists.json"

    def _load_lists(self) -> dict:
        """Load all lists. Migrates old grocery_list.json on first access.

        Structure: {
            "grocery": {
                "owner": "shared",
                "items": [{"name": "...", "added_by": "...", "added_at": "...", "done": false}, ...]
            },
            ...
        }
        """
        path = self._lists_path()
        if not path:
            return {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        # Auto-migrate old grocery_list.json if it exists
        old_grocery = path.parent / "grocery_list.json"
        if old_grocery.exists():
            try:
                old_data = json.loads(old_grocery.read_text())
                old_items = old_data.get("items", [])
                migrated_items = []
                for item in old_items:
                    migrated_items.append({
                        "name": item.get("name", ""),
                        "added_by": item.get("added_by", ""),
                        "added_at": item.get("added_at", ""),
                        "done": False,
                    })
                lists = {"grocery": {"owner": "shared", "items": migrated_items}}
                self._save_lists(lists)
                return lists
            except Exception:
                pass
        return {}

    def _save_lists(self, data: dict) -> None:
        path = self._lists_path()
        if path:
            path.write_text(json.dumps(data, indent=2))

    async def _handle_manage_list(self, inp: dict, *, user_id: str | None = None) -> str:
        from datetime import datetime

        action = inp["action"]
        list_name = inp.get("list_name", "").strip().lower()
        raw_items = inp.get("items", [])
        owner_pref = inp.get("owner", "shared")

        lists = await asyncio.to_thread(self._load_lists)

        if action == "list_all":
            if not lists:
                return "No lists exist yet. Add items to create one."
            lines = ["**Your Lists**\n"]
            for name, lst in sorted(lists.items()):
                lst_owner = lst.get("owner", "shared")
                if lst_owner != "shared" and lst_owner != user_id:
                    continue
                count = len(lst.get("items", []))
                done = sum(1 for i in lst.get("items", []) if i.get("done"))
                owner_label = "shared" if lst_owner == "shared" else "personal"
                if done:
                    lines.append(f"- **{name}** ({count} items, {done} done) [{owner_label}]")
                else:
                    lines.append(f"- **{name}** ({count} items) [{owner_label}]")
            if len(lines) == 1:
                return "No lists visible to you."
            return "\n".join(lines)

        if not list_name:
            return "list_name is required for this action."

        # Resolve the list — check for personal or shared
        lst = lists.get(list_name)
        if lst and lst.get("owner") not in ("shared", user_id, None):
            return f"You don't have access to the '{list_name}' list."

        if action == "show":
            if not lst or not lst.get("items"):
                return f"The '{list_name}' list is empty."
            return self._format_list(list_name, lst)

        if action == "clear":
            if not lst or not lst.get("items"):
                return f"The '{list_name}' list is already empty."
            count = len(lst["items"])
            lst["items"] = []
            await asyncio.to_thread(self._save_lists, lists)
            return f"Cleared {count} item(s) from the '{list_name}' list."

        if action == "add":
            if not raw_items:
                return "No items specified to add."
            # Create list on the fly if it doesn't exist
            if not lst:
                owner = user_id if owner_pref == "personal" and user_id else "shared"
                lst = {"owner": owner, "items": []}
                lists[list_name] = lst
            added, already = [], []
            for name in raw_items:
                name = name.strip()
                if not name:
                    continue
                if any(i["name"].lower() == name.lower() for i in lst["items"]):
                    already.append(name)
                    continue
                lst["items"].append({
                    "name": name,
                    "added_by": user_id or "",
                    "added_at": datetime.now().isoformat(),
                    "done": False,
                })
                added.append(name)
            await asyncio.to_thread(self._save_lists, lists)
            parts = []
            if added:
                parts.append(f"Added to '{list_name}': {', '.join(added)}")
            if already:
                parts.append(f"Already on the list: {', '.join(already)}")
            parts.append(f"\n{self._format_list(list_name, lst)}")
            return "\n".join(parts)

        if action == "remove":
            if not lst:
                return f"The '{list_name}' list doesn't exist."
            if not raw_items:
                return "No items specified to remove."
            removed, not_found = [], []
            for name in raw_items:
                name = name.strip()
                if not name:
                    continue
                q = name.lower()
                matches = [i for i, item in enumerate(lst["items"]) if q in item["name"].lower()]
                if matches:
                    for idx in sorted(matches, reverse=True):
                        removed.append(lst["items"].pop(idx)["name"])
                else:
                    not_found.append(name)
            await asyncio.to_thread(self._save_lists, lists)
            parts = []
            if removed:
                parts.append(f"Removed from '{list_name}': {', '.join(removed)}")
            if not_found:
                parts.append(f"Not found: {', '.join(not_found)}")
            if lst["items"]:
                parts.append(f"\n{self._format_list(list_name, lst)}")
            else:
                parts.append(f"\nThe '{list_name}' list is now empty.")
            return "\n".join(parts)

        if action == "mark_done":
            if not lst:
                return f"The '{list_name}' list doesn't exist."
            if not raw_items:
                return "No items specified to mark as done."
            marked, not_found = [], []
            for name in raw_items:
                q = name.strip().lower()
                if not q:
                    continue
                found = False
                for item in lst["items"]:
                    if q in item["name"].lower() and not item.get("done"):
                        item["done"] = True
                        marked.append(item["name"])
                        found = True
                        break
                if not found:
                    not_found.append(name.strip())
            await asyncio.to_thread(self._save_lists, lists)
            parts = []
            if marked:
                parts.append(f"Marked done: {', '.join(marked)}")
            if not_found:
                parts.append(f"Not found or already done: {', '.join(not_found)}")
            parts.append(f"\n{self._format_list(list_name, lst)}")
            return "\n".join(parts)

        if action == "mark_undone":
            if not lst:
                return f"The '{list_name}' list doesn't exist."
            if not raw_items:
                return "No items specified to mark as undone."
            marked, not_found = [], []
            for name in raw_items:
                q = name.strip().lower()
                if not q:
                    continue
                found = False
                for item in lst["items"]:
                    if q in item["name"].lower() and item.get("done"):
                        item["done"] = False
                        marked.append(item["name"])
                        found = True
                        break
                if not found:
                    not_found.append(name.strip())
            await asyncio.to_thread(self._save_lists, lists)
            parts = []
            if marked:
                parts.append(f"Marked undone: {', '.join(marked)}")
            if not_found:
                parts.append(f"Not found or not done: {', '.join(not_found)}")
            parts.append(f"\n{self._format_list(list_name, lst)}")
            return "\n".join(parts)

        return f"Unknown action: {action}"

    @staticmethod
    def _format_list(list_name: str, lst: dict) -> str:
        items = lst.get("items", [])
        if not items:
            return f"The '{list_name}' list is empty."
        lines = [f"**{list_name.title()} List** ({len(items)} items)\n"]
        for i, item in enumerate(items, 1):
            done_mark = "\u2705 " if item.get("done") else ""
            strike = f"~~{item['name']}~~" if item.get("done") else item["name"]
            added = item.get("added_by", "")
            ts = item.get("added_at", "")
            suffix = ""
            if added or ts:
                parts = []
                if added:
                    parts.append(added)
                if ts:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(ts)
                        parts.append(dt.strftime("%b %d"))
                    except ValueError:
                        pass
                suffix = f"  _({', '.join(parts)})_"
            lines.append(f"{i}. {done_mark}{strike}{suffix}")
        return "\n".join(lines)
