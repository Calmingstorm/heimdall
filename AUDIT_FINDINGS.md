# Loki Code Audit Findings

## Session 1: `src/discord/client.py` — Central Hub (Routing, Tool Loop, Streaming, Attachments, Handoff)

| # | File:Line | Severity | Category | Description |
|---|-----------|----------|----------|-------------|
| 1 | client.py:260,360,1041,1074,1372-1378 | critical | bug | Shared `self._system_prompt` mutated concurrently across channels |
| 2 | client.py:1216,773 | critical | bug | `set_user_context()` sets shared state on `ToolExecutor`, races between channels |
| 3 | client.py:113,1393,1179-1182 | high | bug | `_pending_files` shared list leaks files between concurrent channels |
| 4 | client.py:1198 | high | bug | Return type annotation is 4-tuple but function returns 5-tuple |
| 5 | client.py:347-362 | medium | bug | `_inject_tool_hints` silently swallows exceptions and can leave stale hints |
| 6 | client.py:43-49 | medium | dead-code | `SECRET_SCRUB_PATTERNS` partially redundant with `_RESPONSE_EXTRA_PATTERNS` |
| 7 | client.py:940-942 | low | dead-code | `is_claude_code_by_keyword()` always returns False — unreachable branch |
| 8 | client.py:263 | low | dead-code | `query` parameter on `_build_system_prompt` is never used inside the function |
| 9 | client.py:300,388 | medium | best-practice | Calls private `_load_memory_for_user()` on `ToolExecutor` — encapsulation violation |
| 10 | client.py:877 | medium | best-practice | Accesses private `self.sessions._sessions` directly — bypasses SessionManager API |
| 11 | client.py:2053-2060 | medium | performance | Background task pruning only triggers when completed > 20, no time-based expiry |
| 12 | client.py:2189-2224 | medium | behavioral | `_on_scheduled_task` silently ignores unknown action types |
| 13 | client.py:155 | low | best-practice | `FullTextIndex` type annotation references name not imported when `search.enabled=False` |
| 14 | client.py:894-896 | low | bug | Channel lock creation is not atomic — TOCTOU race on `_channel_locks` dict |
| 15 | client.py:1171-1172 | low | behavioral | Voice callback called with unscrubbed response after error paths |

---

**Details:**

#### Issue 1: Shared `self._system_prompt` mutated concurrently across channels
- **File**: `src/discord/client.py:260,360,1041,1052,1074,1272,1372-1378`
- **Severity**: critical
- **Category**: bug (race condition)
- **Description**: `self._system_prompt` is a single instance attribute shared across all channels. It is mutated in multiple places:
  - Line 1041/1052/1074: `self._system_prompt = self._build_system_prompt(channel=..., user_id=...)` — rebuilds with channel-specific and user-specific context
  - Line 360: `self._system_prompt += f"\n\n{hints}"` — appends tool hints (non-atomic)
  - Lines 1372-1378: Rebuilt inside `_run_tool` when skills are created/edited/deleted
  - Line 1272: Set inside `_process_with_tools` during Haiku→Sonnet escalation

  The per-channel lock (line 894-898) only serialises messages *within the same channel*. Two concurrent messages from different channels both mutate `self._system_prompt`, meaning Channel A's user-specific prompt (with their memory, permissions, personality) can leak into Channel B's LLM call.
- **Impact**: Cross-channel context leakage. User A's memory/personality injected into User B's request. Channel-specific personality directives sent to wrong channel. Tool hints from one query applied to another.
- **Suggested fix**: Make the system prompt a local variable passed through the call chain, or store it per-channel. Never mutate shared instance state for per-request data.

#### Issue 2: `set_user_context()` sets shared state on `ToolExecutor`, races between channels
- **File**: `src/discord/client.py:1216` → `src/tools/executor.py:771-773`
- **Severity**: critical
- **Category**: bug (race condition)
- **Description**: `self.tool_executor.set_user_context(user_id)` at line 1216 sets `self._current_user_id` on the shared `ToolExecutor` instance. This is used later by memory-related tool handlers (`_handle_memory_set` at executor.py:817, `_handle_manage_list` at executor.py:935) to scope data per-user. Since two channels can run `_process_with_tools` concurrently (different channel locks), Channel A's `user_id` can be overwritten by Channel B before Channel A's tool calls read it.
- **Impact**: Memory writes (remember/forget) could be scoped to the wrong user. User A's "remember my server password is X" gets stored under User B's memory scope. List operations could modify another user's lists.
- **Suggested fix**: Pass `user_id` as a parameter to each tool execution call instead of storing it on the executor instance. Or use `contextvars.ContextVar` for async-safe per-task state.

#### Issue 3: `_pending_files` shared list leaks files between concurrent channels
- **File**: `src/discord/client.py:113,1393,1179-1182,2259-2261`
- **Severity**: high
- **Category**: bug (race condition)
- **Description**: `self._pending_files` is a shared instance-level list. Skill execution appends to it (line 1393 via `_skill_file` callback). After response processing, it's read and cleared (lines 1179-1182, 2259-2261). If two channels process messages concurrently, Channel A's skill-generated files can be attached to Channel B's response, or cleared before Channel A reads them.
- **Impact**: Files generated by skills could be sent to the wrong Discord channel, or lost entirely.
- **Suggested fix**: Make `_pending_files` a local variable threaded through the call chain, or use a per-channel dict keyed by `channel_id`.

#### Issue 4: Return type annotation mismatch on `_process_with_tools`
- **File**: `src/discord/client.py:1198`
- **Severity**: high
- **Category**: bug
- **Description**: The function signature declares `-> tuple[str, bool, bool, list[str]]` (4-tuple), but every return statement returns a 5-tuple `(text, already_sent, is_error, tools_used, handoff)`. The docstring (lines 1199-1204) also only documents 4 values and omits the `handoff` flag. All callers correctly unpack 5 values, so this doesn't crash at runtime, but:
  - Static type checkers will flag this
  - The docstring misleads anyone reading the code
  - The 5th value (`handoff`) is undocumented
- **Impact**: Maintenance hazard. A future developer adding a new caller could unpack 4 values and crash, or miss handling the `handoff` flag.
- **Suggested fix**: Update the signature to `-> tuple[str, bool, bool, list[str], bool]` and document the `handoff` parameter in the docstring.

#### Issue 5: `_inject_tool_hints` silently swallows exceptions and can leave stale hints
- **File**: `src/discord/client.py:347-362`
- **Severity**: medium
- **Category**: bug
- **Description**: `_inject_tool_hints` (line 360) appends hints to `self._system_prompt` with `+=`. The caller wraps it in `try/except Exception: pass` (lines 1043-1045, 1054-1056, 1076-1078). Two problems:
  1. If the function fails partway through (e.g., embedder timeout), partial state could exist.
  2. The `+=` mutates the shared instance variable (ties into Issue 1).
  3. The broad `except Exception: pass` means errors like `AttributeError`, `TypeError`, or even `KeyboardInterrupt` subtypes are silently eaten, making debugging impossible.
- **Impact**: Hints from a previous request persist when the current request's hint generation fails, causing the LLM to receive stale/wrong tool recommendations.
- **Suggested fix**: Build the full prompt (with hints) as a local variable and pass it to the LLM call. Log the exception at DEBUG level instead of silently swallowing it.

#### Issue 6: `SECRET_SCRUB_PATTERNS` partially redundant with `_RESPONSE_EXTRA_PATTERNS`
- **File**: `src/discord/client.py:43-49,59-63`
- **Severity**: medium
- **Category**: dead-code
- **Description**: `SECRET_SCRUB_PATTERNS` (lines 43-49) is used only by `_check_for_secrets` (line 537) to detect secrets in *user input* and delete the message. `_RESPONSE_EXTRA_PATTERNS` (lines 59-63) is used by `scrub_response_secrets` (line 74) to redact secrets from *LLM output*. Two of the four patterns in `SECRET_SCRUB_PATTERNS` are identical copies of the two patterns in `_RESPONSE_EXTRA_PATTERNS` (Slack tokens and natural language passwords). These serve different purposes (detection vs scrubbing) but the duplication is confusing and the patterns should be shared constants.
- **Impact**: Maintenance burden — changing one without the other creates inconsistency. The intent is unclear to future developers.
- **Suggested fix**: Extract shared patterns to a single list and compose `SECRET_SCRUB_PATTERNS` and `_RESPONSE_EXTRA_PATTERNS` from it.

#### Issue 7: `is_claude_code_by_keyword()` always returns False — dead branch
- **File**: `src/discord/client.py:940-942`, `src/discord/routing.py:51-54`
- **Severity**: low
- **Category**: dead-code
- **Description**: `is_claude_code_by_keyword()` in `routing.py` always returns `False` (docstring says "Deprecated"). The `elif` branch at line 940-942 that checks it is therefore unreachable dead code. The log message "Message matched claude_code keyword" can never fire.
- **Impact**: No runtime impact, but misleads code readers into thinking keyword-based claude_code routing still exists.
- **Suggested fix**: Remove the `elif is_claude_code_by_keyword(content)` branch entirely. Remove the function from routing.py and the import at line 37.

#### Issue 8: `query` parameter on `_build_system_prompt` is unused
- **File**: `src/discord/client.py:263-345`
- **Severity**: low
- **Category**: dead-code
- **Description**: The `query` parameter defined at line 266 is never referenced anywhere in the 82-line function body. It is passed from callers at lines 1041, 1052, 1074 but has no effect. It appears to be a remnant from a planned feature (query-dependent prompt customisation) that was never implemented.
- **Impact**: No runtime impact, but adds confusion.
- **Suggested fix**: Remove the `query` parameter from the signature and all call sites.

#### Issue 9: Calls private `_load_memory_for_user()` on ToolExecutor
- **File**: `src/discord/client.py:300,388`
- **Severity**: medium
- **Category**: best-practice
- **Description**: `_build_system_prompt` and `_build_chat_system_prompt` both call `self.tool_executor._load_memory_for_user(user_id)` — a private method (leading underscore) on another class. This violates encapsulation and tightly couples the client to the executor's internal memory format.
- **Impact**: If `ToolExecutor` refactors its memory storage, `client.py` breaks. The memory loading involves disk I/O (`json.loads(path.read_text())`) on every prompt build, which could also be a performance concern.
- **Suggested fix**: Add a public `get_user_memory(user_id)` method on `ToolExecutor` (or a dedicated `MemoryStore` class).

#### Issue 10: Accesses private `self.sessions._sessions` directly
- **File**: `src/discord/client.py:877`
- **Severity**: medium
- **Category**: best-practice
- **Description**: Thread context inheritance at line 877 accesses `self.sessions._sessions.get(parent_id)` — a private attribute of `SessionManager`. This bypasses any session management logic (e.g., lazy loading, expiry checks, locking).
- **Impact**: Fragile coupling. If `SessionManager` refactors `_sessions`, this breaks. No session-level locking is respected.
- **Suggested fix**: Add a public `get_session(channel_id) -> Session | None` method to `SessionManager`.

#### Issue 11: Background task pruning only size-based, no time expiry
- **File**: `src/discord/client.py:2053-2060`
- **Severity**: medium
- **Category**: performance
- **Description**: Completed background tasks are only pruned when `len(completed) > self._background_tasks_max` (20). This means:
  1. Up to 20 completed tasks are kept indefinitely with no time-based expiry.
  2. Pruning only triggers when creating a new task — if no new tasks are created, old results linger forever.
  3. The pruning loop removes from the front of a list rebuilt from dict iteration order, which isn't guaranteed to be chronological.
- **Impact**: Gradual memory growth over long bot uptime. Stale task results from weeks ago persist.
- **Suggested fix**: Add a timestamp to `BackgroundTask` and prune tasks older than N hours regardless of count. Run periodic cleanup.

#### Issue 12: `_on_scheduled_task` silently ignores unknown action types
- **File**: `src/discord/client.py:2189-2224`
- **Severity**: medium
- **Category**: behavioral
- **Description**: The `_on_scheduled_task` handler has explicit cases for `"digest"`, `"reminder"`, `"check"`, and `"workflow"` action types. If a schedule has any other action type (e.g., a typo like `"remider"` or a future action type), the function silently does nothing — no log, no error, no notification to the user.
- **Impact**: Users could create schedules that silently never fire. Debugging missed scheduled tasks would be very difficult.
- **Suggested fix**: Add an `else` clause that logs a warning: `log.warning("Unknown scheduled action type: %s", schedule["action"])`.

#### Issue 13: `FullTextIndex` type annotation with conditional import
- **File**: `src/discord/client.py:155,164`
- **Severity**: low
- **Category**: best-practice
- **Description**: Line 155 uses `FullTextIndex` in a type annotation (`self._fts_index: FullTextIndex | None = None`) but the import at line 164 only happens inside an `if config.search.enabled:` block. This works at runtime because `from __future__ import annotations` (line 1) defers annotation evaluation, but it would fail static type checkers and IDE analysis when `search.enabled=False`.
- **Impact**: Type checker warnings, potential `NameError` if annotations are ever evaluated at runtime (e.g., via `get_type_hints()`).
- **Suggested fix**: Move the `FullTextIndex` import to the top of the file, or use a string literal annotation `"FullTextIndex | None"`.

#### Issue 14: Channel lock creation is not atomic (TOCTOU race)
- **File**: `src/discord/client.py:894-896`
- **Severity**: low
- **Category**: bug
- **Description**: The channel lock creation pattern:
  ```python
  if channel_id not in self._channel_locks:
      self._channel_locks[channel_id] = asyncio.Lock()
  lock = self._channel_locks[channel_id]
  ```
  This is technically a TOCTOU (time-of-check/time-of-use) race. In asyncio single-threaded execution, this is safe because there's no `await` between check and set. However, if the code is ever refactored to add an `await` between these lines, or if Discord.py dispatches events from multiple threads, the lock could be created twice. The idiomatic fix is `self._channel_locks.setdefault(channel_id, asyncio.Lock())`.
- **Impact**: Extremely unlikely to cause issues in current single-threaded asyncio context, but a latent fragility.
- **Suggested fix**: Replace with `lock = self._channel_locks.setdefault(channel_id, asyncio.Lock())`.

#### Issue 15: Voice callback called with potentially wrong response after error paths
- **File**: `src/discord/client.py:1171-1172`
- **Severity**: low
- **Category**: behavioral
- **Description**: At line 1171-1172, `voice_callback(response)` is called regardless of whether `is_error` is True. This means if an error occurred (e.g., "Something went wrong: ..."), the error message text is spoken aloud in the voice channel via TTS. While the `is_error` branch at lines 1167-1169 removes the user message from history, it doesn't skip the voice callback.
- **Impact**: Users in voice chat hear error messages spoken aloud, which is a poor UX. Additionally, at line 1150, `response` is scrubbed via `scrub_response_secrets()`, but the voice callback is called *after* scrubbing, so that's at least safe.
- **Suggested fix**: Guard the voice callback: `if voice_callback and not is_error:`.

---

### Session 1 Summary

| Severity | Count |
|----------|-------|
| Critical | 2 |
| High | 2 |
| Medium | 6 |
| Low | 5 |
| **Total** | **15** |

**Key takeaways**: The two critical issues (#1 and #2) are the most urgent — shared mutable state (`self._system_prompt` and `self.tool_executor._current_user_id`) is modified per-request without synchronisation, allowing cross-channel/cross-user data leakage. These are inherent design issues from the bot originally being single-user/single-channel and not fully adapted for concurrent multi-channel operation. The fix requires making per-request state (system prompt, user context) flow through function parameters rather than stored on shared instances.

---

## Session 2: `src/tools/executor.py` + `src/tools/skill_manager.py` + `src/tools/skill_context.py` — Tool Execution, Skill Loading, Sandboxing

| # | File:Line | Severity | Category | Description |
|---|-----------|----------|----------|-------------|
| 1 | skill_context.py:77-84,190-202 vs executor.py:775-813 | critical | bug | SkillContext memory read/write uses flat schema, corrupts executor's scoped memory structure |
| 2 | skill_context.py:182-184 | critical | security | `execute_tool()` lets skills call any built-in tool (including `run_command`, `write_file`) without approval |
| 3 | executor.py:655-663 | high | security | `_handle_incus_exec` doesn't shell-quote command — allows host shell escape |
| 4 | skill_manager.py:169-193 | high | bug | `edit_skill` doesn't validate that new code's `SKILL_DEFINITION.name` matches filename |
| 5 | skill_manager.py:73-122 | high | security | Skills execute arbitrary Python in the bot's process with no sandboxing |
| 6 | executor.py:775-800,881-926 | medium | bug | Memory and lists files have no locking — concurrent read-modify-write races |
| 7 | executor.py:784,800,897,926; skill_context.py:194,202; skill_manager.py:152,175 | medium | best-practice | Synchronous file I/O in async handlers blocks the event loop |
| 8 | skill_context.py:98-134 | medium | security | `http_get`/`http_post` allow SSRF — skills can hit any internal service |
| 9 | background_task.py:183-225 | medium | security | `_execute_tool` bypasses per-tool approval — all delegated steps run unchecked |
| 10 | skill_manager.py:124-128 vs 73-74 | medium | bug | `_unload_skill` module name uses skill name but `_load_skill` uses file stem — mismatch for manually placed skills |
| 11 | executor.py:93 | medium | bug | `_format_metric_labels` mutates input dict via `.pop()` — only safe because caller creates a copy |
| 12 | skill_context.py:49-51 | medium | best-practice | `run_on_host` calls private `_run_on_host` — encapsulation violation |
| 13 | executor.py:313-320 | low | security | `read_file` has no path validation and doesn't require approval — any file readable |
| 14 | executor.py:503-508 | low | bug | `_handle_query_prometheus_range` curl command has fragile nested quoting |
| 15 | skill_manager.py:82 | low | best-practice | Skill module added to `sys.modules` before validation — leaked if validation fails |
| 16 | web.py:31-42 | low | bug | `_HTMLToText` skip tag tracking uses single boolean — nested skip tags break state |

---

**Details:**

#### Issue 1: SkillContext memory corrupts executor's scoped memory structure
- **File**: `src/tools/skill_context.py:77-84,190-202` vs `src/tools/executor.py:775-813`
- **Severity**: critical
- **Category**: bug (data corruption)
- **Description**: Both `ToolExecutor` and `SkillContext` operate on the same `memory.json` file (same `memory_path` passed from `client.py:190,204,210`), but with **incompatible schemas**:
  - `ToolExecutor._load_all_memory()` (line 775) reads memory as a **scoped structure**: `{"global": {"key": "val"}, "user_123": {"key": "val"}}`
  - `SkillContext._load_memory()` (line 190) reads the same file as a **flat dict** and returns whatever is in it
  - `SkillContext._save_memory()` (line 198) writes a flat dict back

  When a skill calls `context.remember("foo", "bar")`:
  1. `_load_memory()` reads `{"global": {"ip": "1.2.3.4"}, "user_123": {"pref": "dark"}}` — returns the entire scoped structure
  2. Sets `data["foo"] = "bar"`, creating `{"global": {"ip": "1.2.3.4"}, "user_123": {"pref": "dark"}, "foo": "bar"}`
  3. Writes this back to memory.json

  Now when `ToolExecutor._load_all_memory()` reads it, it finds `"global"` key (no migration triggered) but also a top-level `"foo": "bar"` entry with a string value instead of a dict. Any iteration expecting dict values will encounter a string and break. Similarly, `context.recall("global")` would return the entire global dict, not a user-stored value.
- **Impact**: **Memory corruption.** Skill memory operations silently corrupt the executor's scoped memory structure. Global and per-user memory entries could be lost or mangled. The corrupted state persists to disk.
- **Suggested fix**: `SkillContext` should either (a) use the executor's scoped memory API (`_load_all_memory`/`_save_all_memory`) with a dedicated skill scope, or (b) use a separate file (e.g., `skill_memory.json`) for skill-specific key-value storage.

#### Issue 2: `execute_tool()` lets skills bypass approval for any tool
- **File**: `src/tools/skill_context.py:182-184`
- **Severity**: critical
- **Category**: security
- **Description**: `SkillContext.execute_tool()` calls `self._executor.execute(tool_name, tool_input)` directly, dispatching to any `_handle_*` method without checking approval. This means a skill can:
  - Call `run_command` to execute arbitrary shell commands on any host
  - Call `write_file` to overwrite files on any host
  - Call `restart_service`, `git_push`, `docker_compose_action`, `incus_delete`, etc.
  - All without any approval check, as long as the skill itself was approved (or has `requires_approval: False`)

  The `create_skill` tool requires approval, but `edit_skill` does **not** (registry.py:563). So an initially harmless skill could be edited to add `execute_tool("run_command", {"host": "server", "command": "..."})` without triggering approval.
- **Impact**: Complete privilege escalation. Any skill, once created, can be edited without approval to execute destructive operations on any host. The approval model is fundamentally bypassed.
- **Suggested fix**: Either (a) `execute_tool` should check `requires_approval()` and refuse to call tools that need approval, or (b) add an allowlist of tools that skills can call via `execute_tool`, or (c) make `edit_skill` require approval too.

#### Issue 3: `_handle_incus_exec` doesn't shell-quote command — host shell escape
- **File**: `src/tools/executor.py:655-663`
- **Severity**: high
- **Category**: security
- **Description**: The `incus_exec` handler:
  ```python
  instance = shlex.quote(inp["instance"])
  command = inp["command"]
  cmd = f"incus exec {instance}"
  cmd += f" -- {command}"
  ```
  The `instance` is shell-quoted, but `command` is **not**. The `-- {command}` separator tells `incus` that everything after is the command, but the entire string is passed to the remote shell via SSH. The shell parses the full string before `incus` sees it.

  For example, if `command` is `ls ; rm -rf /`, the SSH shell executes `incus exec myinstance -- ls ; rm -rf /` as two commands: `incus exec myinstance -- ls` followed by `rm -rf /` on the **host** (not the container).
- **Impact**: The LLM-controlled `command` parameter can escape the Incus container and execute arbitrary commands on the host machine (the Incus host, typically the desktop with the RTX 4070 Ti).
- **Suggested fix**: Shell-quote the entire command, or use `shlex.quote(command)` and wrap: `incus exec {instance} -- sh -c {shlex.quote(command)}`.

#### Issue 4: `edit_skill` doesn't validate SKILL_DEFINITION.name matches filename
- **File**: `src/tools/skill_manager.py:169-193`
- **Severity**: high
- **Category**: bug
- **Description**: `create_skill` (line 161-164) validates that `skill.name == name` and rejects mismatches. However, `edit_skill` has no such check. If the new code changes `SKILL_DEFINITION["name"]` to a different value:
  1. `_unload_skill(name)` removes the old skill from `_skills` and `sys.modules`
  2. `_load_skill(path)` loads the new module with the old file stem
  3. `self._skills[name] = skill` stores the skill under the original name
  4. But `skill.name` (from SKILL_DEFINITION) is now different
  5. `get_tool_definitions()` returns the tool with the new name from SKILL_DEFINITION
  6. The Anthropic API sees a tool with the new name, but `has_skill(new_name)` returns False
  7. The tool call falls through to `executor.execute()` which returns "Unknown tool"
- **Impact**: Editing a skill to change its SKILL_DEFINITION name silently breaks the skill. The LLM will try to call it by the new name, get "Unknown tool" errors, and the user won't understand why.
- **Suggested fix**: Add the same name validation from `create_skill` to `edit_skill`: reject if `skill.name != name`.

#### Issue 5: Skills execute arbitrary Python in the bot's process with no sandboxing
- **File**: `src/tools/skill_manager.py:73-122` (loading), `245-277` (execution)
- **Severity**: high
- **Category**: security
- **Description**: User-created skills are arbitrary Python files loaded via `importlib` and executed directly in the bot's process. The skill code has:
  - Full access to the Python runtime (can import any module, access `os`, `subprocess`, `socket`, etc.)
  - Access to the `SkillContext` which provides SSH to all hosts, HTTP to any URL, and `execute_tool` for any built-in tool
  - Access to the `ToolExecutor` instance (reachable via `context._executor`) with config containing SSH key paths, host addresses, etc.
  - Code execution at **import time** (`spec.loader.exec_module(module)` at line 83) — malicious code doesn't even need to be in the `execute()` function

  The only protection is a 120-second timeout on `execute()` (line 268), which doesn't apply to import-time code, and the skill name regex (line 19) which prevents directory traversal in filenames.
- **Impact**: A malicious or poorly-written skill can exfiltrate secrets, SSH to hosts, corrupt the bot's state, or crash the process. Since `edit_skill` doesn't require approval, an approved skill can be replaced with malicious code without triggering any check.
- **Suggested fix**: At minimum, (a) make `edit_skill` require approval, (b) add an import allowlist validator before executing the module. For stronger isolation, run skills in a subprocess with limited environment. Document the trust model explicitly.

#### Issue 6: Memory and lists files have no locking — concurrent read-modify-write races
- **File**: `src/tools/executor.py:775-800` (memory), `881-926` (lists)
- **Severity**: medium
- **Category**: bug (race condition)
- **Description**: `_load_all_memory()`/`_save_all_memory()` and `_load_lists()`/`_save_lists()` perform read-modify-write cycles on JSON files without any locking. Since tool calls from different channels can run concurrently (different channel locks), two concurrent memory or list operations can race:
  1. Channel A reads memory.json: `{"global": {"a": "1"}}`
  2. Channel B reads memory.json: `{"global": {"a": "1"}}`
  3. Channel A adds key "b", writes: `{"global": {"a": "1", "b": "2"}}`
  4. Channel B adds key "c", writes: `{"global": {"a": "1", "c": "3"}}` — **overwrites A's change**
- **Impact**: Memory or list modifications can be silently lost during concurrent operations.
- **Suggested fix**: Use an `asyncio.Lock` around read-modify-write cycles, or use atomic file writes with `fcntl.flock`.

#### Issue 7: Synchronous file I/O in async handlers blocks the event loop
- **File**: `src/tools/executor.py:784,800,897,926`; `src/tools/skill_context.py:194,202`; `src/tools/skill_manager.py:152,175`
- **Severity**: medium
- **Category**: best-practice
- **Description**: Multiple async handlers call synchronous file I/O directly:
  - `self._memory_path.read_text()` / `.write_text()` (executor.py:784,800)
  - `path.read_text()` / `path.write_text()` (executor.py for lists, skill_manager for skill files)
  - `self._memory_path.read_text()` / `.write_text()` (skill_context.py:194,202)

  These are called from `async def` handlers via the `_handle_*` methods. Each `Path.read_text()`/`write_text()` call is blocking I/O that holds up the asyncio event loop. While individual file reads are fast, they become a concern under concurrent load or with slow storage.
- **Impact**: Event loop stalls during file I/O. Under load, this delays all other async tasks (Discord message handling, SSH operations, etc.).
- **Suggested fix**: Use `asyncio.to_thread(path.read_text)` or `aiofiles` for file I/O in async context.

#### Issue 8: `http_get`/`http_post` allow SSRF — skills can hit any internal service
- **File**: `src/tools/skill_context.py:98-134`
- **Severity**: medium
- **Category**: security
- **Description**: `SkillContext.http_get()` and `http_post()` make arbitrary HTTP requests with no URL validation. Skills can target:
  - Internal services (Prometheus at `http://127.0.0.1:9090`, Grafana, Gitea, etc.)
  - Cloud metadata endpoints (`http://169.254.169.254/`)
  - Docker socket via HTTP (`http://localhost:2375/`)
  - Any service on the internal network (`http://192.168.1.x:...`)

  Unlike `run_on_host` which is limited to configured hosts, HTTP has no restrictions.
- **Impact**: A skill (which can be edited without approval) can probe and interact with any HTTP service reachable from the bot container.
- **Suggested fix**: Add an allowlist of permitted URL patterns/hosts, or block private IP ranges and localhost by default.

#### Issue 9: Background task `_execute_tool` bypasses per-tool approval
- **File**: `src/discord/background_task.py:183-225`
- **Severity**: medium
- **Category**: security
- **Description**: When a background task runs, `_execute_tool()` calls `executor.execute()` and `skill_manager.execute()` directly without checking `requires_approval()` for individual tools. The `BLOCKED_TOOLS` set (line 29-35) blocks some tools, but **not** `run_command`, `restart_service`, `write_file`, `git_push`, `incus_exec`, `incus_delete`, or `docker_compose_action`.

  The design intent is "approve once for the whole batch," but the user only sees the step descriptions when approving `delegate_task`, not the actual commands. The LLM constructs the steps, so a prompt injection or misinterpretation could include destructive steps.
- **Impact**: Destructive tools that normally require per-use approval run unchecked inside background tasks.
- **Suggested fix**: Either (a) add `run_command`, `write_file`, `restart_service`, etc. to `BLOCKED_TOOLS`, or (b) show the actual tool inputs (not just descriptions) in the approval prompt, or (c) check `requires_approval()` per-step with a flag to batch-approve.

#### Issue 10: `_unload_skill` module name mismatch for manually placed skills
- **File**: `src/tools/skill_manager.py:124-128` vs `73-74`
- **Severity**: medium
- **Category**: bug
- **Description**: `_load_skill` (line 74) creates the module name from the file stem: `f"loki_skill_{path.stem}"`. `_unload_skill` (line 126) creates it from the skill name: `f"loki_skill_{name}"`. If a skill file is manually placed in `data/skills/` (not through `create_skill`) where the filename stem differs from `SKILL_DEFINITION.name`, the module name won't match.

  For example: file `my_tool.py` with `SKILL_DEFINITION = {"name": "my_custom_tool", ...}`:
  - `_load_skill` registers `loki_skill_my_tool` in `sys.modules`
  - `_skills["my_custom_tool"] = skill`
  - `_unload_skill("my_custom_tool")` removes `loki_skill_my_custom_tool` (wrong!) from `sys.modules`
  - `loki_skill_my_tool` remains in `sys.modules` forever
- **Impact**: Module leak in `sys.modules`. On re-import, the stale module could be used instead of the new file contents.
- **Suggested fix**: Store the module name on `LoadedSkill` and use it in `_unload_skill`, or enforce name-stem equality in `_load_all` like `create_skill` does.

#### Issue 11: `_format_metric_labels` mutates input dict via `.pop()`
- **File**: `src/tools/executor.py:91-97`
- **Severity**: medium
- **Category**: bug (latent)
- **Description**: `_format_metric_labels(metric: dict)` calls `metric.pop("__name__", "")` at line 93, which **mutates the passed dict**. The callers (`_format_vector` at line 107, `_format_matrix` at line 127) create copies with `dict(item.get("metric", {}))`, so this is currently safe. However:
  1. The function's type signature suggests it's a pure formatting function
  2. Any future caller that passes the original dict would have `__name__` silently removed
  3. The mutation is non-obvious and undocumented
- **Impact**: Latent bug — safe today, but fragile. A future caller skipping the `dict()` copy would lose data.
- **Suggested fix**: Use `name = metric.get("__name__", "")` and filter `__name__` from the label iteration: `{k: v for k, v in metric.items() if k != "__name__"}`.

#### Issue 12: `run_on_host` calls private `_run_on_host` method
- **File**: `src/tools/skill_context.py:49-51`
- **Severity**: medium
- **Category**: best-practice
- **Description**: `SkillContext.run_on_host()` delegates to `self._executor._run_on_host(alias, command)` — a private method. This creates tight coupling between SkillContext and ToolExecutor's internals. If `_run_on_host` is refactored (e.g., signature change, renaming), SkillContext silently breaks.
- **Impact**: Encapsulation violation. Maintenance risk.
- **Suggested fix**: Add a public `run_on_host(alias, command)` method to `ToolExecutor` that wraps `_run_on_host`.

#### Issue 13: `read_file` has no path validation and doesn't require approval
- **File**: `src/tools/executor.py:313-320`, `src/tools/registry.py:176-197`
- **Severity**: low
- **Category**: security
- **Description**: The `read_file` tool (`requires_approval: False`) reads any file on any configured host with no path restrictions. The LLM can read `/etc/shadow`, SSH private keys, application secrets, database files, etc. The path is shell-quoted (preventing injection) but not validated against an allowlist or restricted to certain directories.

  This is likely intentional (the bot needs to read config files for infrastructure management), but it means any user with bot access can read any file on any host.
- **Impact**: Sensitive files on managed hosts are readable without approval. Combined with the permission system (default tier is "user"), any user can have the LLM read sensitive files.
- **Suggested fix**: Consider adding path restrictions (e.g., block `/etc/shadow`, `~/.ssh/`, etc.) or making `read_file` require approval for sensitive paths.

#### Issue 14: `_handle_query_prometheus_range` has fragile nested shell quoting
- **File**: `src/tools/executor.py:503-508`
- **Severity**: low
- **Category**: bug (latent)
- **Description**: The curl command for Prometheus range queries uses complex nested quoting:
  ```python
  cmd = (
      f"curl -s 'http://...?query={safe_query}"
      f"&start='$(date -d '-{shlex.quote(duration)}' ...)'"
      f"&end='$(date ...)'"
      f"&step={safe_step}'"
  )
  ```
  The single quotes break and reform around `$(...)` command substitutions, with `shlex.quote(duration)` adding its own quoting layer. While this works for simple durations like `"1h"` or `"6h"`, the quoting is fragile:
  - `shlex.quote` may add its own single quotes around the duration (e.g., `'1h'`), creating `'-'1h''` which shells handle but is confusing
  - If `date -d` fails (e.g., unsupported on macOS), the curl URL will contain literal `$(date ...)` text
  - The interleaved quoting makes this very hard to audit for injection
- **Impact**: Fragile quoting could produce malformed URLs with unusual duration inputs. Not exploitable due to `shlex.quote`, but hard to maintain.
- **Suggested fix**: Calculate timestamps in Python and construct the curl URL with all values URL-encoded, avoiding shell command substitution entirely.

#### Issue 15: Skill module added to `sys.modules` before validation
- **File**: `src/tools/skill_manager.py:82-96`
- **Severity**: low
- **Category**: best-practice
- **Description**: At line 82, the loaded module is added to `sys.modules` before validation:
  ```python
  sys.modules[module_name] = module
  spec.loader.exec_module(module)  # executes module code
  # ... validation follows at line 86-103
  # if validation fails, line 89/95: del sys.modules[module_name]
  ```
  If `exec_module` raises an exception, the cleanup at line 121 (`sys.modules.pop(module_name, None)`) handles it. But if `exec_module` succeeds and a later validation step fails, the module is removed from `sys.modules` but any side effects from importing (global state changes, background threads started, etc.) persist.
- **Impact**: Import-time side effects from invalid skill files persist even after the skill is rejected.
- **Suggested fix**: Validate the module's exports in a try/finally that always cleans up sys.modules on failure. Consider pre-parsing the AST to check for `SKILL_DEFINITION` and `execute` before executing the module.

#### Issue 16: `_HTMLToText` skip tag tracking uses single boolean
- **File**: `src/tools/web.py:31-42`
- **Severity**: low
- **Category**: bug
- **Description**: The HTML-to-text converter uses a single `self._skip` boolean to track whether content should be skipped:
  ```python
  def handle_starttag(self, tag, attrs):
      if tag in self._skip_tags:
          self._skip = True
  def handle_endtag(self, tag):
      if tag in self._skip_tags:
          self._skip = False
  ```
  This breaks with nested skip tags. For example: `<nav>text<footer>text</footer>more text</nav>` — the `</footer>` end tag sets `_skip = False`, so "more text" inside `<nav>` is incorrectly included.
- **Impact**: Minor — some hidden text (from nested nav/footer/script tags) may leak into web search results. Does not affect correctness of tool operations.
- **Suggested fix**: Use a counter instead of a boolean: increment on start, decrement on end, skip when > 0.

---

### Session 2 Summary

| Severity | Count |
|----------|-------|
| Critical | 2 |
| High | 3 |
| Medium | 7 |
| Low | 4 |
| **Total** | **16** |

**Key takeaways**: The two critical issues are:
1. **Memory corruption** (#1) — `SkillContext` and `ToolExecutor` operate on the same `memory.json` file with incompatible schemas. Any skill using `context.remember()` will corrupt the executor's scoped memory structure. This is a data-loss bug.
2. **Approval bypass via skills** (#2) — `SkillContext.execute_tool()` gives skills unrestricted access to all built-in tools without approval checks. Combined with `edit_skill` not requiring approval (registry.py:563), this creates a privilege escalation path: create a benign skill (approved), then edit it to call destructive tools (no approval needed).

The security model for skills needs a fundamental rethink. Skills currently have the same level of access as the bot itself, with the only gate being the initial `create_skill` approval. The `edit_skill` → `execute_tool` path bypasses the entire approval system.

---

## Session 3: `src/llm/anthropic.py` + `src/llm/circuit_breaker.py` + `src/llm/openai_codex.py` — LLM Integration, Retry Logic, Streaming

Also covers: `src/llm/usage_tracker.py`, `src/llm/codex_auth.py`, `src/llm/system_prompt.py`

| # | File:Line | Severity | Category | Description |
|---|-----------|----------|----------|-------------|
| 1 | anthropic.py:293, circuit_breaker.py:84-89 | high | bug | `_call_with_retry` records failure per retry attempt — one failed request opens the circuit breaker |
| 2 | circuit_breaker.py:79-89, anthropic.py:73-79 | medium | bug | Circuit breaker not thread-safe — stream thread races with event loop thread |
| 3 | sessions/manager.py:172-174, learning/reflector.py:149-151 | medium | bug | Compaction and reflection bypass circuit breaker, retry logic, and budget checks |
| 4 | anthropic.py:75-83, client.py:1511-1522 | medium | bug | `ChatStreamResult` delivers partial text as "error text" on stream failure |
| 5 | openai_codex.py:190 | medium | behavioral | Codex `_read_stream` returns "(no response)" as a successful response |
| 6 | openai_codex.py:63-85 | medium | behavioral | `_convert_messages` drops non-string content — Codex gets conversation history with gaps |
| 7 | codex_auth.py:155-168 | medium | bug | `build_auth_url` doesn't URL-encode parameters — spaces in scopes break URL spec |
| 8 | usage_tracker.py:48,73-85 | medium | performance | `UsageTracker._save()` does synchronous file I/O on every API call |
| 9 | anthropic.py:16-23 | low | security | Secret scrubbing patterns miss AWS keys, GitHub tokens, and cloud credentials |
| 10 | anthropic.py:258-268, client.py:948-962 | low | behavioral | `classify_message` spends Haiku tokens even when budget is exceeded |
| 11 | anthropic.py:189 | low | best-practice | `start_stream` uses deprecated `asyncio.get_event_loop()` |
| 12 | openai_codex.py:109-117 | low | bug | Codex 401 auth retry consumes a retry attempt, reducing retries for transient errors |
| 13 | anthropic.py:120,218,269,279 | low | behavioral | Single circuit breaker shared between Haiku classification and Sonnet processing |

---

**Details:**

#### Issue 1: `_call_with_retry` opens circuit breaker too aggressively
- **File**: `src/llm/anthropic.py:278-304`, `src/llm/circuit_breaker.py:84-89`
- **Severity**: high
- **Category**: bug (behavioral)
- **Description**: `_call_with_retry` calls `self.breaker.record_failure()` on **every** transient error (rate limit, 500, connection error) at line 293 — once per retry attempt. With `MAX_RETRIES = 3` and `failure_threshold = 3` (circuit breaker default), a **single** failed request exhausting all retries records 3 failures, which immediately opens the circuit breaker.

  Timeline of a single rate-limited request:
  1. Attempt 1: `RateLimitError` → `record_failure()` (count=1)
  2. Attempt 2: `RateLimitError` → `record_failure()` (count=2)
  3. Attempt 3: `RateLimitError` → `record_failure()` (count=3) → **breaker opens**
  4. All subsequent requests for 60 seconds get `CircuitOpenError` → routed to Codex

  The circuit breaker is designed to detect **sustained** outages, but this implementation treats retry attempts as independent failures. One burst of rate limiting on a single request blocks the entire Anthropic API for 60 seconds.
- **Impact**: The bot drops to Codex-only mode after a single rate-limited request. Users lose access to Claude tools for 60 seconds unnecessarily. If Codex is also unavailable, the bot becomes completely unresponsive.
- **Suggested fix**: Only call `record_failure()` once per `_call_with_retry` invocation — after all retries are exhausted — not per attempt. Move `record_failure()` from line 293 to just before the final `raise last_error` at line 304.

#### Issue 2: Circuit breaker not thread-safe — stream thread races with event loop
- **File**: `src/llm/circuit_breaker.py:79-89`, `src/llm/anthropic.py:64-83`
- **Severity**: medium
- **Category**: bug (race condition)
- **Description**: `ChatStreamResult._run_stream()` runs in a thread pool executor (line 61) and calls `self._breaker.record_success()` (line 74) or `self._breaker.record_failure()` (line 79) from that **worker thread**. Meanwhile, the main asyncio event loop thread may simultaneously call `self.breaker.check()`, `record_failure()`, or `record_success()` from `_call_with_retry()` or `classify_message()`.

  `CircuitBreaker` uses plain instance attributes (`_failure_count`, `_state`, `_last_failure_time`) with no threading synchronization. While Python's GIL prevents memory corruption, `_failure_count += 1` in `record_failure()` is **not** atomic — it compiles to separate `LOAD_ATTR`, `LOAD_CONST`, `BINARY_ADD`, `STORE_ATTR` bytecodes with potential GIL releases between them. Two threads could read the same `_failure_count`, both add 1, and store the same result (lost update).

  More concerning: the stream thread calling `record_success()` (setting `_failure_count = 0`, `_state = "closed"`) while `_call_with_retry` is recording failures could reset the breaker mid-failure sequence, preventing the breaker from opening when it should.
- **Impact**: Breaker state could become inconsistent under concurrent streaming + non-streaming API calls. The breaker might fail to open during sustained outages or might close prematurely.
- **Suggested fix**: Add a `threading.Lock` to `CircuitBreaker` around state mutations, or move breaker recording from the stream thread to the async caller (record after `await stream.wait()`).

#### Issue 3: Session compaction and reflection bypass circuit breaker, retries, and budget
- **File**: `src/sessions/manager.py:172-174`, `src/learning/reflector.py:149-151,230-232`
- **Severity**: medium
- **Category**: bug
- **Description**: Both `SessionManager.compact()` and `ConversationReflector.reflect()`/`_consolidate()` call the raw Anthropic SDK client directly:
  ```python
  response = await asyncio.to_thread(
      claude_client.client.messages.create,
      model=HAIKU_MODEL, ...
  )
  ```
  This bypasses three safety layers:
  1. **Circuit breaker** — no `self.breaker.check()` call, so no fast-fail when the API is known to be down. The thread will block for the full HTTP timeout (potentially 30+ seconds).
  2. **Retry logic** — no `_call_with_retry()`, so transient errors (rate limits, 500s) cause immediate failure instead of retrying.
  3. **Budget check** — no `is_over_budget()` guard, so compaction/reflection continue burning tokens after the daily budget is exceeded.

  These operations are triggered automatically (compaction on session overflow, reflection on conversation end), so they run without user awareness.
- **Impact**: During API outages, compaction and reflection hang for the full timeout instead of fast-failing (60s recovery vs 30+ second block). Background operations can push token usage past the budget without any guard. The reflector catches exceptions (line 160-162) but the session manager at line 170-186 would propagate them.
- **Suggested fix**: Add a `haiku_call()` method to `ClaudeClient` that wraps Haiku API calls with circuit breaker checks, retry logic, and budget guards. Have compaction and reflection use this instead of the raw SDK client.

#### Issue 4: `ChatStreamResult` delivers partial text as "error text" on stream failure
- **File**: `src/llm/anthropic.py:64-90`, `src/discord/client.py:1511-1534`
- **Severity**: medium
- **Category**: bug
- **Description**: When `_run_stream()` encounters an error mid-stream (line 75-79), it stores the exception as `self._error` and signals completion via `None` on the queue. The consumer in `_stream_iteration` (client.py:1511-1534) first iterates through **all text deltas that arrived before the error**, accumulating them into `accumulated`. Then it checks `stream.message` (which is `None` because the stream failed), and sets `error_text = accumulated`.

  Scenario:
  1. Stream sends text deltas: "The server is running normally with " (partial sentence)
  2. Network error occurs mid-stream
  3. Consumer iterates and accumulates: "The server is running normally with "
  4. `final_msg is None`, `stream.error` is set
  5. Error handlers check for specific error patterns ("image", "rate", "overloaded") — none match
  6. `error_text = error_text or "Something went wrong..."` — since `error_text` is truthy (the partial text), it stays as "The server is running normally with "
  7. User sees "The server is running normally with " — a partial response with no indication it was cut off

  The streaming preview (if created) is deleted at line 1529-1533, so the user only sees the final send. But the partial text is indistinguishable from a valid (if oddly truncated) response.
- **Impact**: Users see truncated/partial responses that appear to be complete. No visual indication that an error occurred. The response could be misleading (e.g., a partial "no issues detected" when the full response would have listed problems).
- **Suggested fix**: When `stream.error` is set and `accumulated` contains text, prepend or append an error indicator: `error_text = accumulated + "\n\n[Response interrupted due to an error. The above may be incomplete.]"`. Or discard accumulated text entirely and use a clear error message.

#### Issue 5: Codex `_read_stream` returns "(no response)" as a successful response
- **File**: `src/llm/openai_codex.py:149-190`
- **Severity**: medium
- **Category**: behavioral
- **Description**: `_read_stream()` returns the literal string `"(no response)"` (line 190) when no text content is extracted from the SSE stream. The caller `_stream_request()` at lines 101-104 treats this as a **valid successful response**:
  ```python
  if resp.status == 200:
      result = await self._read_stream(resp)
      self.breaker.record_success()
      return result
  ```
  The circuit breaker records success (preventing it from opening), and the response flows all the way to Discord as if it were the LLM's actual reply.

  This can happen when:
  - The Codex API returns 200 with an empty response (content policy/moderation block)
  - The SSE event format changes and text arrives in unrecognized event types
  - The stream is truncated at the HTTP level (200 status was already sent before content)

  There is no way for the caller or user to distinguish "(no response)" from a genuine empty result.
- **Impact**: Users see "(no response)" in Discord with no indication of failure. The circuit breaker considers this "success" so it won't trigger protective behavior. The "(no response)" string is saved to conversation history (client.py:1154), polluting context for future messages.
- **Suggested fix**: Return an empty string or raise an exception when no content is received. Have the caller check for empty/None and provide a user-friendly error message. At minimum, log a warning in `_read_stream`.

#### Issue 6: `_convert_messages` drops non-string content — Codex gets history gaps
- **File**: `src/llm/openai_codex.py:63-85`
- **Severity**: medium
- **Category**: behavioral
- **Description**: `CodexChatClient._convert_messages()` at line 69 silently skips any message where `content` is not a string:
  ```python
  if not isinstance(content, str):
      continue
  ```
  In the Anthropic API message format, `content` can be a list of content blocks (text, tool_use, tool_result, image). This means:
  - Assistant messages with tool calls (content is a list of tool_use blocks) are **dropped**
  - User messages with tool results (content is a list of tool_result blocks) are **dropped**
  - Messages with image blocks are **dropped**

  When the bot falls back from Sonnet to Codex (budget exceeded mid-conversation, or circuit breaker trip), the conversation history may be rich with tool interaction messages. All of these get silently removed, creating gaps:

  Original history: User→"check disk"→Assistant→[tool_use]→User→[tool_result]→Assistant→"Disk is 80% full"→User→"what about memory?"
  After conversion: User→"check disk"→Assistant→"Disk is 80% full"→User→"what about memory?"

  The missing tool interaction means Codex has no context about what actually happened between the first user message and the assistant's response.
- **Impact**: Codex receives conversations with missing context, leading to confused or nonsensical responses. Follow-up questions that reference tool results make no sense without the intermediate messages.
- **Suggested fix**: Extract text from list-format content blocks (e.g., `[b["text"] for b in content if b.get("type") == "text"]`), or format tool_use/tool_result blocks as summary text (e.g., "[Used tool: check_disk → 80% used]").

#### Issue 7: `build_auth_url` doesn't URL-encode parameters
- **File**: `src/llm/codex_auth.py:155-168`
- **Severity**: medium
- **Category**: bug
- **Description**: `build_auth_url()` constructs the OAuth authorization URL query string with raw string interpolation:
  ```python
  query = "&".join(f"{k}={v}" for k, v in params.items())
  ```
  Several parameter values contain characters that must be percent-encoded per RFC 3986:
  - `REDIRECT_URI = "http://localhost:1455/auth/callback"` — contains `:`, `/`
  - `SCOPES = "openai profile email offline_access"` — contains **spaces**
  - `code_challenge` — base64url-encoded, may contain `+` or other special chars

  Spaces in the `scope` parameter produce an invalid URL like `...&scope=openai profile email offline_access&...` where everything after the first space is ambiguous. The OpenAI auth endpoint likely applies lenient parsing, but this is technically invalid and could break with stricter server implementations or intermediary proxies.
- **Impact**: Auth flow could fail if the OpenAI endpoint or any intermediary enforces strict URL parsing. Spaces in the scope parameter are the most likely failure point.
- **Suggested fix**: Replace the manual string joining with `urllib.parse.urlencode(params)` which handles all encoding correctly.

#### Issue 8: `UsageTracker._save()` performs synchronous file I/O on every API call
- **File**: `src/llm/usage_tracker.py:43-48,73-85`
- **Severity**: medium
- **Category**: performance
- **Description**: `UsageTracker.record()` (line 43-48) calls `self._save()` which does `path.write_text(json.dumps(...))` — synchronous blocking file I/O. This is called after **every** LLM API call:
  - After `chat()` at anthropic.py:147
  - After `classify_message()` at anthropic.py:265
  - After stream completion at client.py:1537
  - After session compaction at manager.py:182
  - After reflection at reflector.py:156,237

  On a typical user message that triggers classification + streaming + 2 tool calls, this produces 3-4 synchronous file writes on the event loop. Additionally, `_load_today()` (called from both `record()` and `is_over_budget()`) does date comparison and potential disk reads on every invocation.

  While individual writes are fast on local SSD, this adds latency to every API interaction and blocks the event loop from processing other events during I/O.
- **Impact**: Event loop blocked during file writes. Adds cumulative latency to every message. Could become noticeable under load or on slower storage.
- **Suggested fix**: Buffer writes — record in memory and flush to disk periodically (e.g., every 10 records or every 30 seconds) using a background task. Or use `asyncio.to_thread()` for the write operation.

#### Issue 9: Secret scrubbing patterns miss common cloud and CI credentials
- **File**: `src/llm/anthropic.py:16-23`
- **Severity**: low
- **Category**: security
- **Description**: `OUTPUT_SECRET_PATTERNS` covers passwords, API keys, generic tokens, SSH private keys, and database URLs. However, several common credential formats are not matched:
  - AWS access keys: `AKIA[0-9A-Z]{16}` (always starts with `AKIA`)
  - GitHub personal access tokens: `ghp_[a-zA-Z0-9]{36}`, `github_pat_*`
  - GitHub OAuth tokens: `gho_*`, `ghu_*`, `ghs_*`
  - Slack webhook URLs: `https://hooks.slack.com/services/T.../B.../...`
  - Generic `Authorization: Bearer ...` headers in command output

  Since the bot manages infrastructure and runs shell commands, tool output could contain these patterns when users ask to check environment variables, inspect config files, or review HTTP traffic.
- **Impact**: Credentials of these types pass through unscrubbed to the LLM. The LLM might echo them in its response, which then goes to Discord (mitigated by `scrub_response_secrets` but it uses the same limited pattern set).
- **Suggested fix**: Add patterns for AWS keys (`re.compile(r"AKIA[0-9A-Z]{16}")`), GitHub tokens (`re.compile(r"gh[pousr]_[a-zA-Z0-9]{36,}")`), and Slack webhooks. Consider integrating a more comprehensive secret detection library.

#### Issue 10: `classify_message` spends tokens even when budget is exceeded
- **File**: `src/llm/anthropic.py:257-268`, `src/discord/client.py:948-962`
- **Severity**: low
- **Category**: behavioral
- **Description**: The comment at client.py:948 says "Always classify, even when budget exceeded, because claude_code and chat are free." However, `classify_message()` itself calls the Haiku API (line 258-264) and records the usage tokens (`self.usage.record()` at line 265-268). There is no budget check inside `classify_message()`.

  This means after the daily budget is exceeded:
  - Each incoming message still costs ~100-500 Haiku tokens for classification
  - These tokens are recorded to the usage tracker, inflating the "exceeded" total
  - On a high-traffic day, post-budget classification could accumulate thousands of extra tokens

  The classification result when budget is exceeded is also partially wasted — if `msg_type == "task"` it's immediately downgraded to `"chat"` at line 965-967, so the LLM call was unnecessary for task-classified messages.
- **Impact**: Minimal direct cost (~$0.001 per classification), but conceptually inconsistent — "budget exceeded" doesn't actually stop all Anthropic API usage. Accumulated post-budget tokens could affect cost tracking.
- **Suggested fix**: When `budget_exceeded` is True, skip classification and default to `"chat"` (since tasks are blocked anyway). The `is_claude_code_by_keyword` check already handles the claude_code case before classification.

#### Issue 11: `start_stream` uses deprecated `asyncio.get_event_loop()`
- **File**: `src/llm/anthropic.py:189`
- **Severity**: low
- **Category**: best-practice
- **Description**: `start_stream()` calls `asyncio.get_event_loop()` to capture the loop reference for the stream thread to use with `run_coroutine_threadsafe()`. In Python 3.10+, `get_event_loop()` is deprecated and emits `DeprecationWarning` when called without a running loop. While this method is always called from an async context (where a loop is running), the modern and explicit API is `asyncio.get_running_loop()`, which raises `RuntimeError` if no loop is running rather than silently creating a new one.
- **Impact**: No runtime impact in normal operation (a loop is always running when this is called). May emit deprecation warnings in tests or edge cases. Could create a detached loop if called outside an async context.
- **Suggested fix**: Replace `asyncio.get_event_loop()` with `asyncio.get_running_loop()` at line 189.

#### Issue 12: Codex 401 auth retry consumes a retry attempt
- **File**: `src/llm/openai_codex.py:109-117`
- **Severity**: low
- **Category**: bug (edge case)
- **Description**: When the Codex API returns 401 (auth expired) on `attempt == 0`, the code refreshes the token and `continue`s the loop. This consumes `attempt=0` for the auth refresh, leaving only `attempt=1` and `attempt=2` for genuine transient errors.

  If the second request (after refresh) hits a 429 or 500, only one more retry remains instead of the expected two. The sequence:
  1. Attempt 0: 401 → refresh token → continue
  2. Attempt 1: 429 → record failure → retry (wait 5s)
  3. Attempt 2: 429 → record failure → raise RuntimeError

  Without the 401, the same transient error would get 3 attempts. The auth refresh silently reduces retry capacity by 33%.
- **Impact**: Slightly reduced resilience to transient errors after auth refresh. Edge case — unlikely to matter in practice since auth refresh is rare.
- **Suggested fix**: Use a separate loop variable for auth retries, or deduct from `attempt` when a 401 refresh is consumed: `attempt -= 1` after the refresh `continue`.

#### Issue 13: Single circuit breaker shared across Haiku and Sonnet
- **File**: `src/llm/anthropic.py:120,218,269,279`
- **Severity**: low
- **Category**: behavioral
- **Description**: `ClaudeClient` has one `CircuitBreaker` instance (`self.breaker` at line 120) shared across all API operations:
  - `classify_message()` uses **Haiku** and records success/failure at lines 269/275
  - `_call_with_retry()` uses **Sonnet** (or model_override) and records at lines 286/293
  - `start_stream()` uses **Sonnet** and the stream thread records at lines 73-79

  A successful Haiku classification could close the breaker even when Sonnet is experiencing model-specific issues. Conversely, Sonnet failures could block Haiku classifications.

  In practice, Haiku and Sonnet share the same Anthropic API infrastructure, so correlated failures are the norm. However, model-specific outages (e.g., Sonnet at capacity while Haiku is fine) would cause the breaker to oscillate — Haiku successes close it, Sonnet failures reopen it, creating an unstable feedback loop.
- **Impact**: Minor — Anthropic API issues typically affect all models simultaneously. But during model-specific degradation, the breaker state may be unreliable, alternately blocking and allowing requests.
- **Suggested fix**: Consider separate breakers for classification (Haiku) and main processing (Sonnet). Alternatively, accept the shared breaker as a reasonable simplification given the shared API backend.

---

### Session 3 Summary

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 1 |
| Medium | 7 |
| Low | 5 |
| **Total** | **13** |

**Key takeaways**:

1. **Circuit breaker aggression** (#1) is the highest-severity issue — a single rate-limited request opens the breaker for 60 seconds, unnecessarily degrading the bot to Codex-only mode. The fix is simple: count failures per invocation, not per retry attempt.

2. **Bypass of safety layers** (#3) — session compaction and reflection call the raw Anthropic SDK directly, skipping the circuit breaker, retries, and budget checks. These background operations can hang during outages and burn tokens past the budget without any guard.

3. **Silent failure modes** (#4, #5, #6) — the LLM layer has several paths where failures are hidden from the user: partial stream responses shown as if complete, empty Codex responses returned as "(no response)", and conversation context silently dropped when converting for Codex. These create confusing user experiences where the bot appears to respond but the response is wrong or incomplete.

4. **The LLM layer is generally well-structured** — retry logic, circuit breaker pattern, budget tracking, and secret scrubbing are all present and correctly applied on the main code paths. The issues are mostly in edge cases and the secondary code paths (compaction, reflection, fallback routing) that weren't given the same treatment as the primary request flow.

---

## Session 4: `src/sessions/manager.py` + `src/learning/reflector.py` + `src/tools/tool_memory.py` — State Management, Persistence, Learning

| # | File:Line | Severity | Category | Description |
|---|-----------|----------|----------|-------------|
| 1 | reflector.py:320-324 vs 217-274 | high | bug | `_parse_entries` strips `user_id` — consolidation permanently loses per-user attribution |
| 2 | manager.py:199-207,250-256 | high | behavioral | Reflection attributes all compacted/archived entries to `last_user_id` only — multi-user misattribution |
| 3 | manager.py:871-891 (client.py) | high | bug | Thread context inheritance runs outside channel lock — race on thread session initialization |
| 4 | reflector.py:280-290 | medium | bug | `_parse_entries` fence-stripping logic is a no-op — includes all non-fence lines regardless of position |
| 5 | tool_memory.py:176-182 | medium | bug | `find_patterns` skips Jaccard fallback when cosine is below threshold — entries silently missed |
| 6 | manager.py:208-211 | medium | bug | Compaction fallback trims messages but doesn't clear stale summary — context gap |
| 7 | manager.py:383-387, client.py:1156 | medium | performance | `save()` writes ALL sessions to disk after every message — only one session changed |
| 8 | reflector.py:67-91, client.py:300,307,388,395 | medium | performance | `get_prompt_section` reads learned.json from disk on every system prompt build — no caching |
| 9 | manager.py:75-93 | medium | performance | `_find_recent_summary` does synchronous I/O on all archive files during session creation |
| 10 | manager.py:296-322 | medium | performance | `search_history` reads archive JSON files synchronously on the event loop |
| 11 | tool_memory.py:121-141 | medium | performance | Embedding vectors (768 floats each) stored in JSON — tool_memory.json grows to 2-3MB |
| 12 | manager.py:236-247 | medium | performance | Archive directory grows without bound — no cleanup of old archives |
| 13 | manager.py:147-211 | medium | behavioral | Compaction holds channel lock during Haiku API call — blocks other messages for that channel |
| 14 | manager.py:150 | low | bug | `COMPACTION_THRESHOLD` is hardcoded at 40 but `keep_count` depends on configurable `max_history` — mismatch possible |
| 15 | tool_memory.py:98-103 | low | bug | `_expire()` uses string comparison for ISO timestamps — fragile if timezone suffix varies |
| 16 | manager.py:217-234, client.py:1155 | low | performance | `prune()` iterates all sessions after every message — O(n) per message |
| 17 | reflector.py:55-65 | low | best-practice | `_load()` and `_save()` use synchronous file I/O under async lock |

---

**Details:**

#### Issue 1: `_parse_entries` strips `user_id` — consolidation permanently loses per-user attribution
- **File**: `src/learning/reflector.py:320-324` (parse) vs `217-274` (consolidate)
- **Severity**: high
- **Category**: bug (data loss)
- **Description**: `_parse_entries()` validates and returns entries with only three fields: `key`, `category`, `content` (lines 320-324). It discards any other fields, including `user_id`. This is fine for new reflections (where `user_id` is added after parsing at lines 180-184), but it breaks consolidation:

  1. `_consolidate()` sends all entries (with `user_id`) to Haiku and asks it to "Preserve the user_id field exactly as-is" (line 225)
  2. The LLM response is parsed through `_parse_entries()` (line 248), which **strips `user_id`**
  3. The recovery logic at lines 264-265 attempts to restore `user_id` from originals by matching `key`:
     ```python
     if "user_id" not in entry and "user_id" in orig:
         entry["user_id"] = orig["user_id"]
     ```
  4. But if the LLM merged two entries into a new key, or renamed a key during consolidation, `orig_by_key[entry["key"]]` won't find the original, and `user_id` is permanently lost

  After one consolidation cycle, user-specific entries (preferences, corrections) may lose their `user_id` and become global entries, leaking User A's preferences into User B's prompt.
- **Impact**: User-specific learned preferences/corrections gradually become global after consolidation. User A's "always use dark mode" preference could end up in User B's system prompt.
- **Suggested fix**: Include `user_id` in the `_parse_entries` validated output: add `"user_id": item.get("user_id")` to the returned dict (line 320-324). This way `user_id` is preserved through the parse→consolidate round-trip without relying on key matching.

#### Issue 2: Reflection attributes all entries to `last_user_id` — multi-user misattribution
- **File**: `src/sessions/manager.py:199-207` (compaction reflection), `250-256` (archive reflection)
- **Severity**: high
- **Category**: behavioral
- **Description**: `Session.last_user_id` (line 39) stores only the **most recent** human user's ID. When compaction or archival triggers reflection, it passes `user_id=session.last_user_id`:

  - Compaction (line 203): `user_id=session.last_user_id`
  - Archive (line 365-366): `user_id=session.last_user_id`

  In multi-user channels, the compacted/archived messages span multiple users. For example:
  1. User A says "I prefer verbose output" (correction/preference)
  2. User B asks a question (becomes `last_user_id`)
  3. Compaction triggers → `user_id = User B's ID`
  4. Reflector records User A's preference under User B's `user_id`

  This means:
  - User A's preferences are now attributed to User B
  - User A won't see their own preferences in future prompts (filtered by `user_id` at reflector.py:84)
  - User B gets User A's preferences injected into their prompt
- **Impact**: In shared channels, user-specific learned context gradually migrates to whoever sent the last message. Preferences and corrections are attributed to the wrong user.
- **Suggested fix**: Track user IDs per-message (already done via the `[display_name]:` prefix in messages), and have the reflector extract user attribution from message content. Or pass the full set of participant user IDs and let the reflector attribute each insight individually.

#### Issue 3: Thread context inheritance runs outside channel lock — race on session initialization
- **File**: `src/discord/client.py:871-891`
- **Severity**: high
- **Category**: bug (race condition)
- **Description**: The thread context inheritance block (lines 871-891) runs **before** the per-channel lock is acquired (line 894-898). It calls `self.sessions.get_or_create(channel_id)` and then checks `if not thread_session.messages:` to decide whether to seed the thread with parent context.

  In asyncio, this is currently safe because there's no `await` in this block — the code runs to completion before any other coroutine can execute. However, if `_find_recent_summary` (called by `get_or_create` at line 66) is ever made async (as it should be, given it does disk I/O), or if any `await` is added to this block, two concurrent messages to the same thread would race:

  1. Message A: `get_or_create` creates session, sees no messages, starts seeding summary
  2. `await` yields control
  3. Message B: `get_or_create` returns same session, also sees no messages, also seeds summary
  4. Both write to `thread_session.summary`, with last-write-wins

  More importantly, this code accesses `self.sessions._sessions` directly (line 877) and mutates a `Session` object that could be simultaneously accessed by the parent channel's lock-holding coroutine (e.g., during `save()` or `prune()`).
- **Impact**: Currently safe due to no `await` in the block, but a latent race condition that will activate if the code is ever made properly async. The direct `_sessions` access bypasses any future session-level locking.
- **Suggested fix**: Move the thread context inheritance inside the per-channel lock. This ensures only one coroutine at a time modifies the thread session.

#### Issue 4: `_parse_entries` fence-stripping logic is a no-op
- **File**: `src/learning/reflector.py:280-290`
- **Severity**: medium
- **Category**: bug
- **Description**: The markdown fence-stripping code at lines 280-290 is logically broken:
  ```python
  inside = False
  for line in lines:
      if line.strip().startswith("```"):
          inside = not inside
          continue
      if inside or not any(line.strip().startswith(c) for c in ("```",)):
          filtered.append(line)
  ```
  For non-fence lines, the second condition is: `inside or not any(line.strip().startswith("```"))`. Since non-fence lines don't start with "\`\`\`", `any(...)` is `False`, and `not False` is `True`. So the condition is always `True` for non-fence lines, regardless of `inside`.

  This means **all** non-fence lines are included in `filtered` — both inside and outside the fences. The intent was to only keep content inside the fences, but the logic includes everything. Example:
  - Input: `"Here are the entries:\n```json\n[{...}]\n```\nHope that helps!"`
  - Expected output: `[{...}]`
  - Actual output: `Here are the entries:\n[{...}]\nHope that helps!`

  The subsequent JSON parsing at lines 292-306 recovers from this via `raw.find("[")` / `raw.rfind("]")` fallback, so this doesn't cause runtime failures — but the fence-stripping logic is dead code that provides false confidence.
- **Impact**: No runtime failures (fallback parsing compensates), but the fence-stripping code misleads developers into thinking it works. If the LLM ever returns `]` after the JSON block (e.g., in trailing text), `rfind("]")` would capture the wrong endpoint.
- **Suggested fix**: Fix the condition to `if inside:` (remove the second disjunct). Or simplify: just use the `find("[")` / `rfind("]")` approach directly without the fence-stripping attempt.

#### Issue 5: `find_patterns` skips Jaccard fallback when cosine is below threshold
- **File**: `src/tools/tool_memory.py:176-182`
- **Severity**: medium
- **Category**: bug
- **Description**: In `find_patterns()`, when both query and entry have embeddings:
  ```python
  if query_embedding and entry_emb:
      score = _cosine(query_embedding, entry_emb)
      if score >= MIN_SEMANTIC_SCORE:
          scored.append((score, entry))
      continue  # <-- unconditional continue
  # Fallback to Jaccard
  entry_kw = set(entry.get("keywords", []))
  score = _jaccard(query_kw, entry_kw)
  ```
  The `continue` at line 182 executes regardless of whether the cosine score met the threshold. When cosine is below `MIN_SEMANTIC_SCORE` (0.65), the entry is skipped entirely — Jaccard fallback never runs.

  This creates an asymmetry: old entries without embeddings get a Jaccard-based match opportunity, but newer entries with embeddings only get cosine matching. An entry with cosine 0.55 (below threshold) and Jaccard 0.40 (well above the 0.15 threshold) is silently dropped.
- **Impact**: Tool pattern hints become less effective over time as more entries gain embeddings. Entries with moderate semantic similarity but high keyword overlap are missed, reducing hint quality.
- **Suggested fix**: Remove the `continue` and let both scoring paths run, taking the max score: `score = max(_cosine(...), _jaccard(...))`. Or restructure to try cosine first, then fall through to Jaccard if below threshold.

#### Issue 6: Compaction fallback trims messages but doesn't clear stale summary
- **File**: `src/sessions/manager.py:208-211`
- **Severity**: medium
- **Category**: bug
- **Description**: When the compaction API call fails (exception at line 208), the fallback trims messages:
  ```python
  except Exception as e:
      log.error("Failed to compact session: %s", e)
      session.messages = session.messages[-self.max_history:]
  ```
  This drops older messages but **does not update or clear `session.summary`**. The existing summary was generated from a previous compaction and describes messages that may have been trimmed by earlier compactions. After this fallback:

  1. `session.summary` describes old messages from a previous compaction
  2. `session.messages` contains only the last `max_history` messages
  3. There's a gap between what the summary covers and what the messages contain
  4. On the next `get_history()` call, the LLM gets the old summary + recent messages with no awareness of the gap

  If this failure happens during the first compaction (no existing summary), the messages are trimmed without any summary at all — all context from the discarded messages is permanently lost.
- **Impact**: Context loss. The LLM loses track of earlier conversation context when the compaction API call fails. The user may notice the bot "forgetting" things that were discussed.
- **Suggested fix**: When the fallback triggers, at minimum log a more prominent warning. If there's no existing summary, consider keeping more messages (don't trim as aggressively). If there is an existing summary, it remains valid for the period it covers — the gap is between the summary and the kept messages.

#### Issue 7: `save()` writes ALL sessions to disk after every message
- **File**: `src/sessions/manager.py:383-387`, `src/discord/client.py:1156`
- **Severity**: medium
- **Category**: performance
- **Description**: After every successful message, `client.py:1156` calls `self.sessions.save()`:
  ```python
  def save(self) -> None:
      for cid, session in self._sessions.items():
          path = self.persist_dir / f"{cid}.json"
          data = asdict(session)
          path.write_text(json.dumps(data, indent=2))
  ```
  This writes **every active session** to disk, even though only the current channel's session changed. With 10 active channels, this is 10 synchronous file writes per message. The writes use `dataclasses.asdict()` (which deep-copies all data) followed by `json.dumps()` and `write_text()` — three operations per session.

  Additionally, sessions with long message histories produce large JSON files. A session with 40 messages (near the compaction threshold) with 500+ character messages generates ~20KB+ of JSON per file.
- **Impact**: Unnecessary disk I/O. With N active sessions, every message produces N file writes instead of 1. On slow storage (NFS, busy disk), this adds noticeable latency. The `asdict()` deep copy also creates temporary memory pressure.
- **Suggested fix**: Track which sessions are dirty and only save those. Or add a `save_session(channel_id)` method that writes only the specified session. Consider debouncing saves (e.g., save at most once every 5 seconds).

#### Issue 8: `get_prompt_section` reads learned.json from disk on every call
- **File**: `src/learning/reflector.py:67-91`, `src/discord/client.py:300,307,388,395`
- **Severity**: medium
- **Category**: performance
- **Description**: `ConversationReflector.get_prompt_section()` at line 74 calls `self._load()` which reads `learned.json` from disk every time. This method is called from:
  - `_build_system_prompt` (client.py:307) — for every task message
  - `_build_chat_system_prompt` (client.py:395) — for every chat message
  - Potentially both in one message lifecycle (classification → routing → prompt build)

  The file only changes when reflection completes (which happens asynchronously after sessions are archived/compacted). Between reflections, every `_load()` call reads the same unchanged file.
- **Impact**: 1-2 unnecessary synchronous disk reads per message. While individual reads are fast on local SSD, they block the event loop and add up under load. The file also gets parsed with `json.loads()` on every read.
- **Suggested fix**: Cache the parsed data in memory. Invalidate the cache when `_save()` is called (after reflection). The `_lock` in `_reflect()` already serializes writes, so a simple "load once, invalidate on save" pattern works.

#### Issue 9: `_find_recent_summary` does synchronous I/O on archive files
- **File**: `src/sessions/manager.py:75-93`
- **Severity**: medium
- **Category**: performance
- **Description**: `_find_recent_summary()` is called from `get_or_create()` (line 66) when a session doesn't exist yet. It:
  1. Globs the archive directory for `{channel_id}_*.json` files (line 83)
  2. Reads each matching file with `path.read_text()` (line 85) — synchronous I/O
  3. Parses JSON and checks timestamps (lines 86-90)

  For an active channel with many archives, this could read dozens of files. All of this is synchronous, blocking the event loop. The method is called from `get_or_create()` which is a sync method, so it cannot use `await`.

  `get_or_create()` is called from:
  - `add_message()` (line 99) — every message
  - `get_history()` / `get_history_with_compaction()` — every message
  - `search_history()` — on search tool use

  However, `_find_recent_summary` only runs when the session doesn't already exist (the first message in a new session), so the impact is limited to session creation.
- **Impact**: Event loop blocked during first message in a channel after bot restart. Duration depends on number of archive files for that channel.
- **Suggested fix**: Pre-compute summaries during `load()` at startup, or make `get_or_create` async and use `asyncio.to_thread()` for the file reads.

#### Issue 10: `search_history` reads archive files synchronously on event loop
- **File**: `src/sessions/manager.py:296-322`
- **Severity**: medium
- **Category**: performance
- **Description**: `search_history()` is an `async def` method, but the archive search at lines 296-322 performs synchronous operations:
  - `archive_dir.glob("*.json")` — synchronous directory scan
  - `path.read_text()` — synchronous file read, for every archive file
  - `json.loads()` — CPU-bound parsing

  All archive files are read and parsed until `limit` results are found. With a large archive directory (months of operation), this could read hundreds of files synchronously.
- **Impact**: Event loop blocked during `search_conversations` tool use. Long-running bot instances with many archived sessions will experience increasingly slow searches.
- **Suggested fix**: Run the archive scan in a thread via `asyncio.to_thread()`, or maintain a lightweight index (e.g., SQLite) of archive summaries that can be searched without reading all files.

#### Issue 11: Embedding vectors in tool_memory.json cause large file size
- **File**: `src/tools/tool_memory.py:121-141`
- **Severity**: medium
- **Category**: performance
- **Description**: `ToolMemory.record()` stores the full embedding vector (line 133: `entry["embedding"] = vector`) in the JSON file. The `nomic-embed-text` model produces 768-dimensional float vectors. Each float serialized in JSON is ~18 characters (e.g., `0.12345678901234`), so each embedding is ~14KB of JSON text.

  With `MAX_ENTRIES = 200` and embeddings enabled, the file grows to:
  - 200 × 14KB (embeddings) + metadata ≈ **2.8MB**

  This file is:
  - Read at startup in `__init__` (line 75)
  - Written after every successful tool loop (line 141, called from client.py:1161)
  - Read during `find_patterns` to compute cosine similarity (every message via `_inject_tool_hints`)

  Each `_save()` serializes the entire 2.8MB to JSON and writes it to disk synchronously.
- **Impact**: Increasingly slow file I/O as entries accumulate. Parsing a 2.8MB JSON file on every hint lookup adds latency. Synchronous writes of 2.8MB block the event loop for noticeable durations.
- **Suggested fix**: Store embeddings separately (e.g., in a binary file or SQLite database). Or don't persist embeddings and re-embed on load (embeddings are deterministic for the same model+text). Or use a more compact serialization (e.g., base64-encoded float32 array).

#### Issue 12: Archive directory grows without bound
- **File**: `src/sessions/manager.py:236-247`
- **Severity**: medium
- **Category**: performance
- **Description**: `_archive_session()` creates new archive files in `persist_dir/archive/` (line 244) but no code ever deletes old archives. Over weeks/months of operation, the archive directory grows without limit.

  The archive directory is scanned by:
  - `_find_recent_summary()` — on new session creation (glob + read all files)
  - `search_history()` — on `search_conversations` tool use (glob + read all files)
  - `backfill_archives()` — on startup (glob + read all files)

  With an active bot creating 5-10 sessions per day, after a year there would be 1,800-3,600 archive files. Each `search_history` call would scan all of them.
- **Impact**: Degrading performance over time. File system operations on directories with thousands of small files become slow. `search_history` and `_find_recent_summary` become increasingly expensive.
- **Suggested fix**: Add archive cleanup — delete archives older than a configurable retention period (e.g., 90 days). Run cleanup periodically or as part of `prune()`. Alternatively, consolidate old archives into a summary index.

#### Issue 13: Compaction holds channel lock during Haiku API call
- **File**: `src/sessions/manager.py:147-211`, `src/discord/client.py:917`
- **Severity**: medium
- **Category**: behavioral
- **Description**: `get_history_with_compaction()` is called inside `_handle_message_inner()` (client.py:917) which runs inside the per-channel lock. When compaction triggers, `_compact()` calls `asyncio.to_thread(claude_client.client.messages.create, ...)` at line 172. While `to_thread` doesn't block the event loop, the channel lock is held during the entire Haiku API call.

  The Haiku call has no explicit timeout — it uses the Anthropic SDK's default (which can be 10+ seconds). During this time:
  - Other messages for the **same channel** queue behind the lock
  - The compaction API call bypasses the circuit breaker (identified in Session 3, Issue 3)
  - If the API is slow or the network has issues, the channel is effectively frozen

  Users may send follow-up messages while waiting, which queue up and are processed after compaction completes, potentially causing a burst of responses.
- **Impact**: Channel message processing pauses during compaction. Users experience delayed responses. In the worst case (API timeout), the channel is blocked for the full timeout duration.
- **Suggested fix**: Release the channel lock before making the API call, and re-acquire after. Or run compaction as a background task and serve stale history in the meantime.

#### Issue 14: `COMPACTION_THRESHOLD` hardcoded vs configurable `max_history`
- **File**: `src/sessions/manager.py:21,150`
- **Severity**: low
- **Category**: bug
- **Description**: `COMPACTION_THRESHOLD = 40` is a module-level constant, while `self.max_history` is configurable via `config.sessions.max_history` (default 50). Compaction triggers when `len(session.messages) > 40` and keeps `keep_count = self.max_history // 2` messages.

  If a user configures `max_history = 20` (expecting short sessions), compaction still won't trigger until 40 messages — double their intended maximum. The `max_history` setting only takes effect in the fallback path (line 211) where messages are trimmed without a summary.

  Conversely, if `max_history = 100`, compaction triggers at 40 messages and keeps 50 — reasonable behavior, but the threshold could be higher.
- **Impact**: The `max_history` config option doesn't control when compaction happens. Users who set a low `max_history` expecting smaller sessions will see sessions grow to 40 messages before any summarization.
- **Suggested fix**: Derive the compaction threshold from `max_history` (e.g., `self.max_history * 0.8`), or make it configurable.

#### Issue 15: `_expire()` uses string comparison for ISO timestamps
- **File**: `src/tools/tool_memory.py:98-103`
- **Severity**: low
- **Category**: bug (latent)
- **Description**: `_expire()` compares ISO timestamp strings:
  ```python
  cutoff = (datetime.now(timezone.utc) - timedelta(days=EXPIRY_DAYS)).isoformat()
  self._entries = [e for e in self._entries if e.get("timestamp", "") >= cutoff]
  ```
  This works because `datetime.now(timezone.utc).isoformat()` produces `2026-03-19T12:34:56+00:00`, and lexicographic comparison of ISO 8601 strings with identical UTC offsets (`+00:00`) is equivalent to chronological comparison.

  However, if any entry has a `Z` suffix (the other valid UTC representation: `2026-03-19T12:34:56Z`), the comparison breaks: `"Z"` has ASCII value 90, while `"+"` has value 43, so `"...+00:00" < "...Z"` — an entry with `Z` suffix would always appear "newer" than the cutoff and never expire.

  Currently, all timestamps are generated internally with consistent `+00:00` format, so this is safe. But if `tool_memory.json` is ever manually edited, migrated, or populated from another source, the inconsistency could cause entries to persist forever.
- **Impact**: Latent — safe with current code, but fragile. Non-expiring entries would cause the file to hit MAX_ENTRIES sooner, evicting legitimate entries.
- **Suggested fix**: Parse timestamps to `datetime` objects for comparison instead of relying on string ordering: `datetime.fromisoformat(e["timestamp"]) >= cutoff_dt`.

#### Issue 16: `prune()` iterates all sessions after every message
- **File**: `src/sessions/manager.py:217-234`, `src/discord/client.py:1155`
- **Severity**: low
- **Category**: performance
- **Description**: `self.sessions.prune()` is called after every successful message (client.py:1155). The method iterates all active sessions to find expired ones:
  ```python
  expired = [cid for cid, s in self._sessions.items()
             if now - s.last_active > self.max_age_seconds]
  ```
  With `max_age_hours = 24` (default), sessions rarely expire during active use. Most `prune()` calls iterate all sessions and find nothing to prune. For a bot with 20 active channels, this is 20 dict lookups and timestamp comparisons per message — negligible individually, but unnecessary.

  Additionally, when pruning does trigger, `_archive_session()` performs synchronous file I/O (line 246: `path.write_text(...)`) inside the channel lock.
- **Impact**: Minimal per-call overhead, but conceptually wasteful. The archival file I/O during pruning adds latency to the triggering message.
- **Suggested fix**: Run pruning on a periodic timer (e.g., every 5 minutes) instead of after every message. Or track session expiry times in a sorted structure for O(1) "is anything expired?" checks.

#### Issue 17: Reflector `_load()` and `_save()` use synchronous file I/O under async lock
- **File**: `src/learning/reflector.py:55-65`
- **Severity**: low
- **Category**: best-practice
- **Description**: `_reflect()` (line 132) acquires `self._lock` (an `asyncio.Lock`), then calls `self._load()` (sync `path.read_text()` + `json.loads()`) and `self._save()` (sync `json.dumps()` + `path.write_text()`). While the lock prevents concurrent reflections from racing, the synchronous I/O blocks the event loop.

  `get_prompt_section()` (line 74) also calls `_load()` without the lock — it's a read-only operation, but it still blocks the event loop.

  The reflector runs in background tasks (manager.py:200,252) so the lock contention is between reflections, not with message processing. But the sync I/O affects the entire event loop.
- **Impact**: Brief event loop stalls during reflection save/load. Since reflections happen infrequently (on session archive/compaction), the impact is minor.
- **Suggested fix**: Use `asyncio.to_thread()` for the file I/O operations, or use `aiofiles`.

---

### Session 4 Summary

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 3 |
| Medium | 10 |
| Low | 4 |
| **Total** | **17** |

**Key takeaways**:

1. **User attribution is the biggest correctness issue** (#1, #2). The reflection system has two paths where per-user learned context gets misattributed: consolidation silently strips `user_id` during JSON parsing, and multi-user channels attribute all learned entries to `last_user_id` only. Over time, this causes cross-user preference leakage where one user's corrections/preferences are applied to another user's prompts. The data corruption is gradual and invisible.

2. **Synchronous I/O is pervasive** (#7, #8, #9, #10, #11, #17). Nearly every file operation in the state management layer uses synchronous `Path.read_text()` / `write_text()` calls from within async contexts. The most impactful are `save()` writing ALL sessions after every message, `get_prompt_section()` reading learned.json on every prompt build, and tool_memory.json growing to multi-megabyte sizes with embedded vectors. Each individual call is fast, but they compound under load.

3. **The compaction path has multiple issues** (#6, #13, #14). Compaction holds the channel lock during an unbounded API call (no timeout, no circuit breaker), the fallback path creates context gaps by trimming without updating the summary, and the trigger threshold is hardcoded independently of the configurable max history size.

4. **The learning system works correctly on the happy path** but accumulates subtle drift over time. The `_parse_entries` fence-stripping bug (#4) is masked by fallback JSON parsing. The `find_patterns` Jaccard skip (#5) degrades hint quality as more entries gain embeddings. The archive directory growth (#12) slowly degrades search performance. None of these cause immediate failures, but all worsen over weeks/months of operation.

---

## Session 5: `src/health/server.py` + `src/scheduler/scheduler.py` + `src/permissions/manager.py` + `src/discord/routing.py` — Webhooks, Scheduling, Permissions, Routing

| # | File:Line | Severity | Category | Description |
|---|-----------|----------|----------|-------------|
| 1 | client.py:2162-2167 | high | security | Scheduled workflows bypass ALLOWED_CHECK_TOOLS — can execute any tool without user interaction |
| 2 | server.py:198 | high | security | Generic webhook uses plain `!=` for secret comparison — timing attack vulnerability |
| 3 | server.py:153-158 | medium | security | Grafana webhook endpoint has no authentication at all |
| 4 | scheduler.py:102-105 | medium | bug | `run_at` is not validated as ISO datetime at creation — fails silently at runtime |
| 5 | scheduler.py:34-44 | medium | best-practice | `_load()` and `_save()` use synchronous file I/O, called from async context |
| 6 | scheduler.py:226-234 | medium | behavioral | Scheduler loop sleeps 60 seconds — one-time tasks can be delayed up to 60s past `run_at` |
| 7 | scheduler.py:167-198 | medium | bug | `fire_triggers` iterates and mutates `_schedules` — potential race with concurrent `_tick` |
| 8 | manager.py:60-62 | medium | best-practice | `_save_overrides()` uses synchronous file I/O, blocks event loop |
| 9 | routing.py:11-38 | medium | behavioral | `_TASK_KEYWORD_PATTERNS` matches common words like "news", "search", "audit" — false positive task classification |
| 10 | server.py:65-71 | low | security | `_verify_hmac_sha256` returns True when no secret is configured — all webhooks accepted |
| 11 | scheduler.py:82-83 | low | bug | Schedule ID `uuid.uuid4().hex[:8]` has no collision check |
| 12 | scheduler.py:70-75 | low | bug | Workflow steps validation doesn't check tool names against any allowlist |
| 13 | manager.py:78-90,92-102 | low | behavioral | `filter_tools` returns `None` for guest but `allowed_tool_names` returns empty set — inconsistent "no access" encoding |
| 14 | routing.py:67-69 | low | behavioral | `/opt/` regex matches any message mentioning `/opt/` paths, not just project-related ones |
| 15 | server.py:92-106 | low | best-practice | `_send` exposes raw exception message in HTTP response body |

---

**Details:**

#### Issue 1: Scheduled workflows bypass ALLOWED_CHECK_TOOLS — can execute any tool without user interaction
- **File**: `src/discord/client.py:2134-2176` (workflow execution), `src/scheduler/scheduler.py:70-75` (workflow validation)
- **Severity**: high
- **Category**: security
- **Description**: The scheduler validates that `"check"` actions can only use tools in `ALLOWED_CHECK_TOOLS` (6 read-only monitoring tools). However, `"workflow"` actions have **no tool restriction**. Workflow step validation (scheduler.py:73-75) only checks that each step is a dict with `tool_name` — it doesn't validate the tool names against any allowlist.

  At execution time, `_run_scheduled_workflow` (client.py:2162-2167) runs each step directly:
  ```python
  if self.skill_manager.has_skill(tool_name):
      output = await self.skill_manager.execute(tool_name, tool_input)
  else:
      output = await self.tool_executor.execute(tool_name, tool_input)
  ```

  This means a workflow can call `run_command`, `write_file`, `restart_service`, `git_push`, `docker_compose_action`, `incus_exec`, `incus_delete`, or any skill — all without approval at execution time. While `schedule_task` requires approval at creation (registry.py:364), once approved, the workflow runs indefinitely (for cron-based schedules) or on every matching webhook trigger.

  The registry description even acknowledges this: *"Workflows can chain any tool (including write tools)"* — but the risk model assumes the user carefully reviews the steps during approval. In practice, the LLM constructs the steps, and the user sees only a description and an "approve" button.
- **Impact**: A single approval for a scheduled workflow grants permanent unattended access to all destructive tools. A malicious prompt injection or LLM hallucination could construct a workflow with `run_command` steps that execute arbitrary commands on any host, running on a cron schedule or webhook trigger with no further human oversight.
- **Suggested fix**: Either (a) restrict workflow steps to `ALLOWED_CHECK_TOOLS` (same as `"check"` actions), or (b) create a separate `ALLOWED_WORKFLOW_TOOLS` allowlist that includes a broader but bounded set of safe tools, or (c) require re-approval for each workflow execution (impractical for recurring schedules), or (d) show the full step details (tool names + inputs) in the approval prompt rather than just the description.

#### Issue 2: Generic webhook uses plain `!=` for secret comparison — timing attack vulnerability
- **File**: `src/health/server.py:198`
- **Severity**: high
- **Category**: security
- **Description**: The generic webhook endpoint authenticates using plain string inequality:
  ```python
  if self._webhook_config.secret and secret_header != self._webhook_config.secret:
      return web.json_response({"error": "invalid secret"}, status=403)
  ```

  Python's `!=` operator for strings performs byte-by-byte comparison and short-circuits on the first differing byte. This leaks timing information: a request where the first character matches takes slightly longer than one where it doesn't.

  In contrast, the Gitea webhook (line 71) correctly uses `hmac.compare_digest()` for constant-time comparison. The same codebase applies two different security standards to the same type of secret validation.
- **Impact**: An attacker can perform a timing attack to brute-force the webhook secret character by character. While network jitter makes this harder in practice, it's a well-known attack vector for endpoints exposed to the internet. The attack complexity is O(n × charset) instead of O(charset^n), reducing a 32-character hex secret from 2^128 to ~512 attempts.
- **Suggested fix**: Replace `secret_header != self._webhook_config.secret` with `not hmac.compare_digest(secret_header, self._webhook_config.secret)`. This is a one-line fix and the `hmac` module is already imported.

#### Issue 3: Grafana webhook endpoint has no authentication
- **File**: `src/health/server.py:153-158`
- **Severity**: medium
- **Category**: security
- **Description**: The Grafana webhook handler (`_webhook_grafana`) accepts any POST request without any authentication:
  ```python
  async def _webhook_grafana(self, request: web.Request) -> web.Response:
      body = await request.read()
      try:
          data = json.loads(body)
      except json.JSONDecodeError:
          return web.json_response({"error": "invalid JSON"}, status=400)
  ```

  Compare this to:
  - **Gitea webhook** (line 108-112): Verifies HMAC-SHA256 signature via `X-Gitea-Signature` header
  - **Generic webhook** (line 197-199): Checks `X-Webhook-Secret` header against configured secret

  The Grafana endpoint has zero authentication. Any attacker who can reach the webhook port (3000) can send fake alerts that will be posted to the configured Discord channel and fire scheduled triggers.
- **Impact**: An attacker can inject fake Grafana alerts into Discord channels, potentially causing unnecessary incident response. More critically, they can fire webhook-triggered schedules — if any workflow-type triggers are configured for Grafana source, this could execute destructive tools (see Issue #1).
- **Suggested fix**: Add HMAC or shared-secret verification to the Grafana endpoint. Grafana supports webhook basic auth and custom headers. At minimum, check the `X-Webhook-Secret` header the same way the generic endpoint does.

#### Issue 4: `run_at` is not validated as ISO datetime at creation
- **File**: `src/scheduler/scheduler.py:102-105`
- **Severity**: medium
- **Category**: bug
- **Description**: When creating a one-time schedule, the `run_at` parameter is stored directly without validation:
  ```python
  else:
      schedule["run_at"] = run_at
      schedule["next_run"] = run_at
      schedule["one_time"] = True
  ```

  If the LLM passes an invalid string (e.g., `"tomorrow at 9am"` instead of converting to ISO format, or a malformed date like `"2026-13-45T99:99"`), the schedule is saved to disk without error. When `_tick()` fires and calls `datetime.fromisoformat(next_run_str)` at line 246, a `ValueError` is raised. This is caught by the outer `except Exception` at line 233, logged, and the scheduler continues — but the invalid schedule persists and throws the same error every 60 seconds forever.
- **Impact**: Invalid one-time schedules never fire and generate an error log every 60 seconds indefinitely. The user gets no feedback at creation time that their schedule is malformed. The error logs can grow significantly over time.
- **Suggested fix**: Validate `run_at` in `Scheduler.add()`:
  ```python
  if run_at:
      try:
          datetime.fromisoformat(run_at)
      except (ValueError, TypeError):
          raise ValueError(f"Invalid ISO datetime for run_at: {run_at!r}")
  ```

#### Issue 5: Scheduler `_load()` and `_save()` use synchronous file I/O
- **File**: `src/scheduler/scheduler.py:34-44`
- **Severity**: medium
- **Category**: best-practice
- **Description**: `_load()` uses `self.data_path.read_text()` and `_save()` uses `self.data_path.write_text()` — both synchronous. While `_load()` is called once at init (acceptable), `_save()` is called from multiple async contexts:
  - `add()` (line 116) — called from `_handle_schedule_task` during tool execution
  - `delete()` (line 207) — called from `_handle_delete_schedule` during tool execution
  - `fire_triggers()` (line 197) — called from webhook handler
  - `_tick()` (line 270) — called from scheduler loop

  All of these run on the asyncio event loop, blocking it during disk writes. The file contains serialized schedule data that grows with the number of active schedules.
- **Impact**: Brief event loop stalls on every schedule modification and every scheduler tick that fires tasks. Under normal operation with few schedules, the impact is minimal. But `_tick` writes on every cycle that fires any task, which could be every 60 seconds for active cron schedules.
- **Suggested fix**: Use `asyncio.to_thread()` for `_save()` calls, or buffer writes and flush periodically.

#### Issue 6: Scheduler loop sleeps 60 seconds — one-time tasks can be delayed up to 60s
- **File**: `src/scheduler/scheduler.py:226-234`
- **Severity**: medium
- **Category**: behavioral
- **Description**: The scheduler loop sleeps for a fixed 60 seconds between ticks:
  ```python
  async def _loop(self) -> None:
      while True:
          try:
              await asyncio.sleep(60)
              await self._tick()
  ```

  For cron-based schedules (minimum granularity 1 minute), this is appropriate. But for one-time `run_at` schedules (which have second-level precision in ISO format), the task may fire up to 60 seconds after the specified time. A reminder set for "2026-03-20T09:00:00" could execute anywhere between 09:00:00 and 09:01:00.

  Additionally, the sleep happens *before* the first tick, so a newly added schedule that is already due won't fire until the current sleep completes — potentially up to 60 seconds after creation.
- **Impact**: One-time reminders and scheduled tasks fire with up to 60 seconds of delay. For reminders ("remind me in 5 minutes"), 60 seconds of jitter is noticeable. For monitoring checks, this is less critical.
- **Suggested fix**: Calculate sleep duration until the next scheduled event rather than using a fixed 60-second interval. At minimum, run `_tick()` immediately when a new schedule is added (using an `asyncio.Event` to wake the loop).

#### Issue 7: `fire_triggers` and `_tick` can mutate `_schedules` concurrently
- **File**: `src/scheduler/scheduler.py:167-198` (fire_triggers), `236-270` (_tick)
- **Severity**: medium
- **Category**: bug (race condition)
- **Description**: `fire_triggers()` is an async method called from the webhook handler (HealthServer → callback → `scheduler.fire_triggers()`). `_tick()` is called from the scheduler's own loop. Both methods:
  1. Iterate over `self._schedules`
  2. Mutate schedule entries (setting `last_run`)
  3. Call `self._save()`

  While asyncio is single-threaded, both methods contain `await` points (`await self._callback(schedule)` in both). During these await points, the event loop can process other tasks — including a webhook arriving and calling `fire_triggers` while `_tick` is mid-iteration.

  Scenario:
  1. `_tick()` iterates schedules, fires schedule A, `await self._callback(schedule_A)` yields
  2. Webhook arrives, `fire_triggers()` starts iterating the same `_schedules` list
  3. `fire_triggers` modifies `schedule.last_run` on a schedule that `_tick` hasn't processed yet
  4. `fire_triggers` calls `self._save()`, writing the modified list to disk
  5. `_tick` resumes, modifies more schedules, calls `self._save()` again — potentially overwriting `fire_triggers`' changes

  The list itself is not modified during iteration (no adds/removes during the loop body), so there's no iteration error. But the concurrent `last_run` mutations and duplicate `_save()` calls create a race on the persisted state.
- **Impact**: `last_run` timestamps could be inconsistent or lost. If `_tick` removes one-time schedules (line 266-267) between `fire_triggers`' iteration start and its `_save()`, the one-time schedule could be re-saved by `fire_triggers`.
- **Suggested fix**: Add an `asyncio.Lock` around `_tick` and `fire_triggers` to ensure mutual exclusion. Or consolidate state mutations behind a single write path.

#### Issue 8: `PermissionManager._save_overrides()` uses synchronous file I/O
- **File**: `src/permissions/manager.py:60-62`
- **Severity**: medium
- **Category**: best-practice
- **Description**: `_save_overrides()` writes the permissions JSON file synchronously:
  ```python
  def _save_overrides(self) -> None:
      self._overrides_path.parent.mkdir(parents=True, exist_ok=True)
      self._overrides_path.write_text(json.dumps(self._overrides, indent=2))
  ```
  This is called from `set_tier()` (line 75), which is called from `_handle_set_permission()` in the Discord client — running on the asyncio event loop. The `mkdir()` call is also synchronous.

  Additionally, `_load_overrides()` at init (line 52) uses synchronous `read_text()`, but this is acceptable since init runs before the event loop starts.
- **Impact**: Brief event loop stall when changing a user's permission tier. Infrequent operation (admin-only), so impact is minimal in practice.
- **Suggested fix**: Use `asyncio.to_thread()` for the write, or accept the blocking call given its rarity.

#### Issue 9: `_TASK_KEYWORD_PATTERNS` matches common words — false positive task classification
- **File**: `src/discord/routing.py:11-38`
- **Severity**: medium
- **Category**: behavioral
- **Description**: Several patterns in `_TASK_KEYWORD_PATTERNS` match common English words that could appear in casual conversation:
  - `\bnews\b` — "Have you heard the news about the game?" → classified as task
  - `\bsearch\b` — "I've been searching for a good recipe" → classified as task
  - `\baudit\b` — "We need to audit our spending" → classified as task
  - `\bdigest\b` — "I can't digest all this information" → classified as task
  - `\bremember\b` — "Remember when we went there?" → classified as task
  - `\brecall\b` — "I recall that being different" → classified as task
  - `\bpurge\b` — "I need to purge my closet" → classified as task
  - `\bretry\b` — "Let me retry that approach" → classified as task
  - `\bproceed\b` — "How should we proceed?" → classified as task

  The comment at lines 7-9 explicitly acknowledges this trade-off: *"Single ambiguous words (check, log, memory, service, find, run, status) are omitted — the Haiku classifier handles those for ~$0.0001."* However, the included words above are equally ambiguous in casual context.

  When `is_task_by_keyword` returns True (line 943-945 of client.py), the message bypasses the Haiku classifier entirely and is routed as a task. This gives the message tools and uses the more expensive Sonnet model, even for casual conversation that happens to contain these words.
- **Impact**: Casual messages containing common words are over-classified as tasks, unnecessarily activating tools and using more expensive Sonnet model. The cost impact is ~$0.01-0.03 per false positive (Sonnet call with tools vs Haiku chat). For a personal bot with limited users, this is a minor cost increase; for casual conversation channels, it could be more noticeable.
- **Suggested fix**: Move single-word ambiguous terms (`news`, `search`, `audit`, `digest`, `remember`, `recall`, `purge`, `retry`, `proceed`) to the Haiku classifier. Only keep multi-word phrases and truly infrastructure-specific terms in the keyword bypass. The ~$0.0001 per Haiku classification is much cheaper than a false-positive Sonnet call.

#### Issue 10: `_verify_hmac_sha256` returns True when no secret is configured
- **File**: `src/health/server.py:65-71`
- **Severity**: low
- **Category**: security
- **Description**: When no webhook secret is configured (`self._webhook_config.secret` is empty string):
  ```python
  if not secret:
      return True  # No secret configured, skip verification
  ```
  This means all Gitea webhooks are accepted without any authentication. While this is intentional (to allow easy setup without a secret), it creates a risk if the operator forgets to configure a secret. There's no warning logged about the missing secret.
- **Impact**: If the webhook port is exposed without a secret configured, any attacker can send fake Gitea events. This is a configuration issue rather than a code bug, but the silent acceptance makes it easy to overlook.
- **Suggested fix**: Log a warning at startup when webhook endpoints are enabled without a secret: `log.warning("Webhook secret not configured — endpoints accept unauthenticated requests")`.

#### Issue 11: Schedule ID has no collision check
- **File**: `src/scheduler/scheduler.py:82-83`
- **Severity**: low
- **Category**: bug (latent)
- **Description**: Schedule IDs are generated with `uuid.uuid4().hex[:8]` — 8 hex characters = 32 bits of entropy. No uniqueness check is performed against existing schedules. While the probability of collision is extremely low for a personal bot (birthday paradox: ~0.1% chance with 1000 schedules), duplicate IDs would cause `delete()` to remove the wrong schedule, since it deletes the first match.
- **Impact**: Negligible in practice. A collision would require ~65,000 active schedules for a 50% chance. But the lack of uniqueness check is a correctness gap.
- **Suggested fix**: Add a simple uniqueness check in `add()`:
  ```python
  existing_ids = {s["id"] for s in self._schedules}
  while schedule_id in existing_ids:
      schedule_id = uuid.uuid4().hex[:8]
  ```

#### Issue 12: Workflow steps validation doesn't check tool names
- **File**: `src/scheduler/scheduler.py:70-75`
- **Severity**: low
- **Category**: bug
- **Description**: Workflow step validation only checks structural correctness:
  ```python
  for i, step in enumerate(steps):
      if not isinstance(step, dict) or "tool_name" not in step:
          raise ValueError(f"Step {i}: must be a dict with 'tool_name'")
  ```
  It does not verify that `tool_name` references an actual tool. An invalid tool name (typo like `"chekc_disk"`) is accepted at creation but fails at runtime with an "Unknown tool" error in `tool_executor.execute()`. The user doesn't find out until the workflow fires — potentially hours or days later for cron schedules.

  For `"check"` actions, the tool name IS validated against `ALLOWED_CHECK_TOOLS`. This inconsistency means workflows have weaker validation than checks.
- **Impact**: Workflows with misspelled tool names fail silently at runtime. The user gets an error posted to Discord when it fires, but may not see it. Combined with the 60-second scheduler resolution, debugging becomes difficult.
- **Suggested fix**: Validate workflow tool names at creation time against the tool registry and skill manager. At minimum, log a warning for unrecognized tool names.

#### Issue 13: `filter_tools` and `allowed_tool_names` encode "no access" differently
- **File**: `src/permissions/manager.py:78-90,92-102`
- **Severity**: low
- **Category**: behavioral
- **Description**: Two methods encode "guest has no tool access" differently:
  - `filter_tools()` returns `None` for guest (line 88)
  - `allowed_tool_names()` returns empty `set()` for guest (line 101)

  And "admin has all tools" differently:
  - `filter_tools()` returns the full `tools` list for admin (line 86)
  - `allowed_tool_names()` returns `None` for admin (line 99)

  The callers handle these correctly:
  - `filter_tools` is checked at client.py:1221-1223 with `if tools is not None and not tools`
  - `allowed_tool_names` is checked at client.py:354 and passed to `format_hints`

  But the inconsistent encoding (None means "no access" in one, "all access" in the other) is confusing and error-prone for any new caller.
- **Impact**: No runtime impact — existing callers handle both correctly. But a future caller could easily misinterpret the return values.
- **Suggested fix**: Document the return value semantics clearly in docstrings, or use a more explicit encoding (e.g., a `ToolAccess` enum or named tuple).

#### Issue 14: `/opt/` regex overly broad
- **File**: `src/discord/routing.py:67-69`
- **Severity**: low
- **Category**: behavioral
- **Description**: The `_SERVER_INDICATORS` regex includes `r"/opt/"` as a server indicator (line 69). This matches any message containing the string `/opt/` anywhere — not just `/opt/project`. A message like "install it to /opt/myapp" or "the config is in /opt/grafana" would route to server even if the user is discussing a different machine or a general concept.

  The pattern `/opt/project` (line 68) is already included separately, so `/opt/` is a broader fallback. Since only the server and desktop hosts have Claude CLI installed, and `/opt/` is more commonly associated with server paths, this is a reasonable heuristic. But it could surprise users who mention `/opt/` in other contexts.
- **Impact**: Minor — messages mentioning any `/opt/` path route Claude Code to the server host. For a personal bot with known infrastructure, this is usually correct but could occasionally route to the wrong host.
- **Suggested fix**: Consider removing the broad `/opt/` pattern and keeping only `/opt/project`. Or add a comment explaining the heuristic.

#### Issue 15: `_send` exposes raw exception message in HTTP response body
- **File**: `src/health/server.py:104-106`
- **Severity**: low
- **Category**: best-practice
- **Description**: When Discord message delivery fails:
  ```python
  except Exception as e:
      log.error("Webhook %s delivery failed: %s", source, e)
      return web.json_response({"error": str(e)}, status=500)
  ```
  The raw exception message is returned in the HTTP response. This could leak internal details (Discord API error messages, network error details, stack trace fragments) to the webhook sender.
- **Impact**: Minor information leakage. For a personal home network bot, the webhook sender is typically a trusted service (Gitea, Grafana). But if the webhook port is exposed externally, this could reveal internal details.
- **Suggested fix**: Return a generic error message in the response: `{"error": "delivery failed"}`. Keep the detailed error in the log.

---

### Session 5 Summary

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 2 |
| Medium | 7 |
| Low | 6 |
| **Total** | **15** |

**Key takeaways**:

1. **Scheduled workflow security** (#1) is the most impactful issue. The scheduler correctly restricts `"check"` actions to read-only monitoring tools, but `"workflow"` actions can execute any tool — including `run_command`, `write_file`, and `restart_service` — without any per-execution approval. A single approval at schedule creation grants permanent unattended access to all destructive tools. Combined with the unauthenticated Grafana webhook (#3), an attacker could potentially trigger destructive workflows by sending fake Grafana alerts.

2. **Webhook authentication is inconsistent** (#2, #3, #10). Gitea webhooks are properly HMAC-verified with constant-time comparison. Generic webhooks use timing-vulnerable `!=` comparison. Grafana webhooks have no authentication at all. This creates a security gradient where the weakest link (Grafana) undermines the protection of the strongest (Gitea), since all three endpoints can fire the same scheduled triggers.

3. **The modules are generally well-structured**. The permission system is clean and correctly applied — admin-only tools are properly restricted, guest users are blocked from tools, and user-tier users get a reasonable set of read-only tools. The routing module's keyword bypass is a sensible optimization that saves Haiku classification costs for obvious task messages. The health server properly separates webhook parsing from delivery.

4. **Operational issues are minor but worth noting**. The 60-second scheduler resolution (#6), synchronous file I/O (#5, #8), and missing `run_at` validation (#4) are quality-of-life issues that affect reliability over time but don't cause data loss or security breaches.

---

## Audit Round Summary (All 5 Sessions)

| Session | Focus Area | Critical | High | Medium | Low | Total |
|---------|-----------|----------|------|--------|-----|-------|
| 1 | discord/client.py | 2 | 2 | 6 | 5 | 15 |
| 2 | tools/executor, skill_manager, skill_context | 2 | 3 | 7 | 4 | 16 |
| 3 | llm/anthropic, circuit_breaker, openai_codex | 0 | 1 | 7 | 5 | 13 |
| 4 | sessions/manager, learning/reflector, tool_memory | 0 | 3 | 10 | 4 | 17 |
| 5 | health/server, scheduler, permissions, routing | 0 | 2 | 7 | 6 | 15 |
| **Total** | | **4** | **11** | **37** | **24** | **76** |

### Top Priority Issues (Recommended Fix Order)

**Critical (fix immediately):**
1. **Session 1, #1-2**: Shared mutable state (`_system_prompt`, `_current_user_id`) causes cross-channel/cross-user data leakage
2. **Session 2, #1**: SkillContext memory operations corrupt executor's scoped memory structure
3. **Session 2, #2**: `execute_tool()` lets skills bypass approval for any tool

**High (fix soon):**
4. **Session 2, #3**: `_handle_incus_exec` host shell escape via unquoted command
5. **Session 5, #1**: Scheduled workflows can execute any tool without per-execution approval
6. **Session 5, #2**: Generic webhook timing attack vulnerability
7. **Session 1, #3**: `_pending_files` shared list leaks files between concurrent channels
8. **Session 4, #1-2**: Reflection system loses/misattributes per-user learned context
9. **Session 3, #1**: Circuit breaker opens too aggressively (one request = trip)

**Systemic patterns across the codebase:**
- **Shared mutable state**: Instance-level attributes used for per-request data (Sessions 1, 2)
- **Synchronous file I/O in async context**: Pervasive across all modules (Sessions 2, 3, 4, 5)
- **Inconsistent security boundaries**: Some paths verify/approve, parallel paths don't (Sessions 2, 3, 5)
- **Silent failure modes**: Errors swallowed, "(no response)" returned as success, invalid data stored (Sessions 3, 4, 5)
