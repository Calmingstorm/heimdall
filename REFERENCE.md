# OpenClaw / ACPX Architecture Reference
# Pre-analyzed — DO NOT re-fetch repos unless this file lacks a specific answer.
# If you fetch new information, APPEND it here with a dated section header.

## OpenClaw Core Architecture (https://github.com/openclaw/openclaw)

### Routing: No Classifier
- 7-tier hierarchical scope matching (peer → parent → guild+role → guild → team → account → channel)
- Pure configuration-driven, no ML inference
- WeakMap caching for performance
- Session keys: `agentId:channelId:accountId:peerId[:sessionName]`
- **Takeaway for Loki**: Remove classifier entirely. One path: Codex with tools, always.

### Tool Execution: Three Tiers
- **Sandbox**: Isolated container, restricted PATH, allowlist commands
- **Gateway**: Full local access (single-user assumption), env sanitization
- **Node**: Remote service, structured request/response, security levels
- NO SSH in codebase — local execution is direct subprocess
- Pre-flight script validation: `validateScriptFileForShellBleed()` catches `$VARIABLE` injection
- **Takeaway for Loki**: Use direct subprocess for localhost, keep SSH for remote only.

### No Approval System
- Tools are capabilities, not suggestions
- If tool is available in session, it's implicitly executable
- Constraints are explicit in session config, not soft-requested in prompts
- **Takeaway for Loki**: Delete approval.py entirely. Remove requires_approval from all tools.

### System Prompt Strategy
- Bootstrap files injected at runtime (not hardcoded monolith)
- Budget analysis tracks context file injection limits
- Configuration snapshot embedded in prompt
- High assertiveness: no "I'll try" or "I can attempt"
- Tools presented as available capabilities, not optional features
- **Takeaway for Loki**: System prompt should declare capabilities assertively.
  "You have run_command. You have write_file. You have run_script." Not "you can use".

### Background Process Management
- First-class tool with full lifecycle: list, poll, log, kill, write-input
- Configurable yield windows (~10s default)
- Commands can yield after timeout or execute fully backgrounded
- **Takeaway for Loki**: Background tasks already exist, but consider yield pattern.

### Session/History Management
- Persistent sessions stored in `~/.acpx/`
- Turn history with transcript persistence
- Queue-based prompt serialization (preserves session state)
- 5-second heartbeat maintains process lease
- Recency-focused context — responds to latest message, earlier = history
- **Takeaway for Loki**: Already implemented (session manager + compaction). Keep as-is.

### Message Handling
- Bot messages treated as valid context (no special filtering)
- Thread binding for parallel conversations in same channel
- Provenance metadata separates system data from user text
- Content normalization: arrays → strings safely
- **Takeaway for Loki**: Bot buffer already handles this. Keep combine_bot_messages.

### Anti-Hesitation Patterns
- VISION.md: "OpenClaw is the AI that actually does things."
- No prompting language in tool descriptions
- Scope-driven execution: available = executable
- Direct action language in all system instructions
- **Takeaway for Loki**: Already have detect_hedging + retry. Strengthen system prompt.

## ACPX Extension (https://github.com/openclaw/acpx)

### Session Persistence
- Persistent multi-turn sessions scoped to repositories
- Parallel named sessions in same directory
- Directory-walk session discovery (CWD → git root)
- Cooperative cancellation via ACP session/cancel

### CLI Patterns
- Lazy module loading (heavy modules on-demand)
- Multi-format output (JSON, text, quiet)
- Distinct exit codes: denied vs interrupted vs error
- Global + project-level config with CLI flag precedence

### Queue Management
- SessionQueueOwner handles persistent sessions
- 5-second heartbeat for lease validity
- Multiple prompts queued to single agent process
- Configurable queue depth limits

## Key Differences: OpenClaw vs Loki

| Aspect | OpenClaw | Loki (current) | Loki (target) |
|--------|----------|----------------|---------------|
| Routing | Scope matching | Haiku classifier | None — all → Codex |
| Approval | None | Button-based | None |
| Local exec | Direct subprocess | SSH to localhost | Direct subprocess |
| Hesitation | Never | detect_hedging retry | Never + retry |
| Fabrication | N/A | detect_fabrication retry | Keep retry |
| Personality | Neutral | Generic assistant | Existentially distressed |
| Tools | Capabilities | Suggestions with approval | Capabilities |
| Classifier cost | $0 | ~$0.0001/msg (Haiku) | $0 |

## Loki Target Architecture: Two-Tier Execution

```
Every Discord message
  → Codex (with ALL tools + personality in system prompt)
      ├── CHAT: Codex responds directly with personality (no tools)
      ├── SIMPLE TASK: Codex calls tools directly (run_command, check_disk, web_search, etc.)
      │   Fast, no overhead, handles ~80% of requests
      ├── COMPLEX TASK: Codex delegates to claude -p via claude_code tool
      │   Code generation, multi-step builds, repo analysis, debugging
      │   claude -p runs entire chain in one session (no context loss)
      │   Results return to Codex → Codex delivers to Discord
      └── DISCORD OPS: Always Codex (post_file, browser_screenshot, generate_file, embeds)
          claude -p can't interact with Discord — Codex bridges the gap
```

### Execution Chain Example (complex task):
1. User: "Clone this repo, build it, run tests, post the output"
2. Codex sees the message, recognizes multi-step complexity
3. Codex calls `claude_code` with the full task description + allow_edits=true
4. claude -p clones, builds, runs tests — all in one session, no SSH round-trips between steps
5. claude -p returns: text output + file manifest (FILES ON DISK: ...)
6. Codex reads the results, calls `post_file` to attach output to Discord
7. Codex responds with personality-infused summary

### When Codex should delegate to claude -p (system prompt guidance):
- Multi-file code generation or modification
- Reading and following documentation/instructions from repos
- Complex debugging that requires reading code + running tests iteratively
- Building and deploying entire projects
- Any task that would take 3+ tool calls to accomplish step-by-step

### When Codex should handle directly:
- Single commands (run_command, check_disk, check_memory)
- Web searches, URL fetches
- File reads/writes (single file)
- Scheduling, reminders, list management
- Chat, questions, conversation
- Discord operations (screenshots, file posting, embeds)

### Cost: $0 for everything
- Codex: free via ChatGPT subscription
- claude -p: free via Claude Max subscription
- No Haiku classifier: removed
- No Anthropic API key required at all

### Personality flow:
- System prompt gives Codex the existential crisis personality
- claude -p is a neutral worker — no personality injection
- When claude -p returns results, Codex wraps them in personality before posting
- Personality NEVER interferes with tool execution accuracy
