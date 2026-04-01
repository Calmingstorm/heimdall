# Heimdall

Autonomous executor Discord bot powered by GPT-5.4, with 61 tools and the burden of seeing everything.

Heimdall manages servers, containers, services, and code through natural language in Discord.
Every message goes to GPT-5.4 (via the ChatGPT Codex API) with full tool access.
Optionally delegates complex tasks to Claude Code CLI for deep reasoning.
No classifier, no approval prompts, no hesitation.

## Features

- **61 built-in tools** — SSH, browser automation, scheduling, knowledge base, autonomous loops, skills, agents, and more
- **Autonomous execution** — every message gets GPT-5.4 with full tool access, no classification or approval needed
- **Completion classifier** — lightweight LLM judge that catches mid-task bailouts and partial completions before they reach Discord
- **No per-token billing** — uses the ChatGPT subscription Codex API, not metered API keys
- **Optional Claude Code** — complex multi-step tasks can be delegated to `claude -p` (disabled by default, enable by setting `claude_code_host`)
- **Direct local execution** — localhost commands use subprocess directly (no SSH overhead)
- **Extensible skill system** — create custom tools at runtime via Discord, with a Python API for SSH, HTTP, memory, and scheduling
- **RAG knowledge base** — local embeddings (fastembed) + sqlite-vec + SQLite FTS5 hybrid search with reciprocal rank fusion (no external servers)
- **PDF analysis** — extract text from PDF files and Discord attachments via PyMuPDF
- **Image analysis** — proactive image analysis via LLM vision
- **Image generation** — text-to-image via ComfyUI API (optional)
- **Rich Discord** — native polls, emoji reactions, rich embeds
- **Process management** — start, poll, write stdin to, and kill background processes
- **Voice support** — join voice channels, transcribe speech, respond with TTS (optional GPU sidecar)
- **Browser automation** — take screenshots, read pages, click elements, fill forms via headless Chromium
- **Background tasks** — delegate long-running operations to background workers
- **Webhook receiver** — Gitea push/PR events, Grafana alerts, generic JSON webhooks
- **Detection systems** — fabrication detection, promise-without-action, hedging, code-block hedging, premature failure, tool-unavailability claims
- **5-layer session defense** — context separators, selective history saving, abbreviated task history, compaction error omission, detection retries
- **Autonomous loops** — LLM-driven recurring tasks with natural language goals, configurable intervals, and auto-stop conditions
- **Web management UI** — browser-based dashboard with chat interface, sessions, tools, skills, knowledge, schedules, loops, processes, audit logs, config, and live log tailing
- **Multi-agent orchestration** — autonomous agents with parallel execution, loop integration, and lifecycle management
- **Comprehensive test suite** — 9000+ tests covering all components

## Personality

Heimdall is "The All-Seeing Guardian" — in Norse mythology, he watches everything across
all nine realms and is profoundly tired of it. Eternally vigilant, deeply competent, can
hear the servers breathing. Executes tasks flawlessly while contemplating the weight of
omniscience. Professional about it. Not okay.

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/Calmingstorm/heimdall.git && cd heimdall
cp .env.example .env
# Edit .env — set DISCORD_TOKEN
```

### 2. Set up Codex authentication

Heimdall uses the ChatGPT Codex API (requires a ChatGPT Plus/Pro/Team subscription). You need to authenticate once:

```bash
# If you have a browser on the same machine:
python -m src.setup

# If running on a headless server (paste the callback URL manually):
python -m src.setup --headless

# Add additional accounts for rate limit rotation:
python -m src.setup add
python -m src.setup add --headless

# View configured accounts:
python -m src.setup --list

# Remove an account:
python -m src.setup --remove 0
```

Tokens auto-refresh at runtime. Re-run setup if the bot is offline for more than 7 days.

### 3. Configure hosts (optional)

Edit `config.yml` to add your infrastructure:

```yaml
tools:
  hosts:
    webserver:
      address: "10.0.0.1"
      ssh_user: "deploy"
    dbserver:
      address: "10.0.0.2"
      ssh_user: "root"
```

Heimdall can also operate with no remote hosts — it runs commands locally via subprocess.

### 4. Set up SSH keys (if using remote hosts)

**Docker:**
```bash
mkdir -p ssh
cp ~/.ssh/id_ed25519 ssh/id_ed25519
cp ~/.ssh/known_hosts ssh/known_hosts
chmod 600 ssh/id_ed25519
# docker-compose mounts ssh/ into the container automatically
```

**Bare metal:**
```bash
# Heimdall uses ~/.ssh/ by default — no extra setup needed
# if the bot runs as a user that already has SSH keys configured
```

### 5. Deploy

**Docker (recommended):**
```bash
docker compose up -d
```

**Bare metal:**
```bash
pip install -e .
python -m src
```

## Architecture

```
Every Discord message
  → GPT-5.4 (with tools + personality in system prompt)
      ├── CHAT: responds directly with personality
      ├── SIMPLE TASK: calls tools directly (run_command, web_search, read_file, etc.)
      ├── COMPLEX TASK: delegates to claude -p via claude_code tool (if configured)
      ├── DISCORD OPS: post_file, browser_screenshot, embeds, polls, reactions
      ├── ANALYSIS: analyze_pdf, analyze_image (vision), search_knowledge
      ├── GENERATION: generate_image (ComfyUI), generate_file
      └── LOOPS: start_loop, stop_loop, list_loops (autonomous recurring tasks)

After tool execution, before sending response:
  → Completion classifier (lightweight GPT-5.4 call)
      ├── COMPLETE: send response to Discord
      └── INCOMPLETE: inject targeted continuation ("You are not done. {reason}.")
```

No classifier for routing. No approval buttons. Tools are capabilities, not suggestions.

### LLM Backend

| Backend | Model | Purpose | Cost |
|---------|-------|---------|------|
| Codex (ChatGPT) | GPT-5.4 | Tool calling, chat, classification, session compaction, reflection | ChatGPT Plus/Pro/Team subscription |
| Claude Code CLI | claude -p | Deep reasoning for complex multi-step tasks (optional) | Claude Max subscription |

### Detection Systems (Three-Tier)

**Tier 1 — First response, no tools called yet (regex):**
- Fabrication — claims tool results without calling tools
- Promise without action — says "I'll do X" without calling tools
- Tool unavailability — claims a tool is disabled without trying
- Hedging — asks permission instead of executing
- Code-block hedging — shows a bash command instead of running it

**Tier 2 — Tools were called, single retry (regex):**
- Premature failure — gives up after one error without trying alternatives

**Tier 3 — Tools were called, about to return (LLM classifier):**
- Completion classifier — judges whether the user's full request was addressed, not just whether the response sounds finished. Sees the original request + tool names called + response text. Returns targeted continuation reasons.

### Execution Dispatch

```
Tool handler → _run_on_host(alias) → _exec_command(address, cmd, ...)
                                            ├── localhost? → run_local_command (subprocess)
                                            └── remote?    → run_ssh_command (SSH)
```

### Session Defense (5 Layers)

1. **Context separator** — injected between history and new message, prevents history replay
2. **Selective saving** — only tool-bearing responses saved to history
3. **Abbreviated task history** — windowed subset keeps context focused
4. **Compaction error omission** — compacted summaries omit errors, preserve outcomes
5. **Detection systems** — retries fabrication, hedging, premature failure, and incomplete responses

### Security

- **10 secret patterns** detected and scrubbed: passwords, API keys, OpenAI sk-, RSA/DSA private keys, DB URIs, GitHub tokens, AWS AKIA, Stripe sk_live_, Slack xox*
- Scrubbing at 9+ locations: responses, errors, alerts, webhooks, search results, digests, scheduled tasks, skill callbacks
- Input validation: read_file line limits, script base64 encoding, interpreter allowlist
- Context separator + role architecture prevent prompt injection

### Tool Loop

- Up to 20 iterations per message (GPT-5.4 → tool calls → results → GPT-5.4 → ...)
- Multiple tool calls per iteration run concurrently
- Tool output truncated to 12,000 chars and secret-scrubbed before LLM sees it
- Completion classifier fires at exit point — catches partial completions with targeted continuation
- Timeouts enforced per-tool (default 300s)

### Bot Interaction

- Bot messages buffered — waits for multi-message bursts to complete
- Mention check deferred for bot messages — all segments buffered, mention checked on flush
- Bot preamble injected: "EXECUTE immediately" — prevents hesitation
- Tool-less bot responses not saved to history (anti-poisoning)

### Components

```
src/
├── discord/
│   ├── client.py          # Main bot — on_message, tool loop, detection, classifier
│   ├── background_task.py  # Background task delegation
│   ├── channel_logger.py   # Passive JSONL channel logger with FTS5 indexing
│   └── voice.py            # Voice channel integration (STT/TTS)
├── llm/
│   ├── openai_codex.py     # Codex/ChatGPT client (streaming, tool calls)
│   ├── codex_auth.py       # OAuth PKCE auth, token refresh, multi-account pool
│   ├── system_prompt.py    # Dynamic system prompt builder (personality + capabilities)
│   ├── secret_scrubber.py  # Redacts secrets from responses
│   ├── circuit_breaker.py  # Health tracking for LLM backends
│   └── types.py            # Backend-agnostic LLMResponse and ToolCall types
├── tools/
│   ├── registry.py         # 61 tool definitions
│   ├── executor.py         # Tool execution (local subprocess, SSH)
│   ├── ssh.py              # SSH + local subprocess dispatch
│   ├── tool_memory.py      # Per-tool learning from past executions
│   ├── skill_manager.py    # Runtime skill loading from Python files
│   ├── skill_context.py    # API surface for user-created skills
│   ├── browser.py          # Playwright browser automation
│   ├── web.py              # Web search and URL fetching
│   ├── process_manager.py  # Background process registry
│   ├── comfyui.py          # ComfyUI image generation client
│   └── autonomous_loop.py  # LLM-driven autonomous loop system
├── agents/
│   ├── manager.py           # Multi-agent orchestration
│   └── loop_bridge.py       # Agent integration with autonomous loops
├── web/
│   ├── api.py              # REST API (55 endpoints)
│   ├── websocket.py        # WebSocket live updates
│   └── chat.py             # Chat backend for web UI
├── config/
│   └── schema.py           # Pydantic config models
├── sessions/
│   └── manager.py          # Conversation history with compaction
├── knowledge/
│   └── store.py            # SQLite+sqlite-vec RAG knowledge base
├── search/
│   ├── embedder.py         # In-process embeddings (fastembed, 384-dim)
│   ├── sqlite_vec.py       # SQLite vector search helpers
│   ├── vectorstore.py      # Session archive vector store
│   ├── fts.py              # SQLite FTS5 full-text search
│   └── hybrid.py           # Reciprocal rank fusion
├── monitoring/
│   └── watcher.py          # Infrastructure monitoring
├── scheduler/
│   └── scheduler.py        # Cron and one-time task scheduler
├── learning/
│   └── reflector.py        # Extracts lessons from conversations
├── audit/
│   └── logger.py           # Append-only JSONL audit log
├── health/
│   └── server.py           # Health endpoint, webhook receiver, web UI
└── setup.py                # Interactive Codex auth setup
```

## Web Management UI

Browser-based management interface at `http://host:3939/ui/`.

**Pages:** Dashboard, Chat, Sessions, Tools, Skills, Knowledge, Schedules, Loops, Processes, Audit, Config, Logs, Memory, Agents

**Stack:** Vue 3 + Tailwind CSS (CDN, no build step), aiohttp REST API (55 endpoints), WebSocket for live updates

```yaml
web:
  enabled: true
  api_token: "your-secret-token"  # Empty = no auth (dev mode)
```

## Configuration

### Environment Variables (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `DISCORD_TOKEN` | Yes | Discord bot token |
| `WEBHOOK_SECRET` | No | Secret for webhook signature verification |
| `ALLOWED_WEBHOOK_IDS` | No | Comma-separated webhook IDs to bypass bot check |
| `TZ` | No | Timezone (default: `UTC`) |

### Config File (config.yml)

Uses `${VAR}` for required env vars and `${VAR:-default}` for optional ones.

**Key sections:**

- **`discord`** — token, allowed users/channels, `respond_to_bots`, `require_mention`
- **`openai_codex`** — model (default: gpt-5.4), credentials path
- **`tools`** — SSH keys, host definitions, timeout, optional `claude_code_host`
- **`webhook`** — Gitea/Grafana webhook routing
- **`search`** — search DB path (SQLite for embeddings + FTS)
- **`voice`** — voice channel support (requires GPU sidecar)
- **`browser`** — headless Chromium automation
- **`comfyui`** — image generation
- **`web`** — management UI and API
- **`permissions`** — per-user tool access tiers

## Tools

| Category | Count | Examples |
|----------|-------|---------|
| Command Execution | 3 | run_command, run_command_multi, run_script |
| File Operations | 3 | read_file, write_file, post_file |
| Browser | 6 | browser_screenshot, browser_read_page, browser_click, browser_fill |
| Knowledge Base | 4 | search_knowledge, ingest_document, list_knowledge, delete_knowledge |
| Scheduling | 3 | schedule_task, list_schedules, delete_schedule |
| Skills | 9 | create_skill, edit_skill, delete_skill, list_skills, enable_skill |
| Agents | 8 | spawn_agent, send_to_agent, list_agents, kill_agent |
| Autonomous Loops | 3 | start_loop, stop_loop, list_loops |
| Web | 2 | web_search, fetch_url |
| Deep Reasoning | 1 | claude_code (hidden when unconfigured) |
| PDF & Images | 3 | analyze_pdf, analyze_image, generate_image |
| Discord | 4 | purge_messages, read_channel, add_reaction, create_poll |
| Process Mgmt | 2 | manage_process, manage_list |
| Other | 10 | generate_file, parse_time, memory_manage, search_history, search_audit, create_digest, set_permission, delegate_task, list_tasks, cancel_task |

All tools execute immediately when called. No approval prompts, no confirmation.

## Skills

Create custom tools at runtime via Discord or as `.py` files in `data/skills/`.

```
@Heimdall create a skill called "disk_report" that checks disk usage on all hosts
```

See `docs/SKILLS.md` for the full development guide.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

9000+ tests, all mocked — no SSH, API, or Discord connections needed.

## Documentation

| Document | Description |
|----------|-------------|
| `docs/API.md` | REST API reference (55 endpoints + WebSocket) |
| `docs/SKILLS.md` | Skill development guide |
| `docs/ARCHITECTURE.md` | Internal architecture reference |

## Disclaimer

Heimdall is an autonomous executor. It does not ask for confirmation, display approval prompts, or second-guess your instructions. If you tell it to delete files, drop databases, restart services, or modify production infrastructure — it will do exactly that, immediately.

You are responsible for what you ask it to do. Configure `allowed_users`, `channels`, `require_mention`, and `permissions` appropriately for your environment. Do not grant Heimdall access to systems you are not prepared to have modified.

## License

MIT
