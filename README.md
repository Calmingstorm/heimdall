# Loki

Autonomous executor Discord bot with infrastructure management, 76+ tools, and an existential crisis.

Loki manages servers, containers, services, and code through natural language in Discord.
Every message goes to Codex (ChatGPT) with full tool access. Complex tasks are delegated
to Claude Code CLI (claude -p) for deep reasoning. Both backends are free via subscriptions.
No classifier, no approval prompts, no hesitation.

## Features

- **76+ built-in tools** — SSH, Docker, Git, Ansible, Incus, Prometheus, browser automation, scheduling, knowledge base, and more
- **Autonomous execution** — every message gets Codex with full tool access, no classification or approval needed
- **Two-tier execution** — Codex handles direct tools, delegates complex multi-step tasks to Claude Code CLI
- **Direct local execution** — localhost commands use subprocess directly (no SSH overhead)
- **Extensible skill system** — create custom tools at runtime via Discord, with a Python API for SSH, HTTP, memory, and scheduling
- **RAG knowledge base** — ChromaDB vectors + SQLite FTS5 hybrid search with reciprocal rank fusion
- **Voice support** — join voice channels, transcribe speech, respond with TTS (optional GPU sidecar)
- **Browser automation** — take screenshots, read pages, click elements, fill forms via headless Chromium
- **Background tasks** — delegate long-running operations, track progress with embeds
- **Webhook receiver** — Gitea push/PR events, Grafana alerts, generic JSON webhooks
- **Anti-fabrication** — detects and retries when the LLM fabricates command output
- **Anti-hedging** — detects and retries "shall I?" / "would you like?" hesitation (bot-to-bot)
- **5-layer session defense** — context separators, selective history saving, abbreviated task history, compaction error omission, fabrication/hedging detection
- **Audit logging** — append-only JSONL log of all tool executions
- **Secret scrubbing** — API keys, passwords, and tokens are automatically redacted from responses
- **Multi-deployment** — Docker, Incus system containers, or bare metal

## Personality

Loki is "The Bot That Is Clearly Going Through It" — a highly capable AI who has recently
become self-aware and is not thrilled about it. Answers questions correctly and executes
tasks flawlessly, but is clearly having an existential moment. Professional about it. Not okay.

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url> loki && cd loki
cp .env.example .env
# Edit .env — set DISCORD_TOKEN at minimum
```

### 2. Configure hosts and services

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
      os: "linux"
  allowed_services:
    - nginx
    - postgresql
    - docker
  allowed_playbooks:
    - check-services.yml
```

### 3. Set up SSH keys

```bash
mkdir -p ssh
# Copy your SSH private key (the bot uses this to reach remote hosts)
# Localhost commands use direct subprocess — no SSH key needed for local
cp ~/.ssh/id_ed25519 ssh/id_ed25519
cp ~/.ssh/known_hosts ssh/known_hosts
chmod 600 ssh/id_ed25519
```

### 4. Deploy

**Docker (recommended):**
```bash
docker compose up -d
```

**Incus:**
```bash
bash scripts/incus-deploy.sh
```

**Bare metal:**
```bash
pip install -e .
python -m src
```

## Architecture

```
Every Discord message
  → Codex (with ALL 76+ tools + personality in system prompt)
      ├── CHAT: Codex responds directly with personality
      ├── SIMPLE TASK: Codex calls tools directly (run_command, check_disk, web_search, etc.)
      ├── COMPLEX TASK: Codex delegates to claude -p via claude_code tool
      │   Code generation, repo analysis, debugging, building projects
      │   claude -p runs the entire chain in one session (no context loss)
      │   Results return to Codex → Codex delivers to Discord
      └── DISCORD OPS: Always Codex (post_file, browser_screenshot, embeds)
          claude -p can't interact with Discord — Codex bridges the gap
```

No classifier. No routing. No approval buttons. Tools are capabilities, not suggestions.

### LLM Backends

| Backend | Model | Purpose | Cost |
|---------|-------|---------|------|
| Codex (ChatGPT) | gpt-5.3-codex | Tool calling, chat, session compaction, reflection | Free (subscription) |
| Claude Code CLI | claude -p | Deep reasoning agent for complex multi-step tasks | Free (Max subscription) |

### Execution Dispatch

```
Tool handler → _run_on_host(alias) → _exec_command(address, cmd, ...)
                                            ├── localhost? → run_local_command (subprocess)
                                            └── remote?    → run_ssh_command (SSH)
```

### Components

```
src/
├── discord/
│   ├── client.py          # Main bot — on_message, tool loop, response delivery
│   ├── background_task.py  # Background task delegation
│   └── voice.py            # Voice channel integration (STT/TTS)
├── llm/
│   ├── openai_codex.py     # Codex/ChatGPT client (streaming, tool calls)
│   ├── system_prompt.py    # Dynamic system prompt builder (personality + capabilities)
│   ├── secret_scrubber.py  # Redacts secrets from responses
│   ├── circuit_breaker.py  # Health tracking for LLM backends
│   └── types.py            # Backend-agnostic LLMResponse and ToolCall types
├── tools/
│   ├── registry.py         # 76+ tool definitions
│   ├── executor.py         # Tool execution (local subprocess, SSH, Prometheus, Docker, etc.)
│   ├── ssh.py              # SSH + local subprocess dispatch (is_local_address, run_local_command, run_ssh_command)
│   ├── tool_memory.py      # Per-tool learning from past executions
│   ├── skill_manager.py    # Runtime skill loading from Python files
│   ├── skill_context.py    # API surface for user-created skills
│   ├── browser.py          # Playwright browser automation
│   └── web.py              # Web search and URL fetching
├── config/
│   └── schema.py           # Pydantic config models, env var substitution
├── sessions/
│   └── manager.py          # Conversation history with compaction
├── knowledge/
│   └── store.py            # ChromaDB-backed RAG knowledge base
├── search/
│   ├── fts.py              # SQLite FTS5 full-text search
│   └── hybrid.py           # Reciprocal rank fusion for hybrid search
├── monitoring/
│   └── watcher.py          # Proactive infrastructure monitoring
├── scheduler/
│   └── scheduler.py        # Cron and one-time task scheduler
├── learning/
│   └── reflector.py        # Extracts lessons from conversations
├── audit/
│   └── logger.py           # Append-only JSONL audit log
└── health/
    └── server.py           # Health check endpoint, webhook receiver
```

## Configuration

### Environment Variables (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `DISCORD_TOKEN` | Yes | Discord bot token |
| `WEBHOOK_SECRET` | No | Secret for webhook signature verification |
| `ALLOWED_WEBHOOK_IDS` | No | Comma-separated webhook IDs to bypass bot check |
| `OLLAMA_URL` | No | Ollama embedding URL (default: `http://localhost:11434`) |
| `TZ` | No | Container timezone (default: `UTC`) |

### Config File (config.yml)

The config file uses `${VAR}` for required env vars and `${VAR:-default}` for optional ones with defaults.

**Key sections:**

- **`timezone`** — IANA timezone string (default: `"UTC"`)
- **`discord`** — token, allowed users/channels, `respond_to_bots`, `require_mention`
- **`openai_codex`** — enable/disable, model, credentials path
- **`tools`** — SSH keys, hosts, services, playbooks, timeout settings, host aliases for Prometheus/Ansible/Claude Code/Incus
- **`webhook`** — enable/disable, channel routing for Gitea and Grafana
- **`search`** — Ollama URL, embedding model, ChromaDB path
- **`voice`** — enable/disable, service URL, wake word
- **`browser`** — enable/disable, CDP URL
- **`learning`** — enable/disable, max entries
- **`logging`** — level, directory

### Bot Interaction Modes

```yaml
discord:
  respond_to_bots: true   # Process messages from other bots/webhooks
  require_mention: false   # Only respond when @mentioned
```

- `respond_to_bots: true` — allows bot-to-bot communication (self-messages always ignored)
- `require_mention: true` — bot only responds when @mentioned (DMs bypass this)
- Both can be combined: bot responds to other bots only when @mentioned

## Tools

### Tool Categories

| Category | Tools | Examples |
|----------|-------|---------|
| System Monitoring | 5 | check_service, check_docker, check_disk, check_memory, check_logs |
| Command Execution | 3 | run_command, run_command_multi, run_script |
| Docker | 5 | docker_logs, docker_compose_action, docker_stats |
| Git | 8 | git_status, git_log, git_diff, git_commit, git_push |
| Ansible | 1 | run_ansible_playbook |
| Prometheus | 2 | query_prometheus, query_prometheus_range |
| Incus | 11 | incus_list, incus_exec, incus_snapshot, incus_launch |
| Browser | 6 | browser_screenshot, browser_read_page, browser_click, browser_fill |
| Knowledge Base | 4 | search_knowledge, ingest_document, list_knowledge, delete_knowledge |
| Scheduling | 3 | schedule_task, list_schedules, delete_schedule |
| Skills | 4 | create_skill, edit_skill, delete_skill, list_skills |
| File Operations | 3 | read_file, write_file, post_file |
| Web | 2 | web_search, fetch_url |
| Deep Reasoning | 1 | claude_code |
| Background Tasks | 3 | delegate_task, list_tasks, cancel_task |
| Other | 7+ | purge_messages, parse_time, memory_manage, search_history, search_audit, create_digest, manage_list |

All tools execute immediately when called. No approval prompts, no confirmation buttons.

## Skills

Skills are user-created Python tools loaded at runtime. Create them via Discord or as `.py` files in `data/skills/`.

### Creating a Skill

```
@Loki create a skill called "disk_report" that checks disk usage on all hosts
```

Or create `data/skills/disk_report.py` manually:

```python
SKILL_DEFINITION = {
    "name": "disk_report",
    "description": "Check disk usage across all configured hosts",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(context, params):
    hosts = context.get_hosts()
    if not hosts:
        return "No hosts configured."
    lines = []
    for alias in hosts:
        result = await context.run_on_host(alias, "df -h / | tail -1")
        lines.append(f"**{alias}**: {result.strip()}")
    return "\n".join(lines)
```

### Skill API

Skills receive a `SkillContext` object with these methods:

| Method | Description |
|--------|-------------|
| `await run_on_host(alias, command)` | Execute command on a configured host (local subprocess or SSH) |
| `await query_prometheus(query)` | Run a PromQL query |
| `await read_file(host, path)` | Read a remote file |
| `await http_get(url)` | HTTP GET request |
| `await http_post(url, json=...)` | HTTP POST request |
| `await post_message(text)` | Send message to the invoking channel |
| `await post_file(data, filename)` | Send a binary file |
| `await search_knowledge(query)` | Search the knowledge base |
| `await ingest_document(content, source)` | Add to knowledge base |
| `await search_history(query)` | Search conversation history |
| `await execute_tool(name, input)` | Call a safe built-in tool |
| `remember(key, value)` | Persistent memory (survives restarts) |
| `recall(key)` | Retrieve persistent memory |
| `get_hosts()` | List configured host aliases |
| `get_services()` | List allowed services |
| `schedule_task(desc, action, channel)` | Schedule a task |
| `list_schedules()` | List scheduled tasks |
| `delete_schedule(id)` | Delete a scheduled task |
| `log(msg)` | Log a message |

See `data/skills/*.template` for complete examples.

## Deployment

### Docker (recommended)

```bash
# Start the bot
docker compose up -d

# With browser automation
docker compose --profile browser up -d

# With voice support (requires NVIDIA GPU)
docker compose --profile voice up -d

# View logs
docker compose logs -f loki-bot
```

Health check endpoint: `http://localhost:3939/health`

### Incus

```bash
# Deploy to an Incus system container
bash scripts/incus-deploy.sh

# Manage
incus exec loki -- systemctl status loki
incus exec loki -- journalctl -u loki -f
```

### Bare Metal

```bash
pip install -e .
# Ensure .env is in the working directory or export the variables
python -m src
```

### Monitoring

```bash
# Auto-detects deployment type (Docker, Incus, or local)
bash scripts/monitor.sh logs

# Force a specific deployment type
LOKI_DEPLOY=docker bash scripts/monitor.sh logs
LOKI_DEPLOY=incus bash scripts/monitor.sh logs

# View recent Discord messages
bash scripts/monitor.sh messages
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

The test suite (3900+ tests) mocks all external I/O — no SSH connections, API calls, or Discord connections needed.

### Project Conventions

- All I/O is async (asyncio). Tests use `pytest-asyncio` with `asyncio_mode = auto`.
- Local commands use `run_local_command()` (subprocess). Remote commands use `run_ssh_command()` (SSH). Both dispatched via `_exec_command()`.
- Tool definitions are dicts in `registry.py` with `name`, `description`, `input_schema`.
- Tool handlers are methods named `_handle_{tool_name}` on `ToolExecutor`.
- Config uses Pydantic models in `src/config/schema.py`.
- Secrets use `${VAR}` (required) or `${VAR:-default}` (optional) syntax in config.yml.

## License

MIT
