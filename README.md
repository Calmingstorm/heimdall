# Loki

AI assistant Discord bot with infrastructure management, tool calling, and extensible skills.

Loki manages servers, containers, services, and code through natural language in Discord.
It uses Codex (ChatGPT) for tool calling and chat, Claude Code CLI for code generation,
and Haiku for message classification — all via free subscription tiers.

## Features

- **67+ built-in tools** — SSH, Docker, Git, Ansible, Incus, Prometheus, browser automation, scheduling, knowledge base, and more
- **AI-powered routing** — messages are classified and routed to the right backend (chat, task with tools, or code generation)
- **Extensible skill system** — create custom tools at runtime via Discord, with a Python API for SSH, HTTP, memory, and scheduling
- **RAG knowledge base** — ChromaDB vectors + SQLite FTS5 hybrid search with reciprocal rank fusion
- **Voice support** — join voice channels, transcribe speech, respond with TTS (optional GPU sidecar)
- **Browser automation** — take screenshots, read pages, click elements, fill forms via headless Chromium
- **Background tasks** — delegate long-running operations, track progress with embeds
- **Webhook receiver** — Gitea push/PR events, Grafana alerts, generic JSON webhooks
- **Approval workflow** — destructive actions require button confirmation before execution
- **Audit logging** — append-only JSONL log of all tool executions
- **Secret scrubbing** — API keys, passwords, and tokens are automatically redacted from responses
- **Multi-deployment** — Docker, Incus system containers, or bare metal

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url> loki && cd loki
cp .env.example .env
# Edit .env — set DISCORD_TOKEN and ANTHROPIC_API_KEY at minimum
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
# Copy your SSH private key (the bot uses this to reach your hosts)
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
Discord message
  │
  ├─ Keyword bypass (docker, ansible, prometheus → task)
  │
  └─ Haiku classifier → route to:
       "chat"        → Codex/ChatGPT (conversation)
       "task"        → Codex with tool calling (67+ tools)
       "claude_code" → claude -p CLI via SSH (code generation)
```

### LLM Backends

| Backend | Model | Purpose | Cost |
|---------|-------|---------|------|
| Codex (ChatGPT) | gpt-5.3-codex | Tool calling, chat, session compaction, reflection | Free (subscription) |
| Claude Code CLI | claude -p | Code generation via temp dir workflow over SSH | Free (Max subscription) |
| Haiku | claude-haiku-4-5 | Message classification only (raw HTTP, no SDK) | ~$0.0002/call |

### Components

```
src/
├── discord/
│   ├── client.py          # Main bot — on_message, tool loop, response delivery
│   ├── routing.py          # Keyword bypass, host routing
│   ├── background_task.py  # Background task delegation
│   ├── approval.py         # Button-based approval for destructive actions
│   └── voice.py            # Voice channel integration (STT/TTS)
├── llm/
│   ├── openai_codex.py     # Codex/ChatGPT client (streaming, tool calls)
│   ├── haiku_classifier.py # Haiku classifier (raw HTTP, Anthropic API)
│   ├── system_prompt.py    # Dynamic system prompt builder
│   ├── secret_scrubber.py  # Redacts secrets from responses
│   └── circuit_breaker.py  # Health tracking for LLM backends
├── tools/
│   ├── registry.py         # 67+ tool definitions
│   ├── executor.py         # Tool execution (SSH, Prometheus, Docker, etc.)
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
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key (Haiku classification) |
| `WEBHOOK_SECRET` | No | Secret for webhook signature verification |
| `ALLOWED_WEBHOOK_IDS` | No | Comma-separated webhook IDs to bypass bot check |
| `OLLAMA_URL` | No | Ollama embedding URL (default: `http://localhost:11434`) |
| `TZ` | No | Container timezone (default: `UTC`) |

### Config File (config.yml)

The config file uses `${VAR}` for required env vars and `${VAR:-default}` for optional ones with defaults.

**Key sections:**

- **`timezone`** — IANA timezone string (default: `"UTC"`)
- **`discord`** — token, allowed users/channels, `respond_to_bots`, `require_mention`
- **`anthropic`** — API key, model, token budget
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
| Command Execution | 2 | run_command, run_command_multi |
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
| Code Generation | 1 | claude_code |
| Background Tasks | 3 | delegate_task, list_tasks, cancel_task |
| Other | 7 | purge_messages, parse_time, memory_manage, search_history, search_audit, create_digest, manage_list |

### Approval Workflow

Tools marked with `requires_approval: true` (like `run_command`, `git_push`, `docker_compose_action`) trigger a Discord button prompt. The user must click Approve or Deny before execution proceeds.

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
    "requires_approval": False,
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
| `await run_on_host(alias, command)` | Execute SSH command on a configured host |
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

The test suite (3000+ tests) mocks all external I/O — no SSH connections, API calls, or Discord connections needed.

### Project Conventions

- All I/O is async (asyncio). Tests use `pytest-asyncio` with `asyncio_mode = auto`.
- SSH commands go through `src/tools/ssh.py:run_ssh_command()`, always mocked in tests.
- Tool definitions are dicts in `registry.py` with `name`, `description`, `input_schema`, `requires_approval`.
- Tool handlers are methods named `_handle_{tool_name}` on `ToolExecutor`.
- Config uses Pydantic models in `src/config/schema.py`.
- Secrets use `${VAR}` (required) or `${VAR:-default}` (optional) syntax in config.yml.

## License

MIT
