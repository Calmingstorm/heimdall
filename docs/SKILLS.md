# Heimdall Skill Development Guide

Skills are user-created Python tools that extend Heimdall at runtime. Each skill is
a `.py` file in `data/skills/` with a definition dict and an async execute function.

## Quick Start

Create a skill via Discord:
```
@Heimdall create a skill called "uptime_report" that checks uptime on all hosts
```

Or create `data/skills/uptime_report.py` manually:

```python
SKILL_DEFINITION = {
    "name": "uptime_report",
    "description": "Check uptime across all configured hosts",
    "input_schema": {
        "type": "object",
        "properties": {},
    },
}

async def execute(context, params):
    hosts = context.get_hosts()
    if not hosts:
        return "No hosts configured."
    lines = []
    for alias in hosts:
        result = await context.run_on_host(alias, "uptime -p")
        lines.append(f"**{alias}**: {result.strip()}")
    return "\n".join(lines)
```

Skills are hot-reloaded when files change. No restart required.

## Skill Structure

Every skill file must have:

1. **`SKILL_DEFINITION`** — a dict with tool metadata
2. **`async def execute(context, params)`** — the entry point

### SKILL_DEFINITION

Required keys:

| Key | Type | Description |
|-----|------|-------------|
| `name` | str | Lowercase, letters/numbers/underscores, max 50 chars |
| `description` | str | What the skill does (shown to the LLM) |
| `input_schema` | dict | JSON Schema for parameters |

Optional keys:

| Key | Type | Description |
|-----|------|-------------|
| `version` | str | Semver version (e.g., `"1.0.0"`) |
| `author` | str | Skill author name |
| `homepage` | str | URL to skill documentation |
| `tags` | list | Category tags (e.g., `["monitoring", "disk"]`) |
| `dependencies` | list | Python packages to auto-install (e.g., `["requests"]`) |
| `config_schema` | dict | JSON Schema for skill-specific config |

### Input Schema

The `input_schema` follows JSON Schema and defines what parameters the LLM passes:

```python
SKILL_DEFINITION = {
    "name": "http_check",
    "description": "Check if a URL is reachable and measure response time",
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to check",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 10)",
            },
        },
        "required": ["url"],
    },
}
```

### Execute Function

```python
async def execute(context, params):
    url = params.get("url", "")
    timeout = params.get("timeout", 10)
    # ... do work ...
    return "Result text"  # Returned to the LLM
```

- `context` — a `SkillContext` instance with the full skill API
- `params` — dict matching `input_schema`, passed by the LLM
- Return a string (sent back to the LLM as tool output)
- Raise an exception to report an error

## SkillContext API

The `context` object provides methods for infrastructure access, messaging,
knowledge, and more. All async methods must be `await`ed.

### Command Execution

```python
# Run a command on a managed host (SSH or local subprocess)
result = await context.run_on_host("webserver", "df -h /")

# Run a PromQL query against Prometheus
metrics = await context.query_prometheus('up{job="nginx"}')

# Read a file from a host
content = await context.read_file("webserver", "/etc/nginx/nginx.conf")
# Optional: limit lines
content = await context.read_file("webserver", "/var/log/syslog", lines=100)
```

### HTTP Requests

```python
# GET request (auto-parses JSON responses)
data = await context.http_get("https://api.example.com/status")
data = await context.http_get("https://api.example.com/items",
                              params={"page": 1}, timeout=30,
                              headers={"X-API-Key": "..."})

# POST request
result = await context.http_post("https://api.example.com/deploy",
                                 json={"version": "1.2.3"})
```

**Blocked URLs**: localhost, 127.0.0.1, private IP ranges, cloud metadata endpoints.

### Messaging

```python
# Send a text message to the invoking Discord channel
await context.post_message("Deployment complete!")

# Send a file as an attachment
import json
data = json.dumps(report, indent=2).encode()
await context.post_file(data, "report.json", caption="Here's the report")
```

### Knowledge Base

```python
# Search the RAG knowledge base
results = await context.search_knowledge("nginx configuration", limit=5)

# Ingest text into the knowledge base
await context.ingest_document(
    "Deployment procedure: 1. Pull latest...",
    source="deployment-guide"
)

# Search conversation history
history = await context.search_history("disk upgrade", limit=10)
```

### Memory (Persistent Key-Value Store)

```python
# Save a value (survives restarts)
context.remember("last_check", "2026-03-27T10:00:00Z")

# Retrieve a value
last_check = context.recall("last_check")  # Returns None if not found
```

Memory is scoped to the skill. Values persist across restarts in the SQLite database.

### Scheduling

```python
# Create a scheduled task
context.schedule_task(
    description="Weekly disk report",
    action="run disk_report skill",
    channel_id="123456789",
    cron="0 9 * * 1"  # Every Monday at 9 AM
)

# List schedules
schedules = context.list_schedules()

# Delete a schedule
context.delete_schedule("schedule-id-123")
```

### Configuration

```python
# Get available hosts
hosts = context.get_hosts()  # ["webserver", "dbserver", ...]

# Get allowed services
services = context.get_services()  # ["nginx", "postgresql", ...]

# Get skill-specific config (set via API or web UI)
threshold = context.get_config("threshold", default=80)
all_config = context.get_all_config()  # {"threshold": 80, ...}
```

### Tool Execution

```python
# Execute a safe built-in tool
result = await context.execute_tool("web_search", {"query": "python asyncio tutorial"})
result = await context.execute_tool("web_search", {"query": "python asyncio"})
```

Only read-only tools are allowed. See the safe tools list below.

### Logging

```python
context.log("Processing host webserver...")
context.log("ERROR: connection timeout")
```

## Safe Tools

Skills can call these built-in tools via `execute_tool()`:

**File**: read_file

**Knowledge**: search_knowledge, list_knowledge

**Web**: web_search, fetch_url, browser_screenshot, browser_read_page, browser_read_table

**History & Audit**: search_history, search_audit

**Scheduling**: list_schedules, list_skills, list_tasks

**Memory**: memory_manage

**Other**: parse_time

Write operations (run_command, write_file, etc.) are blocked for security.
Use `run_on_host()` for direct command execution instead.

## Sandbox Limits

Skills run within resource limits to prevent abuse:

| Limit | Value |
|-------|-------|
| Execution timeout | 120 seconds |
| Max output | 50,000 characters |
| Max tool calls | 50 per execution |
| Max HTTP requests | 20 per execution |
| Max messages sent | 10 per execution |
| Max files sent | 10 per execution |

Exceeding limits raises an error that stops the skill.

## Blocked File Paths

Skills cannot read sensitive files:

- `.env` — environment variables
- `config.yml` — bot configuration
- SSH keys (`id_rsa`, `id_ed25519`, etc.)
- `.ssh/` directories
- `credentials.json`
- `.kube/config`

## Dependencies

Skills can declare Python package dependencies that are auto-installed:

```python
SKILL_DEFINITION = {
    "name": "weather_check",
    "description": "Get current weather for a city",
    "dependencies": ["httpx"],
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"],
    },
}
```

- Max 10 dependencies per skill
- Packages are installed via pip with a 120-second timeout
- Dependencies are also auto-detected from imports in the source code

## Skill-Specific Configuration

Skills can define a config schema and read values at runtime:

```python
SKILL_DEFINITION = {
    "name": "alert_checker",
    "description": "Check alerts from monitoring system",
    "config_schema": {
        "type": "object",
        "properties": {
            "threshold": {
                "type": "integer",
                "description": "Alert threshold percentage",
                "default": 80,
            },
            "webhook_url": {
                "type": "string",
                "description": "Webhook for notifications",
            },
        },
    },
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(context, params):
    threshold = context.get_config("threshold", default=80)
    # ...
```

Config values are set via the web UI or API (`PUT /api/skills/{name}/config`).

## Installing Skills from URL

Skills can be installed from a URL:

```
@Heimdall install skill from https://example.com/skills/my_skill.py
```

- Max download size: 256 KB
- Allowed schemes: http, https
- Download timeout: 30 seconds

## Validation

The web UI and API validate skill code before saving:

- Syntax errors caught at parse time
- Missing `SKILL_DEFINITION` or `execute` function flagged
- Non-async `execute` generates a warning
- Name format validated: lowercase, letters/numbers/underscores, max 50 chars
- Version format validated: semver (`1.0.0`, `1.0.0-beta`)

## Handoff

Skills can signal that their output should go directly to the LLM for further
processing (rather than being treated as a final tool result):

```python
async def execute(context, params):
    data = await context.run_on_host("webserver", "df -h")
    return {"output": data, "handoff": True}
```

When all tools in an iteration return `handoff=True`, the tool loop returns
control to the LLM for a natural language response.

## Examples

### HTTP Health Check

```python
SKILL_DEFINITION = {
    "name": "http_health",
    "description": "Check HTTP status of URLs",
    "version": "1.0.0",
    "tags": ["monitoring", "http"],
    "input_schema": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "URLs to check",
            },
        },
        "required": ["urls"],
    },
}

async def execute(context, params):
    urls = params.get("urls", [])
    results = []
    for url in urls:
        try:
            data = await context.http_get(url, timeout=10)
            results.append(f"**{url}**: OK")
        except Exception as e:
            results.append(f"**{url}**: FAILED — {e}")
    return "\n".join(results) or "No URLs provided."
```

### Multi-Host Disk Report

```python
SKILL_DEFINITION = {
    "name": "disk_report",
    "description": "Generate disk usage report across all hosts",
    "version": "1.0.0",
    "tags": ["monitoring", "disk"],
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(context, params):
    hosts = context.get_hosts()
    if not hosts:
        return "No hosts configured."

    lines = ["# Disk Usage Report\n"]
    for alias in hosts:
        result = await context.run_on_host(alias, "df -h / | tail -1")
        fields = result.split()
        if len(fields) >= 5:
            lines.append(f"**{alias}**: {fields[4]} used ({fields[2]}/{fields[1]})")
        else:
            lines.append(f"**{alias}**: {result.strip()}")

    return "\n".join(lines)
```

### Scheduled Alert with Memory

```python
SKILL_DEFINITION = {
    "name": "cert_monitor",
    "description": "Check SSL certificate expiry and alert if close",
    "version": "1.0.0",
    "tags": ["security", "ssl"],
    "config_schema": {
        "type": "object",
        "properties": {
            "domains": {"type": "array", "items": {"type": "string"}},
            "warn_days": {"type": "integer", "default": 30},
        },
    },
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(context, params):
    domains = context.get_config("domains", default=[])
    warn_days = context.get_config("warn_days", default=30)

    if not domains:
        return "No domains configured. Set config via web UI."

    alerts = []
    for domain in domains:
        result = await context.run_on_host(
            "localhost",
            f"echo | openssl s_client -servername {domain} -connect {domain}:443 2>/dev/null"
            f" | openssl x509 -noout -dates 2>/dev/null | grep notAfter"
        )
        if "notAfter" in result:
            alerts.append(f"**{domain}**: expires {result.split('=')[1].strip()}")
        else:
            alerts.append(f"**{domain}**: could not check certificate")

    context.remember("last_check", str(context.recall("check_count") or 0))
    return "\n".join(alerts) or "All certificates OK."
```

## Managing Skills via Web UI

The web management UI at `http://host:3939/ui/#/skills` provides:

- **Code editor** — syntax-highlighted Python editor
- **Create/edit/delete** — full CRUD operations
- **Validation** — real-time syntax and structure checking
- **Test** — execute skills with custom parameters
- **Enable/disable** — toggle skills without deleting
- **Configuration** — set skill-specific config values

## Template Files

Three templates are included in `data/skills/`:

- `example_skill.py.template` — comprehensive example with all API methods
- `http_health_check.py.template` — HTTP monitoring skill
- `system_info.py.template` — system information gathering skill

Copy and rename (remove `.template`) to use as starting points.
