# Heimdall REST API Reference

Heimdall exposes a REST API on port 3939 for web-based management. The API shares
the health server and is available when `web.enabled: true` in config.

**Base URL**: `http://host:3939/api`

## Authentication

All endpoints (except `/api/auth/login`) require a Bearer token when `web.api_token`
is set in config. If no token is configured, auth is disabled (dev mode).

```
Authorization: Bearer <your-api-token>
```

Login returns a session cookie as an alternative to the Bearer token.

### POST /api/auth/login

Authenticate and receive a session.

**Body**: `{"token": "your-api-token"}`

**Response**: `{"ok": true}` with `Set-Cookie` header.

**Errors**: `401` if token is invalid.

### POST /api/auth/logout

End the current session.

**Response**: `{"ok": true}` with cleared cookie.

### GET /api/auth/session

Check if the current session is valid.

**Response**: `{"authenticated": true}` or `{"authenticated": false}`.

---

## Status & System

### GET /api/status

Bot status, uptime, connected guilds, and feature flags.

**Response**:
```json
{
  "status": "online",
  "uptime": 3600.5,
  "guilds": 3,
  "channels": 12,
  "sessions": 5,
  "tools": 80,
  "skills": 3,
  "loops": 1,
  "agents": 0,
  "processes": 2,
  "features": {
    "voice": false,
    "browser": true,
    "monitoring": true,
    "learning": true,
    "web_ui": true,
    "comfyui": false
  }
}
```

### POST /api/reload

Reload configuration, skills, and caches. Equivalent to the `/reload` Discord command.

**Response**: `{"ok": true}`

### GET /api/config

Get current configuration with sensitive fields redacted.

**Response**: Config object with passwords/tokens replaced by `"***"`.

### PUT /api/config

Update configuration values. Only supports a subset of safe config keys.

**Body**: `{"key": "value", ...}`

**Response**: `{"ok": true}`

---

## Sessions

Manage conversation history per Discord channel.

### GET /api/sessions

List all active sessions.

**Response**:
```json
[
  {
    "channel_id": "123456789",
    "message_count": 42,
    "last_activity": "2026-03-27T10:30:00Z"
  }
]
```

### GET /api/sessions/{channel_id}

Get messages for a specific channel session.

**Query params**: `limit` (int, default 50), `offset` (int, default 0)

**Response**:
```json
{
  "channel_id": "123456789",
  "messages": [
    {"role": "user", "content": "check disk on webserver", "timestamp": "..."},
    {"role": "assistant", "content": "Disk usage on webserver: 45%", "timestamp": "..."}
  ],
  "total": 42
}
```

### GET /api/sessions/{channel_id}/export

Export full session as downloadable JSON.

**Response**: JSON file download with all messages.

### DELETE /api/sessions/{channel_id}

Delete a single session.

**Response**: `{"ok": true}`

### POST /api/sessions/clear-bulk

Delete multiple sessions at once.

**Body**: `{"channel_ids": ["123", "456"]}`

**Response**: `{"ok": true, "cleared": 2}`

### POST /api/sessions/clear-all

Delete all sessions.

**Response**: `{"ok": true}`

---

## Chat

Send messages to Heimdall via the web interface.

### POST /api/chat

Send a chat message and receive a response.

**Body**:
```json
{
  "content": "check disk usage on all hosts",
  "channel_id": "web-chat-123",
  "user_id": "web-user",
  "username": "admin"
}
```

**Response**:
```json
{
  "content": "Disk usage across hosts:\n- webserver: 45%\n- dbserver: 62%",
  "tools_used": ["run_command"],
  "is_error": false
}
```

Chat is also available via WebSocket for real-time streaming (see WebSocket section).

---

## Tools

### GET /api/tools

List all available tools with definitions.

**Response**:
```json
[
  {
    "name": "run_command",
    "description": "Execute a shell command on a host",
    "input_schema": {...},
    "pack": null
  }
]
```

### GET /api/tools/packs

List tool packs and their status.

**Response**:
```json
{
  "available": {
    "systemd": {"tools": 3, "enabled": true},
    "incus": {"tools": 11, "enabled": true},
    "ansible": {"tools": 1, "enabled": false},
    "prometheus": {"tools": 4, "enabled": true},
    "comfyui": {"tools": 1, "enabled": false}
  },
  "enabled": ["systemd", "incus", "prometheus"]
}
```

### PUT /api/tools/packs

Enable or disable tool packs.

**Body**: `{"enabled": ["systemd", "incus", "prometheus"]}`

**Response**: `{"ok": true}`

### GET /api/tools/stats

Tool execution statistics from the audit log.

**Response**: Object with per-tool execution counts.

---

## Skills

Manage user-created Python tools. Skills are `.py` files in `data/skills/`.

### GET /api/skills

List all loaded skills.

**Response**:
```json
[
  {
    "name": "disk_report",
    "description": "Check disk usage across all hosts",
    "version": "1.0.0",
    "author": "admin",
    "enabled": true,
    "tags": ["monitoring"]
  }
]
```

### POST /api/skills

Create a new skill.

**Body**:
```json
{
  "name": "disk_report",
  "code": "SKILL_DEFINITION = {...}\nasync def execute(context, params): ..."
}
```

**Response**: `{"ok": true, "name": "disk_report"}`

### GET /api/skills/{name}

Get skill source code.

**Response**: `{"name": "disk_report", "code": "...", "definition": {...}}`

### PUT /api/skills/{name}

Update skill source code.

**Body**: `{"code": "..."}`

**Response**: `{"ok": true}`

### DELETE /api/skills/{name}

Delete a skill.

**Response**: `{"ok": true}`

### POST /api/skills/{name}/test

Execute a skill with test parameters.

**Body**: `{"params": {...}}` (optional)

**Response**: `{"ok": true, "output": "...", "duration": 1.23}`

### POST /api/skills/{name}/enable

Enable a disabled skill.

**Response**: `{"ok": true}`

### POST /api/skills/{name}/disable

Disable a skill without deleting it.

**Response**: `{"ok": true}`

### POST /api/skills/validate

Validate skill code without saving.

**Body**: `{"code": "..."}`

**Response**:
```json
{
  "valid": true,
  "diagnostics": [],
  "definition": {"name": "...", "description": "..."}
}
```

### GET /api/skills/{name}/config

Get skill-specific configuration.

**Response**: `{"config": {...}}`

### PUT /api/skills/{name}/config

Update skill-specific configuration.

**Body**: `{"config": {...}}`

**Response**: `{"ok": true}`

---

## Knowledge Base

Manage the RAG knowledge base (local embeddings + SQLite FTS5).

### GET /api/knowledge

List all ingested knowledge sources.

**Response**:
```json
[
  {
    "source": "deployment-guide",
    "chunks": 24,
    "ingested_at": "2026-03-20T15:00:00Z"
  }
]
```

### POST /api/knowledge

Ingest a new document.

**Body**:
```json
{
  "content": "Document text content...",
  "source": "deployment-guide"
}
```

**Response**: `{"ok": true, "chunks": 24}`

### DELETE /api/knowledge/{source}

Delete a knowledge source and its chunks.

**Response**: `{"ok": true}`

### GET /api/knowledge/{source}/chunks

View chunks for a specific source.

**Response**: `{"source": "...", "chunks": [{"text": "...", "index": 0}]}`

### POST /api/knowledge/{source}/reingest

Re-process a source (re-chunk and re-embed).

**Response**: `{"ok": true, "chunks": 24}`

### GET /api/knowledge/search

Search the knowledge base.

**Query params**: `q` (string, required), `limit` (int, default 5)

**Response**:
```json
{
  "results": [
    {"text": "...", "source": "...", "score": 0.85}
  ]
}
```

---

## Schedules

Manage cron jobs and one-time tasks.

### GET /api/schedules

List all scheduled tasks.

**Response**:
```json
[
  {
    "id": "abc123",
    "description": "Daily disk check",
    "cron": "0 9 * * *",
    "action": "check disk on all hosts",
    "channel_id": "123456",
    "next_run": "2026-03-28T09:00:00Z",
    "enabled": true
  }
]
```

### POST /api/schedules

Create a new schedule.

**Body**:
```json
{
  "description": "Daily disk check",
  "cron": "0 9 * * *",
  "action": "check disk on all hosts",
  "channel_id": "123456789"
}
```

For one-time tasks, use `run_at` instead of `cron`:
```json
{
  "description": "Restart nginx tonight",
  "run_at": "2026-03-27T22:00:00Z",
  "action": "restart nginx on webserver",
  "channel_id": "123456789"
}
```

**Response**: `{"ok": true, "id": "abc123"}`

### DELETE /api/schedules/{schedule_id}

Delete a schedule.

**Response**: `{"ok": true}`

### POST /api/schedules/{schedule_id}/run

Trigger a schedule immediately (does not affect next scheduled run).

**Response**: `{"ok": true}`

### POST /api/schedules/validate-cron

Validate a cron expression.

**Body**: `{"cron": "0 9 * * *"}`

**Response**: `{"valid": true, "next_runs": ["2026-03-28T09:00:00Z", ...]}`

---

## Autonomous Loops

Manage LLM-driven recurring tasks.

### GET /api/loops

List all active and recent loops.

**Response**:
```json
[
  {
    "id": "loop-abc",
    "goal": "Monitor disk usage, warn if above 80%",
    "mode": "silent",
    "interval": 300,
    "iteration_count": 12,
    "status": "running",
    "stop_condition": "when told to stop",
    "started_at": "2026-03-27T08:00:00Z"
  }
]
```

### POST /api/loops

Start a new autonomous loop.

**Body**:
```json
{
  "goal": "Check system health every 5 minutes",
  "channel_id": "123456789",
  "interval": 300,
  "mode": "notify",
  "max_iterations": 100,
  "stop_condition": "when all services are healthy for 1 hour"
}
```

**Modes**: `act` (do + report), `notify` (check + report), `silent` (report only if notable)

**Response**: `{"ok": true, "id": "loop-abc"}`

### DELETE /api/loops/{loop_id}

Stop and remove a loop.

**Response**: `{"ok": true}`

### POST /api/loops/{loop_id}/restart

Restart a stopped loop.

**Response**: `{"ok": true}`

### POST /api/loops/stop-all

Stop all running loops.

**Response**: `{"ok": true, "stopped": 3}`

---

## Agents

Manage autonomous agents spawned by loops or the orchestrator.

### GET /api/agents

List all active agents.

**Response**:
```json
[
  {
    "id": "agent-xyz",
    "label": "disk-check",
    "goal": "Check disk on webserver",
    "status": "running",
    "iteration_count": 3,
    "tools_used": ["run_command"],
    "started_at": "2026-03-27T10:00:00Z"
  }
]
```

### DELETE /api/agents/{agent_id}

Kill a running agent.

**Response**: `{"ok": true}`

---

## Background Processes

Manage background processes started via `manage_process`.

### GET /api/processes

List tracked background processes.

**Response**:
```json
[
  {
    "pid": "proc-123",
    "command": "tail -f /var/log/syslog",
    "host": "webserver",
    "status": "running",
    "started_at": "2026-03-27T10:00:00Z"
  }
]
```

### DELETE /api/processes/{pid}

Kill a background process.

**Response**: `{"ok": true}`

---

## Audit Log

Search the append-only tool execution log.

### GET /api/audit

Search audit entries.

**Query params**:
- `tool` (string) — filter by tool name
- `user` (string) — filter by user
- `host` (string) — filter by host
- `q` (string) — full-text search
- `since` (ISO datetime) — entries after this time
- `until` (ISO datetime) — entries before this time
- `limit` (int, default 100) — max results
- `offset` (int, default 0) — pagination offset

**Response**:
```json
{
  "entries": [
    {
      "timestamp": "2026-03-27T10:30:00Z",
      "tool": "run_command",
      "user": "admin",
      "host": "webserver",
      "input": {"command": "uptime"},
      "output": "up 42 days",
      "duration": 0.5,
      "success": true
    }
  ],
  "total": 1234
}
```

---

## Memory

Manage persistent key-value storage (global + per-user scopes).

### GET /api/memory

List all memory scopes and their keys.

**Response**:
```json
{
  "scopes": {
    "global": ["last_deploy", "maintenance_mode"],
    "user:admin": ["preferences", "shortcuts"]
  }
}
```

### GET /api/memory/{scope}/{key}

Get a memory value.

**Response**: `{"scope": "global", "key": "last_deploy", "value": "2026-03-27"}`

### PUT /api/memory/{scope}/{key}

Set a memory value.

**Body**: `{"value": "2026-03-27"}`

**Response**: `{"ok": true}`

### DELETE /api/memory/{scope}/{key}

Delete a memory entry.

**Response**: `{"ok": true}`

### POST /api/memory/bulk-delete

Delete multiple memory entries.

**Body**: `{"keys": [{"scope": "global", "key": "old_key"}]}`

**Response**: `{"ok": true, "deleted": 1}`

---

## WebSocket

Real-time communication channel for log tailing, events, and chat.

**Endpoint**: `ws://host:3939/api/ws`

**Authentication**: Pass token as query parameter: `ws://host:3939/api/ws?token=your-token`

### Subscribe to Logs

```json
{"subscribe": "logs"}
```

Server sends 50 most recent log lines immediately, then streams new lines:
```json
{"type": "log", "line": "2026-03-27 10:30:00 INFO Tool run_command completed"}
```

### Subscribe to Events

```json
{"subscribe": "events"}
```

Server streams infrastructure events:
```json
{"type": "event", "event": "tool_executed", "data": {...}}
```

### Unsubscribe

```json
{"unsubscribe": "logs"}
```

### Chat via WebSocket

```json
{
  "type": "chat",
  "content": "check disk on webserver",
  "channel_id": "web-chat-123",
  "user_id": "web-user",
  "username": "admin"
}
```

Response:
```json
{
  "type": "chat_response",
  "content": "Disk usage: 45%",
  "tools_used": ["run_command"],
  "is_error": false
}
```

### Heartbeat

Client should send pings to keep the connection alive:
```json
{"type": "ping", "ts": 1711536600}
```

Server responds:
```json
{"type": "pong", "ts": 1711536600}
```

The server sends keepalive pings every 30 seconds. Connection closes after missed pings.

---

## Rate Limiting

All API endpoints are rate-limited to **120 requests per 60 seconds per IP**.

Exceeding the limit returns `429 Too Many Requests`.

## Security Headers

All responses include:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Content-Security-Policy` (restrictive policy)

## Error Format

All errors follow a consistent format:

```json
{"error": "Description of what went wrong"}
```

HTTP status codes: `400` (bad request), `401` (unauthorized), `404` (not found), `429` (rate limited), `500` (server error).
