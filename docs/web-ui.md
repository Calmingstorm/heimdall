# Web Management UI

Browser-based management interface at `http://host:3939/ui/`.

## Setup

```yaml
web:
  enabled: true
  api_token: "your-secret-token"  # Empty = no auth (dev mode)
```

## Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Bot status, uptime, connected guilds, recent activity |
| **Chat** | Web-based chat interface with real-time WebSocket |
| **Sessions** | View active conversations, message history, clear sessions |
| **Tools** | Browse all tools, search and filter by category |
| **Skills** | Create, edit, delete runtime skills with code editor |
| **Knowledge** | Browse, search, ingest, delete knowledge base documents |
| **Schedules** | Manage cron jobs, one-time tasks, webhook triggers |
| **Loops** | View and control autonomous loops |
| **Processes** | Monitor background processes |
| **Agents** | View and manage autonomous agents |
| **Audit** | Searchable tool execution log with filters |
| **Config** | View configuration (sensitive fields redacted) |
| **Logs** | Live log tail via WebSocket with level filtering |
| **Memory** | Browse and edit persistent memory |

## Tech Stack

- **Backend**: aiohttp REST API (55 endpoints) + WebSocket
- **Frontend**: Vue 3 + Tailwind CSS + Vue Router (all CDN, no build step)
- **Auth**: Bearer token in `Authorization` header
- **Security**: Rate limiting (120 req/60s/IP), security headers, input validation
