# Configuration

## Environment Variables (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `DISCORD_TOKEN` | Yes | Discord bot token |
| `WEBHOOK_SECRET` | No | Secret for webhook signature verification |
| `ALLOWED_WEBHOOK_IDS` | No | Comma-separated webhook IDs to bypass bot check |
| `TZ` | No | Timezone (default: `UTC`) |

## Config File (config.yml)

Uses `${VAR}` for required env vars and `${VAR:-default}` for optional ones.

### Discord

```yaml
discord:
  token: "${DISCORD_TOKEN}"
  allowed_users: []      # User IDs (empty = allow all)
  channels: []           # Channel IDs (empty = all channels)
  respond_to_bots: false # Process messages from other bots
  require_mention: false # Only respond when @mentioned
```

### LLM Backend

```yaml
openai_codex:
  enabled: true
  model: "gpt-5.4"
  max_tokens: 4096
  credentials_path: "./data/codex_auth.json"
```

### Tools

```yaml
tools:
  enabled: true
  ssh_key_path: "~/.ssh/id_ed25519"
  ssh_known_hosts_path: "~/.ssh/known_hosts"
  hosts: {}
  command_timeout_seconds: 300
```

### Bot Interaction Modes

```yaml
discord:
  respond_to_bots: true   # Process messages from other bots/webhooks
  require_mention: true    # Only respond when @mentioned
```

- `respond_to_bots: true` — allows bot-to-bot communication (self-messages always ignored)
- `require_mention: true` — bot only responds when @mentioned (DMs bypass this)
- Both can be combined: bot responds to other bots only when @mentioned

### Permissions

```yaml
permissions:
  default_tier: "user"
  tiers:
    admin:
      allowed_tools: ["*"]
    user:
      allowed_tools: ["run_command", "read_file", "web_search"]
      blocked_tools: ["purge_messages"]
```

### All Sections

| Section | Purpose |
|---------|---------|
| `timezone` | IANA timezone string |
| `discord` | Bot token, user/channel access, bot interaction |
| `openai_codex` | Model, credentials |
| `tools` | SSH, hosts, timeouts, Claude Code |
| `context` | Context directory, system prompt token budget |
| `sessions` | History limits, persistence |
| `webhook` | Gitea/Grafana webhook routing |
| `learning` | Lesson extraction settings |
| `search` | Search database path |
| `logging` | Log level and directory |
| `voice` | Voice channel support |
| `browser` | Headless Chromium automation |
| `comfyui` | Image generation |
| `monitoring` | Infrastructure checks and alerts |
| `permissions` | Per-user tool access tiers |
| `web` | Management UI and API |
