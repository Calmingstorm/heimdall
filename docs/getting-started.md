# Getting Started

## Prerequisites

- Python 3.12+
- A [Discord bot token](https://discord.com/developers/applications)
- A ChatGPT Plus, Pro, or Team subscription (for the Codex API)

## 1. Clone and Configure

```bash
git clone https://github.com/Calmingstorm/heimdall.git && cd heimdall
cp .env.example .env
```

Edit `.env` and set your `DISCORD_TOKEN`.

## 2. Set Up Codex Authentication

Heimdall uses the ChatGPT Codex API. You need to authenticate once per account:

```bash
# Browser on the same machine:
python -m src.setup

# Headless server (paste the callback URL manually):
python -m src.setup --headless
```

### Multiple Accounts

Add additional accounts for rate limit rotation:

```bash
python -m src.setup add
python -m src.setup add --headless
```

### Managing Accounts

```bash
# List configured accounts:
python -m src.setup --list

# Remove an account:
python -m src.setup --remove 0
```

Tokens auto-refresh at runtime. Re-run setup if the bot is offline for more than 7 days.

## 3. Configure Hosts (Optional)

Edit `config.yml` to add remote infrastructure:

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

## 4. Set Up SSH Keys (If Using Remote Hosts)

```bash
cp ~/.ssh/id_ed25519 ssh/id_ed25519
cp ~/.ssh/known_hosts ssh/known_hosts
chmod 600 ssh/id_ed25519
```

Localhost commands use direct subprocess — no SSH key needed for local.

## 5. Deploy

### Docker (Recommended)

```bash
docker compose up -d
```

### Bare Metal

```bash
pip install -e .
python -m src
```

## 6. Verify

- Health check: `http://localhost:3939/health`
- Web UI: `http://localhost:3939/ui/`
- Send a message to your bot in Discord: `@Heimdall what can you do?`

## Optional Features

### Claude Code (Deep Reasoning)

If you have a Claude Max subscription, Heimdall can delegate complex tasks to `claude -p`:

```yaml
tools:
  claude_code_host: "localhost"   # or a remote host
  claude_code_user: "deploy"
  claude_code_dir: "/opt/project"
```

When `claude_code_host` is not set, the `claude_code` tool is hidden from the model entirely.

### Browser Automation

```yaml
browser:
  enabled: true
  cdp_url: "ws://localhost:3000"
```

### Image Generation (ComfyUI)

```yaml
comfyui:
  enabled: true
  url: "http://localhost:8188"
```

### Voice Support

```yaml
voice:
  enabled: true
  voice_service_url: "ws://localhost:3940/ws"
  wake_word: "heimdall"
```
