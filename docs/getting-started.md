# Getting Started

## Prerequisites

- **Linux** (Ubuntu/Debian recommended)
- Python 3.12+
- A [Discord bot token](https://discord.com/developers/applications)
- A ChatGPT Plus, Pro, or Team subscription (for the Codex API)

## Installation

Choose one of three deployment methods:

### Option A: .deb Package (Ubuntu/Debian)

The easiest way to install on Ubuntu/Debian. The package handles user creation, directory setup, Python venv, systemd service, and all dependencies.

```bash
# Download the latest release
curl -LO https://github.com/calmingstorm/heimdall/releases/latest/download/heimdall_amd64.deb

# Install (creates heimdall user, venv, systemd service)
sudo dpkg -i heimdall_amd64.deb
```

After install, the package prints setup instructions. Run the wizard:

```bash
sudo -u heimdall /opt/heimdall/.venv/bin/python -m src.setup wizard
```

Then start the service:

```bash
sudo systemctl start heimdall
```

**FHS layout:**

| Path | Purpose |
|------|---------|
| `/opt/heimdall/` | Application code + Python venv |
| `/etc/heimdall/config.yml` | Configuration file |
| `/etc/heimdall/.env` | Environment variables (secrets) |
| `/var/lib/heimdall/` | Runtime data (sessions, skills, knowledge, search) |
| `/var/log/heimdall/` | Log files |

The app directory uses symlinks so Heimdall sees its normal flat layout: `/opt/heimdall/config.yml` -> `/etc/heimdall/config.yml`, `/opt/heimdall/data` -> `/var/lib/heimdall`, etc.

**Upgrading:**

```bash
sudo dpkg -i heimdall_X.Y.Z_amd64.deb
sudo systemctl restart heimdall
```

Existing configuration is preserved on upgrade. To reconfigure after an upgrade:

```bash
sudo -u heimdall /opt/heimdall/.venv/bin/python -m src.setup wizard --reconfigure
```

**Uninstalling:**

```bash
sudo dpkg -r heimdall      # Remove (keeps config and data)
sudo dpkg -P heimdall      # Purge (removes everything)
```

### Option B: Docker

```bash
git clone https://github.com/Calmingstorm/heimdall.git && cd heimdall
cp .env.example .env
```

Edit `.env` and set your `DISCORD_TOKEN`, then:

```bash
docker compose up -d
```

On first boot, open `http://localhost:3939/ui/` in your browser. If Heimdall detects it hasn't been configured yet (no valid Discord token), a web setup wizard will appear automatically.

You can also pull pre-built images from GHCR:

```bash
docker pull ghcr.io/calmingstorm/heimdall:latest
# Or a specific version:
docker pull ghcr.io/calmingstorm/heimdall:v1.0.0
```

**Docker ports:**
- `3939` (host) -> `3000` (container) — web UI and health endpoint
- Health check: `http://localhost:3939/health`
- Web UI: `http://localhost:3939/ui/`

### Option C: Bare Metal (from source)

```bash
git clone https://github.com/Calmingstorm/heimdall.git && cd heimdall

# Install dependencies
pip install -e .              # Core only
pip install -e ".[all]"       # All features (PDF, browser, voice)

# Run the setup wizard
python -m src.setup wizard

# Or configure manually:
cp .env.example .env && $EDITOR .env

# Start
python -m src
```

## Setup Wizard

The interactive setup wizard walks through all required configuration. It's available in two forms:

### CLI Wizard

```bash
python -m src.setup wizard
```

Steps:
1. **Discord token** — prompts for your bot token, validates it with a test API call
2. **Codex auth** — runs the OAuth flow for the ChatGPT API (browser or headless)
3. **Hosts** — add remote servers (name, IP/hostname, SSH user)
4. **Features** — toggle browser automation, voice, ComfyUI, Claude Code
5. **Web UI token** — generate or set an API token for the management UI
6. **Write config** — generates `config.yml` and `.env` from your answers
7. **Start service** — optionally start Heimdall via systemd (if detected)

Options:
- `--headless` — use headless OAuth flow (paste callback URL instead of browser)
- `--check` — check if setup is needed (exit code 0 = configured, 1 = needs setup)
- `--reconfigure` — load existing config as defaults, then re-run the wizard
- `--config-path PATH` — custom config.yml location
- `--env-path PATH` — custom .env location

### Web Wizard (First-Boot)

When Heimdall starts and detects unconfigured state (missing or placeholder Discord token), the web UI at `http://host:3000/ui/` shows a setup wizard instead of the normal dashboard.

The web wizard has the same fields as the CLI wizard (minus Codex OAuth) and writes config files on submit. After setup, Heimdall restarts automatically and the wizard never appears again.

No authentication is required for the web wizard (it runs before auth is configured).

## Codex Authentication

Heimdall uses the ChatGPT Codex API, which requires a ChatGPT Plus/Pro/Team subscription. Authentication is separate from the setup wizard:

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
python -m src.setup --list        # List configured accounts
python -m src.setup --remove 0    # Remove account by index
```

Tokens auto-refresh at runtime. Re-run setup if the bot is offline for more than 7 days.

## Configure Hosts (Optional)

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

Heimdall can operate with no remote hosts — it runs commands locally via subprocess.

## SSH Keys (If Using Remote Hosts)

**Docker:**
```bash
mkdir -p ssh
cp ~/.ssh/id_ed25519 ssh/id_ed25519
cp ~/.ssh/known_hosts ssh/known_hosts
chmod 600 ssh/id_ed25519
# docker-compose mounts ssh/ into the container automatically
```

**Bare metal / .deb:**
```bash
# Heimdall uses ~/.ssh/ of the running user (heimdall for .deb installs)
# Copy your SSH key to the heimdall user if needed:
sudo mkdir -p /home/heimdall/.ssh
sudo cp ~/.ssh/id_ed25519 /home/heimdall/.ssh/
sudo chown -R heimdall:heimdall /home/heimdall/.ssh
sudo chmod 600 /home/heimdall/.ssh/id_ed25519
```

Note: The `.deb` package creates a system user with `/usr/sbin/nologin` shell and no home directory. For SSH access, you may need to create `/home/heimdall/.ssh/` manually.

## Verify

After starting Heimdall:

- **Health check:** `http://localhost:3000/health` (bare metal/.deb) or `http://localhost:3939/health` (Docker)
- **Web UI:** `http://localhost:3000/ui/` (bare metal/.deb) or `http://localhost:3939/ui/` (Docker)
- **Discord:** Send a message to your bot: `@Heimdall what can you do?`
- **Version:** `python -m src --version`

## Systemd Service Management

For `.deb` installs, Heimdall runs as a systemd service:

```bash
sudo systemctl start heimdall     # Start
sudo systemctl stop heimdall      # Stop
sudo systemctl restart heimdall   # Restart
sudo systemctl status heimdall    # Check status
journalctl -u heimdall -f         # Tail logs
```

The service is configured with `Restart=always` and `RestartSec=5`, so it will automatically restart on crashes or after web wizard configuration.

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
