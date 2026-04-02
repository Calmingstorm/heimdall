# Packaging and Releases

How to build `.deb` packages, Docker images, and create releases for Heimdall.

## Overview

Heimdall uses three distribution channels:

| Format | Tool | Registry |
|--------|------|----------|
| `.deb` package | [nfpm](https://nfpm.goreleaser.com/) | GitHub Releases |
| Docker image | Docker + Buildx | GHCR (`ghcr.io/calmingstorm/heimdall`) |
| Source | git | GitHub |

All packaging is automated via the GitHub Actions release workflow. Push a `v*` tag to trigger a release.

## Building a .deb Package Locally

### Prerequisites

Install [nfpm](https://nfpm.goreleaser.com/install/):

```bash
# From GitHub releases
curl -sfL https://github.com/goreleaser/nfpm/releases/latest/download/nfpm_linux_amd64.tar.gz | tar xz -C /usr/local/bin nfpm

# Or via Go
go install github.com/goreleaser/nfpm/v2/cmd/nfpm@latest
```

### Build

```bash
# Default version (1.0.0)
nfpm package --config packaging/nfpm.yml --packager deb --target dist/

# Custom version
VERSION=2.1.0 nfpm package --config packaging/nfpm.yml --packager deb --target dist/
```

The `.deb` file will be written to `dist/`.

### What the Package Contains

| Source | Destination |
|--------|-------------|
| `src/` | `/opt/heimdall/src/` |
| `ui/` | `/opt/heimdall/ui/` |
| `pyproject.toml` | `/opt/heimdall/pyproject.toml` |
| `config.yml` | `/opt/heimdall/config.yml` |
| `.env.example` | `/opt/heimdall/.env.example` |
| `packaging/heimdall.service` | `/usr/lib/systemd/system/heimdall.service` |

Directories created: `/etc/heimdall/`, `/var/lib/heimdall/`, `/var/log/heimdall/`.

### Package Scripts

**postinstall.sh** runs after `dpkg -i`:
- Creates `heimdall` system user and group
- Creates FHS directories and data subdirectories
- Copies config template to `/etc/heimdall/` (only on fresh install, preserves on upgrade)
- Creates symlinks from `/opt/heimdall/` to FHS paths
- Creates Python venv and runs `pip install`
- Sets ownership and permissions (`.env` gets `0600`)
- Enables the systemd service

**preremove.sh** runs before `dpkg -r`:
- Stops the service (if running)
- Disables the service (if enabled)
- Reloads systemd daemon

### Dependencies

The `.deb` requires:
- `python3.12`
- `python3.12-venv`
- `openssh-client`

Recommends: `python3-pip`

## Building a Docker Image Locally

```bash
docker build -t heimdall:local .
```

The Dockerfile installs all dependencies (including optional ones like PDF, browser, voice) and runs as the `heimdall` user.

## Creating a Release

Releases are fully automated via GitHub Actions.

### Triggering a Release

```bash
git tag v1.2.3
git push origin v1.2.3
```

This triggers `.github/workflows/release.yml`, which runs three jobs:

### Job 1: build-deb

1. Checks out the code
2. Extracts version from the git tag (`v1.2.3` -> `1.2.3`)
3. Updates `pyproject.toml` version to match the tag
4. Installs nfpm
5. Builds the `.deb` package with `VERSION` env var
6. Uploads the `.deb` as a build artifact

### Job 2: build-docker

1. Checks out the code
2. Updates `pyproject.toml` version to match the tag
3. Logs into GHCR with `GITHUB_TOKEN`
4. Sets up Docker Buildx (for layer caching)
5. Builds and pushes the Docker image with three tags:
   - `ghcr.io/calmingstorm/heimdall:latest`
   - `ghcr.io/calmingstorm/heimdall:v1.2.3`
   - `ghcr.io/calmingstorm/heimdall:1.2.3`

### Job 3: create-release

Runs after both build jobs complete:

1. Downloads the `.deb` artifact
2. Generates a changelog from commits since the previous tag
3. Creates a GitHub Release with:
   - The `.deb` file attached
   - Changelog in the release body
   - Auto-generated contributor notes

### Required Permissions

The workflow needs:
- `contents: write` — to create GitHub Releases
- `packages: write` — to push Docker images to GHCR

These are configured in the workflow file. No additional secrets are needed beyond `GITHUB_TOKEN`.

## Version Management

The version flows from git tag through the build system:

```
git tag v1.2.3
  -> release workflow extracts "1.2.3"
    -> sed updates pyproject.toml to version = "1.2.3"
    -> VERSION=1.2.3 passed to nfpm (overrides ${VERSION:-1.0.0} default)
    -> Docker image built with updated pyproject.toml
```

At runtime, `src/version.py` resolves the version with a 3-tier cascade:

1. `importlib.metadata.version("heimdall")` — works in .deb installs (pip-installed package)
2. Parse `pyproject.toml` directly — works in development mode
3. Fallback to `"0.0.0-dev"` — if neither source is available

The version appears in:
- `python -m src --version` — CLI flag
- `/api/status` — REST API response
- Web UI dashboard — next to uptime

## Packaging Files

| File | Purpose |
|------|---------|
| `packaging/nfpm.yml` | nfpm configuration for .deb builds |
| `packaging/heimdall.service` | Systemd unit file |
| `packaging/postinstall.sh` | Post-install script (user, dirs, venv, symlinks) |
| `packaging/preremove.sh` | Pre-remove script (stop, disable service) |
| `.github/workflows/release.yml` | GitHub Actions release workflow |
| `src/version.py` | Runtime version resolution |
| `src/packaging/validate.py` | Validation functions for packaging artifacts (used by tests) |

## Systemd Service

The service file (`packaging/heimdall.service`) is installed to `/usr/lib/systemd/system/`.

Key directives:
- `Type=simple` — Heimdall is a long-running process
- `User=heimdall`, `Group=heimdall` — runs as dedicated system user
- `WorkingDirectory=/opt/heimdall`
- `ExecStart=/opt/heimdall/.venv/bin/python -m src`
- `EnvironmentFile=/etc/heimdall/.env` — loads secrets from .env
- `Restart=always`, `RestartSec=5` — auto-restart on any exit (including web wizard SIGTERM)

Security hardening:
- `NoNewPrivileges=yes`
- `ProtectSystem=strict` — read-only filesystem (except ReadWritePaths)
- `ProtectHome=yes`
- `PrivateTmp=yes`
- `ReadWritePaths=/var/lib/heimdall /var/log/heimdall`
- `LimitNOFILE=65536` — for aiohttp connections

## Testing Packaging

The test suite includes validators for all packaging artifacts:

```bash
# Run all packaging tests
python -m pytest tests/test_packaging.py tests/test_packaging_integration.py -q

# Run version management tests
python -m pytest tests/test_version_management.py -q

# Run setup wizard tests
python -m pytest tests/test_setup_wizard.py tests/test_setup_wizard_web.py -q
```

These tests validate:
- Systemd service file syntax and required directives
- Shell script syntax (`bash -n`), operations, and variable expansion
- nfpm config structure, dependencies, and file references
- Workflow YAML structure, jobs, actions, and version handling
- Wizard config generation, token validation, and end-to-end flow
- Cross-component consistency (ports, paths, versions match everywhere)
