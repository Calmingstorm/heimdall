"""Round 11: Incus deployment support and multi-deployment verification tests."""

from pathlib import Path
import subprocess

import pytest

from src.config.schema import (
    Config,
    SearchConfig,
    ToolsConfig,
    load_config,
    _substitute_env_vars,
)


class TestOllamaUrlDeploymentAgnostic:
    """ollama_url should default to localhost, not Docker-specific host."""

    def test_schema_default_uses_localhost(self):
        sc = SearchConfig()
        assert sc.ollama_url == "http://localhost:11434"
        assert "host.docker.internal" not in sc.ollama_url

    def test_config_yml_uses_env_var_for_ollama(self):
        content = Path("config.yml").read_text()
        assert "${OLLAMA_URL:-" in content
        assert "localhost:11434" in content

    def test_config_yml_no_hardcoded_docker_internal(self):
        """config.yml should not hardcode host.docker.internal."""
        content = Path("config.yml").read_text()
        assert "host.docker.internal" not in content

    def test_docker_compose_sets_ollama_url(self):
        """docker-compose.yml should set OLLAMA_URL for Docker deployments."""
        content = Path("docker-compose.yml").read_text()
        assert "OLLAMA_URL" in content
        assert "host.docker.internal" in content

    def test_config_loads_with_custom_ollama_url(self, monkeypatch):
        monkeypatch.setenv("DISCORD_TOKEN", "test-token")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("OLLAMA_URL", "http://10.0.0.5:11434")
        cfg = load_config("config.yml")
        assert cfg.search.ollama_url == "http://10.0.0.5:11434"

    def test_config_loads_with_default_ollama_url(self, monkeypatch):
        monkeypatch.setenv("DISCORD_TOKEN", "test-token")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.delenv("OLLAMA_URL", raising=False)
        cfg = load_config("config.yml")
        assert cfg.search.ollama_url == "http://localhost:11434"


class TestNoDockerOnlyAssumptions:
    """Source code should not have Docker-specific hostnames."""

    def _get_source_files(self):
        result = subprocess.run(
            ["git", "ls-files", "--", "src/"],
            capture_output=True, text=True, check=True,
        )
        return [f for f in result.stdout.strip().split("\n") if f.endswith(".py")]

    def test_no_docker_internal_in_source(self):
        """No Python source file should reference host.docker.internal."""
        for f in self._get_source_files():
            content = Path(f).read_text()
            assert "host.docker.internal" not in content, (
                f"{f} contains Docker-specific hostname"
            )

    def test_no_docker_internal_in_config_yml(self):
        content = Path("config.yml").read_text()
        assert "host.docker.internal" not in content


class TestMultiDeploymentConfig:
    """Config supports Docker, Incus, and bare metal deployments."""

    def test_ssh_paths_configurable(self):
        """SSH paths can be overridden for different deployments."""
        # Docker/Incus default
        tc = ToolsConfig()
        assert tc.ssh_key_path == "/app/.ssh/id_ed25519"
        # Bare metal override
        tc2 = ToolsConfig(
            ssh_key_path="/home/deploy/.ssh/id_ed25519",
            ssh_known_hosts_path="/home/deploy/.ssh/known_hosts",
        )
        assert tc2.ssh_key_path == "/home/deploy/.ssh/id_ed25519"

    def test_config_yml_has_ssh_path_comment(self):
        """config.yml documents SSH path alternatives for different deployments."""
        content = Path("config.yml").read_text()
        assert "Bare metal" in content or "bare metal" in content

    def test_voice_service_url_configurable(self):
        """Voice service URL can point to any host, not just Docker container."""
        from src.config.schema import VoiceConfig
        vc = VoiceConfig(voice_service_url="ws://10.0.0.5:3940/ws")
        assert vc.voice_service_url == "ws://10.0.0.5:3940/ws"

    def test_browser_cdp_url_configurable(self):
        """Browser CDP URL can point to any host."""
        from src.config.schema import BrowserConfig
        bc = BrowserConfig(cdp_url="ws://10.0.0.5:3000?token=custom")
        assert bc.cdp_url == "ws://10.0.0.5:3000?token=custom"


class TestIncusDeployScript:
    """Incus deployment script exists and is well-formed."""

    def test_incus_deploy_script_exists(self):
        script = Path("scripts/incus-deploy.sh")
        assert script.exists()

    def test_incus_deploy_script_executable(self):
        script = Path("scripts/incus-deploy.sh")
        assert script.stat().st_mode & 0o111  # has execute bit

    def test_incus_deploy_script_has_shebang(self):
        content = Path("scripts/incus-deploy.sh").read_text()
        assert content.startswith("#!/bin/bash")

    def test_incus_deploy_creates_loki_user(self):
        content = Path("scripts/incus-deploy.sh").read_text()
        assert "useradd" in content and "loki" in content

    def test_incus_deploy_creates_systemd_service(self):
        content = Path("scripts/incus-deploy.sh").read_text()
        assert "systemd" in content or "loki.service" in content

    def test_incus_deploy_installs_dependencies(self):
        content = Path("scripts/incus-deploy.sh").read_text()
        assert "pip install" in content or "pip3 install" in content

    def test_incus_deploy_pushes_ssh_keys(self):
        content = Path("scripts/incus-deploy.sh").read_text()
        assert ".ssh" in content

    def test_incus_deploy_sets_permissions(self):
        content = Path("scripts/incus-deploy.sh").read_text()
        assert "chmod" in content and "700" in content


class TestMonitorScript:
    """Monitor script supports multiple deployment types."""

    def test_monitor_script_supports_docker(self):
        content = Path("scripts/monitor.sh").read_text()
        assert "docker" in content.lower()

    def test_monitor_script_supports_incus(self):
        content = Path("scripts/monitor.sh").read_text()
        assert "incus" in content.lower()

    def test_monitor_script_supports_local(self):
        content = Path("scripts/monitor.sh").read_text()
        assert "local" in content.lower()
        assert "LOKI_LOG_FILE" in content

    def test_monitor_script_auto_detects(self):
        content = Path("scripts/monitor.sh").read_text()
        assert "detect_deploy" in content or "LOKI_DEPLOY" in content

    def test_monitor_script_configurable_deploy_type(self):
        content = Path("scripts/monitor.sh").read_text()
        assert "LOKI_DEPLOY" in content


class TestEnvExample:
    """Env example covers deployment-specific variables."""

    def test_env_example_has_ollama_url(self):
        content = Path(".env.example").read_text()
        assert "OLLAMA_URL" in content

    def test_env_example_documents_docker_vs_local(self):
        content = Path(".env.example").read_text()
        # Should mention Docker and non-Docker usage
        assert "Docker" in content or "docker" in content
        assert "localhost" in content


class TestIncusToolsIntact:
    """Verify existing Incus tool framework is intact after changes."""

    def test_incus_host_config_field(self):
        tc = ToolsConfig()
        assert hasattr(tc, "incus_host")
        assert tc.incus_host == ""

    def test_incus_host_configurable(self):
        tc = ToolsConfig(incus_host="myhost")
        assert tc.incus_host == "myhost"

    def test_incus_tools_in_registry(self):
        from src.tools.registry import TOOLS
        incus_tools = [t for t in TOOLS if t["name"].startswith("incus_")]
        assert len(incus_tools) >= 10  # 11 Incus tools
