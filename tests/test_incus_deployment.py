"""Round 11: Incus deployment support and multi-deployment verification tests."""

from pathlib import Path
import subprocess

import pytest

from src.config.schema import ToolsConfig


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


_SCRIPTS_DIR_EXISTS = Path("scripts").exists()


@pytest.mark.skipif(not _SCRIPTS_DIR_EXISTS, reason="scripts/ not present (stripped for public repo)")
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

    def test_incus_deploy_creates_heimdall_user(self):
        content = Path("scripts/incus-deploy.sh").read_text()
        assert "useradd" in content and "heimdall" in content

    def test_incus_deploy_creates_systemd_service(self):
        content = Path("scripts/incus-deploy.sh").read_text()
        assert "systemd" in content or "heimdall.service" in content

    def test_incus_deploy_installs_dependencies(self):
        content = Path("scripts/incus-deploy.sh").read_text()
        assert "pip install" in content or "pip3 install" in content

    def test_incus_deploy_pushes_ssh_keys(self):
        content = Path("scripts/incus-deploy.sh").read_text()
        assert ".ssh" in content

    def test_incus_deploy_sets_permissions(self):
        content = Path("scripts/incus-deploy.sh").read_text()
        assert "chmod" in content and "700" in content


@pytest.mark.skipif(not _SCRIPTS_DIR_EXISTS, reason="scripts/ not present (stripped for public repo)")
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
        assert "HEIMDALL_LOG_FILE" in content

    def test_monitor_script_auto_detects(self):
        content = Path("scripts/monitor.sh").read_text()
        assert "detect_deploy" in content or "HEIMDALL_DEPLOY" in content

    def test_monitor_script_configurable_deploy_type(self):
        content = Path("scripts/monitor.sh").read_text()
        assert "HEIMDALL_DEPLOY" in content


class TestEnvExample:
    """Env example covers deployment-specific variables."""

    def test_env_example_documents_docker_vs_local(self):
        content = Path(".env.example").read_text()
        # Should mention Docker and non-Docker usage
        assert "Docker" in content or "docker" in content
        assert "localhost" in content


