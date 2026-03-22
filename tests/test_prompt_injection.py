"""Tests for system prompt runtime injections in _build_system_prompt and _build_chat_system_prompt.

These methods inject up to 6 types of runtime context into the base prompt templates:
1. Persistent memory (key-value pairs from disk)
2. Learned context (cross-conversation reflections)
3. User-created skills list (full prompt only)
4. Recent tool actions per-channel (full prompt only)
5. Channel personality (from Discord channel topic)
6. Voice channel info

Previously untested — the base templates in system_prompt.py had tests, but the
injection logic in client.py did not.
"""
from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub(**overrides):
    """Create a minimal LokiBot stub with real _build_system_prompt/_build_chat_system_prompt."""
    stub = MagicMock()

    # Config: hosts, services, playbooks
    host_mock = MagicMock()
    host_mock.ssh_user = "root"
    host_mock.address = "192.168.1.3"
    stub.config.tools.hosts = {"desktop": host_mock}
    stub.config.tools.allowed_services = ["nginx", "docker"]
    stub.config.tools.allowed_playbooks = ["update.yml"]

    # Context loader
    stub.context_loader.context = "Test infra context."

    # Voice manager (default: disabled)
    stub.voice_manager = overrides.get("voice_manager", None)

    # Tool executor memory (per-user scoped)
    stub.tool_executor._load_memory_for_user = MagicMock(
        return_value=overrides.get("memory", {})
    )

    # Reflector
    if "reflector" in overrides:
        stub.reflector = overrides["reflector"]
    else:
        stub.reflector = MagicMock()
        stub.reflector.get_prompt_section = MagicMock(
            return_value=overrides.get("learned", "")
        )

    # Skill manager
    if "skill_manager" in overrides:
        stub.skill_manager = overrides["skill_manager"]
    else:
        stub.skill_manager = MagicMock()
        stub.skill_manager.list_skills = MagicMock(
            return_value=overrides.get("skills", [])
        )

    # Recent actions
    stub._recent_actions = overrides.get("recent_actions", {})
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = overrides.get("expiry", 3600)

    # Bind the real methods
    stub._build_system_prompt = LokiBot._build_system_prompt.__get__(stub)
    stub._build_chat_system_prompt = LokiBot._build_chat_system_prompt.__get__(stub)

    return stub


def _make_channel(channel_id="123", topic=None):
    """Create a mock Discord channel."""
    channel = MagicMock()
    channel.id = channel_id
    channel.topic = topic
    return channel


# ---------------------------------------------------------------------------
# Memory injection
# ---------------------------------------------------------------------------

class TestMemoryInjection:
    """Persistent memory should be injected into both full and chat prompts."""

    def test_full_prompt_includes_memory(self):
        stub = _make_bot_stub(memory={"owner": "Aaron", "timezone": "ET"})
        prompt = stub._build_system_prompt()
        assert "## Persistent Memory" in prompt
        assert "- **owner**: Aaron" in prompt
        assert "- **timezone**: ET" in prompt

    def test_chat_prompt_includes_memory(self):
        stub = _make_bot_stub(memory={"owner": "Aaron"})
        prompt = stub._build_chat_system_prompt()
        assert "## Persistent Memory" in prompt
        assert "- **owner**: Aaron" in prompt

    def test_full_prompt_no_memory_section_when_empty(self):
        stub = _make_bot_stub(memory={})
        prompt = stub._build_system_prompt()
        assert "## Persistent Memory" not in prompt

    def test_chat_prompt_no_memory_section_when_empty(self):
        stub = _make_bot_stub(memory={})
        prompt = stub._build_chat_system_prompt()
        assert "## Persistent Memory" not in prompt

    def test_memory_format_multiple_entries(self):
        stub = _make_bot_stub(memory={"a": "1", "b": "2", "c": "3"})
        prompt = stub._build_system_prompt()
        # All entries present
        assert "- **a**: 1" in prompt
        assert "- **b**: 2" in prompt
        assert "- **c**: 3" in prompt

    def test_memory_calls_load_memory(self):
        stub = _make_bot_stub(memory={"k": "v"})
        stub._build_system_prompt()
        stub.tool_executor._load_memory_for_user.assert_called()

    def test_chat_prompt_calls_load_memory(self):
        stub = _make_bot_stub(memory={"k": "v"})
        stub._build_chat_system_prompt()
        stub.tool_executor._load_memory_for_user.assert_called()


# ---------------------------------------------------------------------------
# Learned context injection
# ---------------------------------------------------------------------------

class TestLearnedContextInjection:
    """Learned context from the reflector should be injected into both prompts."""

    def test_full_prompt_includes_learned(self):
        stub = _make_bot_stub(learned="## Learned Behaviors\n- Be concise")
        prompt = stub._build_system_prompt()
        assert "## Learned Behaviors" in prompt
        assert "- Be concise" in prompt

    def test_chat_prompt_includes_learned(self):
        stub = _make_bot_stub(learned="## Learned Behaviors\n- No emojis")
        prompt = stub._build_chat_system_prompt()
        assert "## Learned Behaviors" in prompt
        assert "- No emojis" in prompt

    def test_full_prompt_no_learned_when_empty(self):
        stub = _make_bot_stub(learned="")
        prompt = stub._build_system_prompt()
        assert "## Learned" not in prompt

    def test_chat_prompt_no_learned_when_empty(self):
        stub = _make_bot_stub(learned="")
        prompt = stub._build_chat_system_prompt()
        assert "## Learned" not in prompt

    def test_no_reflector_attribute_does_not_crash_full(self):
        stub = _make_bot_stub()
        # Remove reflector entirely to simulate it not being configured
        del stub.reflector
        prompt = stub._build_system_prompt()
        assert isinstance(prompt, str)
        assert "## Learned" not in prompt

    def test_no_reflector_attribute_does_not_crash_chat(self):
        stub = _make_bot_stub()
        del stub.reflector
        prompt = stub._build_chat_system_prompt()
        assert isinstance(prompt, str)
        assert "## Learned" not in prompt

    def test_learned_calls_get_prompt_section(self):
        stub = _make_bot_stub(learned="some content")
        stub._build_system_prompt()
        stub.reflector.get_prompt_section.assert_called()


# ---------------------------------------------------------------------------
# Skills injection (full prompt only)
# ---------------------------------------------------------------------------

class TestSkillsInjection:
    """User-created skills should only appear in the full system prompt."""

    def test_full_prompt_includes_skills(self):
        skills = [
            {"name": "ping_host", "description": "Ping a host and return latency"},
            {"name": "backup_db", "description": "Backup the database"},
        ]
        stub = _make_bot_stub(skills=skills)
        prompt = stub._build_system_prompt()
        assert "## User-Created Skills" in prompt
        assert "- `ping_host`: Ping a host and return latency" in prompt
        assert "- `backup_db`: Backup the database" in prompt

    def test_full_prompt_no_skills_section_when_empty(self):
        stub = _make_bot_stub(skills=[])
        prompt = stub._build_system_prompt()
        assert "## User-Created Skills" not in prompt

    def test_chat_prompt_does_not_include_skills(self):
        """Chat prompt should NOT inject skills — they require tools to execute."""
        skills = [{"name": "test_skill", "description": "A test skill"}]
        stub = _make_bot_stub(skills=skills)
        prompt = stub._build_chat_system_prompt()
        assert "## User-Created Skills" not in prompt
        assert "test_skill" not in prompt

    def test_no_skill_manager_attribute_does_not_crash(self):
        stub = _make_bot_stub()
        del stub.skill_manager
        prompt = stub._build_system_prompt()
        assert isinstance(prompt, str)
        assert "## User-Created Skills" not in prompt

    def test_skills_calls_list_skills(self):
        skills = [{"name": "s1", "description": "d1"}]
        stub = _make_bot_stub(skills=skills)
        stub._build_system_prompt()
        stub.skill_manager.list_skills.assert_called()


# ---------------------------------------------------------------------------
# Recent actions injection (full prompt only)
# ---------------------------------------------------------------------------

class TestRecentActionsInjection:
    """Recent tool executions should only appear in the full prompt, per-channel."""

    def test_full_prompt_includes_recent_actions(self):
        now = time.time()
        actions = {"123": [(now, "- [14:30] `check_disk`(host=server) → OK (120ms)")]}
        stub = _make_bot_stub(recent_actions=actions)
        channel = _make_channel("123")
        prompt = stub._build_system_prompt(channel=channel)
        assert "## Recent Actions" in prompt
        assert "check_disk" in prompt

    def test_full_prompt_no_actions_when_empty(self):
        stub = _make_bot_stub(recent_actions={})
        channel = _make_channel("123")
        prompt = stub._build_system_prompt(channel=channel)
        assert "## Recent Actions" not in prompt

    def test_full_prompt_no_actions_without_channel(self):
        """Without a channel, recent actions section is not injected."""
        now = time.time()
        actions = {"123": [(now, "- action entry")]}
        stub = _make_bot_stub(recent_actions=actions)
        prompt = stub._build_system_prompt(channel=None)
        assert "## Recent Actions" not in prompt

    def test_chat_prompt_does_not_include_recent_actions(self):
        """Chat prompt should NOT inject recent actions."""
        now = time.time()
        actions = {"123": [(now, "- action entry")]}
        stub = _make_bot_stub(recent_actions=actions)
        channel = _make_channel("123")
        prompt = stub._build_chat_system_prompt(channel=channel)
        assert "## Recent Actions" not in prompt

    def test_channel_isolation(self):
        """Actions from other channels should not appear."""
        now = time.time()
        actions = {
            "chan-A": [(now, "- action from A")],
            "chan-B": [(now, "- action from B")],
        }
        stub = _make_bot_stub(recent_actions=actions)
        channel = _make_channel("chan-A")
        prompt = stub._build_system_prompt(channel=channel)
        assert "action from A" in prompt
        assert "action from B" not in prompt

    def test_expired_actions_filtered(self):
        """Actions older than expiry should not appear."""
        old_ts = time.time() - 7200  # 2 hours ago
        new_ts = time.time()
        actions = {
            "123": [
                (old_ts, "- old action"),
                (new_ts, "- new action"),
            ]
        }
        stub = _make_bot_stub(recent_actions=actions, expiry=3600)
        channel = _make_channel("123")
        prompt = stub._build_system_prompt(channel=channel)
        assert "new action" in prompt
        assert "old action" not in prompt

    def test_all_expired_no_section(self):
        """If all actions are expired, no Recent Actions section appears."""
        old_ts = time.time() - 7200
        actions = {"123": [(old_ts, "- old action")]}
        stub = _make_bot_stub(recent_actions=actions, expiry=3600)
        channel = _make_channel("123")
        prompt = stub._build_system_prompt(channel=channel)
        assert "## Recent Actions" not in prompt

    def test_actions_capped_at_10(self):
        """Only the last 10 non-expired actions are included."""
        now = time.time()
        actions = {"123": [(now, f"- action {i}") for i in range(15)]}
        stub = _make_bot_stub(recent_actions=actions)
        channel = _make_channel("123")
        prompt = stub._build_system_prompt(channel=channel)
        # Last 10 should be present (actions 5-14)
        assert "action 14" in prompt
        assert "action 5" in prompt
        # First 5 should be trimmed by [-10:]
        assert "action 4" not in prompt


# ---------------------------------------------------------------------------
# Channel personality injection
# ---------------------------------------------------------------------------

class TestChannelPersonalityInjection:
    """Channel topic should be injected as personality in both prompts."""

    def test_full_prompt_includes_personality(self):
        stub = _make_bot_stub()
        channel = _make_channel(topic="Respond like a pirate")
        prompt = stub._build_system_prompt(channel=channel)
        assert "## Channel Personality" in prompt
        assert "Respond like a pirate" in prompt
        assert "keeping all other rules intact" in prompt

    def test_chat_prompt_includes_personality(self):
        stub = _make_bot_stub()
        channel = _make_channel(topic="Be extra formal")
        prompt = stub._build_chat_system_prompt(channel=channel)
        assert "## Channel Personality" in prompt
        assert "Be extra formal" in prompt

    def test_no_personality_without_channel(self):
        stub = _make_bot_stub()
        prompt = stub._build_system_prompt(channel=None)
        assert "## Channel Personality" not in prompt

    def test_no_personality_when_topic_empty(self):
        stub = _make_bot_stub()
        channel = _make_channel(topic="")
        prompt = stub._build_system_prompt(channel=channel)
        assert "## Channel Personality" not in prompt

    def test_no_personality_when_topic_whitespace(self):
        stub = _make_bot_stub()
        channel = _make_channel(topic="   ")
        prompt = stub._build_system_prompt(channel=channel)
        assert "## Channel Personality" not in prompt

    def test_no_personality_when_topic_none(self):
        stub = _make_bot_stub()
        channel = _make_channel(topic=None)
        prompt = stub._build_system_prompt(channel=channel)
        assert "## Channel Personality" not in prompt

    def test_topic_stripped(self):
        """Leading/trailing whitespace in topic should be stripped."""
        stub = _make_bot_stub()
        channel = _make_channel(topic="  pirate mode  ")
        prompt = stub._build_system_prompt(channel=channel)
        assert "pirate mode" in prompt
        # Should not have leading/trailing whitespace in the injected topic
        assert "  pirate mode  " not in prompt

    def test_channel_without_topic_attribute(self):
        """Channels without a topic attribute should not crash."""
        stub = _make_bot_stub()
        channel = MagicMock(spec=[])  # No topic attribute
        channel.id = "123"
        prompt = stub._build_system_prompt(channel=channel)
        assert "## Channel Personality" not in prompt


# ---------------------------------------------------------------------------
# Voice info injection
# ---------------------------------------------------------------------------

class TestVoiceInfoInjection:
    """Voice manager state should be reflected in the prompt."""

    def test_full_prompt_no_voice_manager(self):
        stub = _make_bot_stub(voice_manager=None)
        prompt = stub._build_system_prompt()
        assert "Voice support is not enabled" in prompt

    def test_chat_prompt_no_voice_manager(self):
        stub = _make_bot_stub(voice_manager=None)
        prompt = stub._build_chat_system_prompt()
        assert "Voice support is not enabled" in prompt

    def test_full_prompt_voice_connected(self):
        vm = MagicMock()
        vm.is_connected = True
        vm.current_channel.name = "general-voice"
        stub = _make_bot_stub(voice_manager=vm)
        prompt = stub._build_system_prompt()
        assert "general-voice" in prompt
        assert "spoken aloud" in prompt

    def test_chat_prompt_voice_connected(self):
        vm = MagicMock()
        vm.is_connected = True
        vm.current_channel.name = "hangout"
        stub = _make_bot_stub(voice_manager=vm)
        prompt = stub._build_chat_system_prompt()
        assert "hangout" in prompt
        assert "spoken aloud" in prompt

    def test_full_prompt_voice_not_connected(self):
        vm = MagicMock()
        vm.is_connected = False
        stub = _make_bot_stub(voice_manager=vm)
        prompt = stub._build_system_prompt()
        assert "not in a voice channel" in prompt
        assert "/voice join" in prompt

    def test_full_prompt_voice_connected_no_channel(self):
        """When connected but current_channel is None, should show 'unknown'."""
        vm = MagicMock()
        vm.is_connected = True
        vm.current_channel = None
        stub = _make_bot_stub(voice_manager=vm)
        prompt = stub._build_system_prompt()
        assert "unknown" in prompt


# ---------------------------------------------------------------------------
# Base template content
# ---------------------------------------------------------------------------

class TestBaseTemplateContent:
    """Verify the base template content is present in the built prompts."""

    def test_full_prompt_includes_host_info(self):
        stub = _make_bot_stub()
        prompt = stub._build_system_prompt()
        assert "`desktop`: root@192.168.1.3" in prompt

    def test_full_prompt_includes_services(self):
        stub = _make_bot_stub()
        prompt = stub._build_system_prompt()
        assert "`nginx`" in prompt
        assert "`docker`" in prompt

    def test_full_prompt_includes_playbooks(self):
        stub = _make_bot_stub()
        prompt = stub._build_system_prompt()
        assert "`update.yml`" in prompt

    def test_full_prompt_includes_context(self):
        stub = _make_bot_stub()
        prompt = stub._build_system_prompt()
        assert "Test infra context." in prompt

    def test_full_prompt_includes_identity(self):
        stub = _make_bot_stub()
        prompt = stub._build_system_prompt()
        assert "Loki" in prompt

    def test_chat_prompt_includes_identity(self):
        stub = _make_bot_stub()
        prompt = stub._build_chat_system_prompt()
        assert "Loki" in prompt

    def test_chat_prompt_omits_hosts(self):
        """Chat prompt should not include host details."""
        stub = _make_bot_stub()
        prompt = stub._build_chat_system_prompt()
        assert "192.168.1.3" not in prompt

    def test_chat_prompt_omits_services(self):
        stub = _make_bot_stub()
        prompt = stub._build_chat_system_prompt()
        assert "Allowed Services" not in prompt

    def test_chat_prompt_omits_playbooks(self):
        stub = _make_bot_stub()
        prompt = stub._build_chat_system_prompt()
        assert "Allowed Playbooks" not in prompt


# ---------------------------------------------------------------------------
# Combined injections
# ---------------------------------------------------------------------------

class TestCombinedInjections:
    """Verify that multiple injections work together correctly."""

    def test_full_prompt_all_injections(self):
        """Full prompt with all 5 optional injections active."""
        now = time.time()
        vm = MagicMock()
        vm.is_connected = True
        vm.current_channel.name = "voice-chan"
        stub = _make_bot_stub(
            memory={"owner": "Aaron"},
            learned="## Learned Behaviors\n- Be concise",
            skills=[{"name": "test_skill", "description": "A test"}],
            recent_actions={"ch1": [(now, "- [14:00] `check_disk` → OK")]},
            voice_manager=vm,
        )
        channel = _make_channel("ch1", topic="Be helpful")
        prompt = stub._build_system_prompt(channel=channel)
        assert "## Persistent Memory" in prompt
        assert "## Learned Behaviors" in prompt
        assert "## User-Created Skills" in prompt
        assert "## Recent Actions" in prompt
        assert "## Channel Personality" in prompt
        assert "voice-chan" in prompt

    def test_chat_prompt_all_injections(self):
        """Chat prompt with all applicable injections (no skills, no actions)."""
        vm = MagicMock()
        vm.is_connected = True
        vm.current_channel.name = "vc"
        stub = _make_bot_stub(
            memory={"k": "v"},
            learned="## Learned Behaviors\n- Rule",
            voice_manager=vm,
        )
        channel = _make_channel("ch1", topic="Friendly tone")
        prompt = stub._build_chat_system_prompt(channel=channel)
        assert "## Persistent Memory" in prompt
        assert "## Learned Behaviors" in prompt
        assert "## Channel Personality" in prompt
        assert "## User-Created Skills" not in prompt
        assert "## Recent Actions" not in prompt

    def test_injection_order_in_full_prompt(self):
        """Injections should appear after the base template, in order:
        memory → learned → skills → actions → personality."""
        now = time.time()
        stub = _make_bot_stub(
            memory={"k": "v"},
            learned="## Learned Behaviors\n- x",
            skills=[{"name": "s", "description": "d"}],
            recent_actions={"123": [(now, "- action")]},
        )
        channel = _make_channel("123", topic="personality text")
        prompt = stub._build_system_prompt(channel=channel)

        # Find positions of each section
        mem_pos = prompt.index("## Persistent Memory")
        learned_pos = prompt.index("## Learned Behaviors")
        skills_pos = prompt.index("## User-Created Skills")
        actions_pos = prompt.index("## Recent Actions")
        personality_pos = prompt.index("## Channel Personality")

        assert mem_pos < learned_pos < skills_pos < actions_pos < personality_pos

    def test_full_prompt_no_injections(self):
        """Full prompt with no optional context should still be valid."""
        stub = _make_bot_stub()
        prompt = stub._build_system_prompt()
        assert "Loki" in prompt
        assert "## Persistent Memory" not in prompt
        assert "## Learned" not in prompt
        assert "## User-Created Skills" not in prompt
        assert "## Recent Actions" not in prompt
        assert "## Channel Personality" not in prompt

    def test_chat_prompt_no_injections(self):
        stub = _make_bot_stub()
        prompt = stub._build_chat_system_prompt()
        assert "Loki" in prompt
        assert "## Persistent Memory" not in prompt
        assert "## Learned" not in prompt
        assert "## Channel Personality" not in prompt
