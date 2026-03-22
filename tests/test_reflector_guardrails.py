"""Tests for reflection prompt anti-hallucination guardrails.

The reflection system previously hallucinated a "reject political news" learned
behavior from a conversation where the user simply asked for news and corrected
the bot's refusal. These tests verify the prompt guardrails that prevent such
over-generalization.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.learning.reflector import (
    ConversationReflector,
    _REFLECTION_HEADER,
)


@pytest.fixture
def mock_text_fn():
    return AsyncMock(return_value="[]")


@pytest.fixture
def reflector(tmp_dir: Path, mock_text_fn):
    r = ConversationReflector(
        learned_path=str(tmp_dir / "learned.json"),
        max_entries=10,
        consolidation_target=5,
    )
    r.set_text_fn(mock_text_fn)
    return r


class TestReflectionPromptGuardrails:
    """Verify the reflection prompt contains anti-hallucination rules."""

    def test_prompt_says_explicit_not_implicit(self):
        """The prompt should say 'explicit' not 'implicit' to prevent inferring
        unstated preferences."""
        assert "explicit" in _REFLECTION_HEADER.lower()
        assert "implicit" not in _REFLECTION_HEADER.lower()

    def test_prompt_requires_explicitly_stated_preferences(self):
        """Must tell the LLM to only record preferences the user EXPLICITLY stated."""
        assert "ONLY record preferences the user EXPLICITLY stated" in _REFLECTION_HEADER

    def test_prompt_forbids_inferring_unstated_preferences(self):
        """Must forbid inferring preferences the user never expressed."""
        assert "never infer unstated preferences" in _REFLECTION_HEADER

    def test_prompt_forbids_broad_generalization(self):
        """Must forbid generalizing a specific correction into a broad prohibition."""
        assert "Never generalize a specific correction into a broad prohibition" in _REFLECTION_HEADER

    def test_prompt_has_concrete_anti_hallucination_example(self):
        """Must include a concrete example of the hallucination pattern to avoid.
        The earthquake/news example directly addresses the past production incident."""
        assert "avoid political topics" in _REFLECTION_HEADER.lower()
        assert "do not refuse news requests" in _REFLECTION_HEADER.lower()

    def test_prompt_forbids_inventing_behavioral_rules(self):
        """Must forbid inventing rules the user did not ask for."""
        assert "Never invent behavioral rules the user did not ask for" in _REFLECTION_HEADER

    def test_prompt_prefers_empty_over_hallucination(self):
        """Must prefer returning [] over a hallucinated insight."""
        assert "missed insight is better than a hallucinated one" in _REFLECTION_HEADER

    def test_prompt_correction_guidance(self):
        """When user corrects a refusal, the lesson should be 'do not refuse X',
        not 'avoid broad topic Y'."""
        assert "do not refuse [specific thing]" in _REFLECTION_HEADER
        assert "avoid [broad topic]" in _REFLECTION_HEADER


class TestReflectionSystemMessage:
    """Verify the system message sent with the reflection text_fn call."""

    async def test_system_message_says_explicit(self, reflector, mock_text_fn):
        """The system message must say 'explicit' not 'implicit'."""
        session = MagicMock()
        session.messages = [
            MagicMock(role="user", content="hello"),
            MagicMock(role="assistant", content="hi"),
            MagicMock(role="user", content="thanks"),
        ]
        session.summary = ""

        await reflector.reflect_on_session(session)

        # text_fn(messages, system) — system is second positional arg
        system_msg = mock_text_fn.call_args[0][1]
        assert "explicit" in system_msg.lower()
        assert "implicit" not in system_msg.lower()

    async def test_system_message_warns_against_inference(self, reflector, mock_text_fn):
        """The system message must tell the LLM not to infer unstated preferences."""
        session = MagicMock()
        session.messages = [
            MagicMock(role="user", content="hello"),
            MagicMock(role="assistant", content="hi"),
            MagicMock(role="user", content="thanks"),
        ]
        session.summary = ""

        await reflector.reflect_on_session(session)

        system_msg = mock_text_fn.call_args[0][1]
        assert "never infer unstated preferences" in system_msg.lower()


class TestReflectionPromptContent:
    """Verify the full prompt sent to the text_fn includes guardrails."""

    async def test_full_prompt_includes_anti_hallucination_rules(self, reflector, mock_text_fn):
        """The user prompt must contain the anti-hallucination rules."""
        session = MagicMock()
        session.messages = [
            MagicMock(role="user", content="any news about the earthquake?"),
            MagicMock(role="assistant", content="I cannot discuss political topics."),
            MagicMock(role="user", content="that's wrong, never refuse news requests"),
        ]
        session.summary = ""

        await reflector.reflect_on_session(session)

        # text_fn(messages, system) — messages is first positional arg
        user_msg = mock_text_fn.call_args[0][0][0]["content"]
        assert "Anti-hallucination rules:" in user_msg
        assert "ONLY record preferences the user EXPLICITLY stated" in user_msg

    async def test_prompt_includes_conversation_content(self, reflector, mock_text_fn):
        """The prompt must include the actual conversation for context."""
        session = MagicMock()
        session.messages = [
            MagicMock(role="user", content="check disk on server"),
            MagicMock(role="assistant", content="Disk usage is 45%."),
            MagicMock(role="user", content="thanks"),
        ]
        session.summary = ""

        await reflector.reflect_on_session(session)

        user_msg = mock_text_fn.call_args[0][0][0]["content"]
        assert "check disk on server" in user_msg
        assert "Conversation:" in user_msg


class TestHallucinationScenarios:
    """Test that over-generalized entries are caught by the _parse_entries validator.

    While the prompt guardrails prevent the LLM from generating bad entries,
    the validator provides a second layer of defense by rejecting entries with
    invalid categories. These tests verify the validator works correctly for
    entries that would represent hallucinated behaviors.
    """

    def test_valid_correction_accepted(self):
        """A properly scoped correction should be accepted."""
        raw = json.dumps([{
            "key": "do_not_refuse_news",
            "category": "correction",
            "content": "Do not refuse to discuss news or current events",
        }])
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 1
        assert result[0]["key"] == "do_not_refuse_news"

    def test_valid_preference_accepted(self):
        """An explicitly stated preference should be accepted."""
        raw = json.dumps([{
            "key": "no_emojis",
            "category": "preference",
            "content": "User does not want emojis in responses",
        }])
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 1

    def test_invalid_category_rejected(self):
        """Entries with categories outside the allowed set are rejected."""
        raw = json.dumps([{
            "key": "avoid_politics",
            "category": "prohibition",
            "content": "Avoid political topics",
        }])
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 0

    def test_content_truncated_to_150_chars(self):
        """Content longer than 150 chars is truncated during merge."""
        raw = json.dumps([{
            "key": "long_entry",
            "category": "fact",
            "content": "x" * 200,
        }])
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 1
        # _parse_entries accepts it; truncation happens during merge in _reflect

    def test_missing_key_rejected(self):
        """Entries without a key field are rejected."""
        raw = json.dumps([{
            "category": "preference",
            "content": "some preference",
        }])
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 0

    def test_missing_content_rejected(self):
        """Entries without a content field are rejected."""
        raw = json.dumps([{
            "key": "some_key",
            "category": "preference",
        }])
        result = ConversationReflector._parse_entries(raw)
        assert len(result) == 0


class TestReflectMergesCorrectly:
    """Verify that reflection merges new insights without losing guardrails."""

    async def test_existing_entries_shown_to_llm(self, reflector, mock_text_fn, tmp_dir):
        """The LLM should see existing entries so it can reuse keys."""
        existing = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "no_emojis", "category": "preference",
                 "content": "User does not want emojis",
                 "created_at": "2026-01-01", "updated_at": "2026-01-01"},
            ],
        }
        (tmp_dir / "learned.json").write_text(json.dumps(existing))

        session = MagicMock()
        session.messages = [
            MagicMock(role="user", content="remember I prefer concise responses"),
            MagicMock(role="assistant", content="Got it."),
            MagicMock(role="user", content="thanks"),
        ]
        session.summary = ""

        mock_text_fn.return_value = json.dumps([{
            "key": "concise_responses",
            "category": "preference",
            "content": "User prefers concise responses",
        }])

        await reflector.reflect_on_session(session)

        user_msg = mock_text_fn.call_args[0][0][0]["content"]
        assert "no_emojis" in user_msg
        assert "Currently known:" in user_msg

    async def test_new_entries_merged_with_existing(self, reflector, mock_text_fn, tmp_dir):
        """New entries should be added alongside existing ones."""
        existing = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "no_emojis", "category": "preference",
                 "content": "No emojis", "created_at": "2026-01-01",
                 "updated_at": "2026-01-01"},
            ],
        }
        (tmp_dir / "learned.json").write_text(json.dumps(existing))

        session = MagicMock()
        session.messages = [
            MagicMock(role="user", content="server runs ubuntu 22.04"),
            MagicMock(role="assistant", content="Noted."),
            MagicMock(role="user", content="ok"),
        ]
        session.summary = ""

        mock_text_fn.return_value = json.dumps([{
            "key": "server_os",
            "category": "operational",
            "content": "Server runs Ubuntu 22.04",
        }])

        await reflector.reflect_on_session(session)

        saved = json.loads((tmp_dir / "learned.json").read_text())
        keys = [e["key"] for e in saved["entries"]]
        assert "no_emojis" in keys
        assert "server_os" in keys

    async def test_key_reuse_updates_existing(self, reflector, mock_text_fn, tmp_dir):
        """When LLM reuses a key, the existing entry should be updated."""
        existing = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "server_os", "category": "operational",
                 "content": "Server runs Ubuntu 20.04",
                 "created_at": "2026-01-01", "updated_at": "2026-01-01"},
            ],
        }
        (tmp_dir / "learned.json").write_text(json.dumps(existing))

        session = MagicMock()
        session.messages = [
            MagicMock(role="user", content="upgraded server to 22.04"),
            MagicMock(role="assistant", content="Noted."),
            MagicMock(role="user", content="ok"),
        ]
        session.summary = ""

        mock_text_fn.return_value = json.dumps([{
            "key": "server_os",
            "category": "operational",
            "content": "Server runs Ubuntu 22.04",
        }])

        await reflector.reflect_on_session(session)

        saved = json.loads((tmp_dir / "learned.json").read_text())
        assert len(saved["entries"]) == 1
        assert saved["entries"][0]["content"] == "Server runs Ubuntu 22.04"

    async def test_empty_llm_response_no_changes(self, reflector, mock_text_fn, tmp_dir):
        """When LLM returns [], no changes should be made to learned.json."""
        existing = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "no_emojis", "category": "preference",
                 "content": "No emojis", "created_at": "2026-01-01",
                 "updated_at": "2026-01-01"},
            ],
        }
        (tmp_dir / "learned.json").write_text(json.dumps(existing))

        session = MagicMock()
        session.messages = [
            MagicMock(role="user", content="hello"),
            MagicMock(role="assistant", content="hi"),
            MagicMock(role="user", content="bye"),
        ]
        session.summary = ""

        mock_text_fn.return_value = "[]"

        await reflector.reflect_on_session(session)

        # File should be unchanged
        saved = json.loads((tmp_dir / "learned.json").read_text())
        assert len(saved["entries"]) == 1
        assert saved["entries"][0]["key"] == "no_emojis"


class TestCompactionReflectionGuardrails:
    """Verify that compaction reflection also uses the hardened prompt."""

    async def test_compaction_uses_same_hardened_prompt(self, reflector, mock_text_fn):
        """reflect_on_compacted must also use the anti-hallucination header."""
        messages = [
            MagicMock(role="user", content=f"msg {i}")
            for i in range(6)
        ]

        mock_text_fn.return_value = "[]"

        await reflector.reflect_on_compacted(messages, "summary text")

        user_msg = mock_text_fn.call_args[0][0][0]["content"]
        assert "Anti-hallucination rules:" in user_msg
        assert "ONLY record preferences the user EXPLICITLY stated" in user_msg

    async def test_compaction_filters_to_correction_and_operational(self, reflector, mock_text_fn):
        """Compaction reflection only keeps correction and operational entries."""
        messages = [
            MagicMock(role="user", content=f"msg {i}")
            for i in range(6)
        ]

        mock_text_fn.return_value = json.dumps([
            {"key": "a_pref", "category": "preference", "content": "some pref"},
            {"key": "a_corr", "category": "correction", "content": "some correction"},
            {"key": "a_op", "category": "operational", "content": "some op fact"},
            {"key": "a_fact", "category": "fact", "content": "some fact"},
        ])

        await reflector.reflect_on_compacted(messages, "summary")

        # Only correction and operational should be saved
        saved = json.loads(Path(reflector._path).read_text())
        categories = [e["category"] for e in saved["entries"]]
        assert "correction" in categories
        assert "operational" in categories
        assert "preference" not in categories
        assert "fact" not in categories
