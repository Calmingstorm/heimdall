"""Tests for discord/routing.py — keyword pre-check for task routing."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Stub out heavy discord extension before any src.discord imports trigger __init__
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.routing import is_task_by_keyword  # noqa: E402


class TestIsTaskByKeyword:
    """Test the word-boundary keyword pre-check for task routing."""

    # --- Infrastructure keywords: should always match ---

    @pytest.mark.parametrize("msg", [
        "restart the nginx service",
        "can you restart prometheus?",
        "deploy the latest changes",
        "deploy to production",
        "run the ansible playbook",
        "ssh into the server",
        "check the playbook status",
        "list docker containers",
        "what's the cpu usage?",
        "disk space on the server",
        "how's the disk looking?",
        "show me prometheus metrics",
        "open grafana dashboard",
        "check the siglos dashboard",
        "create an incus container",
        "run journalctl on the server",
        "systemctl status nginx",
    ])
    def test_infra_keywords_match(self, msg):
        assert is_task_by_keyword(msg) is True

    # --- Action directives: should always match ---

    @pytest.mark.parametrize("msg", [
        "try again",
        "ok go ahead",
        "yes, do it",
        "do it now",
        "proceed with the plan",
        "retry the command",
    ])
    def test_action_directives_match(self, msg):
        assert is_task_by_keyword(msg) is True

    # --- File/bot operations: should always match ---

    @pytest.mark.parametrize("msg", [
        "write file to /etc/hosts",
        "read file /var/log/syslog",
        "create file called test.py",
        "save note about the meeting",
        "clear chat history",
        "purge old messages",
        "wipe the conversation",
    ])
    def test_file_bot_ops_match(self, msg):
        assert is_task_by_keyword(msg) is True

    # --- Search / news: should match ---

    @pytest.mark.parametrize("msg", [
        "search for python tutorials",
        "web search for recipes",
        "search the archives",
        "search about climate change",
        "search my history",
        "look up the weather",
        "what's the headline today?",
        "any current events I should know about?",
        "news about the election",
        "latest news on AI",
        "any news from today?",
        "what's the news?",
    ])
    def test_search_news_match(self, msg):
        assert is_task_by_keyword(msg) is True

    # --- Audit: should match ---

    def test_audit_matches(self):
        assert is_task_by_keyword("show me the audit log") is True
        assert is_task_by_keyword("run an audit") is True

    # --- Case insensitivity ---

    def test_case_insensitive(self):
        assert is_task_by_keyword("RESTART the server") is True
        assert is_task_by_keyword("Deploy NOW") is True
        assert is_task_by_keyword("SSH into desktop") is True
        assert is_task_by_keyword("Docker containers") is True

    # --- FALSE POSITIVES that the old substring matching would have caught ---
    # These are the key improvements: casual chat that should NOT be forced to task.

    @pytest.mark.parametrize("msg", [
        # "log" substring: log cabin, dialogue, catalog, apologize
        "I'd love to stay in a log cabin",
        "that was a great dialogue",
        "check the catalog for me",
        "I apologize for the confusion",
        # "check" substring: various casual uses
        "check this out, it's cool",
        "let me check on that",
        "reality check moment",
        "checkbook balance",
        # "memory" substring: casual memory references
        "I have a good memory",
        "fond memory of childhood",
        "that brings back memories",
        # "service" substring: non-infra uses
        "great customer service",
        "service industry trends",
        # "skill" substring: casual references
        "that's a valuable skill to have",
        "she's very skilled at painting",
        "skillful negotiation",
        # "find" substring: casual usage
        "I find it interesting",
        "I find that hard to believe",
        # "run" substring: casual usage
        "I went for a run today",
        "in the long run it doesn't matter",
        "run of the mill conversation",
        # "status" substring: non-infra uses
        "what's the status quo?",
        "social status doesn't matter",
        # "remember" alone: casual usage (now requires context word)
        "I remember when I was young",
        "do you remember the 90s?",
        # "search" alone: casual usage (now requires multi-word)
        "I did a search of my feelings",
        "the search continues",
        # "news" alone: casual usage (now requires multi-word)
        "that's not news to me",
        "she broke the news gently",
        # "article": non-news
        "the definite article in English is 'the'",
        # "digest": non-bot
        # "my name": casual introduction
        "my name is TestUser",
    ])
    def test_false_positives_eliminated(self, msg):
        """These messages should NOT match — they're casual chat, not tasks.

        The old substring matching incorrectly routed all of these to
        the tool loop instead of the chat path.
        """
        assert is_task_by_keyword(msg) is False

    # --- Word boundary correctness ---

    def test_remember_tightened(self):
        """'remember' now only matches command-like 'remember this/that'."""
        # Should match: directive form
        assert is_task_by_keyword("remember this for later") is True
        assert is_task_by_keyword("remember that I like pizza") is True
        # Should NOT match: casual reminiscence
        assert is_task_by_keyword("I remember the good old days") is False
        assert is_task_by_keyword("I remember when I was young") is False
        assert is_task_by_keyword("do you remember me?") is False
        assert is_task_by_keyword("do you remember the 90s?") is False

    def test_search_tightened(self):
        """'search' now requires multi-word patterns."""
        assert is_task_by_keyword("search for recipes") is True
        assert is_task_by_keyword("web search") is True
        assert is_task_by_keyword("search the logs") is True
        assert is_task_by_keyword("search about python") is True
        # These should NOT match
        assert is_task_by_keyword("I'm researching the topic") is False
        assert is_task_by_keyword("stop searching for meaning") is False
        assert is_task_by_keyword("the search continues") is False
        assert is_task_by_keyword("I did a search") is False

    def test_news_tightened(self):
        """'news' now requires multi-word patterns."""
        assert is_task_by_keyword("news about the election") is True
        assert is_task_by_keyword("latest news on tech") is True
        assert is_task_by_keyword("any news today?") is True
        assert is_task_by_keyword("what's the news?") is True
        # Should NOT match casual usage
        assert is_task_by_keyword("that's not news to me") is False
        assert is_task_by_keyword("she broke the news") is False

    def test_deploy_word_boundary(self):
        assert is_task_by_keyword("deploy the app") is True
        # "redeployment" — 'deploy' is a substring but has no word boundary before it
        # Actually \bdeploy\b matches within "redeployment" — no, "redeploy" has 're' before 'deploy'
        # \b is between 're' (word) and 'd' (word) — no boundary. So \bdeploy\b does NOT match.
        assert is_task_by_keyword("the redeployment was smooth") is False

    def test_disk_word_boundary(self):
        assert is_task_by_keyword("check disk usage") is True
        assert is_task_by_keyword("the disk is full") is True
        # "disked" is not a real word, but test boundary
        assert is_task_by_keyword("she disked the field") is False

    def test_purge_word_boundary(self):
        assert is_task_by_keyword("purge the cache") is True
        assert is_task_by_keyword("expurgate the text") is False

    # --- Edge cases ---

    def test_empty_string(self):
        assert is_task_by_keyword("") is False

    def test_only_whitespace(self):
        assert is_task_by_keyword("   ") is False

    def test_keyword_at_start(self):
        assert is_task_by_keyword("restart now") is True

    def test_keyword_at_end(self):
        assert is_task_by_keyword("please restart") is True

    def test_keyword_is_entire_message(self):
        assert is_task_by_keyword("restart") is True
        assert is_task_by_keyword("deploy") is True
        assert is_task_by_keyword("retry") is True

    def test_keyword_with_punctuation(self):
        """Word boundaries should work with surrounding punctuation."""
        assert is_task_by_keyword("restart!") is True
        assert is_task_by_keyword("can you deploy?") is True
        assert is_task_by_keyword("(docker) containers") is True

    def test_multi_word_keyword_spacing(self):
        """Multi-word keywords like 'try again' need exact spacing."""
        assert is_task_by_keyword("try again please") is True
        assert is_task_by_keyword("please try again") is True
        # "try" alone should not match
        assert is_task_by_keyword("I'll try my best") is False
        # "again" alone should not match
        assert is_task_by_keyword("once again, hello") is False

    def test_current_events_singular_and_plural(self):
        assert is_task_by_keyword("any current events?") is True
        assert is_task_by_keyword("what's the current event?") is True

    def test_purely_casual_messages(self):
        """Common casual messages should never trigger the keyword fast-path."""
        casual = [
            "hey what's up?",
            "how are you doing?",
            "tell me a joke",
            "what do you think about AI?",
            "thanks for the help!",
            "that's really interesting",
            "good morning!",
            "can you explain quantum physics?",
            "what's your favorite color?",
            "I'm bored, entertain me",
        ]
        for msg in casual:
            assert is_task_by_keyword(msg) is False, f"False positive: {msg!r}"
