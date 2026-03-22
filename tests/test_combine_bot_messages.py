"""Tests for combine_bot_messages() — bot buffer code block merging (Round 2)."""
from __future__ import annotations

from src.discord.client import combine_bot_messages


class TestCombineBotMessages:
    """Unit tests for the combine_bot_messages helper."""

    def test_empty_list(self):
        assert combine_bot_messages([]) == ""

    def test_single_message(self):
        assert combine_bot_messages(["hello"]) == "hello"

    def test_plain_text_messages(self):
        """Regular messages without code blocks join with double newline."""
        result = combine_bot_messages(["hello", "world"])
        assert result == "hello\n\nworld"

    def test_split_code_block_no_extra_blank_line(self):
        """A code block split across two messages should not get a blank line."""
        msg1 = "```bash\nline1"
        msg2 = "line2\n```"
        result = combine_bot_messages([msg1, msg2])
        assert result == "```bash\nline1\nline2\n```"

    def test_split_code_block_three_messages(self):
        """Code block split across three messages — all joined with \\n."""
        msg1 = "```bash\nline1"
        msg2 = "line2"
        msg3 = "line3\n```"
        result = combine_bot_messages([msg1, msg2, msg3])
        assert result == "```bash\nline1\nline2\nline3\n```"

    def test_adjacent_code_blocks_merged(self):
        """Two adjacent code blocks should merge into one."""
        msg1 = "```bash\npart1\n```"
        msg2 = "```bash\npart2\n```"
        result = combine_bot_messages([msg1, msg2])
        assert "```bash\npart1\npart2\n```" == result

    def test_adjacent_code_blocks_no_language(self):
        """Adjacent code blocks without language markers merge too."""
        msg1 = "```\npart1\n```"
        msg2 = "```\npart2\n```"
        result = combine_bot_messages([msg1, msg2])
        assert "```\npart1\npart2\n```" == result

    def test_text_before_code_block(self):
        """Text before a code block is preserved normally."""
        msg1 = "Here is the script:"
        msg2 = "```bash\necho hello\n```"
        result = combine_bot_messages([msg1, msg2])
        assert result == "Here is the script:\n\n```bash\necho hello\n```"

    def test_text_after_code_block(self):
        """Text after a code block is preserved normally."""
        msg1 = "```bash\necho hello\n```"
        msg2 = "Run this please"
        result = combine_bot_messages([msg1, msg2])
        assert result == "```bash\necho hello\n```\n\nRun this please"

    def test_separate_code_blocks_with_text_between(self):
        """Code blocks separated by text should NOT merge."""
        msg1 = "File 1:\n```python\ncode1\n```"
        msg2 = "File 2:\n```python\ncode2\n```"
        result = combine_bot_messages([msg1, msg2])
        # Both blocks should remain separate (text between them)
        assert "code1" in result
        assert "code2" in result
        assert result.count("```") == 4  # two open + two close

    def test_three_adjacent_code_blocks_merged(self):
        """Three adjacent code blocks all merge into one."""
        msg1 = "```bash\npart1\n```"
        msg2 = "```bash\npart2\n```"
        msg3 = "```bash\npart3\n```"
        result = combine_bot_messages([msg1, msg2, msg3])
        assert "```bash\npart1\npart2\npart3\n```" == result

    def test_split_code_preserves_indentation(self):
        """Indentation inside split code blocks is preserved."""
        msg1 = "```python\ndef foo():"
        msg2 = "    return 42\n```"
        result = combine_bot_messages([msg1, msg2])
        assert "    return 42" in result
        assert result.count("```") == 2  # one open, one close

    def test_heredoc_in_split_code_block(self):
        """Heredoc content inside a split code block stays intact."""
        msg1 = "```bash\ncat <<'EOF'"
        msg2 = "hello world"
        msg3 = "EOF\n```"
        result = combine_bot_messages([msg1, msg2, msg3])
        # All three joined with \n (not \n\n) since inside code block
        assert "cat <<'EOF'\nhello world\nEOF" in result

    def test_mixed_text_and_code(self):
        """Complex mix: text, then split code, then text."""
        parts = [
            "Here's a script:",
            "```bash\necho hello",
            "echo world\n```",
            "That should work!",
        ]
        result = combine_bot_messages(parts)
        assert "Here's a script:" in result
        assert "echo hello\necho world" in result
        assert "That should work!" in result
