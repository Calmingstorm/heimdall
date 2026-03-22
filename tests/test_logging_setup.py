"""Tests for logging/logger.py coverage gaps.

Targets uncovered lines: 10-31 (setup_logging function).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.logging.logger import setup_logging, get_logger


class TestSetupLogging:
    def test_creates_log_directory(self, tmp_path: Path):
        """setup_logging creates the log directory (lines 10-11)."""
        log_dir = tmp_path / "logs"
        assert not log_dir.exists()
        setup_logging(level="DEBUG", log_dir=str(log_dir))
        assert log_dir.exists()

    def test_sets_root_logger_level(self, tmp_path: Path):
        """setup_logging sets the root logger level (lines 13-14)."""
        # Clean up any existing handlers first
        root = logging.getLogger("ansiblex")
        original_handlers = root.handlers.copy()
        root.handlers.clear()

        try:
            setup_logging(level="WARNING", log_dir=str(tmp_path / "logs"))
            assert root.level == logging.WARNING
        finally:
            # Restore original state
            root.handlers = original_handlers

    def test_adds_console_handler(self, tmp_path: Path):
        """setup_logging adds a StreamHandler (lines 21-23)."""
        root = logging.getLogger("ansiblex")
        original_handlers = root.handlers.copy()
        root.handlers.clear()

        try:
            setup_logging(level="INFO", log_dir=str(tmp_path / "logs"))
            stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)
                              and not isinstance(h, logging.FileHandler)]
            assert len(stream_handlers) >= 1
        finally:
            root.handlers = original_handlers

    def test_adds_rotating_file_handler(self, tmp_path: Path):
        """setup_logging adds a RotatingFileHandler (lines 25-31)."""
        from logging.handlers import RotatingFileHandler

        root = logging.getLogger("ansiblex")
        original_handlers = root.handlers.copy()
        root.handlers.clear()

        try:
            log_dir = tmp_path / "logs"
            setup_logging(level="INFO", log_dir=str(log_dir))
            file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
            assert len(file_handlers) >= 1
            # Verify the log file path
            assert "ansiblex.log" in file_handlers[0].baseFilename
        finally:
            root.handlers = original_handlers

    def test_invalid_level_defaults_to_info(self, tmp_path: Path):
        """Invalid level string defaults to INFO via getattr fallback (line 14)."""
        root = logging.getLogger("ansiblex")
        original_handlers = root.handlers.copy()
        root.handlers.clear()

        try:
            setup_logging(level="BOGUS", log_dir=str(tmp_path / "logs"))
            assert root.level == logging.INFO
        finally:
            root.handlers = original_handlers


class TestGetLogger:
    def test_returns_namespaced_logger(self):
        """get_logger returns logger under ansiblex namespace."""
        logger = get_logger("test_module")
        assert logger.name == "ansiblex.test_module"
