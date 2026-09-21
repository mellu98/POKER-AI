"""
Logging utility for Poker AI Assistant.
Handles file and console logging with different levels.
"""
import logging
import os
from datetime import datetime
from pathlib import Path

class PokerLogger:
    """Custom logger for poker AI system."""

    def __init__(self, name: str, log_dir: str = "logs"):
        """
        Initialize logger.

        Args:
            name: Logger name (usually __name__)
            log_dir: Directory for log files
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # File handler for errors
        error_handler = logging.FileHandler(
            log_path / "errors.log",
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_format = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_format)

        # File handler for all logs
        info_handler = logging.FileHandler(
            log_path / "detection_accuracy.log",
            encoding='utf-8'
        )
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(error_format)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)

        # Add handlers
        self.logger.addHandler(error_handler)
        self.logger.addHandler(info_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)

# Create default logger
logger = PokerLogger("poker_ai")
