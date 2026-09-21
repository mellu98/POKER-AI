"""
Configuration loader for Poker AI Assistant.
Handles loading and saving JSON configuration files.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

class ConfigLoader:
    """Load and manage configuration files."""

    def __init__(self, config_dir: str = "config"):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._configs = {}

    def load(self, filename: str) -> Dict[str, Any]:
        """
        Load configuration file.

        Args:
            filename: Name of config file (e.g., 'settings.json')

        Returns:
            Configuration dictionary
        """
        filepath = self.config_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self._configs[filename] = config
        return config

    def save(self, filename: str, config: Dict[str, Any]):
        """
        Save configuration to file.

        Args:
            filename: Name of config file
            config: Configuration dictionary to save
        """
        filepath = self.config_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        self._configs[filename] = config

    def get(self, filename: str, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            filename: Config file name
            key_path: Dot-separated path (e.g., 'capture.interval_seconds')
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config = ConfigLoader()
            >>> interval = config.get('settings.json', 'capture.interval_seconds', 2.5)
        """
        if filename not in self._configs:
            self.load(filename)

        config = self._configs[filename]
        keys = key_path.split('.')

        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, filename: str, key_path: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            filename: Config file name
            key_path: Dot-separated path
            value: Value to set
        """
        if filename not in self._configs:
            self.load(filename)

        config = self._configs[filename]
        keys = key_path.split('.')

        # Navigate to parent dict
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set value
        current[keys[-1]] = value

        # Save to file
        self.save(filename, config)

# Create default config loader
config_loader = ConfigLoader()
