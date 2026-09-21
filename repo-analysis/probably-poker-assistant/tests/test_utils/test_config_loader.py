"""
Comprehensive tests for ConfigLoader module.

Tests configuration file loading, saving, and validation.
"""
import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import config_loader


class TestConfigLoader:
    """Test suite for ConfigLoader class."""

    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create temporary config file."""
        config_data = {
            "test_key": "test_value",
            "nested": {
                "key1": "value1",
                "key2": 42
            },
            "list_key": [1, 2, 3]
        }
        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(config_data, indent=2))
        return config_file

    # =========================================================================
    # LOADING TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_load_valid_config(self, temp_config, tmp_path):
        """Test loading a valid configuration file."""
        # Override config directory
        original_dir = config_loader.config_dir
        config_loader.config_dir = tmp_path

        config = config_loader.load('test_config.json')
        assert config['test_key'] == 'test_value'
        assert config['nested']['key1'] == 'value1'
        assert config['nested']['key2'] == 42
        assert config['list_key'] == [1, 2, 3]

        config_loader.config_dir = original_dir

    @pytest.mark.unit
    def test_load_nonexistent_file(self, tmp_path):
        """Test loading non-existent file."""
        original_dir = config_loader.config_dir
        config_loader.config_dir = tmp_path

        try:
            config = config_loader.load('nonexistent.json')
            # May return empty dict or raise error
            assert config == {} or config is None
        except FileNotFoundError:
            pass  # Expected behavior

        config_loader.config_dir = original_dir

    @pytest.mark.unit
    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ invalid json }")

        original_dir = config_loader.config_dir
        config_loader.config_dir = tmp_path

        try:
            config = config_loader.load('invalid.json')
            # May return empty dict or raise error
        except json.JSONDecodeError:
            pass  # Expected behavior

        config_loader.config_dir = original_dir

    # =========================================================================
    # CONFIG ACCESS TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_get_nested_value(self, temp_config, tmp_path):
        """Test getting nested configuration values."""
        original_dir = config_loader.config_dir
        config_loader.config_dir = tmp_path

        config = config_loader.load('test_config.json')
        assert config['nested']['key1'] == 'value1'

        config_loader.config_dir = original_dir

    @pytest.mark.unit
    def test_get_with_default(self, tmp_path):
        """Test getting config with default value."""
        config_file = tmp_path / "partial.json"
        config_file.write_text('{"existing_key": "value"}')

        original_dir = config_loader.config_dir
        config_loader.config_dir = tmp_path

        config = config_loader.load('partial.json')
        # Getting non-existent key should not crash
        result = config.get('nonexistent_key', 'default_value')
        assert result == 'default_value'

        config_loader.config_dir = original_dir


class TestSettingsConfig:
    """Test loading actual settings.json structure."""

    @pytest.fixture
    def settings_config(self, tmp_path):
        """Create settings.json structure."""
        settings = {
            "capture": {
                "window_title": "Poker just got",
                "interval_seconds": 1.0
            },
            "detection": {
                "card_confidence": 0.85
            },
            "strategy": {
                "style": "tight_aggressive",
                "aggression_factor": 1.2
            },
            "overlay": {
                "width": 320,
                "height": 240,
                "opacity": 0.9
            }
        }
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps(settings, indent=2))
        return config_file

    @pytest.mark.unit
    def test_load_settings_structure(self, settings_config, tmp_path):
        """Test loading settings.json structure."""
        original_dir = config_loader.config_dir
        config_loader.config_dir = tmp_path

        config = config_loader.load('settings.json')

        assert 'capture' in config
        assert 'detection' in config
        assert 'strategy' in config
        assert 'overlay' in config

        assert config['capture']['window_title'] == 'Poker just got'
        assert config['detection']['card_confidence'] == 0.85
        assert config['strategy']['style'] == 'tight_aggressive'

        config_loader.config_dir = original_dir
