"""
Comprehensive tests for AnchorManager module.

Tests anchor configuration loading and region calculation.
"""
import pytest
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.capture.anchor_manager import AnchorManager


class TestAnchorManager:
    """Test suite for AnchorManager class."""

    @pytest.fixture
    def mock_config_loader(self):
        """Create mock config_loader."""
        return MagicMock()

    @pytest.fixture
    def anchor_manager(self, tmp_path):
        """Create AnchorManager instance with temp directory."""
        with patch('src.capture.anchor_manager.config_loader') as mock_loader:
            mock_loader.load.side_effect = FileNotFoundError()
            manager = AnchorManager(anchor_dir=str(tmp_path / "anchors"))
        return manager

    # =========================================================================
    # CONFIGURATION LOADING TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_load_config(self, tmp_path):
        """Test loading anchor configuration."""
        config = {
            "active_anchor": "test_anchor",
            "regions": {
                "hole_cards": {"off_x": 100, "off_y": 200, "w": 150, "h": 100},
                "community_cards": {"off_x": 400, "off_y": 300, "w": 350, "h": 100}
            }
        }

        with patch('src.capture.anchor_manager.config_loader') as mock_loader:
            mock_loader.load.return_value = config
            manager = AnchorManager(anchor_dir=str(tmp_path / "anchors"))

        assert manager.active_anchor_name == "test_anchor"
        assert "hole_cards" in manager.relative_regions
        assert "community_cards" in manager.relative_regions

    @pytest.mark.unit
    def test_load_config_missing_file(self, tmp_path):
        """Test loading non-existent config file."""
        with patch('src.capture.anchor_manager.config_loader') as mock_loader:
            mock_loader.load.side_effect = FileNotFoundError()
            manager = AnchorManager(anchor_dir=str(tmp_path / "anchors"))

        # Should handle gracefully - no anchor set
        assert manager.active_anchor_name is None
        assert manager.relative_regions == {}

    # =========================================================================
    # REGION CALCULATION TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_get_absolute_regions(self, tmp_path):
        """Test calculating absolute region positions from anchor."""
        config = {
            "active_anchor": "test_anchor",
            "regions": {
                "hole_cards": {"off_x": 100, "off_y": 200, "w": 150, "h": 100},
                "pot_amount": {"off_x": 50, "off_y": 50, "w": 100, "h": 30}
            }
        }

        with patch('src.capture.anchor_manager.config_loader') as mock_loader:
            mock_loader.load.return_value = config
            manager = AnchorManager(anchor_dir=str(tmp_path / "anchors"))

        anchor_pos = (500, 400)  # Anchor found at this position
        regions = manager.get_absolute_regions(anchor_pos)

        # Regions should be offset by anchor position
        assert regions is not None
        assert 'hole_cards' in regions
        # hole_cards: (500+100, 400+200, 150, 100) = (600, 600, 150, 100)
        assert regions['hole_cards'] == (600, 600, 150, 100)
        # pot_amount: (500+50, 400+50, 100, 30) = (550, 450, 100, 30)
        assert regions['pot_amount'] == (550, 450, 100, 30)

    @pytest.mark.unit
    def test_get_regions_without_anchor(self, anchor_manager):
        """Test getting regions when no anchor is set."""
        anchor_manager.active_anchor_name = None
        anchor_manager.relative_regions = {}

        regions = anchor_manager.get_absolute_regions((0, 0))
        # Should return empty dict
        assert regions == {}

    # =========================================================================
    # ANCHOR DETECTION TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_find_anchor_mocked(self, tmp_path):
        """Test anchor detection with mocked template matching."""
        config = {
            "active_anchor": "test_anchor",
            "regions": {}
        }

        with patch('src.capture.anchor_manager.config_loader') as mock_loader:
            mock_loader.load.return_value = config
            manager = AnchorManager(anchor_dir=str(tmp_path / "anchors"))

        # Set up a mock anchor image
        manager.active_anchor_img = np.zeros((50, 50), dtype=np.uint8)

        # Mock screen image
        mock_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Mock cv2.matchTemplate to return a match
        with patch('cv2.matchTemplate') as mock_match:
            mock_result = np.zeros((1030, 1870), dtype=np.float32)
            mock_result[50, 100] = 0.9  # High confidence match
            mock_match.return_value = mock_result

            with patch('cv2.minMaxLoc', return_value=(0.1, 0.9, (0, 0), (100, 50))):
                result = manager.find_anchor(mock_screen)
                # Should return position (x, y, w, h)
                assert result == (100, 50, 50, 50)

    @pytest.mark.unit
    def test_find_anchor_no_template(self, anchor_manager):
        """Test anchor detection when no anchor image loaded."""
        anchor_manager.active_anchor_img = None

        mock_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = anchor_manager.find_anchor(mock_screen)
        assert result is None

    @pytest.mark.unit
    def test_find_anchor_low_confidence(self, tmp_path):
        """Test anchor detection with low confidence match."""
        config = {
            "active_anchor": "test_anchor",
            "regions": {}
        }

        with patch('src.capture.anchor_manager.config_loader') as mock_loader:
            mock_loader.load.return_value = config
            manager = AnchorManager(anchor_dir=str(tmp_path / "anchors"))

        manager.active_anchor_img = np.zeros((50, 50), dtype=np.uint8)
        mock_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

        with patch('cv2.matchTemplate') as mock_match:
            mock_result = np.zeros((1030, 1870), dtype=np.float32)
            mock_match.return_value = mock_result

            with patch('cv2.minMaxLoc', return_value=(0.1, 0.5, (0, 0), (100, 50))):
                # 0.5 < 0.8 threshold
                result = manager.find_anchor(mock_screen)
                assert result is None


class TestAnchorManagerIntegration:
    """Integration tests for AnchorManager."""

    @pytest.mark.integration
    def test_full_workflow(self, tmp_path):
        """Test complete anchor workflow."""
        config = {
            "active_anchor": "test",
            "regions": {
                "hole_cards": {"off_x": 100, "off_y": 200, "w": 150, "h": 100}
            }
        }

        with patch('src.capture.anchor_manager.config_loader') as mock_loader:
            mock_loader.load.return_value = config
            manager = AnchorManager(anchor_dir=str(tmp_path / "anchors"))

        assert manager.active_anchor_name == "test"
        assert "hole_cards" in manager.relative_regions

        # Test getting absolute regions
        regions = manager.get_absolute_regions((500, 300))
        assert regions['hole_cards'] == (600, 500, 150, 100)

    @pytest.mark.integration
    def test_add_relative_region(self, tmp_path):
        """Test adding a new relative region."""
        with patch('src.capture.anchor_manager.config_loader') as mock_loader:
            mock_loader.load.side_effect = FileNotFoundError()
            manager = AnchorManager(anchor_dir=str(tmp_path / "anchors"))

        # Add a region with anchor at (500, 400) and region at (600, 500, 150, 100)
        manager.add_relative_region(
            "test_region",
            anchor_pos=(500, 400),
            region_rect=(600, 500, 150, 100)
        )

        assert "test_region" in manager.relative_regions
        region = manager.relative_regions["test_region"]
        assert region["off_x"] == 100  # 600 - 500
        assert region["off_y"] == 100  # 500 - 400
        assert region["w"] == 150
        assert region["h"] == 100
