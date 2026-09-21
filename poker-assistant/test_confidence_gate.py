"""Focused, dependency-free tests for the vision confidence safety gate."""
import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ui.confidence_gate import confidence_block_reasons


class ConfidenceGateTests(unittest.TestCase):
    def test_accepts_complete_confident_state(self):
        self.assertEqual(
            confidence_block_reasons(
                {
                    "hole": ["As", "Kh"], "board": ["Qd", "Jh", "2c"],
                    "to_call": 20, "position": "BTN", "stage": "flop",
                    "confidence": {"hole": 0.9, "cards": 0.9, "to_call": 0.9,
                                   "position": 0.9, "stage": 0.9},
                }
            ),
            [],
        )

    def test_blocks_all_uncertainty_sources(self):
        reasons = confidence_block_reasons(
            {
                "is_uncertain": True,
                "uncertainty_reasons": ["OCR unreadable", "OCR unreadable"],
                "estimated_fields": ["to_call"],
                "confidence": {"hole": 0.2, "cards": 0.1, "to_call": 0.4,
                               "position": 0.5, "stage": 0.0},
            }
        )
        self.assertEqual(
            reasons,
            [
                "state is uncertain",
                "OCR unreadable",
                "to_call is estimated",
                "low confidence: hole",
                "low confidence: cards",
                "low confidence: to_call",
                "low confidence: position",
                "low confidence: stage",
            ],
        )

    def test_blocks_estimated_to_call_source(self):
        self.assertEqual(
            confidence_block_reasons({"to_call_source": "estimated"}),
            ["to_call is estimated"],
        )

    def test_controller_waits_without_calling_equity_or_engine(self):
        calls = {"equity": 0, "engine": 0}

        def calculate_equity(*_args, **_kwargs):
            calls["equity"] += 1
            return 0.5

        class FakeOverlay:
            def __init__(self):
                self.updates = []

            def update(self, **fields):
                self.updates.append(fields)

        class FakeEngine:
            def recommend(self, **_kwargs):
                calls["engine"] += 1
                return {"action": "c"}

        dependencies = {
            "assistant_engine": types.SimpleNamespace(AssistantEngine=FakeEngine),
            "confidence_gate": sys.modules["ui.confidence_gate"],
            "equity_service": types.SimpleNamespace(
                calculate_equity=calculate_equity,
                get_hand_strength_class=lambda *_args: None,
            ),
            "state_extractor": types.SimpleNamespace(get_extractor=lambda *_args: None),
            "overlay": types.SimpleNamespace(PokerOverlay=FakeOverlay),
        }
        with patch.dict(sys.modules, dependencies):
            spec = importlib.util.spec_from_file_location(
                "controller_confidence_gate_test", Path("ui/controller.py")
            )
            controller_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(controller_module)

        controller = controller_module.AssistantController.__new__(
            controller_module.AssistantController
        )
        controller.mode = "manual"
        controller._manual_state = {
            "hole": ["As", "Kh"],
            "board": ["Qd", "Jh", "2c"],
            "pot": 120,
            "to_call": 20,
            "position": "BTN",
            "stage": "flop",
            "is_uncertain": True,
        }
        controller.overlay = FakeOverlay()
        controller.engine = FakeEngine()

        controller._tick()

        self.assertEqual(calls, {"equity": 0, "engine": 0})
        self.assertEqual(
            controller.overlay.updates,
            [{
                "status": "WAIT | FLOP | BTN | state is uncertain",
                "hand": "As Kh",
                "board": "Qd Jh 2c",
                "equity": 0.0,
                "action": "WAIT",
                "sizing": "",
            }],
        )

    def test_extractor_marks_fallback_to_call_as_estimated(self):
        tree = ast.parse(Path("vision/state_extractor.py").read_text())
        assignments = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "confidence"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "to_call"
                for target in node.targets
            )
        ]
        self.assertTrue(
            any(isinstance(node.value, ast.Constant) and node.value.value == 0.0
                for node in assignments)
        )

    def test_api_scripts_require_environment_key_without_embedded_secret(self):
        for filename in (
            "test_llm_api.py",
            "test_llm_api_synthetic.py",
            "test_api_direct.py",
        ):
            source = Path(filename).read_text()
            self.assertIn('os.getenv("OPENROUTER_API_KEY")', source)
            self.assertNotIn("sk-or-v1-", source)

    def test_config_has_no_api_key_value(self):
        config = Path("config.yaml").read_text()
        self.assertIn('api_key: ""', config)


if __name__ == "__main__":
    unittest.main()
