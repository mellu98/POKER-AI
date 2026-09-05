"""Quick local test for LLM vision extractor (no API call)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vision"))

from llm_vision_extractor import (
    LLMVisionExtractor,
    _encode_frame_to_base64,
    _extract_json_from_text,
    _normalize_card,
)

import numpy as np


def test_init():
    ext = LLMVisionExtractor(api_key="dummy")
    assert ext.api_key == "dummy"
    print("INIT OK")


def test_encode():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    b64 = _encode_frame_to_base64(frame)
    assert len(b64) > 100
    print(f"ENCODE OK: {len(b64)} chars")


def test_json_extract_plain():
    text = '{"hole": ["As", "Kh"], "pot": 120}'
    result = _extract_json_from_text(text)
    assert result["hole"] == ["As", "Kh"]
    print(f"JSON EXTRACT OK: {result}")


def test_json_extract_markdown():
    text = '```json\n{"hole": ["As", "Kh"]}\n```'
    result = _extract_json_from_text(text)
    assert result["hole"] == ["As", "Kh"]
    print(f"MARKDOWN JSON OK: {result}")


def test_frame_hash():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    h1 = LLMVisionExtractor._frame_hash(frame)
    h2 = LLMVisionExtractor._frame_hash(frame)
    assert h1 == h2
    assert len(h1) == 64
    print(f"FRAME HASH OK: {h1}")


def test_normalize_card():
    assert _normalize_card("TD") == "Td"
    assert _normalize_card("as") == "As"
    assert _normalize_card("7h") == "7h"
    print("NORMALIZE CARD OK")


def test_skip_logic():
    ext = LLMVisionExtractor(api_key="dummy", cooldown_seconds=10.0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # First call should NOT skip
    assert not ext._should_skip(frame)
    # Manually set cache
    ext._last_call_time = __import__("time").time()
    ext._last_frame_hash = ext._frame_hash(frame)
    # Same frame within cooldown SHOULD skip
    assert ext._should_skip(frame)
    print("SKIP LOGIC OK")


if __name__ == "__main__":
    test_init()
    test_encode()
    test_json_extract_plain()
    test_json_extract_markdown()
    test_frame_hash()
    test_normalize_card()
    test_skip_logic()
    print("\nAll local tests passed!")
