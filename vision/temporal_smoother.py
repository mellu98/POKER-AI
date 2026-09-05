"""Temporal consensus smoothing for vision-extracted poker table states.

A single misread frame should not be enough to change the assistant's
recommendation. :class:`TemporalSmoother` keeps a short ring buffer of raw
states and returns a consensus state where each field is accepted only after
enough recent frames agree.
"""
from __future__ import annotations

from collections import deque
from statistics import median
from typing import Any, Optional


class TemporalSmoother:
    """Smooth raw vision states using per-field temporal consensus.

    Parameters
    ----------
    window_size:
        Maximum number of recent frames to keep in the consensus window.
    agreement_threshold:
        Minimum fraction of ``window_size`` frames that must agree on a discrete
        value before it is accepted.
    min_samples:
        Minimum number of agreeing frames required for any consensus decision.
    """

    def __init__(
        self,
        window_size: int,
        agreement_threshold: float,
        min_samples: int,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 <= agreement_threshold <= 1.0:
            raise ValueError("agreement_threshold must be between 0.0 and 1.0")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")
        self.window_size = window_size
        self.agreement_threshold = agreement_threshold
        self.min_samples = min_samples
        self._buffer: deque[dict] = deque(maxlen=window_size)

    def update(self, raw_state: dict) -> dict:
        """Add ``raw_state`` to the window and return a consensus state.

        The returned dict is a shallow copy of ``raw_state`` with smoothed
        values for ``hole``, ``board``, ``pot``, ``to_call``, ``stack``,
        ``position`` and ``stage``. Confidence and uncertainty metadata are
        updated to reflect the consensus stability.
        """
        self._buffer.append(raw_state)
        state = dict(raw_state)

        # Cards: per-slot majority vote.
        hole, hole_stable = self._smooth_card_group("hole", num_slots=2)
        board, board_stable = self._smooth_card_group("board", num_slots=5)
        state["hole"] = hole
        state["board"] = board

        # Numeric values: median + coefficient-of-variation confidence.
        pot, pot_conf, pot_stable = self._smooth_numeric("pot")
        to_call, tc_conf, tc_stable = self._smooth_numeric("to_call")
        stack, stack_conf, stack_stable = self._smooth_numeric("stack")
        state["pot"] = pot
        state["to_call"] = to_call
        if "stack" in state:
            state["stack"] = stack

        # Position: majority vote.
        position, pos_conf, pos_stable = self._smooth_position()
        state["position"] = position

        # Stage: derived from the consensus board length.
        stage, _stage_conf, stage_stable = self._derive_stage(board)
        state["stage"] = stage

        # Build confidence map, keeping any unrelated keys from the raw state.
        confidence = dict(state.get("confidence") or {})
        hole_conf = 1.0 if (hole_stable and len(hole) == 2) else 0.0
        board_conf = (
            1.0
            if (board_stable and len(board) in (0, 3, 4, 5))
            else 0.0
        )
        confidence["hole"] = hole_conf
        confidence["cards"] = min(hole_conf, board_conf)
        confidence["pot"] = pot_conf
        confidence["to_call"] = tc_conf
        confidence["stack"] = stack_conf
        confidence["position"] = pos_conf
        confidence["stage"] = board_conf if stage_stable else 0.0
        state["confidence"] = confidence

        # Collect uncertainty reasons. Keep reasons already produced upstream.
        reasons = list(state.get("uncertainty_reasons") or [])
        if not hole_stable or len(hole) != 2:
            reasons.append("hole unstable across frames")
        if not board_stable:
            reasons.append("board unstable across frames")
        if not pot_stable:
            reasons.append("pot unstable across frames")
        if not tc_stable:
            reasons.append("to_call unstable across frames")
        if not stack_stable:
            reasons.append("stack unstable across frames")
        if not pos_stable:
            reasons.append("position unstable across frames")
        if not stage_stable:
            reasons.append("stage unstable across frames")

        # De-duplicate while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for reason in reasons:
            if reason and reason not in seen:
                seen.add(reason)
                deduped.append(reason)

        state["uncertainty_reasons"] = deduped
        state["is_uncertain"] = bool(deduped)
        return state

    def is_stable(self, field_value_history: list[Any]) -> tuple[bool, Any]:
        """Return ``(is_consensus, consensus_value)`` for a field history.

        Discrete values use majority vote; numeric values use the median. The
        result is stable only when enough frames agree, respecting both
        ``agreement_threshold`` and ``min_samples``.
        """
        values = [v for v in field_value_history if v is not None]
        if not values:
            return False, None

        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            if len(values) < self.min_samples:
                return False, None
            return True, median(values)

        # Discrete majority vote.
        counts: dict[Any, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1

        winner, winner_count = max(counts.items(), key=lambda item: item[1])
        required = self.agreement_threshold * self.window_size
        if winner_count >= required and winner_count >= self.min_samples:
            return True, winner
        return False, None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _required_agreement(self) -> float:
        return self.agreement_threshold * self.window_size

    def _smooth_card_group(self, field: str, num_slots: int) -> tuple[list[str], bool]:
        """Return the consensus card list and whether the group is stable."""
        slot_histories: list[list[Optional[str]]] = [[] for _ in range(num_slots)]
        for raw in self._buffer:
            cards = raw.get(field) or []
            for i in range(num_slots):
                card = cards[i] if i < len(cards) else None
                slot_histories[i].append(card)

        result: list[Optional[str]] = []
        attempted = 0
        stable_count = 0
        for hist in slot_histories:
            non_empty = [v for v in hist if v not in (None, "")]
            if non_empty:
                attempted += 1
                stable, card = self.is_stable(non_empty)
                if stable:
                    stable_count += 1
                    result.append(card)
                else:
                    result.append(None)
            else:
                result.append(None)

        filtered = [c for c in result if c is not None]
        is_stable_group = attempted == 0 or stable_count == attempted
        return filtered, is_stable_group

    def _smooth_numeric(self, field: str) -> tuple[Any, float, bool]:
        """Return ``(value, confidence, is_stable)`` for a numeric field."""
        latest = self._buffer[-1] if self._buffer else {}
        values = [
            v
            for v in (raw.get(field) for raw in self._buffer)
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]

        if not values:
            return latest.get(field, 0), 0.0, False

        value = median(values)
        confidence = self._numeric_confidence(values)
        stable = len(values) >= self.min_samples and confidence >= 0.7
        return value, confidence, stable

    def _numeric_confidence(self, values: list[float]) -> float:
        """Confidence based on the coefficient of variation, clamped to [0, 1]."""
        n = len(values)
        if n == 0:
            return 0.0
        if n == 1:
            return 1.0

        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std = variance ** 0.5
        cv = std / (abs(mean) + 1e-9)
        return max(0.0, min(1.0, 1.0 - cv))

    def _smooth_position(self) -> tuple[str, float, bool]:
        """Return ``(position, confidence, is_stable)``."""
        latest = self._buffer[-1] if self._buffer else {}
        positions = [
            p
            for p in (raw.get("position") for raw in self._buffer)
            if p is not None and p != ""
        ]

        if not positions:
            return latest.get("position", ""), 0.0, False

        stable, position = self.is_stable(positions)
        if stable:
            return position, 1.0, True

        # Return the most recent raw value while signalling uncertainty.
        counts: dict[Any, int] = {}
        for p in positions:
            counts[p] = counts.get(p, 0) + 1
        top_count = max(counts.values())
        confidence = min(1.0, top_count / self.window_size)
        return latest.get("position", positions[-1]), confidence, False

    def _derive_stage(self, board: list[str]) -> tuple[str, float, bool]:
        """Derive the poker street from the consensus board length."""
        n = len(board)
        if n == 0:
            return "preflop", 1.0, True
        if n == 3:
            return "flop", 1.0, True
        if n == 4:
            return "turn", 1.0, True
        if n == 5:
            return "river", 1.0, True
        return "unknown", 0.0, False
