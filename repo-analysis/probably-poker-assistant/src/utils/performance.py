"""
Performance Monitor for Poker AI Assistant.

Tracks timing of each pipeline step to identify bottlenecks.
Provides statistics and warnings for slow operations.

Usage:
    perf = PerformanceMonitor()
    with perf.track("screen_capture"):
        screen = capture_screen()
    stats = perf.get_stats()
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.utils.logger import logger


@dataclass
class StepStats:
    """Statistics for a single pipeline step."""
    count: int
    avg_ms: float
    min_ms: float
    max_ms: float
    total_ms: float
    last_ms: float


class PerformanceMonitor:
    """
    Monitors performance of game loop pipeline steps.

    Tracks timing for:
    - screen_capture
    - anchor_detection
    - card_detection
    - ocr_reading
    - decision_engine
    - total_frame
    """

    def __init__(self, max_samples: int = 100, warning_threshold_ms: float = 100.0):
        """
        Initialize performance monitor.

        Args:
            max_samples: Maximum samples to keep per step (rolling window)
            warning_threshold_ms: Log warning if step exceeds this threshold
        """
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.max_samples = max_samples
        self.warning_threshold_ms = warning_threshold_ms
        self.enabled = True

        logger.info(f"PerformanceMonitor initialized (threshold: {warning_threshold_ms}ms)")

    @contextmanager
    def track(self, step_name: str):
        """
        Context manager to track timing of a code block.

        Args:
            step_name: Name of the step being tracked

        Usage:
            with perf.track("screen_capture"):
                screen = grab_screen()
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Store timing (rolling window)
            self.timings[step_name].append(elapsed_ms)
            if len(self.timings[step_name]) > self.max_samples:
                self.timings[step_name].pop(0)

            # Log warning if slow
            if elapsed_ms > self.warning_threshold_ms:
                logger.warning(f"Slow {step_name}: {elapsed_ms:.1f}ms (threshold: {self.warning_threshold_ms}ms)")

    def get_stats(self, step_name: Optional[str] = None) -> Dict[str, StepStats]:
        """
        Get statistics for tracked steps.

        Args:
            step_name: Specific step to get stats for, or None for all steps

        Returns:
            Dictionary of step_name -> StepStats
        """
        result = {}

        steps_to_check = [step_name] if step_name else self.timings.keys()

        for name in steps_to_check:
            if name not in self.timings or not self.timings[name]:
                continue

            samples = self.timings[name]
            result[name] = StepStats(
                count=len(samples),
                avg_ms=sum(samples) / len(samples),
                min_ms=min(samples),
                max_ms=max(samples),
                total_ms=sum(samples),
                last_ms=samples[-1]
            )

        return result

    def get_summary(self) -> str:
        """
        Get human-readable summary of all step timings.

        Returns:
            Formatted string with timing statistics
        """
        stats = self.get_stats()
        if not stats:
            return "No performance data collected"

        lines = ["Performance Summary:"]
        lines.append("-" * 60)
        lines.append(f"{'Step':<20} {'Avg':>10} {'Min':>10} {'Max':>10} {'Count':>8}")
        lines.append("-" * 60)

        for name, s in sorted(stats.items()):
            lines.append(f"{name:<20} {s.avg_ms:>9.1f}ms {s.min_ms:>9.1f}ms {s.max_ms:>9.1f}ms {s.count:>8}")

        # Calculate total frame time
        if 'total_frame' in stats:
            fps = 1000 / stats['total_frame'].avg_ms if stats['total_frame'].avg_ms > 0 else 0
            lines.append("-" * 60)
            lines.append(f"Effective FPS: {fps:.1f}")

        return "\n".join(lines)

    def log_summary(self):
        """Log performance summary at INFO level."""
        logger.info("\n" + self.get_summary())

    def check_bottleneck(self) -> Optional[str]:
        """
        Identify the slowest step in the pipeline.

        Returns:
            Name of the bottleneck step, or None if no data
        """
        stats = self.get_stats()
        if not stats:
            return None

        # Exclude total_frame from bottleneck check
        step_stats = {k: v for k, v in stats.items() if k != 'total_frame'}
        if not step_stats:
            return None

        bottleneck = max(step_stats.items(), key=lambda x: x[1].avg_ms)
        return bottleneck[0]

    def reset(self):
        """Clear all timing data."""
        self.timings.clear()
        logger.debug("Performance data reset")

    def enable(self):
        """Enable performance tracking."""
        self.enabled = True
        logger.info("Performance monitoring enabled")

    def disable(self):
        """Disable performance tracking."""
        self.enabled = False
        logger.info("Performance monitoring disabled")

    def set_warning_threshold(self, threshold_ms: float):
        """Set the warning threshold in milliseconds."""
        self.warning_threshold_ms = threshold_ms
        logger.debug(f"Warning threshold set to {threshold_ms}ms")
