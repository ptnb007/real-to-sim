"""Small, dependency-light reactive safety layer for RGB-D wheeled robots."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class AvoidanceConfig:
    stop_distance: float = 0.42
    slow_distance: float = 0.85
    side_distance: float = 0.34
    turn_speed: float = 0.72
    turn_hold_frames: int = 18
    valid_min_depth: float = 0.06
    percentile: float = 18.0


class ReactiveAvoider:
    """Override forward velocity when metric depth reports a nearby obstacle.

    The chosen escape direction is held briefly.  This hysteresis prevents the
    left/right oscillation that otherwise occurs when both sides have similar
    clearance.
    """

    def __init__(self, config: AvoidanceConfig | None = None) -> None:
        self.config = config or AvoidanceConfig()
        self._turn_direction = 0
        self._hold_remaining = 0

    def reset(self) -> None:
        self._turn_direction = 0
        self._hold_remaining = 0

    def _clearance(self, depth: np.ndarray) -> Dict[str, float]:
        image = np.squeeze(np.asarray(depth, dtype=np.float32))
        if image.ndim != 2:
            return {"left": np.inf, "center": np.inf, "right": np.inf, "valid_fraction": 0.0}
        height, width = image.shape
        # Ignore ceiling and the lowest band where the robot can see its own body/floor.
        roi = image[int(height * 0.28) : int(height * 0.78), int(width * 0.08) : int(width * 0.92)]
        thirds = np.array_split(roi, 3, axis=1)
        values = []
        valid_total = 0
        for sector in thirds:
            valid = sector[np.isfinite(sector) & (sector > self.config.valid_min_depth)]
            valid_total += valid.size
            values.append(float(np.percentile(valid, self.config.percentile)) if valid.size else np.inf)
        return {
            "left": values[0],
            "center": values[1],
            "right": values[2],
            "valid_fraction": float(valid_total / max(roi.size, 1)),
        }

    def command(self, depth: np.ndarray, linear: float, angular: float) -> Tuple[float, float, Dict[str, float | str]]:
        clearance = self._clearance(depth)
        left, center, right = clearance["left"], clearance["center"], clearance["right"]
        mode = "clear"

        # Depth loss must not create a false emergency turn. The global policy
        # remains responsible until enough valid pixels return.
        if clearance["valid_fraction"] < 0.08:
            return linear, angular, {**clearance, "mode": "depth_invalid"}

        if center < self.config.stop_distance and linear > 0.0:
            mode = "blocked_turn"
            if self._hold_remaining <= 0:
                # Positive angular velocity is a left turn.
                self._turn_direction = 1 if left >= right else -1
                self._hold_remaining = self.config.turn_hold_frames
            elif self._turn_direction > 0 and right > left + 0.30:
                self._turn_direction = -1
            elif self._turn_direction < 0 and left > right + 0.30:
                self._turn_direction = 1
            linear = 0.0
            angular = self._turn_direction * self.config.turn_speed
            self._hold_remaining -= 1
        elif center < self.config.slow_distance and linear > 0.0:
            mode = "slow"
            scale = np.clip(
                (center - self.config.stop_distance) / (self.config.slow_distance - self.config.stop_distance),
                0.12,
                1.0,
            )
            linear *= float(scale)
            # Gently bias away from the closer side without reversing the policy turn.
            if left < self.config.side_distance:
                angular = min(angular, -0.25)
            elif right < self.config.side_distance:
                angular = max(angular, 0.25)
        else:
            self._hold_remaining = max(0, self._hold_remaining - 1)
            if left < self.config.side_distance and linear > 0.0:
                mode = "edge_right"
                angular = min(angular, -0.22)
            elif right < self.config.side_distance and linear > 0.0:
                mode = "edge_left"
                angular = max(angular, 0.22)

        return float(linear), float(angular), {**clearance, "mode": mode}

