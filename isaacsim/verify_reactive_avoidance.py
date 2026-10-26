import numpy as np

from reactive_avoidance import AvoidanceConfig, ReactiveAvoider


def scene(left: float, center: float, right: float) -> np.ndarray:
    image = np.full((120, 180), 3.0, dtype=np.float32)
    band = image[34:94, 14:166]
    width = band.shape[1] // 3
    band[:, :width] = left
    band[:, width : 2 * width] = center
    band[:, 2 * width :] = right
    return image


avoider = ReactiveAvoider(AvoidanceConfig(turn_hold_frames=3))
linear, angular, info = avoider.command(scene(3, 3, 3), 0.25, 0.0)
assert linear == 0.25 and angular == 0.0 and info["mode"] == "clear"

linear, _, info = avoider.command(scene(3, 0.65, 3), 0.25, 0.0)
assert 0.0 < linear < 0.25 and info["mode"] == "slow"

linear, angular, info = avoider.command(scene(1.8, 0.25, 0.55), 0.25, 0.0)
assert linear == 0.0 and angular > 0 and info["mode"] == "blocked_turn"

# Small clearance fluctuations must not immediately flip the escape turn.
_, angular2, _ = avoider.command(scene(0.70, 0.25, 0.75), 0.25, 0.0)
assert angular2 > 0

invalid = np.zeros((120, 180), dtype=np.float32)
linear, angular, info = avoider.command(invalid, 0.2, 0.1)
assert linear == 0.2 and angular == 0.1 and info["mode"] == "depth_invalid"
print("REACTIVE_AVOIDANCE_VERIFIED")
