"""Offline checks for goal conversion and iPlanner trajectory tracking."""

import math
import numpy as np

from internnav_iplanner import rho_theta_to_goal, trajectory_command


x, y = rho_theta_to_goal([2.0, math.pi / 2])
assert abs(x) < 1e-6 and abs(y - 2.0) < 1e-6

straight = np.zeros((1, 51, 3), dtype=np.float32)
straight[0, :, 0] = np.linspace(0.0, 2.0, 51)
linear, angular, _ = trajectory_command(straight)
assert linear > 0.0 and abs(angular) < 1e-6

left = straight.copy()
left[0, :, 1] = np.linspace(0.0, 0.8, 51)
linear, angular, _ = trajectory_command(left)
assert linear >= 0.0 and angular > 0.0

right = straight.copy()
right[0, :, 1] = np.linspace(0.0, -0.8, 51)
linear, angular, _ = trajectory_command(right)
assert linear >= 0.0 and angular < 0.0

print("INTERNNAV_TRACKER_VERIFIED")
