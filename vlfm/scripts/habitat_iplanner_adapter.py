"""Use InternNav/NavDP iPlanner as a local controller for Habitat actions."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class HabitatIPlannerAdapter:
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3

    def __init__(
        self,
        image_intrinsic: np.ndarray,
        max_depth: float,
        device: str = "cuda:0",
        turn_threshold_degrees: float = 25.0,
        lookahead_m: float = 0.45,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        planner_root = repo_root / "third_party/NavDP/baselines/iplanner"
        checkpoint = planner_root / "checkpoints/iplanner.pth"
        config = planner_root / "configs/iplanner.yaml"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing iPlanner checkpoint: {checkpoint}")
        sys.path.insert(0, str(planner_root))
        from iplanner_agent import IPlannerAgent

        self.agent = IPlannerAgent(
            np.asarray(image_intrinsic, dtype=np.float32),
            model_path=str(checkpoint),
            model_config_path=str(config),
            device=device,
        )
        self.max_depth = float(max_depth)
        self.turn_threshold = math.radians(turn_threshold_degrees)
        self.turn_release = math.radians(8.0)
        self.turning = 0
        self.lookahead_m = float(lookahead_m)
        self.last_debug: dict[str, Any] = {"active": False}

    def act(
        self,
        depth: np.ndarray,
        rho_theta: Any,
        fallback_action: int,
        stop_called: bool = False,
    ) -> int:
        if stop_called or fallback_action == self.STOP:
            self.last_debug = {"active": False, "reason": "stop", "action": self.STOP}
            return self.STOP

        polar = np.asarray(rho_theta, dtype=np.float32).reshape(-1)
        if polar.size < 2 or not np.all(np.isfinite(polar[:2])):
            self.last_debug = {"active": False, "reason": "no_goal", "action": fallback_action}
            return fallback_action

        rho, theta = float(polar[0]), float(polar[1])
        goal = np.array([[rho * math.cos(theta), rho * math.sin(theta), 0.0]], dtype=np.float32)
        depth_image = np.asarray(depth, dtype=np.float32).squeeze()
        if depth_image.size == 0:
            return fallback_action
        # Habitat's configured depth observation is normalized to [0, 1].
        if float(np.nanmax(depth_image)) <= 1.01:
            depth_image = depth_image * self.max_depth
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=self.max_depth, neginf=0.0)

        _, trajectory, fear = self.agent.step_pointgoal(
            depth_image[None, ..., None], goal
        )
        points = trajectory.detach().float().cpu().numpy()[0, :, :2]
        distances = np.linalg.norm(points, axis=1)
        candidates = np.flatnonzero(distances >= self.lookahead_m)
        index = int(candidates[0]) if len(candidates) else int(np.argmax(distances))
        x, y = (float(v) for v in points[index])
        heading = math.atan2(y, max(x, 1e-4))
        # Habitat turns by a fixed 30 degrees. A single low threshold makes the
        # controller alternate LEFT/RIGHT around the desired heading. Start a
        # turn only for a substantial error and keep its direction until the
        # error is small or crosses zero; then advance instead of reversing.
        if self.turning and (abs(heading) <= self.turn_release or heading * self.turning <= 0.0):
            self.turning = 0
        if self.turning == 0:
            if heading > self.turn_threshold:
                self.turning = 1
            elif heading < -self.turn_threshold:
                self.turning = -1
        if self.turning > 0:
            action = self.TURN_LEFT
        elif self.turning < 0:
            action = self.TURN_RIGHT
        else:
            action = self.MOVE_FORWARD
        self.last_debug = {
            "active": True,
            "action": action,
            "goal_xy": goal[0, :2].tolist(),
            "lookahead_xy": [x, y],
            "heading_degrees": math.degrees(heading),
            "turning_state": self.turning,
            "fear": float(fear.detach().float().cpu().reshape(-1)[0]),
            "trajectory": points.tolist(),
        }
        return action

    def draw(self, size: int = 250) -> np.ndarray:
        canvas = np.full((size, size, 3), 248, dtype=np.uint8)
        cv2.putText(canvas, "InternNav iPlanner", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (25, 25, 25), 1)
        origin = np.array([size // 2, size - 22], dtype=np.int32)
        cv2.circle(canvas, tuple(origin), 5, (30, 30, 30), -1)
        if not self.last_debug.get("active"):
            cv2.putText(canvas, "waiting for VLFM goal", (18, size // 2), cv2.FONT_HERSHEY_SIMPLEX, .45, (90, 90, 90), 1)
            return canvas
        points = np.asarray(self.last_debug["trajectory"], dtype=np.float32)
        scale = 62.0
        pixels = np.column_stack((origin[0] - points[:, 1] * scale, origin[1] - points[:, 0] * scale)).astype(np.int32)
        pixels[:, 0] = np.clip(pixels[:, 0], 4, size - 5)
        pixels[:, 1] = np.clip(pixels[:, 1], 34, size - 5)
        cv2.polylines(canvas, [pixels], False, (35, 175, 65), 3, cv2.LINE_AA)
        lx, ly = self.last_debug["lookahead_xy"]
        lookahead = (int(origin[0] - ly * scale), int(origin[1] - lx * scale))
        cv2.circle(canvas, lookahead, 6, (30, 60, 220), -1)
        label = f"a={self.last_debug['action']} heading={self.last_debug['heading_degrees']:.1f}"
        cv2.putText(canvas, label, (8, size - 5), cv2.FONT_HERSHEY_SIMPLEX, .4, (20, 20, 20), 1)
        cv2.putText(canvas, f"risk={self.last_debug['fear']:.3f}", (8, 43), cv2.FONT_HERSHEY_SIMPLEX, .4, (20, 20, 20), 1)
        return canvas
