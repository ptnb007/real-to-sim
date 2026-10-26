"""Small HTTP client and trajectory tracker for InternNav/NavDP iPlanner."""

from __future__ import annotations

import io
import json
import math
import uuid
import urllib.request
from typing import Any

import numpy as np
from PIL import Image


def rho_theta_to_goal(rho_theta: Any) -> tuple[float, float]:
    """Convert VLFM's polar local goal to iPlanner's x-forward/y-left goal."""
    values = np.asarray(rho_theta, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("rho_theta must contain distance and bearing")
    rho = max(0.0, float(values[0]))
    theta = float(values[1])
    return rho * math.cos(theta), rho * math.sin(theta)


def trajectory_command(
    trajectory: Any,
    *,
    lookahead_m: float = 0.45,
    max_linear: float = 0.25,
    max_angular: float = 0.8,
) -> tuple[float, float, dict[str, float]]:
    """Follow an iPlanner trajectory using a dependency-free pure-pursuit rule."""
    points = np.asarray(trajectory, dtype=np.float64)
    while points.ndim > 2:
        points = points[0]
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
        raise ValueError(f"Unexpected iPlanner trajectory shape: {points.shape}")
    xy = points[:, :2]
    finite = np.all(np.isfinite(xy), axis=1)
    xy = xy[finite]
    if len(xy) == 0:
        raise ValueError("iPlanner returned no finite trajectory point")

    distances = np.linalg.norm(xy, axis=1)
    candidates = np.flatnonzero(distances >= lookahead_m)
    index = int(candidates[0]) if len(candidates) else int(np.argmax(distances))
    x, y = (float(v) for v in xy[index])
    heading = math.atan2(y, max(x, 1e-4))
    target_distance = math.hypot(x, y)

    angular = float(np.clip(1.8 * heading, -max_angular, max_angular))
    heading_scale = float(np.clip(1.0 - abs(heading) / 1.25, 0.0, 1.0))
    linear = min(max_linear, 0.7 * target_distance) * heading_scale
    if x < 0.02 or abs(heading) > 1.35:
        linear = 0.0
    return linear, angular, {
        "lookahead_x": round(x, 3),
        "lookahead_y": round(y, 3),
        "heading_error": round(heading, 3),
        "trajectory_points": int(len(xy)),
    }


class IPlannerClient:
    def __init__(self, server: str, timeout: float = 30.0) -> None:
        self.server = server.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _image_bytes(rgb: np.ndarray, depth_m: np.ndarray) -> tuple[bytes, bytes]:
        rgb_buffer = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3], "RGB").save(
            rgb_buffer, format="JPEG", quality=86
        )
        depth_mm10 = np.clip(
            np.nan_to_num(depth_m, nan=0.0, posinf=6.5, neginf=0.0) * 10000.0,
            0,
            65535,
        ).astype(np.uint16)
        depth_buffer = io.BytesIO()
        Image.fromarray(depth_mm10, mode="I;16").save(depth_buffer, format="PNG")
        return rgb_buffer.getvalue(), depth_buffer.getvalue()

    @staticmethod
    def _multipart(fields: dict[str, str], files: dict[str, tuple[str, str, bytes]]) -> tuple[bytes, str]:
        boundary = "----CodexIPlanner" + uuid.uuid4().hex
        body = bytearray()
        for name, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.extend(value.encode())
            body.extend(b"\r\n")
        for name, (filename, content_type, data) in files.items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
            )
            body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
            body.extend(data)
            body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())
        return bytes(body), f"multipart/form-data; boundary={boundary}"

    def reset(self, intrinsics: np.ndarray, width: int, height: int) -> dict[str, Any]:
        # The official agent scales this matrix from the declared 640x360 input.
        intrinsic = np.asarray(intrinsics, dtype=np.float64).copy()
        intrinsic[0, :] *= 640.0 / float(width)
        intrinsic[1, :] *= 360.0 / float(height)
        payload = json.dumps(
            {"intrinsic": intrinsic.tolist(), "stop_threshold": [0.2], "batch_size": 1}
        ).encode()
        request = urllib.request.Request(
            self.server + "/navigator_reset", data=payload, method="POST"
        )
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode())

    def plan(self, rgb: np.ndarray, depth_m: np.ndarray, goal_x: float, goal_y: float) -> dict[str, Any]:
        rgb_bytes, depth_bytes = self._image_bytes(rgb, depth_m)
        body, content_type = self._multipart(
            {"goal_data": json.dumps({"goal_x": [goal_x], "goal_y": [goal_y]})},
            {
                "image": ("rgb.jpg", "image/jpeg", rgb_bytes),
                "depth": ("depth.png", "image/png", depth_bytes),
            },
        )
        request = urllib.request.Request(self.server + "/pointgoal_step", data=body, method="POST")
        request.add_header("Content-Type", content_type)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode())
