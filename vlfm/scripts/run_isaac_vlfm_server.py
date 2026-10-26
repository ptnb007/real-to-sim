"""Run VLFM in WSL and expose one policy step to Isaac Sim over localhost."""

import argparse
import io
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vlfm.policy.isaacsim_policy import IsaacSimITMPolicyV2
from vlfm.utils.geometry_utils import get_fov, wrap_heading, xyz_yaw_to_tf_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12190)
    parser.add_argument("--goal", default="person")
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--mock", action="store_true", help="Test transport without loading VLFM models")
    return parser.parse_args()


class IsaacObservationAdapter:
    def __init__(self, goal: str, max_depth: float) -> None:
        self.goal = goal
        self.max_depth = max_depth
        self.start_xy: np.ndarray | None = None
        self.start_yaw: float | None = None

    def reset(self) -> None:
        self.start_xy = None
        self.start_yaw = None

    def convert(self, packet: Dict[str, np.ndarray]) -> Dict[str, Any]:
        rgb = np.asarray(packet["rgb"], dtype=np.uint8)[..., :3]
        depth_m = np.asarray(packet["depth_m"], dtype=np.float32)
        depth_m = np.squeeze(depth_m)
        if depth_m.ndim != 2:
            raise ValueError(f"Expected a 2D depth image, received shape {depth_m.shape}")
        depth_m = np.nan_to_num(depth_m, nan=self.max_depth, posinf=self.max_depth, neginf=0.0)
        depth = np.clip(depth_m, 0.0, self.max_depth) / self.max_depth
        position = np.asarray(packet["position"], dtype=np.float64)
        yaw = float(np.asarray(packet["yaw"]).reshape(-1)[0])
        intrinsics = np.asarray(packet["intrinsics"], dtype=np.float64)

        if self.start_xy is None:
            self.start_xy = position[:2].copy()
            self.start_yaw = yaw

        assert self.start_yaw is not None
        offset = position[:2] - self.start_xy
        c, s = np.cos(-self.start_yaw), np.sin(-self.start_yaw)
        robot_xy = np.array([[c, -s], [s, c]]) @ offset
        robot_heading = wrap_heading(yaw - self.start_yaw)

        camera_height = float(np.asarray(packet.get("camera_height", [0.18])).reshape(-1)[0])
        camera_xyz = np.array([robot_xy[0], robot_xy[1], camera_height])
        tf_camera_to_episode = xyz_yaw_to_tf_matrix(camera_xyz, robot_heading)
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        hfov = get_fov(fx, depth.shape[1])

        # VLFM expects normalized depth. One simulated RGB-D camera is reused for
        # obstacle, language-value and object point-cloud maps.
        nav_depth = cv2.resize(depth, (240, 212), interpolation=cv2.INTER_AREA)
        obstacle_map_depths = [
            (depth, tf_camera_to_episode, 0.0, self.max_depth, fx, fy, hfov),
            (None, tf_camera_to_episode, 0.0, self.max_depth, fx, fy, hfov),
        ]
        value_map_rgbd = [(rgb, depth, tf_camera_to_episode, 0.0, self.max_depth, hfov)]
        object_map_rgbd = [(rgb, depth, tf_camera_to_episode, 0.0, self.max_depth, fx, fy)]

        return {
            "nav_depth": nav_depth,
            "robot_xy": robot_xy,
            "robot_heading": robot_heading,
            "objectgoal": self.goal,
            "obstacle_map_depths": obstacle_map_depths,
            "value_map_rgbd": value_map_rgbd,
            "object_map_rgbd": object_map_rgbd,
        }


def load_policy() -> IsaacSimITMPolicyV2:
    config_dir = str((REPO_ROOT / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="experiments/reality")
    return IsaacSimITMPolicyV2.from_config(cfg)


def create_app(goal: str, max_depth: float, mock: bool = False) -> Flask:
    app = Flask(__name__)
    adapter = IsaacObservationAdapter(goal=goal, max_depth=max_depth)
    policy = None if mock else load_policy()
    lock = threading.Lock()
    mask = torch.zeros(1, 1, device="cpu" if mock else "cuda", dtype=torch.bool)
    latest_frames: Dict[str, bytes] = {}
    latest_status: Dict[str, Any] = {
        "goal": goal,
        "step": 0,
        "target_detected": False,
        "stop_called": False,
        "linear": 0.0,
        "angular": 0.0,
        "state": "等待 Isaac Sim 数据",
    }

    def encode_rgb(image: np.ndarray) -> bytes:
        image = np.asarray(image)
        if image.ndim == 2:
            image = cv2.applyColorMap(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        else:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 86])
        if not ok:
            raise RuntimeError("Could not encode dashboard frame")
        return encoded.tobytes()

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ready", "goal": adapter.goal})

    @app.get("/status")
    def status() -> Any:
        return jsonify(latest_status)

    @app.get("/frame/<name>.jpg")
    def frame(name: str) -> Any:
        data = latest_frames.get(name)
        if data is None:
            return Response(status=404)
        return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/dashboard")
    def dashboard() -> Any:
        return Response(
            """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>VLFM × Isaac Sim 实时搜索</title><style>
body{margin:0;background:#0b1020;color:#eef3ff;font-family:'Microsoft YaHei',sans-serif}
header{padding:18px 24px;background:#121a30;border-bottom:1px solid #263250;display:flex;justify-content:space-between;align-items:center}
h1{font-size:22px;margin:0}.badge{padding:7px 12px;border-radius:18px;background:#173a2b;color:#78f0ad}
#status{padding:12px 24px;color:#b9c7e6}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:0 24px 24px}
.card{background:#121a30;border:1px solid #263250;border-radius:12px;overflow:hidden}.card h2{font-size:15px;margin:0;padding:10px 14px;background:#17213b}
.card img{display:block;width:100%;height:330px;object-fit:contain;background:#050810}.metrics{display:flex;gap:18px;flex-wrap:wrap}
.value{color:#68a8ff;font-weight:700}@media(max-width:900px){.grid{grid-template-columns:1fr}.card img{height:auto}}
</style></head><body><header><h1>VLFM × Isaac Sim 实时人体搜索</h1><span class='badge' id='state'>等待数据</span></header>
<div id='status'><div class='metrics'>目标：<span class='value' id='goal'>person</span> 步数：<span class='value' id='step'>0</span> 发现目标：<span class='value' id='detected'>否</span> 线速度：<span class='value' id='linear'>0</span> 角速度：<span class='value' id='angular'>0</span></div></div>
<main class='grid'><section class='card'><h2>Isaac Sim 相机 / 目标检测</h2><img id='camera'></section>
<section class='card'><h2>深度图</h2><img id='depth'></section>
<section class='card'><h2>VLFM 障碍与探索地图</h2><img id='obstacle'></section>
<section class='card'><h2>VLFM 语言价值地图</h2><img id='value'></section></main>
<script>const names=['camera','depth','obstacle','value'];async function tick(){const t=Date.now();
for(const n of names){document.getElementById(n).src='/frame/'+n+'.jpg?t='+t}
try{const s=await(await fetch('/status?t='+t)).json();for(const k of ['goal','step','linear','angular'])document.getElementById(k).textContent=s[k];document.getElementById('detected').textContent=s.target_detected?'是':'否';document.getElementById('state').textContent=s.state}catch(e){} }
tick();setInterval(tick,800);</script></body></html>""",
            mimetype="text/html",
        )

    @app.post("/reset")
    def reset() -> Any:
        nonlocal mask
        with lock:
            adapter.goal = request.args.get("goal", adapter.goal)
            adapter.reset()
            if policy is not None:
                policy._reset()
            mask = torch.zeros_like(mask)
        return jsonify({"status": "reset", "goal": adapter.goal})

    @app.post("/act")
    def act() -> Any:
        nonlocal mask
        with lock:
            with np.load(io.BytesIO(request.get_data()), allow_pickle=False) as packet:
                observations = adapter.convert({key: packet[key] for key in packet.files})
            rgb = observations["object_map_rgbd"][0][0]
            depth = observations["object_map_rgbd"][0][1]
            latest_frames["camera"] = encode_rgb(rgb)
            latest_frames["depth"] = encode_rgb(depth)
            if policy is None:
                latest_status.update({"step": 1, "linear": 0.25, "angular": 0.0, "state": "通信测试"})
                return jsonify(
                    {
                        "linear": 0.25,
                        "angular": 0.0,
                        "arm_yaw": -1.0,
                        "target_detected": False,
                        "stop_called": False,
                        "step": 1,
                        "mock": True,
                        "rgb_shape": list(observations["object_map_rgbd"][0][0].shape),
                        "depth_shape": list(observations["object_map_rgbd"][0][1].shape),
                    }
                )
            action = policy.get_action(observations, mask, deterministic=True)
            mask = torch.ones_like(mask)
            info = action.get("info", {})
            rho_theta = np.asarray(action.get("rho_theta", [0.0, 0.0]), dtype=np.float32).reshape(-1)
            for source, target in (
                ("annotated_rgb", "camera"),
                ("obstacle_map", "obstacle"),
                ("value_map", "value"),
            ):
                if source in info:
                    latest_frames[target] = encode_rgb(info[source])
            latest_status.update(
                {
                    "goal": adapter.goal,
                    "step": int(policy._num_steps),
                    "target_detected": bool(info.get("target_detected", False)),
                    "stop_called": bool(info.get("stop_called", False)),
                    "linear": round(float(action.get("linear", 0.0)), 3),
                    "angular": round(float(action.get("angular", 0.0)), 3),
                    "goal_distance": round(float(rho_theta[0]), 3),
                    "goal_bearing": round(float(rho_theta[1]), 3),
                    "state": "已到达目标" if info.get("stop_called", False) else ("正在接近目标" if info.get("target_detected", False) else "正在探索"),
                }
            )
            return jsonify(
                {
                    "linear": float(action.get("linear", 0.0)),
                    "angular": float(action.get("angular", 0.0)),
                    "arm_yaw": float(action.get("arm_yaw", -1.0)),
                    "target_detected": bool(info.get("target_detected", False)),
                    "stop_called": bool(info.get("stop_called", False)),
                    "step": int(policy._num_steps),
                    "rho_theta": rho_theta[:2].tolist(),
                }
            )

    return app


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    app = create_app(goal=args.goal, max_depth=args.max_depth, mock=args.mock)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
