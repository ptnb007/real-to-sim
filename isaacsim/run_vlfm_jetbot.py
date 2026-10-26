"""Drive the verified Isaac Sim Jetbot with VLFM running in WSL."""

import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

sys.argv.append("--/app/vulkan=false")

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": os.environ.get("VLFM_HEADLESS", "0") == "1",
        "renderer": "RaytracedLighting",
        "multi_gpu": False,
        "width": 960,
        "height": 540,
    }
)

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.robot.wheeled_robots.controllers import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.sensors.camera import Camera

from reactive_avoidance import ReactiveAvoider
from internnav_iplanner import IPlannerClient, rho_theta_to_goal, trajectory_command


SCENE = os.environ.get(
    "VLFM_ISAAC_SCENE",
    r"C:\Users\woom\Desktop\woom\isaacsim\jetbot_scene.usda",
)
SERVER = os.environ.get("VLFM_SERVER", "http://127.0.0.1:12190")
IPLANNER_SERVER = os.environ.get("INTERNNAV_SERVER", "http://127.0.0.1:8888")
USE_IPLANNER = os.environ.get("INTERNNAV_ENABLED", "1") == "1"
GOAL = os.environ.get("VLFM_GOAL", "person")
POLICY_SECONDS = float(os.environ.get("VLFM_ACTION_SECONDS", "0.35"))
MAX_STEPS = int(os.environ.get("VLFM_MAX_STEPS", "750"))
CAMERA_HEIGHT = 0.18
CAMERA_FORWARD = 0.08


def yaw_from_quaternion(q: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def post_json(path: str, body: bytes = b"") -> dict:
    req = urllib.request.Request(SERVER + path, data=body, method="POST")
    if body:
        req.add_header("Content-Type", "application/x-npz")
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server() -> None:
    for _ in range(60):
        try:
            with urllib.request.urlopen(SERVER + "/health", timeout=2) as response:
                status = json.loads(response.read().decode("utf-8"))
                print("VLFM_SERVER_READY", status, flush=True)
                return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise RuntimeError("VLFM server is not reachable at " + SERVER)


def main() -> None:
    if not open_stage(SCENE):
        raise RuntimeError("Could not open scene: " + SCENE)

    world = World(stage_units_in_meters=1.0)
    jetbot = world.scene.add(
        WheeledRobot(
            prim_path="/World/Jetbot",
            name="vlfm_jetbot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        )
    )
    camera = Camera(
        prim_path="/World/VLFM_Camera",
        frequency=10,
        resolution=(320, 240),
    )
    controller = DifferentialController(
        name="vlfm_differential_controller",
        wheel_radius=0.035,
        wheel_base=0.10,
        max_linear_speed=0.35,
        max_angular_speed=1.2,
        max_wheel_speed=12.0,
    )
    avoider = ReactiveAvoider()
    iplanner = IPlannerClient(IPLANNER_SERVER)
    iplanner_ready = False

    def place_camera() -> None:
        current_position, current_orientation = jetbot.get_world_pose()
        current_yaw = yaw_from_quaternion(np.asarray(current_orientation))
        cam_position = np.asarray(current_position, dtype=np.float64).copy()
        cam_position[0] += CAMERA_FORWARD * math.cos(current_yaw)
        cam_position[1] += CAMERA_FORWARD * math.sin(current_yaw)
        cam_position[2] = CAMERA_HEIGHT
        camera.set_world_pose(
            position=cam_position,
            orientation=np.asarray(current_orientation),
            camera_axes="world",
        )

    world.reset()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    wait_for_server()
    post_json("/reset?goal=" + urllib.parse.quote(GOAL))

    start_position, start_orientation = jetbot.get_world_pose()
    start_yaw = yaw_from_quaternion(np.asarray(start_orientation))
    print(f"VLFM_START goal={GOAL!r} server={SERVER}", flush=True)

    for policy_step in range(MAX_STEPS):
        position, orientation = jetbot.get_world_pose()
        yaw = yaw_from_quaternion(np.asarray(orientation))
        place_camera()
        for _ in range(3):
            world.step(render=True)

        rgba = camera.get_rgba()
        depth_m = camera.get_depth(device="cpu")
        if rgba is None or depth_m is None:
            continue
        if hasattr(depth_m, "numpy"):
            depth_m = depth_m.numpy()
        intrinsics = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)
        packet = io.BytesIO()
        np.savez_compressed(
            packet,
            rgb=np.asarray(rgba, dtype=np.uint8)[..., :3],
            depth_m=np.asarray(depth_m, dtype=np.float32),
            position=np.asarray(position, dtype=np.float64),
            yaw=np.asarray([yaw], dtype=np.float64),
            intrinsics=intrinsics,
            camera_height=np.asarray([CAMERA_HEIGHT], dtype=np.float64),
        )
        action = post_json("/act", packet.getvalue())
        planned_command = None
        if USE_IPLANNER and action.get("arm_yaw", -1.0) == -1.0 and not action.get("stop_called"):
            try:
                if not iplanner_ready:
                    result = iplanner.reset(intrinsics, rgba.shape[1], rgba.shape[0])
                    iplanner_ready = True
                    print("INTERNNAV_READY", result, flush=True)
                goal_x, goal_y = rho_theta_to_goal(action.get("rho_theta", [0.0, 0.0]))
                if goal_x * goal_x + goal_y * goal_y > 1e-4:
                    plan = iplanner.plan(
                        np.asarray(rgba, dtype=np.uint8)[..., :3],
                        np.asarray(depth_m, dtype=np.float32),
                        goal_x,
                        goal_y,
                    )
                    planned_command = trajectory_command(plan["trajectory"])
                    print(
                        "INTERNNAV_PLAN",
                        json.dumps(
                            {"goal_xy": [round(goal_x, 3), round(goal_y, 3)], **planned_command[2]},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            except Exception as error:
                iplanner_ready = False
                print(f"INTERNNAV_FALLBACK {type(error).__name__}: {error}", flush=True)
        print("VLFM_ACTION", json.dumps(action, ensure_ascii=False), flush=True)

        if action.get("stop_called"):
            jetbot.apply_wheel_actions(controller.forward(command=np.array([0.0, 0.0])))
            print("VLFM_TARGET_REACHED", flush=True)
            break

        arm_yaw = float(action.get("arm_yaw", -1.0))
        # Each loop advances one rendered sensing frame and one commanded
        # physics frame, hence 30 loop iterations per simulated second.
        action_frames = max(1, int(POLICY_SECONDS * 30))
        for _ in range(action_frames):
            if arm_yaw != -1.0:
                _, current_orientation = jetbot.get_world_pose()
                current_yaw = yaw_from_quaternion(np.asarray(current_orientation))
                error = wrap_angle(start_yaw + arm_yaw - current_yaw)
                linear, angular = 0.0, float(np.clip(2.0 * error, -1.0, 1.0))
            else:
                if planned_command is not None:
                    linear, angular, _ = planned_command
                else:
                    linear = float(np.clip(action.get("linear", 0.0), -1.0, 1.0)) * 0.25
                    angular = float(np.clip(action.get("angular", 0.0), -1.0, 1.0)) * 0.8
                if abs(linear) < 1e-4 and abs(angular) < 1e-4 and not action.get("target_detected"):
                    # An empty/unbounded scene has no frontier, so VLFM returns
                    # zero. Keep scanning safely instead of appearing frozen.
                    angular = 0.32
            # Keep the sensor attached to the moving chassis and reassess depth
            # during the action window instead of blindly driving for 0.7 s.
            place_camera()
            world.step(render=True)
            live_depth = camera.get_depth(device="cpu")
            if live_depth is not None:
                if hasattr(live_depth, "numpy"):
                    live_depth = live_depth.numpy()
                linear, angular, safety = avoider.command(live_depth, linear, angular)
                if safety["mode"] != "clear":
                    print("LOCAL_AVOIDANCE", json.dumps(safety, ensure_ascii=False), flush=True)
            jetbot.apply_wheel_actions(controller.forward(command=np.array([linear, angular])))
            world.step(render=True)

        if not simulation_app.is_running():
            break

    final_position, _ = jetbot.get_world_pose()
    distance = float(np.linalg.norm(np.asarray(final_position)[:2] - np.asarray(start_position)[:2]))
    print(f"VLFM_RUN_COMPLETE distance_m={distance:.4f}", flush=True)


try:
    main()
finally:
    simulation_app.close()
