"""Safe Galaxea R1 Pro chassis demo for the focus_v4 Isaac Sim scene.

The controller follows Galaxea's official ``wheel_move.py`` convention:
steering joints are held by position and rolling joints are driven by velocity.
It adds a speed ramp and a roll/pitch watchdog so a bad command stops instead
of continuing to push a falling robot.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

GUI_MODE = "--gui" in sys.argv
REVIEW_MODE = "--review" in sys.argv
RECORD_MODE = "--record" in sys.argv
RENDER = GUI_MODE or REVIEW_MODE
sys.argv.append("--/app/vulkan=false")
from isaacsim import SimulationApp

APP = SimulationApp({"headless": not GUI_MODE, "multi_gpu": False, "width": 960, "height": 720})

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage


HERE = Path(__file__).resolve().parent
SCENE = HERE / "focus_v4" / "focused_zone_with_official_r1pro.usda"
REPORT = HERE / "focus_v4" / "r1pro_safe_motion_verification.json"
if REVIEW_MODE:
    REPORT = HERE / "focus_v4" / "review_robot" / "isaac_motion_verification.json"
    REPORT.parent.mkdir(exist_ok=True)
ROBOT_PATH = "/World/R1Pro/r1_pro_with_gripper/base_link"

# Stable pose copied from Galaxea's official galaxea_isaac_tutorial.
STABLE_POSE = {
    "steer_motor_joint1": 0.0,
    "steer_motor_joint2": 0.0,
    "steer_motor_joint3": 0.0,
    "torso_joint1": 0.250,
    "torso_joint2": -0.622,
    "torso_joint3": -0.532,
    "torso_joint4": 0.0,
    "left_arm_joint1": -0.13901277,
    "left_arm_joint2": 0.48248085,
    "left_arm_joint3": -0.0478383,
    "left_arm_joint4": -1.66591489,
    "left_arm_joint5": 0.90680851,
    "left_arm_joint6": -0.40558298,
    "left_arm_joint7": -0.73308511,
    "left_gripper_finger_joint1": 0.0,
    "left_gripper_finger_joint2": 0.0,
    "right_arm_joint1": -0.69229362,
    "right_arm_joint2": -0.99383404,
    "right_arm_joint3": 1.09085957,
    "right_arm_joint4": -1.74389787,
    "right_arm_joint5": -0.88277021,
    "right_arm_joint6": -0.59911489,
    "right_arm_joint7": 0.50699574,
    "right_gripper_finger_joint1": 0.0,
    "right_gripper_finger_joint2": 0.0,
}

WHEEL_NAMES = [f"wheel_motor_joint{i}" for i in (1, 2, 3)]


def roll_pitch_deg(q_wxyz):
    """Return roll and pitch in degrees for a wxyz quaternion."""
    w, x, y, z = [float(v) for v in q_wxyz]
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    return math.degrees(roll), math.degrees(pitch)


def ramp(frame, up=60, hold=90, down=60, peak=0.8):
    """Smooth wheel-speed profile in rad/s (about walking-demo crawl speed)."""
    if frame < up:
        return peak * 0.5 * (1.0 - math.cos(math.pi * frame / up))
    if frame < up + hold:
        return peak
    k = frame - up - hold
    if k < down:
        return peak * 0.5 * (1.0 + math.cos(math.pi * k / down))
    return 0.0


try:
    open_stage(str(SCENE))
    if REVIEW_MODE:
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        from isaacsim.core.utils.viewports import set_camera_view
        viewport = get_active_viewport()
        viewport.camera_path = "/World/ReviewCamera"
        viewport.set_texture_resolution((960, 720))
        from isaacsim.core.utils.stage import get_current_stage
        get_current_stage().GetPrimAtPath('/World/ReviewCamera').GetAttribute('focalLength').Set(32.)
        set_camera_view(np.array([-3.2, -3.8, 4.8]), np.array([.15, -.25, .75]),
                        camera_prim_path="/World/ReviewCamera", viewport_api=viewport)
    captures = []
    capture_index = 0
    def record_frame():
        global capture_index
        if REVIEW_MODE and RECORD_MODE:
            frames = REPORT.parent / 'frames'
            frames.mkdir(exist_ok=True)
            captures.append(capture_viewport_to_file(viewport, str(frames / f'{capture_index:04d}.png')))
            capture_index += 1
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    robot = Articulation(ROBOT_PATH)
    world.reset()
    robot.initialize()

    dof_names = list(robot.dof_names)
    pose_names = [name for name in STABLE_POSE if name in dof_names]
    pose_ids = np.asarray([dof_names.index(name) for name in pose_names], dtype=np.int32)
    pose_values = np.asarray([STABLE_POSE[name] for name in pose_names], dtype=np.float32)
    wheel_ids = np.asarray([dof_names.index(name) for name in WHEEL_NAMES], dtype=np.int32)

    # Settle in the official stable pose before any wheel command.
    robot.set_joint_positions(pose_values, joint_indices=pose_ids)
    robot.set_joint_velocities(np.zeros(3, dtype=np.float32), joint_indices=wheel_ids)
    for _ in range(90):
        robot.set_joint_positions(pose_values, joint_indices=pose_ids)
        robot.set_joint_velocities(np.zeros(3, dtype=np.float32), joint_indices=wheel_ids)
        world.step(render=RENDER)

    p0, q0 = robot.get_world_poses()
    samples = []
    aborted = False
    total_frames = 60 + 90 + 60

    for frame in range(total_frames):
        speed = ramp(frame)
        robot.set_joint_positions(pose_values, joint_indices=pose_ids)
        robot.set_joint_velocities(
            np.full(3, speed, dtype=np.float32), joint_indices=wheel_ids
        )
        world.step(render=RENDER)
        pos, quat = robot.get_world_poses()
        if frame % 6 == 0:
            record_frame()
        roll, pitch = roll_pitch_deg(quat[0])
        samples.append({
            "frame": frame,
            "wheel_speed_rad_s": speed,
            "xyz_m": np.asarray(pos[0], dtype=float).tolist(),
            "roll_deg": roll,
            "pitch_deg": pitch,
        })
        if max(abs(roll), abs(pitch)) > 12.0:
            aborted = True
            break

    # Explicit brake/settle phase; never reverse abruptly.
    for _ in range(60):
        robot.set_joint_positions(pose_values, joint_indices=pose_ids)
        robot.set_joint_velocities(np.zeros(3, dtype=np.float32), joint_indices=wheel_ids)
        world.step(render=RENDER)

    p1, q1 = robot.get_world_poses()
    delta = np.asarray(p1[0], dtype=float) - np.asarray(p0[0], dtype=float)
    max_tilt = max(
        (max(abs(s["roll_deg"]), abs(s["pitch_deg"])) for s in samples),
        default=0.0,
    )
    distance = float(np.linalg.norm(delta[:2]))
    passed = (not aborted) and distance > 0.03 and max_tilt < 12.0
    result = {
        "status": "passed" if passed else "failed",
        "method": "official Galaxea velocity-wheel / position-steer convention with ramp",
        "scene": str(SCENE),
        "robot_path": ROBOT_PATH,
        "peak_wheel_speed_rad_s": 0.8,
        "planar_distance_m": distance,
        "displacement_xyz_m": delta.tolist(),
        "max_abs_roll_or_pitch_deg": max_tilt,
        "tilt_watchdog_aborted": aborted,
        "start_orientation_wxyz": np.asarray(q0[0], dtype=float).tolist(),
        "end_orientation_wxyz": np.asarray(q1[0], dtype=float).tolist(),
        "sample_count": len(samples),
        "samples": samples[::15],
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("R1PRO_SAFE_MOTION " + json.dumps(result, ensure_ascii=False))
    if REVIEW_MODE:
        if RECORD_MODE:
            for frame in range(total_frames):
                robot.set_joint_positions(pose_values, joint_indices=pose_ids)
                robot.set_joint_velocities(np.full(3, -ramp(frame), dtype=np.float32), joint_indices=wheel_ids)
                world.step(render=True)
                if frame % 6 == 0:
                    record_frame()
                _, quat = robot.get_world_poses()
                if max(abs(v) for v in roll_pitch_deg(quat[0])) > 12:
                    raise RuntimeError('Reverse motion tilt watchdog')
            robot.set_joint_velocities(np.zeros(3, dtype=np.float32), joint_indices=wheel_ids)
            for _ in range(60):
                robot.set_joint_positions(pose_values, joint_indices=pose_ids)
                world.step(render=True)
        for _ in range(30):
            world.render()
        capture = capture_viewport_to_file(viewport, str(REPORT.parent / "isaac_robot_review.png"))
        for _ in range(60):
            APP.update()
        print("ISAAC_REVIEW_CAPTURE_REQUESTED", flush=True)
    if not passed:
        raise RuntimeError("Safe chassis verification did not pass; see report")
    if GUI_MODE:
        print("Demo complete. Close the Isaac Sim window when you are finished viewing it.")
        while APP.is_running():
            robot.set_joint_positions(pose_values, joint_indices=pose_ids)
            robot.set_joint_velocities(
                np.zeros(3, dtype=np.float32), joint_indices=wheel_ids
            )
            world.step(render=True)
finally:
    APP.close()
