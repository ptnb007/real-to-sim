"""Run a short Galaxea R1 Pro base-motion test in focus_v4."""

import json
import sys
from pathlib import Path

import numpy as np

sys.argv.append("--/app/vulkan=false")
from isaacsim import SimulationApp

APP = SimulationApp({"headless": True, "multi_gpu": False})

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import open_stage


HERE = Path(__file__).resolve().parent
SCENE = HERE / "focus_v4" / "focused_zone_with_r1pro.usda"
REPORT = HERE / "focus_v4" / "r1pro_drive_verification.json"


try:
    open_stage(str(SCENE))
    world = World(stage_units_in_meters=1.0)
    robot = Articulation("/World/R1Pro/base_link")
    world.reset()
    robot.initialize()

    names = list(robot.dof_names)
    steer_names = [f"steer_motor_joint{i}" for i in (1, 2, 3)]
    wheel_names = [f"wheel_motor_joint{i}" for i in (1, 2, 3)]
    steer_ids = np.array([names.index(name) for name in steer_names], dtype=np.int32)
    wheel_ids = np.array([names.index(name) for name in wheel_names], dtype=np.int32)

    robot.set_joint_positions(np.zeros(3, dtype=np.float32), joint_indices=steer_ids)
    # URDF import gives every revolute joint a position drive.  A wheel with
    # non-zero stiffness fights the velocity command and appears locked, so
    # switch the three continuous wheel joints to velocity-drive gains.
    robot.set_gains(
        kps=np.zeros(3, dtype=np.float32),
        kds=np.full(3, 250.0, dtype=np.float32),
        joint_indices=wheel_ids,
    )
    start_position, start_orientation = robot.get_world_poses()
    start_wheel_positions = np.asarray(robot.get_joint_positions(joint_indices=wheel_ids), dtype=float)

    # Keep the automated check short because focus_v4 contains hundreds of
    # meshes and the 8 GB GPU advances this scene relatively slowly.
    commanded_frames = 30
    for _ in range(commanded_frames):
        robot.set_joint_efforts(
            np.full(3, 18.0, dtype=np.float32), joint_indices=wheel_ids
        )
        world.step(render=False)

    robot.set_joint_efforts(np.zeros(3, dtype=np.float32), joint_indices=wheel_ids)
    for _ in range(10):
        world.step(render=False)

    end_position, end_orientation = robot.get_world_poses()
    start_xyz = np.asarray(start_position[0], dtype=float)
    end_xyz = np.asarray(end_position[0], dtype=float)
    displacement = end_xyz - start_xyz
    travelled = float(np.linalg.norm(displacement[:2]))
    end_wheel_positions = np.asarray(robot.get_joint_positions(joint_indices=wheel_ids), dtype=float)
    end_wheel_velocities = np.asarray(robot.get_joint_velocities(joint_indices=wheel_ids), dtype=float)

    result = {
        "status": "passed" if travelled > 0.05 else "failed",
        "scene": str(SCENE),
        "robot": "Galaxea R1 Pro",
        "steering_joints": steer_names,
        "wheel_joints": wheel_names,
        "wheel_effort_nm": 18.0,
        "commanded_frames": commanded_frames,
        "start_xyz_m": start_xyz.tolist(),
        "end_xyz_m": end_xyz.tolist(),
        "displacement_xyz_m": displacement.tolist(),
        "planar_distance_m": travelled,
        "start_wheel_positions_rad": start_wheel_positions.tolist(),
        "end_wheel_positions_rad": end_wheel_positions.tolist(),
        "end_wheel_velocities_rad_s": end_wheel_velocities.tolist(),
        "start_orientation_wxyz": np.asarray(start_orientation[0], dtype=float).tolist(),
        "end_orientation_wxyz": np.asarray(end_orientation[0], dtype=float).tolist(),
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("R1PRO_DRIVE_VERIFIED " + json.dumps(result, ensure_ascii=False))
    if result["status"] != "passed":
        raise RuntimeError("Wheel commands did not move the R1 Pro base")
finally:
    APP.close()
