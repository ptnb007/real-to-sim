"""Headless Isaac Sim test: approach and physically lift the demo can."""

import json
import sys
import traceback
import asyncio
import xml.etree.ElementTree as ET
from pathlib import Path

GUI_DEMO = "--demo-gui" in sys.argv
GRIPPER_TEST = "--gripper-test" in sys.argv
PATH_TEST = "--path-test" in sys.argv
CALIBRATION_TEST = "--calibration-test" in sys.argv
SETTLE_TEST = "--settle-test" in sys.argv
RECORD_DATA = "--record-data" in sys.argv
TEST_CYCLES = 1
PAUSE_SELF_TEST = "--pause-self-test" in sys.argv
if PAUSE_SELF_TEST:
    sys.argv.remove("--pause-self-test")
if "--cycles" in sys.argv:
    option_index = sys.argv.index("--cycles")
    TEST_CYCLES = int(sys.argv[option_index + 1])
    del sys.argv[option_index:option_index + 2]
if GUI_DEMO:
    sys.argv.remove("--demo-gui")
if GRIPPER_TEST:
    sys.argv.remove("--gripper-test")
if PATH_TEST:
    sys.argv.remove("--path-test")
if CALIBRATION_TEST:
    sys.argv.remove("--calibration-test")
if SETTLE_TEST:
    sys.argv.remove("--settle-test")
if RECORD_DATA:
    sys.argv.remove("--record-data")
sys.argv.append("--/app/vulkan=false")

from isaacsim import SimulationApp

APP = SimulationApp(
    {"headless": not GUI_DEMO, "renderer": "RaytracedLighting", "multi_gpu": False, "width": 1280, "height": 720}
)

import numpy as np
import omni.usd
import omni.kit.app
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.rotations import rot_matrix_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
SCENE = HERE / "robot_lab_grasp_demo.usda"
URDF = HERE / "r1pro_grasp_demo.urdf"
ARM_SIDE = "right"
IK_CONFIG = HERE / "grasp_demo_robot_description_right.yaml"
REPORT = HERE / "grasp_demo_report.json"
ROBOT_PATH = "/World/GalaxeaR1Pro/Robot"
CAN_PATH = "/World/DemoCan"
PHYSX_BODIES: dict[str, SingleRigidPrim] = {}
DATA_RECORDER = None
ARM_LIMITS = None
READY_Q = np.asarray([-0.5, -0.46, 0.65, -1.23, -2.27, -0.37, -1.44])


def wait_for_play(world: World) -> None:
    if GUI_DEMO or PAUSE_SELF_TEST:
        while APP.is_running() and not world.is_playing():
            APP.update()
        if not APP.is_running():
            raise KeyboardInterrupt


def simulation_step(world: World) -> None:
    """Advance physics and synchronously sample the configured data streams."""
    wait_for_play(world)
    world.step(render=GUI_DEMO or RECORD_DATA)
    if DATA_RECORDER is not None:
        DATA_RECORDER.step()


def advance(world: World, frames: int = 90) -> None:
    for _ in range(frames):
        simulation_step(world)


def tune_articulation_drives(stage, names: list[str]) -> None:
    """Give imported URDF joints damped position drives within effort limits."""
    for name in names:
        # The URDF importer puts the articulation below /World/.../Robot, so
        # all imported joint prims are children of this namespaced joint scope.
        joint_path = f"{ROBOT_PATH}/joints/{name}"
        joint = stage.GetPrimAtPath(joint_path)
        if not joint.IsValid():
            continue
        effort = 100.0 if name.startswith("torso_joint") else 55.0
        if "_arm_joint" in name:
            index = int(name.rsplit("joint", 1)[1])
            effort = [55.0, 55.0, 25.0, 25.0, 18.0, 18.0, 18.0][index - 1]
        elif "gripper_finger_joint" in name:
            effort = 35.0
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.GetStiffnessAttr().Set(1800.0)
        drive.GetDampingAttr().Set(120.0)
        drive.GetMaxForceAttr().Set(effort)


def move_joint_positions(
    world: World, robot: Robot, joint_indices: np.ndarray, target_positions: np.ndarray, frames: int = 60
) -> None:
    indices = np.asarray(joint_indices, dtype=np.int32)
    target = np.asarray(target_positions, dtype=np.float64)
    start = np.asarray(robot.get_joint_positions(joint_indices=indices), dtype=np.float64)
    for alpha in np.linspace(1.0 / frames, 1.0, frames):
        wait_for_play(world)
        blend = 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5
        positions = start + blend * (target - start)
        robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=positions, joint_indices=indices)
        )
        simulation_step(world)


def move_ik_joints(
    world: World,
    joint_subset,
    action: ArticulationAction,
    frames: int = 180,
) -> None:
    """Track an IK joint target with the articulation drives and active physics."""
    indices = np.asarray(joint_subset.joint_indices, dtype=np.int32)
    start = np.asarray(joint_subset.get_joint_positions(), dtype=np.float64)
    target = np.asarray(action.joint_positions, dtype=np.float64)
    if ARM_LIMITS is not None and len(target) == 7:
        if np.any(target < ARM_LIMITS[:, 0] - 1e-5) or np.any(target > ARM_LIMITS[:, 1] + 1e-5):
            raise RuntimeError("IK solution exceeds the demonstration joint limits")
    frames = max(frames, int(np.ceil(np.max(np.abs(target - start)) * 1.875 / 0.45 * 60)))
    for alpha in np.linspace(0.0, 1.0, frames):
        wait_for_play(world)
        blend = 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5
        positions = start + blend * (target - start)
        joint_subset.apply_action(joint_positions=positions)
        simulation_step(world)


def move_cartesian_to(
    world: World,
    robot: Robot,
    arm_kinematics: ArticulationKinematicsSolver,
    target_position: np.ndarray,
    target_orientation: np.ndarray,
) -> tuple[ArticulationAction, bool]:
    """Follow a short-step Cartesian path, warm-starting every IK solve."""
    start_position, _ = arm_kinematics.compute_end_effector_pose()
    joint_subset = arm_kinematics.get_joints_subset()
    distance = float(np.linalg.norm(np.asarray(target_position) - np.asarray(start_position)))
    if 1e-6 < distance <= 0.12:
        # The straight-down grasp segment is near a Lula branch boundary:
        # intermediate samples can fail even though the final pose is valid.
        # Solve that endpoint directly, then use the physical articulation
        # drives for a slow, smooth joint-space transition.
        final_action, success = arm_kinematics.compute_inverse_kinematics(
            target_position=np.asarray(target_position),
            target_orientation=target_orientation,
            position_tolerance=0.025,
        )
        if not success:
            return final_action, False
        move_ik_joints(world, joint_subset, final_action, frames=240)
        for _ in range(90):
            simulation_step(world)
        achieved, _ = arm_kinematics.compute_end_effector_pose()
        if np.linalg.norm(np.asarray(target_position) - np.asarray(achieved)) > 0.025:
            return final_action, False
        return final_action, True
    # Dense Cartesian samples prevent Lula warm-start IK from jumping to a
    # distant branch on the near-singular vertical descent at the table edge.
    path_steps = max(1, int(np.ceil(distance / 0.01)))
    final_action = None
    for fraction in np.linspace(1.0 / path_steps, 1.0, path_steps):
        waypoint = np.asarray(start_position) + fraction * (
            np.asarray(target_position) - np.asarray(start_position)
        )
        final_action, success = arm_kinematics.compute_inverse_kinematics(
            target_position=waypoint,
            target_orientation=target_orientation,
            position_tolerance=0.025,
        )
        if not success:
            return final_action, False
        move_ik_joints(
            world,
            joint_subset,
            final_action,
            # Allow the PhysX drives to converge; short position teleports
            # produced large end-effector errors and invalid contact poses.
            frames=18,
        )
    # Let the physical joint drives catch up to the final setpoint before
    # reading the achieved pose or planning the next segment.
    for _ in range(90):
        simulation_step(world)
    # Close the residual error with measured-pose feedback. Lula's model and
    # the reconstructed USD articulation differ slightly at the wrist.
    for _ in range(3):
        achieved, _ = arm_kinematics.compute_end_effector_pose()
        residual = np.asarray(target_position) - np.asarray(achieved)
        if np.linalg.norm(residual) < 0.012:
            break
        corrected, ok = arm_kinematics.compute_inverse_kinematics(
            target_position=np.asarray(target_position),
            target_orientation=target_orientation,
            position_tolerance=0.008,
        )
        if not ok:
            break
        move_ik_joints(world, joint_subset, corrected, frames=90)
        for _ in range(90):
            simulation_step(world)
    return final_action, final_action is not None


def get_can_position(stage) -> np.ndarray:
    if CAN_PATH in PHYSX_BODIES:
        return np.asarray(PHYSX_BODIES[CAN_PATH].get_current_dynamic_state().position, dtype=np.float64)
    prim = stage.GetPrimAtPath(CAN_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"Demo target is missing: {CAN_PATH}")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.asarray(transform.ExtractTranslation(), dtype=np.float64)


def get_can_z(stage) -> float:
    return float(get_can_position(stage)[2])


def get_gripper_link_positions(stage) -> dict[str, list[float]]:
    prefix = f"{ARM_SIDE}_gripper"
    result = {}
    for name in (f"{prefix}_link", f"{prefix}_finger_link1", f"{prefix}_finger_link2"):
        path = f"/World/GalaxeaR1Pro/Robot/{name}"
        if path in PHYSX_BODIES:
            position = PHYSX_BODIES[path].get_current_dynamic_state().position
        else:
            position = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()
        result[name] = np.asarray(position, dtype=np.float64).tolist()
    return result


def main() -> dict:
    global ARM_LIMITS
    global DATA_RECORDER
    REPORT.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    if not all(path.is_file() for path in (SCENE, URDF, IK_CONFIG)):
        raise FileNotFoundError("Scene, source URDF, or IK configuration is missing")

    context = omni.usd.get_context()
    if not context.open_stage(str(SCENE)):
        raise RuntimeError(f"Could not open stage: {SCENE}")
    for _ in range(240):
        APP.update()
        if context.get_stage() is not None:
            break
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not finish loading the grasp-demo stage")
    if GUI_DEMO:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.asarray([2.4, -1.2, 1.8]), target=np.asarray([0.65, 0.25, 0.85]))

    robot_prim = stage.GetPrimAtPath(ROBOT_PATH)
    if not robot_prim.IsValid():
        # Some USD reference resolvers keep the referenced source prim name.
        candidate = "/World/GalaxeaR1Pro/Robot/r1pro_urdf/base_link"
        if stage.GetPrimAtPath(candidate).IsValid():
            robot_path = candidate
        else:
            raise RuntimeError(f"R1 Pro articulation prim not found at {ROBOT_PATH}")
    else:
        robot_path = ROBOT_PATH

    for finger in (1, 2):
        collision_root = stage.GetPrimAtPath(f"{ROBOT_PATH}/right_gripper_finger_link{finger}/collisions")
        if collision_root.IsValid():
            collision_root.SetInstanceable(False)
    adjusted_contacts = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and ("right_gripper_finger" in str(prim.GetPath()) or str(prim.GetPath()).startswith(CAN_PATH)):
            contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            contact.CreateContactOffsetAttr(0.001)
            contact.CreateRestOffsetAttr(0.0)
            adjusted_contacts.append(str(prim.GetPath()))
            if "right_gripper_finger" in str(prim.GetPath()):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
    print("CONTACT_SHAPES " + json.dumps(adjusted_contacts), flush=True)
    limits_by_name = {joint.attrib["name"]: joint.find("limit").attrib
                      for joint in ET.parse(URDF).getroot().findall("joint")
                      if joint.attrib["name"].startswith("right_arm_joint")}
    ARM_LIMITS = np.asarray([[float(limits_by_name[f"right_arm_joint{i}"][key])
                             for key in ("lower", "upper")] for i in range(1, 8)])
    for i in range(1, 8):
        joint = stage.GetPrimAtPath(f"{ROBOT_PATH}/joints/right_arm_joint{i}")
        joint.GetAttribute("physics:lowerLimit").Set(float(np.degrees(ARM_LIMITS[i - 1, 0])))
        joint.GetAttribute("physics:upperLimit").Set(float(np.degrees(ARM_LIMITS[i - 1, 1])))
        joint.GetAttribute("physxJoint:maxJointVelocity").Set(float(np.degrees(0.6)))
    world = World(stage_units_in_meters=1.0)
    # core.prims.Articulation is the low-level view and has no controller API.
    # The Core API Robot wrapper owns the articulation controller in Isaac 5.1.
    robot = Robot(prim_path=robot_path, name="r1pro_grasp_demo")
    if RECORD_DATA:
        from grasp_data_recorder import GraspDataRecorder
        DATA_RECORDER = GraspDataRecorder(HERE, world, robot, stage, sample_every=6)
        DATA_RECORDER.initialize_sensors()
    # Sensors that depend on the PhysX scene must exist before the reset which
    # initializes the simulation views and starts the timeline.
    world.reset()
    robot.initialize()
    if RECORD_DATA:
        advance(world, 30)
    controller = robot.get_articulation_controller()
    names = list(robot.dof_names)
    if PAUSE_SELF_TEST:
        world.pause()
        paused_q = np.asarray(robot.get_joint_positions()).copy()
        pause_observations = []
        async def resume_for_check():
            for _ in range(15):
                await omni.kit.app.get_app().next_update_async()
                pause_observations.append(float(np.max(np.abs(np.asarray(robot.get_joint_positions()) - paused_q))))
            world.play()
        asyncio.ensure_future(resume_for_check())
        simulation_step(world)
        check = {"paused_updates": len(pause_observations),
                 "max_joint_change_while_paused": max(pause_observations),
                 "resumed": world.is_playing()}
        (HERE / "pause_resume_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
        if check["max_joint_change_while_paused"] > 1e-6 or not check["resumed"]:
            raise RuntimeError("Pause/resume validation failed")
    # The exported articulation layer carries non-zero rigid-body velocity
    # samples from an earlier sim session. Clear those before evaluating grasp.
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            if rb.GetVelocityAttr():
                rb.GetVelocityAttr().Set(Gf.Vec3f(0.0))
            if rb.GetAngularVelocityAttr():
                rb.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0))
    for body_path in [CAN_PATH] + [
        f"/World/GalaxeaR1Pro/Robot/{ARM_SIDE}_gripper{suffix}"
        for suffix in ("_link", "_finger_link1", "_finger_link2")
    ] + [
        f"/World/GalaxeaR1Pro/Robot/{ARM_SIDE}_arm_base_link",
        *[f"/World/GalaxeaR1Pro/Robot/{ARM_SIDE}_arm_link{i}" for i in range(1, 8)],
    ]:
        body = SingleRigidPrim(prim_path=body_path, name="grasp_probe_" + body_path.rsplit("/", 1)[-1])
        body.initialize()
        PHYSX_BODIES[body_path] = body
    if SETTLE_TEST:
        points = {}
        for path in ("/World/GraspTableSupport", CAN_PATH, "/World/Environment/rear_table/rear_table__3_collider"):
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                mat = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                points[path] = {
                    "world_origin": list(mat.ExtractTranslation()),
                    "schemas": list(prim.GetAppliedSchemas()),
                    "collision": bool(UsdPhysics.CollisionAPI(prim)) if prim.HasAPI(UsdPhysics.CollisionAPI) else False,
                }
        advance(world, 240)
        snapshots = []
        for elapsed in (0, 1, 10, 60, 120, 240):
            if elapsed:
                advance(world, elapsed - (snapshots[-1]["frame"] if snapshots else 0))
            snapshots.append({"frame": elapsed, "can": get_can_position(stage).tolist()})
        result = {"status": "settle_test", "prim_info": points, "snapshots": snapshots}
        print("SETTLE_TEST_RESULT " + json.dumps(result), flush=True)
        REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    if not names:
        raise RuntimeError("R1 Pro articulation initialized without any joints")
    tune_articulation_drives(stage, names)
    # Hold the torso upright during this arm-only demonstration. These drive
    # limits are simulation settings, not hardware actuator specifications.
    torso_indices = np.asarray([names.index(f"torso_joint{i}") for i in range(1, 5)], dtype=np.int32)
    controller.set_gains(kps=np.asarray([10000.0 if "gripper_finger" in n else 1800.0 for n in names]),
                         kds=np.asarray([200.0 if "gripper_finger" in n else 120.0 for n in names]))
    controller.set_max_efforts(np.full(4, 1000.0), joint_indices=torso_indices)
    move_joint_positions(world, robot, torso_indices, np.asarray([0.5, -0.8, -0.3, 0.0]), frames=240)
    advance(world, 120)
    required = [f"{ARM_SIDE}_arm_joint{i}" for i in range(1, 8)] + [
        f"{ARM_SIDE}_gripper_finger_joint1",
        f"{ARM_SIDE}_gripper_finger_joint2",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"Missing articulation DOFs: {missing}; found {names}")

    solver = LulaKinematicsSolver(robot_description_path=str(IK_CONFIG), urdf_path=str(URDF))
    # Lula otherwise assumes the robot base is at the USD world origin, while
    # this reconstructed scene places the mobile base several meters away.
    base_position, base_orientation = robot.get_world_pose()
    solver.set_robot_base_pose(base_position, base_orientation)
    arm_kinematics = ArticulationKinematicsSolver(robot, solver, f"{ARM_SIDE}_gripper_link")
    gripper_indices = np.asarray([names.index(name) for name in required[7:]], dtype=np.int32)
    if CALIBRATION_TEST:
        solver_indices = np.asarray([names.index(name) for name in solver.get_joint_names()], dtype=np.int32)
        frames = [f"{ARM_SIDE}_arm_base_link"] + [f"{ARM_SIDE}_arm_link{i}" for i in range(1, 8)] + [f"{ARM_SIDE}_gripper_link"]

        def sample_calibration(label: str) -> None:
            base_pos, base_quat = robot.get_world_pose()
            q = np.asarray(robot.get_joint_positions(joint_indices=solver_indices), dtype=np.float64)
            solver.set_robot_base_pose(base_pos, base_quat)
            physical_ee_path = f"/World/GalaxeaR1Pro/Robot/{ARM_SIDE}_gripper_link"
            physical_state = PHYSX_BODIES[physical_ee_path].get_current_dynamic_state()
            fk_position, fk_rotation = arm_kinematics.compute_end_effector_pose()
            chain_checks = []
            for frame in frames:
                fk_frame_position, _ = solver.compute_forward_kinematics(frame, q)
                prim_path = f"/World/GalaxeaR1Pro/Robot/{frame}"
                physx_frame_position = PHYSX_BODIES[prim_path].get_current_dynamic_state().position
                chain_checks.append({
                    "frame": frame,
                    "lula_fk_world_m": np.asarray(fk_frame_position).tolist(),
                    "physx_world_m": np.asarray(physx_frame_position).tolist(),
                    "delta_m": (np.asarray(physx_frame_position) - np.asarray(fk_frame_position)).tolist(),
                })
            arm_base_check = chain_checks[0]
            print("CALIBRATION_POSE_CHECK " + json.dumps({
                "sample": label,
                "robot_base_position_m": np.asarray(base_pos).tolist(),
                "solver_joint_names": list(solver.get_joint_names()),
                "solver_joint_positions_rad": q.tolist(),
                "applied_joint_position_targets": np.asarray(controller.get_applied_action().joint_positions).reshape(-1).tolist(),
                "lula_fk_gripper_position_m": np.asarray(fk_position).tolist(),
                "physx_gripper_position_m": np.asarray(physical_state.position).tolist(),
                "position_delta_physx_minus_lula_m": (np.asarray(physical_state.position) - np.asarray(fk_position)).tolist(),
                "torso_joint_positions_rad": [float(robot.get_joint_positions()[names.index(name)]) for name in ("torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4")],
                "arm_base_position_delta_m": arm_base_check["delta_m"],
            }, ensure_ascii=False), flush=True)

        sample_calibration("frame_0")
        elapsed = 0
        for frames_to_advance in (1, 9, 50):
            advance(world, frames_to_advance)
            elapsed += frames_to_advance
            sample_calibration(f"frame_{elapsed}")
        return {"status": "calibration_pose_check_complete"}
    if GRIPPER_TEST:
        advance(world, 60)
        print("GRIPPER_TEST_START " + json.dumps({
            "names": [names[int(i)] for i in gripper_indices],
            "indices": gripper_indices.tolist(),
            "positions": robot.get_joint_positions(joint_indices=gripper_indices).tolist(),
        }), flush=True)
        move_joint_positions(world, robot, gripper_indices, np.asarray([0.05, -0.05]), frames=120)
        print("GRIPPER_TEST_OPEN " + json.dumps({
            "positions": robot.get_joint_positions(joint_indices=gripper_indices).tolist(),
            "targets": robot.get_articulation_controller().get_applied_action().joint_positions[gripper_indices].tolist(),
        }), flush=True)
        move_joint_positions(world, robot, gripper_indices, np.asarray([0.0, 0.0]), frames=120)
        advance(world, 120)
        print("GRIPPER_TEST_CLOSE " + json.dumps({
            "positions": robot.get_joint_positions(joint_indices=gripper_indices).tolist(),
            "targets": robot.get_articulation_controller().get_applied_action().joint_positions[gripper_indices].tolist(),
        }), flush=True)
        return {"status": "gripper_test_complete"}
    base_position, base_orientation = robot.get_world_pose()
    arm_indices = np.asarray([names.index(f"right_arm_joint{i}") for i in range(1, 8)], dtype=np.int32)
    move_joint_positions(world, robot, arm_indices, READY_Q, frames=600)
    advance(world, 120)
    ee_position, ee_orientation = arm_kinematics.compute_end_effector_pose()
    # Keep position IK unconstrained while the wrist/tool frame is being
    # calibrated; the physical finger collision shapes remain authoritative.
    gripper_orientation = np.asarray([1.0, 0.0, 0.0, 0.0])
    print(
        "GRASP_DIAGNOSTICS "
        + json.dumps(
            {
                "robot_base_position_m": np.asarray(base_position).tolist(),
                "robot_base_orientation_wxyz": np.asarray(base_orientation).tolist(),
                f"{ARM_SIDE}_gripper_position_m": np.asarray(ee_position).tolist(),
                "left_arm_joint_positions_rad_or_m": np.asarray(robot.get_joint_positions()).tolist(),
            }
        ),
        flush=True,
    )

    # Let the table collision and dynamic target settle before manipulation.
    advance(world, 120)
    can_start_z = get_can_z(stage)
    can_initial_position = get_can_position(stage)

    # Parallel-jaw demo stroke: fingers move symmetrically by 50 mm each.
    move_joint_positions(world, robot, gripper_indices, np.asarray([0.05, -0.05]), frames=60)

    # Position-only IK waypoints around the modeled object; each is solved using
    # the current joint state as the warm start and verified by Lula.
    # Use the settled rigid-body pose: the reconstructed tabletop collision
    # determines the final can height, not its initial stage placement.
    # The USD scene is translated relative to the authored asset, and the
    # tabletop collision can settle the rigid body at an offset from its
    # authored transform.  Always grasp the measured PhysX pose.
    can_center = np.asarray(can_initial_position, dtype=np.float64)
    # With the jaw span now horizontal, center the hand directly over the
    # reconstructed object's measured tabletop position.
    gripper_tcp_offset = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    grasp_xy = can_center + gripper_tcp_offset
    waypoints = [
        # First lift the hand clear of the tabletop, then translate over the
        # target and descend; this avoids sweeping the fingers across the part.
        ("above_target", grasp_xy + np.asarray([0.0, 0.0, 0.20])),
        # Keep the wrist at a reachable height; the finger geometry extends
        # below the link origin to pinch the object from its sides.
        ("grasp", grasp_xy + np.asarray([0.0, 0.0, 0.10])),
    ]
    ik_results = []
    reachability_probe = []
    observed_end_effector_poses = []
    target_fell_during_approach = False
    clearance_action = ArticulationAction(joint_positions=READY_Q, joint_indices=arm_indices)
    above_action = None
    grasp_action = None
    for label, target in waypoints:
        action, success = move_cartesian_to(
            world, robot, arm_kinematics, target, gripper_orientation
        )
        ik_results.append({"waypoint": label, "target_m": target.tolist(), "ik_success": bool(success)})
        if not success:
            for dx in (-0.90, -0.60, -0.30, 0.0, 0.30, 0.60, 0.90):
                for dy in (-0.90, -0.60, -0.30, 0.0, 0.30, 0.60, 0.90):
                    for dz in (-0.20, 0.0, 0.20):
                        probe_target = target + np.asarray([dx, dy, dz])
                        _, probe_success = arm_kinematics.compute_inverse_kinematics(
                            target_position=probe_target,
                            target_orientation=gripper_orientation,
                            position_tolerance=0.025,
                        )
                        if probe_success:
                            reachability_probe.append(
                                {"target_m": probe_target.tolist(), "ik_success": True}
                            )
            REPORT.write_text(
                json.dumps(
                    {
                        "status": "ik_unreachable",
                        "failed_waypoint": label,
                        "base_position_m": np.asarray(base_position).tolist(),
                        "left_gripper_position_m": np.asarray(ee_position).tolist(),
                        "reachability_probe": reachability_probe,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise RuntimeError(f"IK did not converge for {label} at {target.tolist()}")
        observed_position, observed_rotation = arm_kinematics.compute_end_effector_pose()
        solver_joint_indices = np.asarray([names.index(name) for name in solver.get_joint_names()], dtype=np.int32)
        achieved_joints = np.asarray(robot.get_joint_positions(joint_indices=solver_joint_indices))
        physical_ee_position = np.asarray(
            PHYSX_BODIES[f"/World/GalaxeaR1Pro/Robot/{ARM_SIDE}_gripper_link"]
            .get_current_dynamic_state().position,
            dtype=np.float64,
        )
        physical_position_error = float(np.linalg.norm(physical_ee_position - target))
        physical_tolerance = 0.12 if label == "above_target" else 0.035
        if physical_position_error > physical_tolerance:
            REPORT.write_text(json.dumps({
                "status": "physical_tracking_error",
                "failed_waypoint": label,
                "target_m": target.tolist(),
                "lula_position_m": np.asarray(observed_position).tolist(),
                "physx_position_m": physical_ee_position.tolist(),
                "physx_error_m": physical_position_error,
                "can_position_m": get_can_position(stage).tolist(),
                "ik_joint_targets": np.asarray(action.joint_positions).tolist(),
                "actual_solver_joint_positions": achieved_joints.tolist(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(f"PhysX gripper missed {label} by {physical_position_error:.3f} m")
        observed_end_effector_poses.append(
            {
                "waypoint": label,
                "position_m": np.asarray(observed_position).tolist(),
                "ik_action_indices": np.asarray(action.joint_indices).tolist(),
                "ik_action_index_names": [names[int(i)] for i in np.asarray(action.joint_indices)],
                "solver_joint_names": list(solver.get_joint_names()),
                "solver_joint_indices": solver_joint_indices.tolist(),
                "can_position_after_waypoint_m": get_can_position(stage).tolist(),
                "can_z_after_waypoint_m": get_can_z(stage),
                "rotation_matrix": np.asarray(observed_rotation).tolist(),
                "position_error_m": float(np.linalg.norm(np.asarray(observed_position) - target)),
                "ik_joint_targets": np.asarray(action.joint_positions).tolist(),
                "actual_solver_joint_positions": achieved_joints.tolist(),
                "actual_gripper_link_positions_m": get_gripper_link_positions(stage),
            }
        )
        if label == "initial_clearance":
            clearance_action = action
        if label == "above_target":
            above_action = action
        if label == "grasp":
            grasp_action = action
        can_after_waypoint = get_can_position(stage)
        if can_after_waypoint[2] < can_start_z - 0.025:
            # Keep the visible demo running through the planned close-and-lift
            # motion even if contact knocked the object away. This is still a
            # fully dynamic PhysX test; it does not attach or reposition the
            # target. The final report will show whether it was actually lifted.
            target_fell_during_approach = True
            print(
                "TARGET_FELL_DURING_APPROACH "
                + json.dumps({
                    "waypoint": label,
                    "can_start_position_m": can_initial_position.tolist(),
                    "can_after_waypoint_m": can_after_waypoint.tolist(),
                }),
                flush=True,
            )
        if PATH_TEST and label == "clearance":
            debug_result = {
                "status": "path_test_complete",
                "waypoint": observed_end_effector_poses[-1],
                "robot_dof_names": names,
            }
            REPORT.write_text(json.dumps(debug_result, ensure_ascii=False, indent=2), encoding="utf-8")
            return debug_result

    # Close the physically simulated gripper around the dynamic target. No
    # attachment, kinematic toggle, or scripted object transform is used.
    move_joint_positions(world, robot, gripper_indices, np.asarray([0.0, 0.0]), frames=60)
    advance(world, 120)
    can_z_after_close = get_can_z(stage)
    can_position_after_close = get_can_position(stage).tolist()
    gripper_links_after_close = get_gripper_link_positions(stage)
    gripper_after_close = np.asarray(robot.get_joint_positions())[gripper_indices].tolist()
    closure_state = {"status": "closed", "can_position_after_close_m": can_position_after_close,
                     "gripper_positions_after_close_m": gripper_after_close,
                     "gripper_link_positions_after_close_m": gripper_links_after_close}
    REPORT.write_text(json.dumps(closure_state, indent=2), encoding="utf-8")
    print("CLOSURE_STATE " + json.dumps(closure_state), flush=True)

    if can_z_after_close < can_start_z - 0.025:
        print(
            "OBJECT_NOT_RETAINED_AFTER_CLOSURE "
            + json.dumps({
                "can_start_z_m": can_start_z,
                "can_z_after_close_m": can_z_after_close,
                "can_position_after_close_m": can_position_after_close,
            }),
            flush=True,
        )

    if clearance_action is None:
        raise RuntimeError("No solved clearance pose is available for the retreat motion")
    # Retract to the already-solved raised pose via physical articulation
    # drives. This avoids a Lula FK edge case on the short vertical lift path.
    if above_action is None:
        raise RuntimeError("No pre-grasp lift pose was recorded")
    move_ik_joints(world, arm_kinematics.get_joints_subset(), above_action, frames=360)
    advance(world, 90)
    observed_lift_position = np.asarray(
        PHYSX_BODIES[f"/World/GalaxeaR1Pro/Robot/{ARM_SIDE}_gripper_link"]
        .get_current_dynamic_state().position,
        dtype=np.float64,
    )

    # Allow free dynamics to reveal whether the object is actually retained.
    advance(world, 90)
    can_final_z = get_can_z(stage)
    can_final_position = get_can_position(stage).tolist()
    base_final_position, base_final_orientation = robot.get_world_pose()
    grasp_displacement = float(can_final_z - can_z_after_close)
    lifted = grasp_displacement > 0.045 and can_final_z > can_start_z - 0.025
    result = {
        "status": "passed" if lifted else "failed",
        "task": "R1 Pro faces the reconstructed table from the front and attempts a dynamic contact grasp of the small brown tabletop object",
        "robot_prim": robot_path,
        "robot_base_final_position_m": np.asarray(base_final_position).tolist(),
        "robot_base_final_orientation_wxyz": np.asarray(base_final_orientation).tolist(),
        "robot_base_xy_m": np.asarray(base_final_position[:2]).tolist(),
        "dof_count": int(robot.num_dof),
        "can_start_z_m": can_start_z,
        "target_fell_during_approach": target_fell_during_approach,
        "can_final_z_m": can_final_z,
        "can_position_after_close_m": can_position_after_close,
        "grasp_model": "dynamic rigid body with PhysX collision/contact; no scripted attachment",
        "can_initial_position_m": can_initial_position.tolist(),
        "can_final_position_m": can_final_position,
        "can_z_after_close_m": can_z_after_close,
        "lift_delta_m": grasp_displacement,
        "gripper_positions_after_close_m": gripper_after_close,
        "gripper_link_positions_after_close_m": gripper_links_after_close,
        "observed_end_effector_poses": observed_end_effector_poses,
        "observed_lift_end_effector_position_m": np.asarray(observed_lift_position).tolist(),
        "waypoints": ik_results,
        "assumptions": [
            "Target proxy matches the reconstructed brown object's 50 mm x 32 mm mesh bounds; its 35 g mass is an illustrative assumption.",
            "R1 Pro finger travel is patched to +/-50 mm from a separate simulation model, not verified against this hardware revision.",
            "A pass requires the dynamic object to rise at least 45 mm after closure and the lift motion.",
            "Robot base pose is approximate and intended for this local simulation demonstration only.",
        ],
    }
    if can_final_z < can_start_z - 0.025:
        result["status"] = "object_fell_off_table"
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("GRASP_DEMO_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    if GUI_DEMO:
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.asarray([1.65, -0.25, 1.35]), target=np.asarray([0.8, 0.35, 0.88]))
        for _ in range(12):
            APP.update()
        capture_viewport_to_file(get_active_viewport(), str(HERE / "grasp_demo_current.png"))
    if not lifted:
        print(
            f"PHYSICAL_GRASP_NOT_ACHIEVED: dynamic object rose only {grasp_displacement:.3f} m; "
            "the object remains governed by PhysX and was not scripted to follow the gripper.",
            flush=True,
        )
    cycle_summaries = [{"cycle": 1, "status": result["status"], "lift_m": grasp_displacement}]
    loop_report = HERE / "grasp_loop_report.json"
    loop_report.write_text(json.dumps({"status": "running" if GUI_DEMO else "testing",
                                      "cycles": cycle_summaries}, indent=2), encoding="utf-8")
    cycle_number = 2
    joint_subset = arm_kinematics.get_joints_subset()
    while lifted and (GUI_DEMO or cycle_number <= TEST_CYCLES):
        wait_for_play(world)
        # Place the carried object at its original tabletop location using
        # the measured object-to-hand offset, avoiding cumulative drift.
        hand_position = np.asarray(PHYSX_BODIES[f"{ROBOT_PATH}/{ARM_SIDE}_gripper_link"]
                                   .get_current_dynamic_state().position)
        carried_offset = get_can_position(stage) - hand_position
        place_target = can_initial_position - carried_offset + np.asarray([0.0, 0.0, 0.003])
        place_action, ok = arm_kinematics.compute_inverse_kinematics(
            target_position=place_target, target_orientation=gripper_orientation,
            position_tolerance=0.01)
        if not ok:
            raise RuntimeError("Could not solve the physical put-down pose")
        move_ik_joints(world, joint_subset, place_action, frames=360)
        advance(world, 60)
        move_joint_positions(world, robot, gripper_indices, np.asarray([0.05, -0.05]), frames=120)
        advance(world, 120)
        placed_position = get_can_position(stage)
        if placed_position[2] < can_start_z - 0.025:
            raise RuntimeError("Object left the tabletop during release")
        move_ik_joints(world, joint_subset, above_action, frames=240)
        advance(world, 60)
        # Warm-start the next cycle from the same elbow/wrist branch.
        center = get_can_position(stage)
        next_above, ok = arm_kinematics.compute_inverse_kinematics(
            target_position=center + np.asarray([0.0, 0.0, 0.20]),
            target_orientation=gripper_orientation, position_tolerance=0.01)
        if not ok:
            raise RuntimeError("Next-cycle approach is outside the limited workspace")
        move_ik_joints(world, joint_subset, next_above, frames=180)
        next_grasp, ok = arm_kinematics.compute_inverse_kinematics(
            target_position=center + np.asarray([0.0, 0.0, 0.10]),
            target_orientation=gripper_orientation, position_tolerance=0.01)
        if not ok:
            raise RuntimeError("Next-cycle grasp is outside the limited workspace")
        move_ik_joints(world, joint_subset, next_grasp, frames=360)
        advance(world, 60)
        move_joint_positions(world, robot, gripper_indices, np.zeros(2), frames=120)
        advance(world, 120)
        before_lift = get_can_z(stage)
        move_ik_joints(world, joint_subset, next_above, frames=360)
        advance(world, 180)
        final_position = get_can_position(stage)
        lifted = final_position[2] - before_lift > 0.045
        lifted = lifted and np.linalg.norm(final_position[:2] - center[:2]) < 0.10
        cycle_summaries.append({"cycle": cycle_number, "status": "passed" if lifted else "failed",
                                "lift_m": float(final_position[2] - before_lift),
                                "placed_position_m": placed_position.tolist(),
                                "object_position_m": final_position.tolist()})
        loop_report.write_text(json.dumps({"status": "running" if GUI_DEMO and lifted else "complete",
                                           "cycles": cycle_summaries,
                                           "joint_limits_rad": ARM_LIMITS.tolist(),
                                           "max_command_speed_rad_s": 0.45}, indent=2), encoding="utf-8")
        print("GRASP_LOOP_CYCLE " + json.dumps(cycle_summaries[-1]), flush=True)
        above_action, grasp_action = next_above, next_grasp
        cycle_number += 1
        if not lifted:
            raise RuntimeError("Repeat grasp failed to retain the dynamic object")
    return result


exit_code = 0
try:
    main()
except KeyboardInterrupt:
    print("Grasp demonstration closed.", flush=True)
except BaseException as exc:
    exit_code = 1
    failure = {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
    if REPORT.is_file():
        try:
            previous = json.loads(REPORT.read_text(encoding="utf-8"))
            if previous.get("status") not in (None, "error"):
                failure = {**previous, **failure, "previous_result": previous}
        except (OSError, json.JSONDecodeError):
            pass
    REPORT.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
    traceback.print_exc()
finally:
    if DATA_RECORDER is not None:
        try:
            manifest = DATA_RECORDER.close()
            print("DATA_RECORDING_RESULT " + json.dumps(manifest), flush=True)
        except BaseException:
            traceback.print_exc()
    if GUI_DEMO and not exit_code:
        print("Grasp demo complete. Close the Isaac Sim window to exit.", flush=True)
        while APP.is_running():
            APP.update()
    else:
        APP.close()
if exit_code:
    raise SystemExit(exit_code)
