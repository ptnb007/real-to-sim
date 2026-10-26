"""Isolated VLFM PersonNav experiment; never modifies the original HM3D tasks.

The simulator knows the inserted target pose. The policy receives only RGB-D,
odometry and the string 'person'. Semantic IDs and world target pose are strictly
evaluation-only. Use --prepare to inspect placement without loading CUDA models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("MAGNUM_LOG", "quiet")

import cv2
import numpy as np

# Run with cwd set to the existing WSL VLFM checkout.
sys.path.insert(0, str(Path.cwd()))
ASSET_ROOT = Path(__file__).resolve().parents[1] / "data/person_search"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "results/person_search"
PERSON_ASSETS = {"cesium": ASSET_ROOT / "CesiumMan.glb",
                 "rocketbox": ASSET_ROOT / "rocketbox/Male_Adult_01_standing.glb"}


def build_env(max_episode_steps=350):
    import vlfm.run  # Registers the baseline's Hydra components.
    import habitat
    from habitat.config.default import patch_config
    from habitat.config import read_write
    from hydra import compose, initialize_config_dir
    from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig

    with initialize_config_dir(version_base=None, config_dir=str(Path.cwd() / "config")):
        cfg = patch_config(compose(config_name="experiments/vlfm_objectnav_hm3d"))
    with read_write(cfg):
        cfg.habitat.dataset.data_path = "data/datasets/objectnav/hm3d/v1/val_mini/val_mini.json.gz"
        cfg.habitat.dataset.split = "val_mini"
        cfg.habitat.environment.iterator_options.shuffle = False
        cfg.habitat.environment.max_episode_steps = max_episode_steps
        cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        cfg.habitat.simulator.habitat_sim_v0.enable_physics = True
        for key in list(cfg.habitat.task.lab_sensors):
            if key not in ("gps_sensor", "compass_sensor", "heading_sensor"):
                del cfg.habitat.task.lab_sensors[key]
        cfg.habitat.task.measurements = {}
        sensors = cfg.habitat.simulator.agents.main_agent.sim_sensors
        semantic = HabitatSimSemanticSensorConfig()
        semantic.height = sensors.rgb_sensor.height
        semantic.width = sensors.rgb_sensor.width
        semantic.position = list(sensors.rgb_sensor.position)
        semantic.hfov = sensors.rgb_sensor.hfov
        sensors.semantic_sensor = semantic
        cfg.habitat_baselines.eval.video_option = ["disk"]
    env = habitat.Env(config=cfg.habitat)
    obs = env.reset()
    return cfg, env, obs


def place_person(env, target=None, yaw=0.0, asset_name="cesium"):
    import habitat_sim
    import magnum as mn
    sim = env.sim
    mgr = sim.get_object_template_manager()
    template = mgr.create_new_template(f"person_search_{asset_name}")
    template.render_asset_handle = str(PERSON_ASSETS[asset_name])
    template.collision_asset_handle = str(PERSON_ASSETS[asset_name])
    template.force_flat_shading = True
    template.orient_up = mn.Vector3(0, 1, 0)
    template.orient_front = mn.Vector3(0, 0, -1)
    tid = mgr.register_template(template)
    objects = sim.get_rigid_object_manager()
    person = objects.add_object_by_template_id(tid)
    bb = person.root_scene_node.cumulative_bb
    print("RAW_PERSON_BOUNDS", bb, flush=True)
    size = np.asarray(bb.size())
    scale = 1.7 / float(size[1])
    objects.remove_object_by_id(person.object_id)
    template.scale = mn.Vector3(scale)
    tid = mgr.register_template(template)
    person = objects.add_object_by_template_id(tid)
    # Habitat 0.2.4 ignores pose setters on STATIC objects. Place while
    # KINEMATIC, verify the pose, and only then freeze the body.
    person.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    person.semantic_id = 10001
    person.rotation = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3.y_axis())
    state = sim.get_agent_state()
    if target is None:
        sim.pathfinder.seed(41)
        candidates = []
        for _ in range(400):
            p = sim.pathfinder.get_random_navigable_point()
            d = np.linalg.norm((p - state.position)[[0, 2]])
            if 3.0 <= d <= 5.0 and abs(p[1] - state.position[1]) < 0.25:
                clearance = sim.pathfinder.distance_to_closest_obstacle(p, 1.0)
                geo = sim.geodesic_distance(state.position, p)
                if np.isfinite(geo) and geo < 7 and clearance > 0.45:
                    candidates.append((clearance - abs(d - 4) * .1, p))
        if not candidates:
            raise RuntimeError("No safe target placement found on the starting floor")
        target = max(candidates, key=lambda item: item[0])[1]
    target = np.asarray(target, dtype=float)
    bb = person.root_scene_node.cumulative_bb
    print("SCALED_PERSON_BOUNDS", bb, "scale", scale, flush=True)
    expected_position = [float(target[0]), float(target[1] - bb.min.y), float(target[2])]
    person.translation = mn.Vector3(*expected_position)
    if not np.allclose(np.asarray(person.translation), expected_position):
        raise RuntimeError("Habitat ignored the requested person placement")
    person.motion_type = habitat_sim.physics.MotionType.STATIC
    print("PLACED_TRANSLATION", person.translation, flush=True)
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    agent_cfg = env._config.simulator.agents.main_agent
    settings.agent_radius = agent_cfg.radius
    settings.agent_height = agent_cfg.height
    if not sim.recompute_navmesh(sim.pathfinder, settings, include_static_objects=True):
        raise RuntimeError("Could not rebuild traversable space around the inserted person")
    return person, target


def observe(env):
    sim_obs = env.sim.get_sensor_observations()
    return env.sim._sensor_suite.get_observations(sim_obs)


def save_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb)[..., :3], cv2.COLOR_RGB2BGR))


def inspect_target(env, target, output):
    from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs
    best = None
    for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        pos = env.sim.pathfinder.snap_point(target + [2 * np.cos(angle), 0, 2 * np.sin(angle)])
        if not np.all(np.isfinite(pos)) or abs(pos[1] - target[1]) > .3:
            continue
        direction = target - pos
        direction[1] = 0
        rotation = quat_to_coeffs(quat_from_two_vectors(np.array([0., 0., -1.]), direction))
        obs = env.sim.get_observations_at(pos, rotation, keep_agent_at_new_pose=False)
        count = int(np.count_nonzero(obs["semantic"] == 10001))
        if best is None or count > best[0]:
            best = (count, obs, pos)
    if best is None or best[0] < 100:
        raise RuntimeError("Inserted person is not visible in evaluation camera probes")
    save_rgb(output / "person_in_room.jpg", best[1]["rgb"])
    np.save(output / "person_probe_semantic.npy", best[1]["semantic"])
    print("TARGET_PROBE", best[0], "pixels", list(best[2]), flush=True)


def run_search(cfg, env, target, max_steps, output, confidence, use_iplanner=False):
    import torch
    import imageio.v2 as imageio
    from vlfm.policy.habitat_policies import HabitatMixin
    from vlfm.policy.itm_policy import ITMPolicyV2

    class PersonPolicy(HabitatMixin, ITMPolicyV2):
        """Language target adapter without HM3D's six-class integer mapping."""
        last_detections = None

        def act_person(self, observations, masks):
            observations["objectgoal"] = "person"
            return ITMPolicyV2.act(self, observations, None, None, masks, True)[0]

        def _get_object_detections(self, rgb):
            det = self._object_detector.predict(rgb, caption="person .")
            det.filter_by_class(["person"])
            det.filter_by_conf(self._non_coco_threshold)
            self.last_detections = det
            return det

    torch.set_num_threads(1)
    torch.manual_seed(0)
    policy = PersonPolicy.from_config(cfg)
    policy._non_coco_threshold = confidence
    output.mkdir(parents=True, exist_ok=True)
    os.environ["IMAGEIO_FFMPEG_EXE"] = str(Path(sys.executable).parents[1] / "lib/python3.9/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")
    writer = imageio.get_writer(str(output / "person_search.mp4"), fps=8, macro_block_size=None)
    iplanner = None
    if use_iplanner:
        from habitat_iplanner_adapter import HabitatIPlannerAdapter
        sensor = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
        width, height = int(sensor.width), int(sensor.height)
        hfov = np.deg2rad(float(sensor.hfov))
        focal = width / (2.0 * np.tan(hfov / 2.0))
        intrinsic = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], dtype=np.float32)
        iplanner = HabitatIPlannerAdapter(intrinsic, float(sensor.max_depth))
        print("INTERNNAV_IPLANNER_READY", flush=True)
    rows, confirmed, stop_called = [], False, False
    path_length, last_match_step = 0., -1000
    false_positive_frames, blocked_forward_steps = 0, 0
    blocked_streak = 0
    previous = np.asarray(env.sim.get_agent_state().position).copy()
    started = time.monotonic()
    # We do not reset or teleport after insertion. Initial observation is at the
    # fixed episode start; there are no evaluation camera probes in a search run.
    obs = observe(env)
    obs.update(env.task.sensor_suite.get_observations(observations=obs, episode=env.current_episode, task=env.task))
    initial_pixels = int(np.count_nonzero(obs["semantic"] == 10001))
    try:
        with torch.inference_mode():
            for step in range(max_steps):
                # Explicit allow-list prevents oracle/semantic information leakage.
                policy_obs = {k: torch.as_tensor(obs[k], device="cuda").unsqueeze(0)
                              for k in ("rgb", "depth", "gps", "compass", "heading")}
                vlfm_action = int(policy.act_person(policy_obs, torch.tensor([[step > 0]], device="cuda")).item())
                info = policy._policy_info
                action = vlfm_action
                if iplanner is not None and "rho_theta" in info:
                    iplanner_action = iplanner.act(
                        obs["depth"], info["rho_theta"], vlfm_action,
                        bool(info.get("stop_called", False)),
                    )
                    # InternNav supplies a learned local trajectory while VLFM's
                    # PointNav controller remains the safety executor.  Only an
                    # agreed command is accepted directly; disagreements fall
                    # back to the controller that owns Habitat's navmesh state.
                    action = iplanner_action if iplanner_action == vlfm_action else vlfm_action
                    iplanner.last_debug["vlfm_action"] = vlfm_action
                    iplanner.last_debug["executed_action"] = action
                    iplanner.last_debug["arbiter"] = (
                        "agreement" if iplanner_action == vlfm_action else "vlfm_navmesh_safety"
                    )
                    print("INTERNNAV_IPLANNER_ACTION", json.dumps(iplanner.last_debug), flush=True)
                recovery_active = bool(
                    iplanner is not None
                    and iplanner.last_debug.get("arbiter") == "vlfm_navmesh_safety"
                )
                truth_mask = (obs["semantic"] == 10001).squeeze()  # Evaluation only.
                h, w = truth_mask.shape
                overlap = 0
                for box in policy.last_detections.boxes:
                    x1, y1, x2, y2 = np.rint(box.detach().cpu().numpy() * [w, h, w, h]).astype(int)
                    overlap = max(overlap, int(truth_mask[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)].sum()))
                matched = overlap >= 50
                if policy.last_detections.num_detections > 0 and not matched:
                    false_positive_frames += 1
                if matched:
                    last_match_step = step
                confirmed = confirmed or matched
                position = np.asarray(env.sim.get_agent_state().position).copy()
                distance = float(np.linalg.norm((position - target)[[0, 2]]))
                row = {"step": step, "action": action, "vlfm_action": vlfm_action,
                       "iplanner": None if iplanner is None else iplanner.last_debug,
                       "recovery_active": recovery_active,
                       "position": position.tolist(),
                       "distance_to_person_center_m": distance,
                       "person_pixels_eval_only": int(truth_mask.sum()),
                       "detections": policy.last_detections.to_json(),
                       "detection_matches_inserted_person": matched,
                       "target_in_policy_map": bool(info.get("target_detected", False)),
                       "stop_called": bool(info.get("stop_called", False))}
                rows.append(row)
                canvas = np.full((720, 1280, 3), 248, dtype=np.uint8)
                rgb = info.get("annotated_rgb", obs["rgb"])
                canvas[80:620, 20:740] = cv2.resize(rgb, (720, 540))
                for key, left in [("obstacle_map", 760), ("value_map", 1010)]:
                    if key in info:
                        map_image = info[key]
                        ys, xs = np.where(np.any(map_image < 235, axis=-1))
                        if len(xs):
                            map_image = map_image[max(0, ys.min()-30):ys.max()+31,
                                                  max(0, xs.min()-30):xs.max()+31]
                        canvas[80:300, left:left+220] = cv2.resize(map_image, (220, 220))
                if iplanner is not None:
                    canvas[350:600, 865:1115] = iplanner.draw(250)
                cv2.putText(canvas, "VLFM PERSON SEARCH | simulated RGB-D + learned policy", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (15, 15, 15), 2)
                controller = "iPlanner" if iplanner is not None and iplanner.last_debug.get("active") else "VLFM"
                cv2.putText(canvas, f"Step {step} | action {action} ({controller}) | matched person: {confirmed} | distance (eval): {distance:.2f} m", (20, 670), cv2.FONT_HERSHEY_SIMPLEX, .65, (15, 15, 15), 2)
                writer.append_data(canvas)
                if matched and not (output / "person_detected.jpg").exists():
                    save_rgb(output / "person_detected.jpg", canvas)
                if step == 0:
                    save_rgb(output / "start.jpg", canvas)
                print(f"PERSON_PROGRESS step={step} action={action} matched={matched} distance={distance:.3f}", flush=True)
                if action == 0:
                    stop_called = bool(info.get("stop_called", False))
                    save_rgb(output / "final.jpg", canvas)
                    break
                obs = env.step(action)
                new_position = np.asarray(env.sim.get_agent_state().position).copy()
                moved = float(np.linalg.norm(new_position - previous))
                path_length += moved
                if action == 1 and moved < .05:
                    blocked_forward_steps += 1
                    blocked_streak += 1
                elif action == 1:
                    blocked_streak = 0
                previous = new_position
                if env.episode_over:
                    break
        final_dist = float(np.linalg.norm((previous - target)[[0, 2]]))
        save_rgb(output / "final.jpg", canvas)
        # Custom person-search criterion, NOT the unmodified HM3D toilet metric.
        success = bool(stop_called and confirmed and final_dist <= 1.2 and len(rows) - 1 - last_match_step <= 5)
        metrics = {"success": success, "episode_count": 1,
                   "task": "custom static person search", "policy": "VLFM + GroundingDINO + InternNav iPlanner" if use_iplanner else "VLFM ITMPolicyV2 + GroundingDINO person adapter",
                   "steps": len(rows), "path_length_m": path_length,
                   "person_confidence_threshold": confidence,
                   "false_positive_detection_frames": false_positive_frames,
                   "blocked_forward_steps": blocked_forward_steps,
                   "distance_to_person_center_m": final_dist, "success_radius_m": 1.2,
                   "stop_called": stop_called, "inserted_person_detection_confirmed": confirmed,
                   "initial_person_visible": initial_pixels > 0,
                   "last_matching_detection_step": last_match_step,
                   "elapsed_seconds": time.monotonic() - started,
                   "target_position_passed_to_policy": False,
                   "iplanner_enabled": use_iplanner,
                   "policy_inputs": ["rgb", "depth", "gps", "compass", "heading", "person"],
                   "not_a_benchmark": True}
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (output / "trajectory.json").write_text(json.dumps(rows, indent=2))
        print("PERSON_RESULT", json.dumps(metrics), flush=True)
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--target", type=float, nargs=3)
    parser.add_argument("--yaw", type=float, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "run_01")
    parser.add_argument("--person-confidence", type=float, default=.65)
    parser.add_argument("--asset", choices=PERSON_ASSETS, default="cesium")
    parser.add_argument("--iplanner", action="store_true")
    args = parser.parse_args()
    if not PERSON_ASSETS[args.asset].is_file():
        raise FileNotFoundError(PERSON_ASSETS[args.asset])
    if args.max_steps < 1 or not 0 <= args.person_confidence <= 1:
        raise ValueError("Invalid step budget or confidence threshold")
    if (args.output / "scene_setup.json").exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg, env, _ = build_env()
    try:
        start = env.sim.get_agent_state()
        person, target = place_person(env, args.target, args.yaw, args.asset)
        metadata = {
            "scene_id": env.current_episode.scene_id,
            "start_position": start.position.tolist(),
            "start_rotation_xyzw": [*start.rotation.imag.tolist(), start.rotation.real],
            "target_floor_position": target.tolist(),
            "asset": str(PERSON_ASSETS[args.asset]),
            "asset_name": args.asset,
            "asset_relative_path": str(PERSON_ASSETS[args.asset].relative_to(ASSET_ROOT.parent.parent)),
            "asset_sha256": hashlib.sha256(PERSON_ASSETS[args.asset].read_bytes()).hexdigest(),
            "person_height_m": 1.7,
            "person_yaw_radians": args.yaw,
            "semantic_id_evaluation_only": 10001,
            "target_position_passed_to_policy": False,
        }
        print(json.dumps(metadata, indent=2), flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "scene_setup.json").write_text(json.dumps(metadata, indent=2))
        if not args.prepare:
            run_search(cfg, env, target, args.max_steps, args.output, args.person_confidence, args.iplanner)
            return
        images = []
        for i in range(12):
            obs = observe(env)
            save_rgb(args.output / f"panorama_{i:02d}.jpg", obs["rgb"])
            pixels = int(np.count_nonzero(obs.get("semantic", []) == 10001))
            print("VIEW", i, "person_pixels", pixels, flush=True)
            images.append(cv2.resize(obs["rgb"][..., :3], (320, 240)))
            env.step(2)
        sheet = np.concatenate([np.concatenate(images[i:i+4], axis=1) for i in range(0, 12, 4)], axis=0)
        save_rgb(args.output / "placement_panorama.jpg", sheet)
        inspect_target(env, target, args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
