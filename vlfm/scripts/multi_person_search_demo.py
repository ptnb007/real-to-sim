"""Multi-room VLFM search for standing, seated, and lying static people.

This is an isolated research demo. Semantic IDs and simulator target poses are
used only for evaluation/association after a policy STOP; the policy receives
RGB-D, odometry, and the literal text goal ``person``.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
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

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from person_search_demo import build_env, observe, save_rgb

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "data/person_search/rocketbox"
OUTPUT_ROOT = ROOT / "results/multi_person_search"
POSES = {
    "standing": ASSET_ROOT / "Male_Adult_01_standing.glb",
    "seated": ASSET_ROOT / "poses/Male_Adult_01_seated.glb",
    "lying": ASSET_ROOT / "poses/Male_Adult_01_lying.glb",
}


def region_for_point(sim, point):
    point = np.asarray(point)
    matches = []
    for region in sim.semantic_scene.regions:
        center, sizes = np.asarray(region.aabb.center), np.asarray(region.aabb.sizes)
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
            continue
        lower, upper = center-sizes/2, center+sizes/2
        if np.all(point >= lower - .05) and np.all(point <= upper + .05):
            volume = float(np.prod(np.maximum(upper - lower, .001)))
            matches.append((volume, region))
    if matches:
        return min(matches, key=lambda item: item[0])[1]
    # HM3D's semantic descriptor used with Habitat-Sim 0.2.4 can omit region
    # AABBs. Assign a navigable point to the region of the nearest annotated
    # scene object instead. This uses semantic metadata only for evaluation and
    # placement; the navigation policy never receives the resulting room ID.
    nearest = None
    for obj in sim.semantic_scene.objects:
        if obj is None or getattr(obj, "region", None) is None:
            continue
        center, sizes = np.asarray(obj.aabb.center), np.asarray(obj.aabb.sizes)
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
            continue
        lower, upper = center-sizes/2, center+sizes/2
        delta = np.maximum(np.maximum(lower[[0,2]]-point[[0,2]], point[[0,2]]-upper[[0,2]]), 0)
        vertical_penalty = 0 if lower[1]-.5 <= point[1] <= upper[1]+.5 else 2
        distance = float(np.linalg.norm(delta) + vertical_penalty)
        if nearest is None or distance < nearest[0]:
            nearest = distance, obj.region
    return None if nearest is None else nearest[1]


def region_record(region):
    if region is None:
        return {"id": None, "category": "unassigned"}
    try:
        category = region.category.name()
    except Exception:
        category = "unknown"
    return {"id": str(region.id), "category": category}


def select_targets(env):
    """Choose reachable, well-separated points in three semantic rooms."""
    sim, start = env.sim, np.asarray(env.sim.get_agent_state().position)
    print("SEMANTIC_REGION_COUNT", len(sim.semantic_scene.regions), flush=True)
    sim.pathfinder.seed(20260907)
    groups = {}
    for _ in range(5000):
        point = np.asarray(sim.pathfinder.get_random_navigable_point())
        if not np.all(np.isfinite(point)) or abs(point[1] - start[1]) > .3:
            continue
        euclidean = float(np.linalg.norm((point-start)[[0, 2]]))
        geodesic = float(sim.geodesic_distance(start, point))
        clearance = float(sim.pathfinder.distance_to_closest_obstacle(point, 1.2))
        if euclidean < 3 or not np.isfinite(geodesic) or geodesic > 30 or clearance < .55:
            continue
        region = region_for_point(sim, point)
        if region is None:
            continue
        record = region_record(region)
        groups.setdefault(record["id"], []).append((clearance + .02*geodesic, point, region, geodesic))
    best = None
    # Search several safe points per room rather than assuming each room's
    # maximum-clearance point also maximizes separation from other rooms.
    for region_ids in itertools.combinations(groups, 3):
        pools = [sorted(groups[key], key=lambda item: item[0], reverse=True)[:50] for key in region_ids]
        for candidate in itertools.product(*pools):
            distances = [float(np.linalg.norm((a[1]-b[1])[[0,2]]))
                         for n, a in enumerate(candidate) for b in candidate[n+1:]]
            if all(distance >= 2.5 for distance in distances):
                score = sum(distances) + .25*sum(item[3] for item in candidate) + sum(item[0] for item in candidate)
                if best is None or score > best[0]:
                    best = score, list(candidate)
    if best is None:
        summary = {key: len(value) for key, value in groups.items()}
        raise RuntimeError(f"Could not find three separated semantic rooms: {summary}")
    chosen = sorted(best[1], key=lambda item: item[3])
    return [{"pose": pose, "position": item[1], "region": item[2], "start_geodesic": item[3]}
            for pose, item in zip(POSES, chosen)]


def add_people(env, targets):
    import habitat_sim
    import magnum as mn
    sim = env.sim
    templates, objects = sim.get_object_template_manager(), sim.get_rigid_object_manager()
    start = np.asarray(sim.get_agent_state().position)
    placed = []
    for index, target in enumerate(targets):
        pose, asset = target["pose"], POSES[target["pose"]]
        template = templates.create_new_template(f"multi_person_{pose}")
        template.render_asset_handle = str(asset)
        template.collision_asset_handle = str(asset)
        template.force_flat_shading = True
        template.orient_up, template.orient_front = mn.Vector3(0, 1, 0), mn.Vector3(0, 0, -1)
        template_id = templates.register_template(template)
        person = objects.add_object_by_template_id(template_id)
        if pose == "standing":
            scale = 1.7 / float(person.root_scene_node.cumulative_bb.size().y)
            objects.remove_object_by_id(person.object_id)
            template.scale = mn.Vector3(scale)
            template_id = templates.register_template(template)
            person = objects.add_object_by_template_id(template_id)
        else:
            scale = 1.0
        semantic_id = 11001 + index
        target_floor = np.asarray(target["position"], dtype=float)
        yaw = math.atan2(start[0]-target_floor[0], start[2]-target_floor[2])
        person.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        person.semantic_id = semantic_id
        person.rotation = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3.y_axis())
        bounds = person.root_scene_node.cumulative_bb
        translation = [float(target_floor[0]), float(target_floor[1]-bounds.min.y), float(target_floor[2])]
        person.translation = mn.Vector3(*translation)
        if not np.allclose(np.asarray(person.translation), translation):
            raise RuntimeError(f"Habitat ignored placement for {pose}")
        person.motion_type = habitat_sim.physics.MotionType.STATIC
        placed.append({**target, "asset": asset, "scale": scale, "yaw": yaw,
                       "semantic_id": semantic_id, "object_id": person.object_id})
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    agent = env._config.simulator.agents.main_agent
    settings.agent_radius, settings.agent_height = agent.radius, agent.height
    if not sim.recompute_navmesh(sim.pathfinder, settings, include_static_objects=True):
        raise RuntimeError("Could not rebuild navmesh around people")
    return placed


def task_observation(env):
    obs = observe(env)
    obs.update(env.task.sensor_suite.get_observations(observations=obs, episode=env.current_episode, task=env.task))
    return obs


def prepare(env, targets, output):
    from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs
    output.mkdir(parents=True, exist_ok=True)
    initial = []
    for i in range(12):
        obs = task_observation(env)
        initial.append(cv2.resize(obs["rgb"][..., :3], (320, 240)))
        env.step(2)
    save_rgb(output / "initial_360_panorama.jpg",
             np.concatenate([np.concatenate(initial[i:i+4], axis=1) for i in range(0, 12, 4)], axis=0))
    for target in targets:
        best = None
        point = target["position"]
        for angle in np.linspace(0, 2*np.pi, 20, endpoint=False):
            camera = env.sim.pathfinder.snap_point(point + [2*np.cos(angle), 0, 2*np.sin(angle)])
            if not np.all(np.isfinite(camera)):
                continue
            direction = point-camera
            direction[1] = 0
            rotation = quat_to_coeffs(quat_from_two_vectors(np.array([0., 0., -1.]), direction))
            rotation = rotation / np.linalg.norm(rotation)
            obs = env.sim.get_observations_at(camera, rotation, keep_agent_at_new_pose=False)
            count = int(np.count_nonzero(obs["semantic"] == target["semantic_id"]))
            if best is None or count > best[0]:
                best = count, obs
        if best is None or best[0] < 100:
            raise RuntimeError(f"{target['pose']} is not visibly placed")
        save_rgb(output / f"placed_{target['pose']}.jpg", best[1]["rgb"])
        print("TARGET_PROBE", target["pose"], best[0], flush=True)


def crop_map(image):
    ys, xs = np.where(np.any(image < 235, axis=-1))
    if len(xs):
        image = image[max(0, ys.min()-30):ys.max()+31, max(0, xs.min()-30):xs.max()+31]
    return cv2.resize(image, (250, 250))


def run_search(cfg, env, targets, output, max_steps, confidence):
    import imageio.v2 as imageio
    import torch
    from vlfm.policy.habitat_policies import HabitatMixin, TorchActionIDs
    from vlfm.policy.itm_policy import ITMPolicyV2

    class MultiPersonPolicy(HabitatMixin, ITMPolicyV2):
        last_detections = None
        scan_steps_remaining = 0
        scan_only = False
        scan_label = None
        frontier_exhausted = False

        def act_person(self, observations, mask):
            observations["objectgoal"] = "person"
            return ITMPolicyV2.act(self, observations, None, None, mask, True)[0]

        def _get_object_detections(self, rgb):
            det = self._object_detector.predict(rgb, caption="person .")
            det.filter_by_class(["person"])
            det.filter_by_conf(self._non_coco_threshold)
            self.last_detections = det
            return det

        def _get_target_object_location(self, position):
            if self.scan_steps_remaining or self.scan_only:
                return None
            return super()._get_target_object_location(position)

        def _explore(self, observations):
            if self.scan_steps_remaining:
                self.scan_steps_remaining -= 1
                os.environ.setdefault("DEBUG_INFO", "")
                os.environ["DEBUG_INFO"] += f"Room scan: {12-self.scan_steps_remaining}/12. "
                return TorchActionIDs.TURN_LEFT
            frontiers = self._observations_cache["frontier_sensor"]
            if len(frontiers) == 0:
                self.frontier_exhausted = True
            elif self.scan_only:
                # Once all people are marked, prioritize systematic coverage
                # over semantic value by expanding the nearest local frontier.
                robot = self._observations_cache["robot_xy"]
                goal = min(frontiers, key=lambda point: np.linalg.norm(point[:2]-robot))
                os.environ["DEBUG_INFO"] = f"Coverage frontier: {len(frontiers)} remain. "
                return self._pointnav(goal[:2], stop=False)
            return super()._explore(observations)

        def _resume_exploration(self, scan_steps):
            self._object_map.reset()
            self._called_stop = False
            self._last_goal = np.zeros(2)
            self._pointnav_policy.reset()
            self.scan_steps_remaining = scan_steps
            self.frontier_exhausted = False

        def continue_after_marking(self, all_found, pose):
            self._resume_exploration(12)
            self.scan_only = all_found
            self.scan_label = pose

        def recover_false_positive(self):
            self._resume_exploration(6)
            self.scan_only = False
            self.scan_label = None

    torch.set_num_threads(1)
    torch.manual_seed(0)
    policy = MultiPersonPolicy.from_config(cfg)
    policy._non_coco_threshold = confidence
    output.mkdir(parents=True, exist_ok=True)
    os.environ["IMAGEIO_FFMPEG_EXE"] = str(Path(sys.executable).parents[1] / "lib/python3.9/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")
    writer = imageio.get_writer(str(output / "multi_person_search.mp4"), fps=8, macro_block_size=None)
    obs = task_observation(env)
    initial_visible = {t["pose"]: int(np.count_nonzero(obs["semantic"] == t["semantic_id"])) for t in targets}
    rows, events, found_ids, seen_ids, recent_matches = [], [], set(), set(), {}
    region_visits, room_scan_counts = set(), {}
    previous = np.asarray(env.sim.get_agent_state().position).copy()
    path_length, blocked, false_frames, rejected_stops, stale_resets = 0., 0, 0, 0, 0
    last_retained_detection_step = -1000
    started, termination = time.monotonic(), "step_budget"
    try:
        with torch.inference_mode():
            for step in range(max_steps):
                policy_obs = {key: torch.as_tensor(obs[key], device="cuda").unsqueeze(0)
                              for key in ("rgb", "depth", "gps", "compass", "heading")}
                was_scanning = policy.scan_steps_remaining > 0
                active_scan_pose = policy.scan_label
                action = int(policy.act_person(policy_obs, torch.tensor([[step > 0]], device="cuda")).item())
                if was_scanning and active_scan_pose:
                    room_scan_counts[active_scan_pose] = room_scan_counts.get(active_scan_pose, 0) + 1
                info, truth = policy._policy_info, obs["semantic"].squeeze()
                if policy.last_detections.num_detections:
                    last_retained_detection_step = step
                height, width = truth.shape
                overlaps = {t["semantic_id"]: 0 for t in targets if t["semantic_id"] not in found_ids}
                for box in policy.last_detections.boxes:
                    x1, y1, x2, y2 = np.rint(box.detach().cpu().numpy()*[width, height, width, height]).astype(int)
                    patch = truth[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
                    for semantic_id in overlaps:
                        overlaps[semantic_id] = max(overlaps[semantic_id], int(np.count_nonzero(patch == semantic_id)))
                matched_ids = [semantic_id for semantic_id, overlap in overlaps.items() if overlap >= 50]
                for semantic_id in matched_ids:
                    recent_matches[semantic_id] = step
                if policy.last_detections.num_detections and not matched_ids:
                    false_frames += 1
                position = np.asarray(env.sim.get_agent_state().position).copy()
                stale_reset = False
                # ObjectPointCloudMap normally retains a one-frame hypothesis
                # indefinitely. Reject it after 12 frames without any repeated
                # person detection. This uses detector history only, not semantic
                # truth or simulator target locations.
                if (info.get("target_detected", False) and
                        step-last_retained_detection_step > 12 and action != 0):
                    stale_resets += 1
                    stale_reset = True
                    policy.recover_false_positive()
                    action = 2
                region = region_for_point(env.sim, position)
                region_id = region_record(region)["id"]
                if region_id:
                    region_visits.add(region_id)
                canvas = np.full((720, 1280, 3), 248, dtype=np.uint8)
                canvas[80:620, 20:740] = cv2.resize(info.get("annotated_rgb", obs["rgb"]), (720, 540))
                if "obstacle_map" in info:
                    canvas[80:330, 865:1115] = crop_map(info["obstacle_map"])
                if "value_map" in info:
                    canvas[350:600, 865:1115] = crop_map(info["value_map"])
                found_poses = [t["pose"] for t in targets if t["semantic_id"] in found_ids]
                cv2.putText(canvas, "VLFM MULTI-ROOM PERSON SEARCH", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .85, (15,15,15), 2)
                cv2.putText(canvas, f"Step {step} | action {action} | found {len(found_ids)}/3: {','.join(found_poses) or '-'}", (20, 666), cv2.FONT_HERSHEY_SIMPLEX, .68, (15,15,15), 2)
                writer.append_data(canvas)
                for semantic_id in matched_ids:
                    if semantic_id not in seen_ids:
                        pose = next(t["pose"] for t in targets if t["semantic_id"] == semantic_id)
                        save_rgb(output / f"first_seen_{pose}.jpg", canvas)
                        seen_ids.add(semantic_id)
                row = {"step": step, "action": action, "position": position.tolist(),
                       "region": region_record(region), "matched_semantic_ids": matched_ids,
                       "found_semantic_ids_before_action": sorted(found_ids),
                       "detections": policy.last_detections.to_json(),
                       "stop_called": bool(info.get("stop_called", False)),
                       "scan_steps_remaining": policy.scan_steps_remaining,
                       "stale_target_reset": stale_reset,
                       "frontier_count": len(policy._observations_cache.get("frontier_sensor", []))}
                rows.append(row)
                print(f"MULTI_PROGRESS step={step} action={action} found={len(found_ids)}/3 matches={matched_ids}", flush=True)
                if action == 0 and info.get("stop_called", False):
                    candidates = [sid for sid, match_step in recent_matches.items()
                                  if sid not in found_ids and step-match_step <= 5]
                    candidates = [sid for sid in candidates if np.linalg.norm((position-next(t["position"] for t in targets if t["semantic_id"] == sid))[[0,2]]) <= 1.25]
                    if candidates:
                        semantic_id = min(candidates, key=lambda sid: np.linalg.norm((position-next(t["position"] for t in targets if t["semantic_id"] == sid))[[0,2]]))
                        target = next(t for t in targets if t["semantic_id"] == semantic_id)
                        found_ids.add(semantic_id)
                        event = {"step": step, "pose": target["pose"], "semantic_id": semantic_id,
                                 "region": region_record(target["region"]),
                                 "distance_m": float(np.linalg.norm((position-target["position"])[[0,2]]))}
                        events.append(event)
                        save_rgb(output / f"found_{len(events):02d}_{target['pose']}.jpg", canvas)
                        env.sim.get_rigid_object_manager().remove_object_by_id(target["object_id"])
                        policy.continue_after_marking(len(found_ids) == len(targets), target["pose"])
                        room_scan_counts[target["pose"]] = 0
                        obs = task_observation(env)
                        continue
                    rejected_stops += 1
                    policy.recover_false_positive()
                    obs = task_observation(env)
                    continue
                if action == 0 and policy.frontier_exhausted:
                    termination = "frontiers_exhausted"
                    break
                obs = env.step(action)
                new_position = np.asarray(env.sim.get_agent_state().position).copy()
                displacement = float(np.linalg.norm(new_position-previous))
                path_length += displacement
                if action == 1 and displacement < .05:
                    blocked += 1
                previous = new_position
                if env.episode_over:
                    termination = "habitat_episode_over"
                    break
        room_scan_complete = (len(found_ids) == len(targets) and
                              room_scan_counts == {pose: 12 for pose in ("standing", "seated", "lying")})
        full_scan_complete = termination == "frontiers_exhausted" and room_scan_complete
        metrics = {"success": bool(room_scan_complete), "task": "custom multi-room multi-pose person search",
                   "steps": len(rows), "path_length_m": path_length, "people_total": len(targets),
                   "people_found": len(found_ids), "found_events": events,
                   "all_pose_types_found": {t["pose"]: t["semantic_id"] in found_ids for t in targets},
                   "distinct_target_regions": len({region_record(t["region"])["id"] for t in targets}),
                   "distinct_regions_visited": len(region_visits), "room_scans_360_completed": room_scan_counts,
                   "room_scan_complete": bool(room_scan_complete),
                   "room_scan_definition": "all three targets found in distinct rooms; 12-view 360-degree scan after each",
                   "full_scan_definition": "room scan complete; no >=1.5m2 map frontier remains",
                   "full_scan_complete": full_scan_complete, "termination": termination,
                   "initial_target_pixels": initial_visible, "confidence_threshold": confidence,
                   "false_positive_detection_frames": false_frames, "blocked_forward_steps": blocked,
                   "unconfirmed_policy_stops_rejected": rejected_stops,
                   "stale_visual_target_resets": stale_resets,
                   "elapsed_seconds": time.monotonic()-started,
                   "target_positions_passed_to_policy": False,
                   "policy_inputs": ["rgb", "depth", "gps", "compass", "heading", "person"],
                   "evaluation_intervention": "confirmed found avatars removed after STOP to prevent rediscovery",
                   "not_a_benchmark": True}
        save_rgb(output / "final.jpg", canvas)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (output / "trajectory.json").write_text(json.dumps(rows, indent=2))
        print("MULTI_RESULT", json.dumps(metrics), flush=True)
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "prepare_01")
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--person-confidence", type=float, default=.58)
    args = parser.parse_args()
    if any(not path.is_file() for path in POSES.values()):
        raise FileNotFoundError("A standing/seated/lying GLB is missing")
    if (args.output / "scene_setup.json").exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg, env, _ = build_env(max_episode_steps=args.max_steps+20)
    try:
        start = env.sim.get_agent_state()
        selected = select_targets(env)
        targets = add_people(env, selected)
        setup = {"scene_id": env.current_episode.scene_id, "start_position": start.position.tolist(),
                 "start_rotation_xyzw": [*start.rotation.imag.tolist(), start.rotation.real],
                 "targets": [{"pose": t["pose"], "floor_position": t["position"].tolist(),
                              "region": region_record(t["region"]), "start_geodesic_m": t["start_geodesic"],
                              "semantic_id_evaluation_only": t["semantic_id"], "scale": t["scale"],
                              "yaw_radians": t["yaw"],
                              "asset_relative_path": str(t["asset"].relative_to(ROOT)),
                              "asset_sha256": hashlib.sha256(t["asset"].read_bytes()).hexdigest()}
                             for t in targets],
                 "target_positions_passed_to_policy": False}
        (args.output / "scene_setup.json").write_text(json.dumps(setup, indent=2))
        print("MULTI_SETUP", json.dumps(setup), flush=True)
        if args.prepare:
            prepare(env, targets, args.output)
        else:
            run_search(cfg, env, targets, args.output, args.max_steps, args.person_confidence)
    finally:
        env.close()


if __name__ == "__main__":
    main()
