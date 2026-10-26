"""Read-only audit of a completed multi-room, multi-pose search artifact."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(value, message):
    if not value:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-full-coverage", action="store_true",
                        help="also require VLFM to exhaust every >=1.5m2 map frontier")
    args = parser.parse_args()
    root = args.run_dir
    metrics = json.loads((root / "metrics.json").read_text())
    setup = json.loads((root / "scene_setup.json").read_text())
    rows = json.loads((root / "trajectory.json").read_text())
    if args.require_full_coverage:
        require(metrics["full_scan_complete"] and metrics["termination"] == "frontiers_exhausted",
                "Whole-scene frontier coverage did not complete")
    require(not setup["target_positions_passed_to_policy"] and
            not metrics["target_positions_passed_to_policy"], "Target coordinate leak flag")
    require(metrics["policy_inputs"] == ["rgb", "depth", "gps", "compass", "heading", "person"],
            "Unexpected policy inputs")
    require(len(set(t["pose"] for t in setup["targets"])) == 3, "Missing pose type")
    require(len(set(t["region"]["id"] for t in setup["targets"])) == 3, "Targets not in distinct rooms")
    repo = Path(__file__).resolve().parents[1]
    for target in setup["targets"]:
        asset = repo / target["asset_relative_path"]
        require(hashlib.sha256(asset.read_bytes()).hexdigest() == target["asset_sha256"], "Asset hash mismatch")
    require(all(value == 0 for value in metrics["initial_target_pixels"].values()), "Target visible initially")
    require([row["step"] for row in rows] == list(range(len(rows))), "Trajectory step gap")
    require(math.dist(rows[0]["position"], setup["start_position"]) < 1e-5, "Starting pose mismatch")
    # The runner accumulates the transition from the initial state to row zero too.
    length = math.dist(setup["start_position"], rows[0]["position"])
    length += sum(math.dist(a["position"], b["position"]) for a, b in zip(rows, rows[1:]))
    # Rows capture the pre-action pose. The metric additionally includes the
    # final action, whose displacement is bounded by one 0.25 m forward step.
    trailing_displacement = metrics["path_length_m"] - length
    require(-1e-3 <= trailing_displacement <= .251, "Path length mismatch")
    transitions = [math.dist(setup["start_position"], rows[0]["position"])]
    transitions += [math.dist(a["position"], b["position"]) for a, b in zip(rows, rows[1:])]
    require(max(transitions, default=0) < .4,
            "Possible trajectory teleport")
    threshold = metrics["confidence_threshold"]
    for row in rows:
        require(all(score >= threshold for score in row["detections"]["logits"]), "Unfiltered score")
        require(all(phrase == "person" for phrase in row["detections"]["phrases"]), "Wrong detection phrase")
    require(metrics["people_found"] == metrics["people_total"] == 3, "Not all people found")
    require(set(event["pose"] for event in metrics["found_events"]) == {"standing", "seated", "lying"},
            "Found-event pose mismatch")
    targets = {target["pose"]: target for target in setup["targets"]}
    for index, event in enumerate(metrics["found_events"], 1):
        position = rows[event["step"]]["position"]
        target = targets[event["pose"]]["floor_position"]
        distance = math.hypot(position[0]-target[0], position[2]-target[2])
        require(abs(distance-event["distance_m"]) < 1e-6 and distance <= 1.25, "Found distance mismatch")
        require((root / f"found_{index:02d}_{event['pose']}.jpg").stat().st_size > 1000,
                "Missing found image")
    require(metrics["room_scans_360_completed"] == {pose: 12 for pose in ("standing", "seated", "lying")},
            "A 360 room scan is incomplete")
    for name in ("multi_person_search.mp4", "final.jpg"):
        require((root / name).stat().st_size > 1000, f"Missing evidence: {name}")
    print(json.dumps({"audit": "passed", "steps": len(rows), "path_length_m": length,
                      "found_order": [e["pose"] for e in metrics["found_events"]],
                      "distinct_target_regions": 3, "room_scans": 3,
                      "room_scan_completion": "three 12-view 360-degree scans",
                      "whole_scene_frontier_completion": metrics["full_scan_complete"],
                      "scope": "one custom scene; evaluation-assisted found-target retirement"}, indent=2))


if __name__ == "__main__":
    main()
