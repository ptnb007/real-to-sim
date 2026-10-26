"""Read-only audit of a completed PersonNav demo and its recorded evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir
    metrics = json.loads((root / "metrics.json").read_text())
    setup = json.loads((root / "scene_setup.json").read_text())
    rows = json.loads((root / "trajectory.json").read_text())
    log = (root / "navigation.log").read_text()
    expected_inputs = ["rgb", "depth", "gps", "compass", "heading", "person"]
    require(metrics["policy_inputs"] == expected_inputs, "Unexpected policy input")
    require(not metrics["target_position_passed_to_policy"], "Target location leaked")
    require(not setup["target_position_passed_to_policy"], "Setup leakage flag")
    require(metrics["steps"] == len(rows), "Step count mismatch")
    require(metrics["success"], "Run does not report success")
    require(math.dist(rows[0]["position"], setup["start_position"]) < 1e-5, "Starting pose mismatch")
    require([r["step"] for r in rows] == list(range(len(rows))), "Missing trajectory steps")
    require(rows[-1]["action"] == 0 and rows[-1]["stop_called"], "No explicit policy STOP")
    require(rows[0]["person_pixels_eval_only"] == 0, "Person visible at initial pose")
    final = rows[-1]["position"]
    target = setup["target_floor_position"]
    distance = math.hypot(final[0] - target[0], final[2] - target[2])
    require(abs(distance - metrics["distance_to_person_center_m"]) < 1e-6, "Final distance mismatch")
    length = sum(math.dist(a["position"], b["position"]) for a, b in zip(rows, rows[1:]))
    require(abs(length - metrics["path_length_m"]) < 1e-4, "Travel length mismatch")
    matches = [r["step"] for r in rows if r["detection_matches_inserted_person"]]
    require(matches and len(rows)-1-matches[-1] <= 5, "No recent real-target detection")
    require(distance <= metrics["success_radius_m"], "Outside custom success radius")
    threshold = metrics.get("person_confidence_threshold", .35)
    if "false_positive_detection_frames" in metrics:
        false_frames = sum(bool(r["detections"]["logits"]) and not r["detection_matches_inserted_person"]
                           for r in rows)
        require(false_frames == metrics["false_positive_detection_frames"], "False-positive frame count mismatch")
    for row in rows:
        require(all(s >= threshold for s in row["detections"]["logits"]), "Unfiltered detection score")
        require(all(p == "person" for p in row["detections"]["phrases"]), "Wrong target class")
    modes = {m: len(re.findall(f"Mode: {m} ", log)) for m in ("initialize", "explore", "navigate")}
    require(modes["explore"] > 0 and modes["navigate"] > 0, "No exploration-to-navigation transition")
    for name in ("person_search.mp4", "start.jpg", "person_detected.jpg", "final.jpg"):
        require((root / name).stat().st_size > 1000, f"Missing evidence: {name}")
    asset = Path(__file__).resolve().parents[1] / setup.get("asset_relative_path", "data/person_search/CesiumMan.glb")
    require(hashlib.sha256(asset.read_bytes()).hexdigest() ==
            setup.get("asset_sha256", "b7001eaeea8254bd44773bcd247e78696d94169388fbb2a1800fc69434e777d9"), "Asset hash mismatch")
    print(json.dumps({"audit": "passed", "steps": len(rows), "modes": modes,
                      "first_confirmed_person_step": matches[0], "distance_m": distance,
                      "path_length_m": length, "scope": "one custom static-person episode"}, indent=2))


if __name__ == "__main__":
    main()
