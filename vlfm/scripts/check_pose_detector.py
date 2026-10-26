"""Offline GroundingDINO check on evaluation-only placement screenshots."""
import hashlib
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path.cwd()))

from vlfm.vlm.grounding_dino import GroundingDINO


def main():
    root = Path(__file__).resolve().parents[1]
    images = root / "results/multi_person_search/prepare_02"
    detector = GroundingDINO(caption="person .")
    result = {}
    for pose in ("standing", "seated", "lying"):
        path = images / f"placed_{pose}.jpg"
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        detections = detector.predict(rgb, caption="person .")
        result[pose] = {"scores": [float(value) for value in detections.logits],
                        "phrases": detections.phrases,
                        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    output = root / "results/multi_person_search/prepare_02/detector_check.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
