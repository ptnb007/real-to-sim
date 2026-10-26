from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


CASE_DIR = Path("/home/woom/Mira-Scene/work/lab_no_robot")
RESULT_DIR = Path("/mnt/c/Users/woom/Desktop/woom/mira_scene/results/lab_no_robot_segmentation")


OBJECTS = [
    {
        "name": "folding room divider",
        "caption": "A large off-white folding room divider forming the central wall of the reconstructed laboratory area.",
        "motion": "fixed",
        "polygon": [(480, 145), (1340, 85), (1380, 680), (1290, 735), (1275, 925), (1110, 920), (970, 760), (755, 720), (520, 760)],
    },
    {
        "name": "laboratory desk",
        "caption": "A white rectangular laboratory desk with a monitor and computer equipment, placed to the left of the divider.",
        "motion": "fixed",
        "polygon": [(455, 505), (800, 535), (825, 650), (745, 680), (730, 820), (500, 805), (445, 690)],
    },
    {
        "name": "upholstered ottoman",
        "caption": "A rectangular pale fabric ottoman with a black tray on top, positioned in front of the divider.",
        "motion": "dynamic",
        "polygon": [(825, 760), (1120, 750), (1240, 1060), (1120, 1190), (825, 1100), (770, 900)],
    },
    {
        "name": "poster display stand",
        "caption": "A freestanding black metal poster display with a rectangular framed sign and circular base.",
        "motion": "dynamic",
        "polygon": [(1280, 380), (1415, 335), (1450, 735), (1395, 790), (1390, 1115), (1350, 1180), (1320, 1100), (1335, 770), (1295, 710)],
    },
]


def main() -> None:
    input_dir = CASE_DIR / "input"
    scene_path = input_dir / "scene.png"
    source = Image.open(scene_path).convert("RGB")
    width, height = source.size
    overlay = source.convert("RGBA")
    colors = [(66, 133, 244, 115), (52, 168, 83, 115), (251, 188, 5, 130), (234, 67, 53, 125)]

    nodes = [
        {"id": "floor", "kind": "static_environment", "name": "floor", "motion": "fixed", "fixed": True, "dynamic": False},
        {"id": "world_anchor", "kind": "virtual_static_anchor", "name": "wall_or_ceiling", "motion": "fixed", "fixed": True, "dynamic": False},
    ]
    edges = []
    foreground = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    for index, item in enumerate(OBJECTS):
        polygon = [(min(max(x, 0), width - 1), min(max(y, 0), height - 1)) for x, y in item["polygon"]]
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
        mask.save(input_dir / f"mask_{index:03d}.png")

        tint = Image.new("RGBA", (width, height), colors[index])
        overlay.alpha_composite(Image.composite(tint, Image.new("RGBA", (width, height)), mask))
        foreground.alpha_composite(Image.composite(source.convert("RGBA"), Image.new("RGBA", (width, height)), mask))

        dynamic = item["motion"] == "dynamic"
        nodes.append(
            {
                "id": f"object_{index:03d}",
                "kind": "object",
                "mask_index": index,
                "mask_file": f"input/mask_{index:03d}.png",
                "name": item["name"],
                "short_name": item["name"],
                "caption": item["caption"],
                "caption_source": "human_guided",
                "name_locked": True,
                "caption_locked": True,
                "motion": item["motion"],
                "fixed": not dynamic,
                "dynamic": dynamic,
                "upright_mode": "auto",
                "confidence": 0.9,
            }
        )
        edges.append(
            {
                "child": f"object_{index:03d}",
                "parent": "floor",
                "relation": "rests_on",
                "supporter_raw": "floor",
                "confidence": 0.9,
                "operational": True,
            }
        )

    graph = {
        "schema": "mira_scene_graph_v1",
        "case": "lab_no_robot",
        "edge_direction": "child_to_direct_supporter",
        "nodes": nodes,
        "edges": edges,
        "vlm_inference": {"model": "manual_bootstrap"},
    }
    (CASE_DIR / "scene_graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    foreground.save(input_dir / "scene_fg.png")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(RESULT_DIR / "instance_overlay.jpg", quality=94)
    (RESULT_DIR / "scene_graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    for index in range(len(OBJECTS)):
        Image.open(input_dir / f"mask_{index:03d}.png").save(RESULT_DIR / f"mask_{index:03d}.png")
    print(RESULT_DIR / "instance_overlay.jpg")


if __name__ == "__main__":
    main()
