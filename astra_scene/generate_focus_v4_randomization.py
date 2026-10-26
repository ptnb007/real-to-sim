"""Generate reproducible focus_v4 domain-randomized USD scene layers."""

import json
import math
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "focus_v4" / "randomized"
COUNT = 12
SEED = 20260924


def quat_z(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) / 2.0
    return math.cos(half), 0.0, 0.0, math.sin(half)


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    samples = []

    for index in range(COUNT):
        x = rng.uniform(0.05, 0.42)
        y = rng.uniform(-0.78, -0.36)
        yaw = rng.uniform(-22.0, 22.0)
        dome_intensity = rng.uniform(420.0, 850.0)
        key_intensity = rng.uniform(1100.0, 2300.0)
        warm = rng.uniform(0.90, 1.00)
        focal_length = rng.uniform(38.0, 50.0)
        qw, qx, qy, qz = quat_z(yaw)

        filename = f"focus_v4_r1pro_random_{index:03d}.usda"
        path = OUT / filename
        path.write_text(
            f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    subLayers = [@../focused_zone_isaac.usda@]
    upAxis = "Z"
)

over "World"
{{
    def Xform "R1Pro" (
        prepend references = @../../../isaacsim/galaxea_r1pro_scene.usda@</r1pro_urdf>
    )
    {{
        quatd xformOp:orient = ({qw:.9f}, {qx:.9f}, {qy:.9f}, {qz:.9f})
        double3 xformOp:translate = ({x:.6f}, {y:.6f}, 0.0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}

    over "IsaacLighting"
    {{
        over "Environment"
        {{
            color3f inputs:color = (1.0, {warm:.6f}, {max(0.82, warm - 0.05):.6f})
            float inputs:intensity = {dome_intensity:.3f}
        }}
        over "SoftKey"
        {{
            float inputs:intensity = {key_intensity:.3f}
        }}
    }}

    over "ReviewCamera"
    {{
        float focalLength = {focal_length:.3f}
    }}
}}
''',
            encoding="utf-8",
        )
        samples.append(
            {
                "index": index,
                "usd": str(path),
                "robot_xyz_m": [x, y, 0.0],
                "robot_yaw_deg": yaw,
                "dome_intensity": dome_intensity,
                "key_intensity": key_intensity,
                "light_warmth": warm,
                "camera_focal_length_mm": focal_length,
            }
        )

    manifest = {
        "scene": "focus_v4",
        "robot": "Galaxea R1 Pro",
        "seed": SEED,
        "count": COUNT,
        "randomized_fields": [
            "robot_xy",
            "robot_yaw",
            "dome_light_intensity",
            "key_light_intensity",
            "light_color_temperature_proxy",
            "camera_focal_length",
        ],
        "samples": samples,
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Generated {COUNT} randomized scenes: {manifest_path}")


if __name__ == "__main__":
    main()
