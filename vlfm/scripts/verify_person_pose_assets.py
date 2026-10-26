"""Read-only structural checks of the two static, textured pose assets."""
import hashlib
import json
from pathlib import Path
import struct


def main():
    root = Path(__file__).resolve().parents[1] / "data/person_search/rocketbox/poses"
    results = []
    for pose in ("seated", "lying"):
        path = root / f"Male_Adult_01_{pose}.glb"
        raw = path.read_bytes()
        magic, version, length = struct.unpack_from("<4sII", raw)
        assert (magic, version, length) == (b"glTF", 2, len(raw))
        size, kind = struct.unpack_from("<I4s", raw, 12)
        assert kind == b"JSON"
        doc = json.loads(raw[20:20+size])
        assert not doc.get("skins") and not doc.get("animations")
        assert len(doc["meshes"]) == 1 and len(doc["images"]) == 3
        assert all("bufferView" in image and "uri" not in image for image in doc["images"])
        assert all("TEXCOORD_0" in primitive["attributes"] for primitive in doc["meshes"][0]["primitives"])
        report = json.loads((root / f"conversion_{pose}.json").read_text())
        assert report["asset_sha256"] == hashlib.sha256(raw).hexdigest()
        assert report["normalized_standing_height_m"] == 1.7
        dimensions = report["blender_z_up_dimensions_m"]
        assert .9 < dimensions[2] < 1.5 if pose == "seated" else .15 < dimensions[2] < .5
        assert (root / f"{pose}_preview.png").stat().st_size > 10000
        results.append({"pose": pose, "embedded_textures": len(doc["images"]),
                        "dimensions_xyz_z_up_m": dimensions, "sha256": report["asset_sha256"]})
    print(json.dumps({"audit": "passed", "assets": results}, indent=2))


if __name__ == "__main__":
    main()
