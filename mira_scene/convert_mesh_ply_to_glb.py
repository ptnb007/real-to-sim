"""Import a vertex-colored PLY surface mesh in Blender and export GLB."""
import bpy
import sys
from pathlib import Path

args = sys.argv[sys.argv.index("--") + 1:]
source, target = map(Path, args)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.ply_import(filepath=str(source), import_colors="SRGB")
mesh_objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if len(mesh_objects) != 1:
    raise RuntimeError(f"Expected one mesh object, got {len(mesh_objects)}")
obj = mesh_objects[0]
if len(obj.data.polygons) < 1:
    raise RuntimeError("Imported mesh contains no polygons")
target.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(filepath=str(target), export_format="GLB")
print(f"GLB_OK vertices={len(obj.data.vertices)} faces={len(obj.data.polygons)} file={target}")

usd_target = target.with_suffix(".usd")
bpy.ops.wm.usd_export(filepath=str(usd_target), selected_objects_only=True)
print(f"USD_OK vertices={len(obj.data.vertices)} faces={len(obj.data.polygons)} file={usd_target}")
