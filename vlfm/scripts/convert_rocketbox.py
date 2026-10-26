"""Convert a licensed Rocketbox FBX into a static textured Habitat GLB.

Run using portable Blender: blender -b -t 2 -P this_file -- --inspect
Only source geometry/textures are used; no AI-generated images.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import struct

import bpy
from mathutils import Matrix, Vector


def aim_limb(rig, parent_name, child_name, desired_world):
    """Aim the anatomical joint segment, not FBX's arbitrary bone display axis."""
    bone = rig.pose.bones[parent_name]
    child = rig.pose.bones[child_name]
    desired_local = rig.matrix_world.to_3x3().inverted() @ desired_world
    delta = (child.head - bone.head).normalized().rotation_difference(desired_local.normalized())
    head = bone.head.copy()
    bone.matrix = Matrix.Translation(head) @ delta.to_matrix().to_4x4() @ Matrix.Translation(-head) @ bone.matrix
    bpy.context.view_layer.update()
    actual = (rig.matrix_world.to_3x3() @ (child.head-bone.head)).normalized()
    if actual.dot(desired_world.normalized()) < .99:
        raise RuntimeError(f"Pose direction check failed: {parent_name}")


def render_preview(asset, destination, seated):
    """CPU-render the exported geometry with neutral lighting for visual QA."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(asset))
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError("Unexpected exported mesh layout")
    mesh = meshes[0]
    bpy.ops.object.select_all(action="DESELECT")
    vertices = [mesh.matrix_world @ v.co for v in mesh.data.vertices]
    lower = Vector(tuple(min(v[i] for v in vertices) for i in range(3)))
    upper = Vector(tuple(max(v[i] for v in vertices) for i in range(3)))
    center = (lower + upper) / 2
    # A simple reference seat is preview-only, not included in the person GLB.
    if seated:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, .08, .47))
        seat = bpy.context.object
        seat.name = "preview_only_reference_seat"
        seat.dimensions = (.5, .48, .055)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -.008))
    floor = bpy.context.object
    mat = bpy.data.materials.new("preview_floor")
    mat.diffuse_color = (.2, .24, .29, 1)
    floor.data.materials.append(mat)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.world = bpy.data.worlds.new("preview_world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.6, .65, .75, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .6
    for location, power, size in [((2, -3, 4), 600, 4), ((-3, -1, 2), 250, 3)]:
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy, light.data.shape, light.data.size = power, "DISK", size
        light.rotation_euler = (center-light.location).to_track_quat("-Z", "Y").to_euler()
    location = center + (Vector((2.5, -4, 2)) if seated else Vector((2.4, -2.8, 3)))
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.rotation_euler = (center-camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 1.6 if seated else 2.3
    scene.camera = camera
    scene.render.resolution_x, scene.render.resolution_y = 1000, 1000
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(destination)
    bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--pose", choices=("standing", "seated", "lying"), default="standing")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    root = Path(__file__).resolve().parents[1] / "data/person_search/rocketbox"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(root / "Male_Adult_01.fbx"), use_anim=False)
    print("IMPORT_INSPECTION", json.dumps({
        "objects": [{"name": o.name, "type": o.type, "dimensions": list(o.dimensions),
                     "materials": [m.name if m else None for m in o.data.materials]
                     if o.type == "MESH" else []} for o in bpy.data.objects],
        "materials": [m.name for m in bpy.data.materials],
        "images": [{"name": i.name, "path": i.filepath} for i in bpy.data.images],
        "bones": {o.name: [b.name for b in o.data.bones] for o in bpy.data.objects if o.type == "ARMATURE"},
    }, indent=2), flush=True)
    if args.inspect:
        return
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if len(meshes) != 1 or "hipoly" not in meshes[0].name:
        raise RuntimeError("Unexpected source mesh layout")
    rig = bpy.data.objects["Bip01"]
    original_height = meshes[0].dimensions.z
    front = sum((rig.matrix_world.to_3x3() @
                 (rig.pose.bones[f"Bip01 {s} Toe0"].head-rig.pose.bones[f"Bip01 {s} Foot"].head)
                 for s in ("L", "R")), Vector())
    front.z = 0
    front.normalize()
    print("ANATOMICAL_FORWARD", tuple(front), flush=True)
    # Lower each upper arm in world space to a relaxed standing pose. Preserve
    # the source elbows/fingers; bake deformation before removing the rig.
    for side in ("L", "R"):
        bone = rig.pose.bones[f"Bip01 {side} UpperArm"]
        shoulder = rig.matrix_world @ bone.head
        center = rig.matrix_world @ rig.pose.bones["Bip01 Spine2"].head
        outward = Vector((shoulder.x-center.x, shoulder.y-center.y, 0)).normalized()
        desired_world = Vector((0, 0, -1)) + outward * .16
        desired_local = rig.matrix_world.to_3x3().inverted() @ desired_world
        # FBX bone local Y is not necessarily the anatomical limb direction.
        # Use shoulder-to-elbow joint positions rather than the display bone tail.
        limb = rig.pose.bones[f"Bip01 {side} Forearm"].head - bone.head
        delta = limb.normalized().rotation_difference(desired_local.normalized())
        head = bone.head.copy()
        bone.matrix = Matrix.Translation(head) @ delta.to_matrix().to_4x4() @ Matrix.Translation(-head) @ bone.matrix
        bpy.context.view_layer.update()
        posed_limb = rig.pose.bones[f"Bip01 {side} Forearm"].head - bone.head
        actual_world = (rig.matrix_world.to_3x3() @ posed_limb).normalized()
        if actual_world.dot(desired_world.normalized()) < .99:
            raise RuntimeError(f"Arm pose did not match requested direction: {side}")
    if args.pose == "seated":
        for side in ("L", "R"):
            prefix = f"Bip01 {side} "
            foot = rig.pose.bones[prefix+"Foot"]
            foot_orientation = foot.matrix.copy()
            aim_limb(rig, prefix+"Thigh", prefix+"Calf", front)
            aim_limb(rig, prefix+"Calf", prefix+"Foot", Vector((0, 0, -1)))
            foot_orientation.translation = foot.head.copy()
            foot.matrix = foot_orientation
            bpy.context.view_layer.update()
            aim_limb(rig, prefix+"Forearm", prefix+"Hand", front + Vector((0, 0, -.35)))
    for mat in bpy.data.materials:
        texture_path = root / f"{mat.name}_color.tga"
        if not texture_path.exists():
            raise FileNotFoundError(texture_path)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        shader = nodes.new("ShaderNodeBsdfPrincipled")
        shader.inputs["Metallic"].default_value = 0
        shader.inputs["Roughness"].default_value = .85
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(str(texture_path), check_existing=True)
        mat.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
        if "opacity" in mat.name:
            mat.node_tree.links.new(texture.outputs["Alpha"], shader.inputs["Alpha"])
            mat.surface_render_method = "DITHERED"
        mat.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
        mat.use_backface_culling = False
    bpy.ops.object.select_all(action="DESELECT")
    mesh = meshes[0]
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.convert(target="MESH")
    matrix = mesh.matrix_world.copy()
    mesh.parent = None
    mesh.matrix_world = matrix
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if args.pose != "standing":
        # Preserve the same adult's size: scale by the SOURCE standing height,
        # never by the seated/lying bounding-box height.
        mesh.data.transform(Matrix.Scale(1.7 / original_height, 4))
        if args.pose == "lying":
            mesh.data.transform(front.rotation_difference(Vector((0, 0, 1))).to_matrix().to_4x4())
        min_z = min(v.co.z for v in mesh.data.vertices)
        mid_x = (min(v.co.x for v in mesh.data.vertices)+max(v.co.x for v in mesh.data.vertices))/2
        # Keep seated hips near the origin; center the lying body horizontally.
        mid_y = 0 if args.pose == "seated" else (min(v.co.y for v in mesh.data.vertices)+max(v.co.y for v in mesh.data.vertices))/2
        mesh.data.transform(Matrix.Translation(Vector((-mid_x, -mid_y, -min_z))))
    destination_root = root if args.pose == "standing" else root / "poses"
    destination_root.mkdir(exist_ok=True)
    destination = destination_root / f"Male_Adult_01_{args.pose}.glb"
    bpy.ops.export_scene.gltf(filepath=str(destination), export_format="GLB",
                              use_selection=True, export_animations=False,
                              export_skins=False, export_yup=True)
    # Habitat uses alpha-cutout hair cards. Make the intended material mode
    # explicit while leaving all exported geometry and PNG pixels unchanged.
    raw = destination.read_bytes()
    json_length = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20:20+json_length])
    for mat in gltf.get("materials", []):
        if "opacity" in mat.get("name", ""):
            mat.update(alphaMode="MASK", alphaCutoff=.5, doubleSided=True)
    content = json.dumps(gltf, separators=(",", ":")).encode()
    content += b" " * (-len(content) % 4)
    remaining = raw[20+json_length:]
    destination.write_bytes(struct.pack("<4sII", b"glTF", 2, 20+len(content)+len(remaining)) +
                            struct.pack("<I4s", len(content), b"JSON") + content + remaining)
    report = {"source_repository": "https://github.com/microsoft/Microsoft-Rocketbox",
              "source_commit": "0943055db6ec570bcef9f2c8b41c9e5467c808f9",
              "license": "MIT; Copyright (c) 2020 Microsoft",
              "blender_version": bpy.app.version_string,
              "changes": ["lower upper arms", "bake static mesh", "relink color textures",
                          "hair alpha cutout", "export embedded GLB"],
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(root.iterdir()) if p.suffix in (".fbx", ".tga")},
              "asset_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
              "asset": destination.name, "triangles": len(mesh.data.loop_triangles),
              "pose": args.pose, "source_standing_height_m": original_height,
              "normalized_standing_height_m": 1.7 if args.pose != "standing" else None,
              "blender_z_up_dimensions_m": [max(v.co[i] for v in mesh.data.vertices)-min(v.co[i] for v in mesh.data.vertices) for i in range(3)]}
    report["changes"].extend([] if args.pose == "standing" else [f"pose: {args.pose}", "normalize stature to 1.7 m", "ground contact at z=0 before Y-up export"])
    report_name = "conversion.json" if args.pose == "standing" else f"conversion_{args.pose}.json"
    (destination_root / report_name).write_text(json.dumps(report, indent=2))
    print("CONVERSION_RESULT", json.dumps(report), flush=True)
    if args.preview:
        render_preview(destination, destination_root / f"{args.pose}_preview.png", args.pose == "seated")


if __name__ == "__main__":
    main()
