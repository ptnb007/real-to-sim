from pathlib import Path
import bpy
from mathutils import Vector

root = Path(__file__).resolve().parent.parent
out = root / 'astra_scene/focus_v4/review_robot'
out.mkdir(exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(root / 'astra_scene/focus_v4/focused_zone.blend'))
before = set(bpy.data.objects)
bpy.ops.wm.usd_import(filepath=str(root / 'vendor/galaxea_isaac_tutorial/object/r1pro/r1pro.usda'))
added = set(bpy.data.objects) - before
for obj in added:
    if obj.parent not in added:
        # Robot USD uses a centimeters layer declaration, while geometry values
        # are meter-sized; Blender applies 0.01 on import. Correct preview only.
        obj.scale *= 100
        obj.location += Vector((.25, -.55, 0))
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 24
scene.cycles.use_denoising = True
scene.render.resolution_x = 1200
scene.render.resolution_y = 1000
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
# Restore an overview independent of any camera imported with the robot.
camera = bpy.data.objects.new('RobotReviewCamera', bpy.data.cameras.new('RobotReviewCamera'))
scene.collection.objects.link(camera)
camera.location = (-6, -5, 5.7)
camera.rotation_euler = (Vector((0, 0, .8)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
camera.data.lens = 43
scene.camera = camera
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'v4_robot_review.blend'))
scene.render.filepath = str(out / 'v4_robot_review.png')
bpy.ops.render.render(write_still=True)
print('ROBOT_REVIEW_COMPLETE', len(added), flush=True)
