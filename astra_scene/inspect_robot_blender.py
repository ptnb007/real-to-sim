import bpy
from mathutils import Vector
from pathlib import Path
p=Path(__file__).resolve().parent/'focus_v4/review_robot/v4_robot_review.blend'
bpy.ops.wm.open_mainfile(filepath=str(p))
for c in bpy.data.collections:
    print('COLLECTION',c.name,c.hide_render)
for o in bpy.data.objects:
    if 'base_link' in o.name or o.name=='Root' or 'visual' in o.name and o.type=='MESH':
        coords=[o.matrix_world @ Vector(v) for v in o.bound_box]
        print('ROBOTOBJ',o.name,o.type,o.hide_render,tuple(o.matrix_world.translation),[(min(v[i] for v in coords),max(v[i] for v in coords)) for i in range(3)])
